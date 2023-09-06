import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym 
from gym.spaces import Discrete, Box

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    n = len(sizes)
    layers = []
    for i in range(n - 1):
        act = activation if i < (n - 2) else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

def reward_to_go(ep_rew): 
    rtg = np.zeros_like(ep_rew)
    for i in reversed(range(len(ep_rew))):
        rtg[i] = ep_rew[i] + (rtg[i+1] if i+1 < len(ep_rew) else 0 )
    return rtg


def train(env_name="CartPole-v0", hidden_sizes=[32], lr=1e-2, epochs=50,
           batch_size=5000, render=False):
    env = gym.make(env_name)
    
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    #make core policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])
    optimizer = Adam(logits_net.parameters(), lr=lr)

    #define function to get the policy distrubution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    #define sample action function
    def get_action(obs):
        return get_policy(obs).sample().item()
    
    #define loss function 
    def compute_loss(obs, act, weights):
        logits_prob = get_policy(obs).log_prob(act)
        return -(logits_prob * weights).mean()


    #training the policy 
    def train_one_epoch():
        batch_obs = []  #for observations
        batch_acts = []  #for actions
        batch_rews = []  #for rewards of an episode
        batch_weights = [] #the one R(tau) we multiply to log probs
        batch_lens = [] # the lengths of an episode

        #reset some env variables
        obs = env.reset()
        done = False
        rews_to_time_t = []

        finished_rendering_this_epoch = False

        while True:
            if not finished_rendering_this_epoch and render:
                env.render()

            #save obs
            batch_obs.append(obs.copy())
            
            #get action
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))

            #do a step in the env
            obs, rew, done, _ =  env.step(act)
            #save action reward
            batch_acts.append(act)
            rews_to_time_t.append(rew)

            if done:
                #train agent 
                ep_return, ep_length = sum(rews_to_time_t), len(rews_to_time_t)
                batch_rews.append(ep_return)
                batch_lens.append(ep_length)

                #calculate weight
                batch_weights += list(reward_to_go(rews_to_time_t))
                obs, done, rews_to_time_t = env.reset(), False, []

                finished_rendering_this_epoch = True

                if len(batch_obs) > batch_size:
                    break

        # take a policy update using policy gradient
        optimizer.zero_grad()
    
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                              act=torch.as_tensor(batch_acts, dtype=torch.int32),
                              weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rews, batch_lens
    
    # The training loop
    for i in range(epochs):
        batch_loss, batch_rews, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t rewards: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.mean(batch_rews), np.mean(batch_lens)))
        

if __name__ == '__main__':
    #import argparse
    train()
            
