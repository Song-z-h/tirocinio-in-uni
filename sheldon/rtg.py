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

def reward_to_go(ep_rews):
    n = len(ep_rews)
    rtg = np.zeros_like(ep_rews)
    for i in reversed(range(n)):
        rtg[i] = ep_rews[i] + (rtg[i+1] if i+1 < n else 0)
    return rtg

def train(env_name="CartPole-v0", hidden_sizes=[32], lr=1e-2, 
          epoch=50, batch_size=5000, render=False):
    #define env
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    logit_net = mlp(sizes=[obs_dim]+hidden_sizes+[act_dim])
    optimizer = Adam(logit_net.parameters(), lr=lr)

    def get_policy(obs):
        return Categorical(logits=logit_net(obs))

    def get_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        log_prob = get_policy(obs).log_prob(act)
        return -(log_prob * weights).mean()

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rews = []
        batch_lens = []

        obs = env.reset()
        done = False
        ep_rews = []

        finished_rendering_for_this_epoch = False

        while True:
            if not finished_rendering_for_this_epoch and render:
                env.render()

            batch_obs.append(obs.copy())

            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            ep_rews.append(rew)
            batch_acts.append(act)

            if done:
                total_ep_return, total_ep_len = sum(ep_rews), len(ep_rews)

                batch_rews.append(total_ep_return)
                batch_lens.append(total_ep_len)
                
                batch_weights += list(reward_to_go(ep_rews=ep_rews))
                #batch_weights += [total_ep_return] * total_ep_len
                obs, ep_rews, done = env.reset(), [], False
            
                finished_rendering_for_this_epoch = True
                
                if len(batch_obs) > batch_size:
                    break
        
        
        optimizer.zero_grad()
        loss = compute_loss(torch.as_tensor(batch_obs, dtype=torch.float32),
                                    torch.as_tensor(batch_acts, dtype=torch.int32),
                                    torch.as_tensor(batch_weights, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        return loss, batch_rews, batch_lens
            
    
    for i in range(epoch):
        batch_loss, batch_rews, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t rewards: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.mean(batch_rews), np.mean(batch_lens)))

if __name__ == "__main__":
    train()