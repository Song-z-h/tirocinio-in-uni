import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym 
from gym.spaces import Discrete, Box
from spinup.utils.mpi_tools import mpi_statistics_scalar

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    n = len(sizes)
    layers = []
    for i in range(n - 1):
        act = activation if i < (n - 2) else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

def value_mlp(input_size, hidden_size, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    layers += [nn.Linear(input_size, hidden_size), activation()]
    layers += [nn.Linear(hidden_size, 1), output_activation()]
    return nn.Sequential(*layers)

def reward_to_go(ep_rews, gamma=0.99):
    n = len(ep_rews)
    rtg = np.zeros_like(ep_rews)
    for i in reversed(range(n)):
        rtg[i] = ep_rews[i] + (gamma * rtg[i + 1] if i+1 < n else 0)
    return rtg

def train(env_name="CartPole-v0", hidden_sizes=[32], lr=1e-2, 
          epoch=50, batch_size=5000, render=False, k_step_value_steps=30,
            lamda=0.97, gamma=0.99):
    #define env
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    logit_net = mlp(sizes=[obs_dim]+hidden_sizes+[act_dim])
    optimizer = Adam(logit_net.parameters(), lr=lr)
    value_net = value_mlp(input_size=obs_dim, hidden_size=hidden_sizes[0])
    value_optimizer = Adam(value_net.parameters(), lr=lr) 

   

    def get_policy(obs):
        return Categorical(logits=logit_net(obs))

    def get_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        log_prob = get_policy(obs).log_prob(act)
        return -(log_prob * weights).mean()
    
    def get_value(obs):
        # get the value estimate from the value function
        Q = value_net(obs)
        return Q

    def compute_value_loss(obs, R):
        value_func = get_value(obs)
        return ((value_func - R)**2).mean()

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rews = []
        batch_lens = []

        #gae estimate buffer
        ep_vals = []
        batch_advs = []

        obs = env.reset()
        done = False
        ep_rews = []

        finished_rendering_for_this_epoch = False

        while True:
            if not finished_rendering_for_this_epoch and render:
                env.render()

            batch_obs.append(obs.copy())

            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            val = get_value(torch.as_tensor(obs, dtype=torch.float32)).item()
            
            obs, rew, done, _ = env.step(act)

            ep_vals.append(val)
            ep_rews.append(rew)
            batch_acts.append(act)

            if done:
                total_ep_return, total_ep_len = sum(ep_rews), len(ep_rews)
                batch_rews.append(total_ep_return)
                batch_lens.append(total_ep_len)
                batch_weights += list(reward_to_go(ep_rews))
                #batch_weights += [total_ep_return] * total_ep_len
                #= −V (st) + rt + γV (st+1)
                ep_vals += [get_value(torch.as_tensor(obs, dtype=torch.float32)).item()]
                deltas = np.array(ep_rews) + gamma * np.array(ep_vals[1:]) - np.array(ep_vals[:-1])
                batch_advs += list(reward_to_go(deltas, gamma * lamda))
                #reset buffers
                obs, ep_rews, ep_vals, done = env.reset(), [], [], False
            
                finished_rendering_for_this_epoch = True
                
                if len(batch_obs) > batch_size:
                    break

        
        adv_mean, adv_std = mpi_statistics_scalar(batch_advs)
        batch_advs = (batch_advs - adv_mean) / adv_std


        optimizer.zero_grad()
        loss = compute_loss(torch.as_tensor(batch_obs, dtype=torch.float32),
                                    torch.as_tensor(batch_acts, dtype=torch.int32),
                                    torch.as_tensor(batch_advs, dtype=torch.float32))
        loss.backward()
        optimizer.step()

        for i in range(k_step_value_steps):
            value_optimizer.zero_grad()
            v_loss = compute_value_loss(torch.as_tensor(batch_obs, dtype=torch.float32),
                                        torch.as_tensor(batch_weights, dtype=torch.float32))
            v_loss.backward()
            value_optimizer.step()

        return loss, batch_rews, batch_lens
            
    
    for i in range(epoch):
        batch_loss, batch_rews, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t rewards: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.mean(batch_rews), np.mean(batch_lens)))

if __name__ == "__main__":
    train(render=False, k_step_value_steps=1, epoch=50)