import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym 
from gym.spaces import Discrete, Box
from spinup.utils.mpi_tools import mpi_statistics_scalar, proc_id
import torch.nn.functional as F

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

def to_tensor(tensor, dtype=torch.float32):
    return torch.as_tensor(tensor, dtype=dtype)

def print_weights(model):
    for name, param in model.named_parameters():
        if hasattr(param, 'data'):
            print(f"Layer: {name}, Weights: {param.data}")

def reward_to_go(ep_rews, gamma=0.99):
    n = len(ep_rews)
    rtg = np.zeros_like(ep_rews, dtype=np.float32)
    for i in reversed(range(n)):
        rtg[i] = ep_rews[i] + (gamma * rtg[i + 1] if i+1 < n else 0)
    return rtg

class EVPG(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, temperature):
        super().__init__()
        self.temperature = temperature
        self.policy_net = mlp(sizes=[obs_dim]+hidden_sizes+[act_dim])
        self.value_net = value_mlp(input_size= (1+obs_dim), hidden_size=hidden_sizes[0])


    def sample_gumbel(self, shape, eps=1e-20):
        u = torch.rand(shape)
        return -torch.log(-torch.log(u + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        gumbel_noise = self.sample_gumbel(logits.shape)
        y = logits + gumbel_noise.detach()
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        return self.gumbel_softmax_sample(logits, temperature)

    def get_action(self, obs):
        return self.get_policy(obs).sample().item()
        #return torch.argmax(logits).item()
    
    def get_action_gumbel_softmax(self, obs):
        logits = self.get_logits(obs)
        return self.gumbel_softmax(logits, self.temperature)
        

    def get_logits(self, obs):
        return self.policy_net(obs)
    
    def get_policy(self, obs):
        logits = self.get_logits(obs)
        return Categorical(logits=logits)
    
    def get_value(self, obs_act):
        #return self.value_net(obs_act)
        Q = self.forward(None, obs_act=obs_act)
        return Q
    
    def forward(self, obs, obs_act=None):
        # this action is not differentiable
        #act_prob = self.get_action_gumbel_softmax(obs)
        #act_prob = self.get_logits(obs)
        
        #act = self.get_action(act_prob)
        if obs_act is None:
            act = self.get_action(obs)
            obs_act = to_tensor(obs.tolist() + [act])
        Q = self.value_net(obs_act)
        return Q


def train(env_name="CartPole-v0", hidden_sizes=[32], lr=1e-2, 
          epoch=50, batch_size=5000, render=False, k_step_value_steps=30,
            lamda=0.97, gamma=0.99, temperature=1, seed=0, temp_decay=False):
    
    total_rewards = []

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    #define env
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    #non end to end
    #logit_net = mlp(sizes=[obs_dim]+hidden_sizes+[act_dim])
    #optimizer = Adam(logit_net.parameters(), lr=lr)
    #value_net = value_mlp(input_size=obs_dim, hidden_size=hidden_sizes[0])
    #value_optimizer = Adam(value_net.parameters(), lr=lr) 

    evpg_net = EVPG(obs_dim, act_dim, hidden_sizes, temperature)
    optimizer = Adam(evpg_net.parameters(), lr=lr)

    def compute_policy_loss(obs, act, weights):
        log_prob = evpg_net.get_policy(obs).log_prob(act)
        return -(log_prob * weights).mean()

    def compute_value_loss(obs, R,):
        Q = evpg_net.get_value(obs)
        return ((Q - R)**2).mean()
    
    def compute_evgp_loss(obs, act, adv, weights, obs_acts):
        policy_loss = compute_policy_loss(obs, act, adv)
        value_loss = compute_value_loss(obs_acts, weights)
        return value_loss, policy_loss, value_loss

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rews = []
        batch_lens = []
        batch_obs_acts = []

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

            val = evpg_net(to_tensor(obs))
            act = evpg_net.get_action(to_tensor(obs))
            obs_act = obs.tolist() + [act]
            batch_obs_acts.append(obs_act)
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
                ep_vals += [evpg_net.get_value(to_tensor(obs_act)).item()]
                deltas = np.array(ep_rews) + gamma * np.array(ep_vals[1:]) - np.array(ep_vals[:-1])
                batch_advs += list(reward_to_go(deltas, gamma * lamda))
                #reset buffers
                obs, ep_rews, ep_vals, done = env.reset(), [], [], False
            
                finished_rendering_for_this_epoch = True
                
                if len(batch_obs) > batch_size:
                    break

        
        adv_mean, adv_std = mpi_statistics_scalar(batch_advs)
        batch_advs = (batch_advs - adv_mean) / adv_std
        adv_mean, adv_std = mpi_statistics_scalar(batch_weights)
        batch_weights = (batch_weights - adv_mean) / adv_std

        optimizer.zero_grad()
        loss, p_loss, v_loss = compute_evgp_loss(to_tensor(batch_obs),
                                    to_tensor(batch_acts, torch.int32),
                                    to_tensor(batch_advs),
                                    to_tensor(batch_weights),
                                    to_tensor(batch_obs_acts))
        loss.backward()
        for name, param in evpg_net.named_parameters():
            print(f"Layer: {name}, Gradients: {param.grad}")
        optimizer.step()
        return loss, batch_rews, batch_lens
            
    temperature_decay = temperature / epoch
    for i in range(epoch):
        batch_loss, batch_rews, batch_lens = train_one_epoch()
        total_rewards += [np.mean(batch_rews)]
        if temp_decay is True:
            temperature -= temperature_decay if temperature > temperature_decay else 0
        print('epoch: %3d \t loss: %.3f \t rewards: %.3f \t ep_len: %.3f \t temp: %.3f '  %
              (i, batch_loss, np.mean(batch_rews), np.mean(batch_lens), temperature))
        
    return total_rewards

if __name__ == "__main__":
    num_simulations = 1
    num_epoch = 50
    #env_name='LunarLander-v2'
    env_name="CartPole-v0"
    #file_name = f'temp_evpg_topk_temp_decay_sim_{num_simulations}_{env_name}.txt'
    all_rewards = np.empty((num_simulations, num_epoch))

    for i in range(num_simulations):
        all_rewards[i, :] = train(env_name=env_name, render=False, 
                                  k_step_value_steps=1, epoch=num_epoch, 
                                  hidden_sizes=[4], temperature=0.5, temp_decay=True)
    average_rewards = np.mean(all_rewards, axis=0)
    #np.savetxt(file_name, average_rewards)
    #np.savetxt('temp_' + file_name, temperatures)