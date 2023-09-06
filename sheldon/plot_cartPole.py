import numpy as np
import matplotlib.pyplot as plt

rewards_nondf = np.loadtxt("evpg_10_CartPole-v0.txt")
rewards_gumbel = np.loadtxt("evpg_gumbel_sim_10_CartPole-v0.txt")
rewards_vpg_nondf = np.loadtxt("vpt_10_CartPole-v0.txt")
rewards_gumbel_temp_decay = np.loadtxt("evpg_gumbel_temp_decay_sim_10_CartPole-v0.txt")
rewards_togo = np.loadtxt("rtg_10_CartPole-v0.txt")
rewards_topk = np.loadtxt("temp_evpg_topk_temp_decay_sim_10_CartPole-v0.txt")

# Plot the rewards on the same plot
#plt.plot(rewards_nondf, label='Rewards nondiff')
plt.plot(rewards_gumbel, label='Rewards gumbel')
#plt.plot(rewards_vpg_nondf, label='Rewards vpg')
plt.plot(rewards_gumbel_temp_decay, label='Rewards gumbel temperature decay')
plt.plot(rewards_topk, label='Rewards topk')
#plt.plot(rewards_togo, label='Rewards rtg')
plt.xlabel('Episode/Iteration')
plt.ylabel('Average Rewards')
plt.title('Average Rewards Comparison in CartPole-V0')
plt.legend()
plt.show()

