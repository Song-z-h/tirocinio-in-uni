import numpy as np
import matplotlib.pyplot as plt

#rewards_nondf = np.loadtxt("average_evpg_nondf_sim_ 10_LunarLander-v2.txt")
rewards_gumbel = np.loadtxt("evpg_gumbel_sim_10_LunarLander-v2.txt")
rewards_vpg_nondf = np.loadtxt("vpt_10_LunarLander-v2.txt")
rewards_rtg = np.loadtxt("rtg_10_LunarLander-v2.txt")
rewards_gumbel_temp_decay = np.loadtxt("temp_evpg_gumbel_temp_decay_sim_10_LunarLander-v2.txt")

# Plot the rewards on the same plot
#plt.plot(rewards_nondf, label='Rewards nondiff')
plt.plot(rewards_gumbel, label='Rewards gumbel')
#plt.plot(rewards_vpg_nondf, label='Rewards vpg')
#plt.plot(rewards_rtg, label='Rewards rtg')
plt.plot(rewards_gumbel_temp_decay, label='Rewards gumbel temp decay')
plt.xlabel('Episode/Iteration')
plt.ylabel('Average Rewards')
plt.title('Average Rewards Comparison in LunarLander-v2')
plt.legend()
plt.show()

