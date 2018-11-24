import matplotlib.pyplot as plt
from all_policies import pi_vi_times_info, print_policy, pi_policy_info, vi_policy_info
from all_grids import all_info
import numpy as np
import pandas as pd

hs = [10, 20, 30, 40]
grid_sizes = [h**2 for h in hs]
transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids = all_info()

# Policy iteration times

for _ in range(10):
    pi_iter_times, vi_times = pi_vi_times_info(hs, transitions_all_grids, rewards_all_grids)
    #print(pi_iter_times)
    plt.plot(grid_sizes, pi_iter_times, linestyle='-', marker='o')
plt.ylabel("time (sec)")
plt.xlabel("grid size")
plt.title("policy iteration times - multiple runs")
plt.legend()
plt.savefig("pi times")
plt.clf()

# Value iteration times

for _ in range(10):
    pi_iter_times, vi_times = pi_vi_times_info(hs, transitions_all_grids, rewards_all_grids)
    plt.plot(grid_sizes, vi_times, linestyle='-', marker='o')
plt.ylabel("time (sec)")
plt.xlabel("grid size")
plt.title("value iteration times - multiple runs")
plt.savefig("vi times")
plt.clf()

# pi_policies = pi_policy_info(hs, transitions_all_grids, rewards_all_grids)
# vi_policies = vi_policy_info(hs, transitions_all_grids, rewards_all_grids)
# for m in range(4):
#     print_policy(hs[m], hs[m], greens_all_grids[m], reds_all_grids[m], pi_policies[m])
#     print_policy(hs[m], hs[m], greens_all_grids[m], reds_all_grids[m], vi_policies[m])


# Policy and value function variation per iteration

df_vi_small = pd.read_table("vi_iters_small_grid.txt", header=None)
df_vi_big = pd.read_table("vi_iters_big_grid.txt", header=None)
df_pi_small = pd.read_table("pi_iters_small_grid.txt", header=None)
df_pi_big = pd.read_table("pi_iters_big_grid.txt", header=None)

vi_small = df_vi_small.values
vi_big = df_vi_big.values
pi_small = df_pi_small.values
pi_big = df_pi_big.values

plt.plot([x+1 for x in range(len(vi_small))], vi_small, label='small grid')
plt.plot([x+1 for x in range(len(vi_big))], vi_big, label='big grid')
plt.ylabel("value variation")
plt.xlabel("iteration")
plt.legend()
plt.title("Variation of Value Function per iteration")
plt.savefig("value_var_all_grids")
plt.clf()

plt.plot([x+1 for x in range(len(pi_small))], pi_small, label='small grid')
plt.plot([x+1 for x in range(len(pi_big))], pi_big, label='big grid')
plt.ylabel("number of different actions")
plt.xlabel("iteration")
plt.legend()
plt.title("Number of different actions per iteration")
plt.savefig("policy_var_all_grids")
plt.clf()
