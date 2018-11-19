import matplotlib.pyplot as plt
from all_policies import pi_vi_times_info
from all_grids import all_info
import numpy as np
import pandas as pd

hs = [10, 24, 33, 40]
grid_sizes = [h**2 for h in hs]
transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids = all_info()

for _ in range(10):
    pi_iter_times, vi_times = pi_vi_times_info(hs, transitions_all_grids, rewards_all_grids)
    print(pi_iter_times)
    plt.plot(grid_sizes, pi_iter_times, linestyle='-', marker='o')
plt.ylabel("time (sec)")
plt.xlabel("grid size")
plt.title("policy iteration times - multiple runs")
plt.legend()
plt.savefig("pi times")
plt.clf()

for _ in range(10):
    pi_iter_times, vi_times = pi_vi_times_info(hs, transitions_all_grids, rewards_all_grids)
    plt.plot(grid_sizes, vi_times, linestyle='-', marker='o')
plt.ylabel("time (sec)")
plt.xlabel("grid size")
plt.title("value iteration times - multiple runs")
plt.savefig("vi times")
plt.clf()


# df_vi_small = pd.read_table("vi_iters_small_grid.txt")
# df_vi_big = pd.read_table("vi_iters_big_grid.txt")
# df_pi_small = pd.read_table("pi_iters_small_grid.txt")
# df_pi_big = pd.read_table("pi_iters_big_grid.txt")

# vi_small = df_vi_small.values
# vi_big = df_vi_big.values
# pi_small = df_pi_small.values
# pi_big = df_pi_big.values

# plt.plot([x+1 for x in range(len(vi_small))], vi_small, label='value iteration')
# plt.plot([x+1 for x in range(len(pi_small))], pi_small, label='policy iteration')
# plt.ylabel("value variation")
# plt.xlabel("iteration")
# plt.legend()
# plt.title("Value Variation per iteration - small grid")
# plt.savefig("value_var_small_grid")
# plt.clf()

# plt.plot([x+1 for x in range(len(vi_small))], vi_small, label='value iteration')
# plt.plot([x+1 for x in range(len(pi_small))], pi_small, label='policy iteration')
# plt.ylabel("value variation")
# plt.xlabel("iteration")
# plt.legend()
# plt.title("Value Variation per iteration - big grid")
# plt.savefig("value_var_big_grid")
# plt.clf()
