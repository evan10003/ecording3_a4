import matplotlib.pyplot as plt
from all_policies import policy_info

hs = [10, 24, 33, 40]
grid_sizes = [h**2 for h in hs]

pi_box = [[], [], [], []]
# for _ in range(25):
#     pi_mod_times, pi_iter_times, pi_matrix_times, vi_times, pi_mod_iters, pi_iter_iters, pi_matrix_iters, vi_iters = policy_info()
#     plt.plot(grid_sizes, pi_iter_times, linestyle='-', marker='o')
# plt.ylabel("time (sec)")
# plt.xlabel("grid size")
# plt.title("policy iteration times - multiple runs")
# plt.legend()
# plt.savefig("pi times")
# plt.clf()

# for _ in range(25):
#     pi_mod_times, pi_iter_times, pi_matrix_times, vi_times, pi_mod_iters, pi_iter_iters, pi_matrix_iters, vi_iters = policy_info()
#     plt.plot(grid_sizes, vi_times, linestyle='-', marker='o')
# plt.ylabel("time (sec)")
# plt.xlabel("grid size")
# plt.title("value iteration times - multiple runs")
# plt.savefig("vi times")
# plt.clf()
