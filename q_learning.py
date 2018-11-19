import numpy as np
from all_grids import all_info
from all_policies import policy_dif, pi_policy_info
import mdptoolbox as tool
import matplotlib.pyplot as plt
import time
import copy

def print_q_policy(h, greens, reds, pol):
    for i in range(h):
        strin = ""
        for j in range(h):
            if [i,j] in greens:
                strin += " g "
            elif [i,j] in reds:
                strin += " r "
            else:
                #print(pol[w*i+j,:])
                p = np.argmax(pol[h*i+j,:])
                #print(p)
                if p == 0:
                    strin += " ^ "
                elif p == 1:
                    strin += " v "
                elif p == 2:
                    strin += " < "
                else:
                    strin += " > "
        print(strin)


def q_learning_iter(T, R, Q, s, gamma, alpha_t, h, l, greens, reds, prob=0.2):
    r = R[s]
    s_y = s//h
    s_x = s%h
    if [s_y,s_x] in greens or [s_y,s_x] in reds:
        for a in range(4):
            Q[s,a] = r
        while True:
            s = np.random.randint(0,l)
            s_y = s//h
            s_x = s%h
            if [s_y,s_x] not in greens and [s_y,s_x] not in reds:
                break
        return Q, s
    a = np.argmax(Q[s])
    if np.random.random() < prob:
        a = np.random.randint(0,4)
    s_new = np.random.choice(range(len(T[a][s])), 1, p=T[a][s])
    s_new = s_new[0]
    x = r + gamma*np.amax(Q[s_new])
    Q[s,a] = (1-alpha_t)*Q[s,a] + alpha_t*x
    return Q, s_new

transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids = all_info()


def run_q_learn(hs, transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids, prob=0.2, alpha=0.9, q0_term=2):
    ql_times = []
    grid_squares = []
    ql_iterations = []
    all_sum_q_values = []
    all_Q = []

    for idx in range(4):
        T = transitions_all_grids[idx]
        R = rewards_all_grids[idx]
        greens = greens_all_grids[idx]
        reds = reds_all_grids[idx]
        h = hs[idx]
        l = h**2
        print("grid size", l)

        start = time.time()
        Q = q0_term+np.random.random((l,4))
        s = None
        while True:
            s = np.random.randint(0,l)
            s_y = s//h
            s_x = s%h
            if [s_y,s_x] not in greens and [s_y,s_x] not in reds:
                break

        sum_q_values = []
        iteration = 0
        prev_sum = np.sum(Q)
        new_sum = np.sum(Q)
        for _ in range(100000):
            if iteration%2000==0 and iteration!=0:
                if np.absolute(new_sum-prev_sum)<50:
                    break
                prev_sum = np.sum(Q)
            Q, s = q_learning_iter(T, R, Q, s, 0.9, alpha, h, l, greens, reds, prob=prob)
            new_sum = np.sum(Q)
            sum_q_values.append(new_sum)

            iteration += 1

        ql_times.append(time.time()-start)
        ql_iterations.append(len(sum_q_values))
        all_sum_q_values.append(sum_q_values)
        grid_squares.append(l)
        all_Q.append(np.array(copy.deepcopy(Q)))
    return all_Q, all_sum_q_values, ql_times, ql_iterations

# for p in range(0, 11):
#     prob = float(p)/10
#     hs = [10,24,33,40]

#     mult_all_Q = []
#     mult_all_sum_q_values = []
#     mult_ql_times = []
#     mult_ql_iterations = []
#     mult_difs = []
#     for _ in range(10):
#         all_Q, all_sum_q_values, ql_times, ql_iterations = run_q_learn(hs, transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids, prob=prob)

#         pi_policies = pi_policy_info(hs, transitions_all_grids, rewards_all_grids)

#         for i in range(4):
#             pi_policies[i] = np.array(pi_policies[i])
#             all_Q[i] = np.argmax(all_Q[i], axis=1)

#         difs = []
#         for i in range(4):
#             difs.append(policy_dif(hs[i], greens_all_grids[i], reds_all_grids[i], pi_policies[i], all_Q[i]))
#         mult_all_Q.append(all_Q)
#         mult_all_sum_q_values.append(all_sum_q_values)
#         mult_ql_times.append(ql_times)
#         mult_ql_iterations.append(ql_iterations)
#         mult_difs.append(difs)

#     for m in range(10):
#         #print(difs)
#         #print([1-float(dif)/(h**2) for h, dif in zip(hs, difs)])
#         plt.plot([h**2 for h in hs], [float(h**2-dif)/(h**2) for h, dif in zip(hs, mult_difs[m])], linestyle='-', marker='o')
#     plt.title("proportion of QL and PI/VI policy agreement - random prob "+str(prob))
#     plt.xlabel("grid size")
#     plt.ylabel("proportion of grid size")
#     plt.legend()
#     plt.savefig("ql_vs_pi_policies_pequals0point"+str(int(prob*10)))
#     plt.clf()

#     # for m in range(10):
#     #     for i in range(4):
#     #         plt.plot(range(ql_iterations[i]), all_sum_q_values[i], label='grid '+str(hs[i])+'x'+str(hs[i]))
#     #     plt.title("sum of q values vs iterations - random prob "+str(prob))
#     #     plt.xlabel("iteration")
#     #     plt.ylabel("sum of q values")
#     #     plt.legend()
#     #     plt.savefig("sum_q_value_curves_pequals0point"+str(int(prob*10)))
#     #     plt.clf()

#     for m in range(10):
#         plt.plot([h**2 for h in hs], mult_ql_times[m], linestyle='-', marker='o')
#     plt.title("Q learning times - random prob "+str(prob))
#     plt.xlabel("grid size")
#     plt.ylabel("time (sec)")
#     plt.savefig("ql_times_pequals0point"+str(int(prob*10)))
#     plt.clf()

#     for m in range(10):
#         plt.plot([h**2 for h in hs], mult_ql_iterations[m], linestyle='-', marker='o')
#     plt.title("Q learning iterations - random prob "+str(prob))
#     plt.xlabel("grid size")
#     plt.ylabel("time (sec)")
#     plt.savefig("ql_iter_pequals0point"+str(int(prob*10)))
#     plt.clf()

for q in range(-2, 3):
    hs = [10,24,33,40]

    mult_all_Q = []
    mult_all_sum_q_values = []
    mult_ql_times = []
    mult_ql_iterations = []
    mult_difs = []
    for _ in range(10):
        all_Q, all_sum_q_values, ql_times, ql_iterations = run_q_learn(hs, transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids, q0_term=q)

        pi_policies = pi_policy_info(hs, transitions_all_grids, rewards_all_grids)

        for i in range(4):
            pi_policies[i] = np.array(pi_policies[i])
            all_Q[i] = np.argmax(all_Q[i], axis=1)

        difs = []
        for i in range(4):
            difs.append(policy_dif(hs[i], greens_all_grids[i], reds_all_grids[i], pi_policies[i], all_Q[i]))
        mult_all_Q.append(all_Q)
        mult_all_sum_q_values.append(all_sum_q_values)
        mult_ql_times.append(ql_times)
        mult_ql_iterations.append(ql_iterations)
        mult_difs.append(difs)

    for m in range(10):
        plt.plot([h**2 for h in hs], [float(h**2-dif)/(h**2) for h, dif in zip(hs, mult_difs[m])], linestyle='-', marker='o')
    plt.title("proportion of QL and PI/VI policy agreement - q0 addition "+str(q))
    plt.xlabel("grid size")
    plt.ylabel("proportion of grid size")
    plt.legend()
    plt.savefig("ql_vs_pi_policies_q0_"+str(q))
    plt.clf()

    for m in range(10):
        plt.plot([h**2 for h in hs], mult_ql_times[m], linestyle='-', marker='o')
    plt.title("Q learning times - q0 addition "+str(q))
    plt.xlabel("grid size")
    plt.ylabel("time (sec)")
    plt.savefig("ql_times_q0_"+str(q))
    plt.clf()

    for m in range(10):
        plt.plot([h**2 for h in hs], mult_ql_iterations[m], linestyle='-', marker='o')
    plt.title("Q learning iterations - q0 addition "+str(q))
    plt.xlabel("grid size")
    plt.ylabel("time (sec)")
    plt.savefig("ql_iter_q0_"+str(q))
    plt.clf()




