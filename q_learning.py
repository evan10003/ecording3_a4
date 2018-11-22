import numpy as np
from all_grids import all_info
from all_policies import policy_dif, pi_policy_info
import mdptoolbox as tool
import matplotlib.pyplot as plt
import time
import copy

def print_policy_dif(h, greens, reds, ql, pi):
    for i in range(h):
        strin = ""
        for j in range(h):
            if [i,j] in greens:
                strin += " g "
            elif [i,j] in reds:
                strin += " r "
            else:
                ql_p = ql[h*i+j]
                pi_p = pi[h*i+j]
                if ql_p == pi_p:
                    p = ql_p
                    if p == 0:
                        strin += " ^ "
                    elif p == 1:
                        strin += " v "
                    elif p == 2:
                        strin += " < "
                    else:
                        strin += " > "
                else:
                    strin += " x "
        print(strin)


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
                p = np.argmax(pol[h*i+j,:], axis=1)
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


def q_learning_anneal_iter(T, R, Q, s, gamma, alpha, h, greens, reds, temp):
    l = h**2
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
    a = np.random.randint(0,4)
    prev_q = Q[s,a]
    s_new = np.random.choice(range(len(T[a][s])), 1, p=T[a][s])
    s_new = s_new[0]
    x = r + gamma*np.amax(Q[s_new])
    new_q = (1-alpha)*Q[s,a] + alpha*x
    if new_q >= prev_q:
        Q[s,a] = new_q
        return Q, s_new
    elif np.random.random() < np.e**((new_q-prev_q)/temp):
        Q[s,a] = new_q
        return Q, s_new
    return Q, s


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
        last_mean = None
        vals = []
        while iteration<300000:
            if iteration%1000==0 and iteration!=0:
                #print(iteration, prev_sum-new_sum)
                sum_q_values.append(np.absolute(prev_sum-new_sum))
                if np.absolute(prev_sum-new_sum) < 7:
                    break
                prev_sum = new_sum
            Q, s = q_learning_iter(T, R, Q, s, 0.9, alpha, h, l, greens, reds, prob=prob)
            new_sum = np.sum(Q)
            # if iteration%100==0 and iteration!=0:
            #     val = np.absolute(prev_sum-new_sum)
            #     #print(iteration//1000, val)
            #     sum_q_values.append(val)
            #     if val < 7:
            #         break
            #     prev_sum = new_sum
            #     vals.append(prev_sum)
            # if iteration==500:
            #     last_mean = np.mean(np.array(vals))
            # elif iteration%500==0 and iteration!=0:
            #     new_mean = np.mean(np.array(vals))
            #     if new_mean >= last_mean:
            #         break
            #     last_mean = new_mean
            #     vals = []
            iteration += 1

        ql_times.append(time.time()-start)
        ql_iterations.append(len(sum_q_values))
        all_sum_q_values.append(sum_q_values)
        grid_squares.append(l)
        all_Q.append(np.array(copy.deepcopy(Q)))
    return all_Q, all_sum_q_values, ql_times, ql_iterations

# for p in range(0, 11):
#     prob = float(p)/10
#     hs = [10,20,30,40]

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
#         plt.plot([h**2 for h in hs], [float(h**2-dif)/(h**2) for h, dif in zip(hs, mult_difs[m])], linestyle='-', marker='o')
#     plt.title("proportion of QL and PI/VI policy agreement - random prob "+str(prob))
#     plt.xlabel("grid size")
#     plt.ylabel("proportion of grid size")
#     plt.legend()
#     plt.savefig("ql_vs_pi_policies_pequals0point"+str(int(prob*10)))
#     plt.clf()

#     if p == 2:
#         for i in range(4):
#             plt.plot(range(len(mult_all_sum_q_values[0][i])), mult_all_sum_q_values[0][i], linestyle='-', label='grid '+str(hs[i])+'x'+str(hs[i]))
#         plt.title("Q Value Variation")
#         plt.xlabel("iteration")
#         plt.ylabel("difference in Q values")
#         plt.legend()
#         plt.savefig("sum_q_value_curves2")
#         plt.clf()

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

# for q in [3]:
#     hs = [10,20,30,40]

#     mult_all_Q = []
#     mult_all_sum_q_values = []
#     mult_ql_times = []
#     mult_ql_iterations = []
#     mult_difs = []
#     for _ in range(10):
#         all_Q, all_sum_q_values, ql_times, ql_iterations = run_q_learn(hs, transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids, q0_term=q)

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
#         plt.plot([h**2 for h in hs], [float(h**2-dif)/(h**2) for h, dif in zip(hs, mult_difs[m])], linestyle='-', marker='o')
#     plt.title("proportion of QL and PI/VI policy agreement - q0 addition "+str(q))
#     plt.xlabel("grid size")
#     plt.ylabel("proportion of grid size")
#     plt.legend()
#     plt.savefig("ql_vs_pi_policies_q0_"+str(q))
#     plt.clf()

#     for m in range(10):
#         plt.plot([h**2 for h in hs], mult_ql_times[m], linestyle='-', marker='o')
#     plt.title("Q learning times - q0 addition "+str(q))
#     plt.xlabel("grid size")
#     plt.ylabel("time (sec)")
#     plt.savefig("ql_times_q0_"+str(q))
#     plt.clf()

#     for m in range(10):
#         plt.plot([h**2 for h in hs], mult_ql_iterations[m], linestyle='-', marker='o')
#     plt.title("Q learning iterations - q0 addition "+str(q))
#     plt.xlabel("grid size")
#     plt.ylabel("time (sec)")
#     plt.savefig("ql_iter_q0_"+str(q))
#     plt.clf()


hs = [10, 20, 30, 40]

mult_all_Q = []
mult_all_sum_q_values = []
mult_ql_times = []
mult_ql_iterations = []
mult_difs = []
for _ in range(10):
    all_Q, all_sum_q_values, ql_times, ql_iterations = run_q_learn(hs, transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids)
    pi_policies = pi_policy_info(hs, transitions_all_grids, rewards_all_grids)

    for i in range(4):
        pi_policies[i] = np.array(pi_policies[i])
        #all_Q[i] = np.argmax(all_Q[i], axis=1)

    difs = []
    for i in range(4):
        difs.append(policy_dif(hs[i], greens_all_grids[i], reds_all_grids[i], pi_policies[i], np.argmax(all_Q[i], axis=1)))
    mult_all_Q.append(all_Q)
    mult_all_sum_q_values.append(all_sum_q_values)
    mult_ql_times.append(ql_times)
    mult_ql_iterations.append(ql_iterations)
    mult_difs.append(difs)

for m in range(10):
    plt.plot([h**2 for h in hs], [float(h**2-dif)/(h**2) for h, dif in zip(hs, mult_difs[m])], linestyle='-', marker='o')
plt.title("proportion of QL and PI/VI policy agreement")
plt.xlabel("grid size")
plt.ylabel("proportion of grid size")
plt.legend()
plt.savefig("ql_vs_pi_policies_initial")
plt.clf()

for m in range(10):
    plt.plot([h**2 for h in hs], mult_ql_times[m], linestyle='-', marker='o')
plt.title("Q learning times")
plt.xlabel("grid size")
plt.ylabel("time (sec)")
plt.savefig("ql_times_initial")
plt.clf()

for m in range(10):
    plt.plot([h**2 for h in hs], mult_ql_iterations[m], linestyle='-', marker='o')
plt.title("Q learning iterations")
plt.xlabel("grid size")
plt.ylabel("iterations")
plt.savefig("ql_iterations_initial")
plt.clf()

for i in range(4):
    plt.plot(range(len(mult_all_sum_q_values[0][i])), mult_all_sum_q_values[0][i], linestyle='-', label='grid '+str(hs[i])+'x'+str(hs[i]))
plt.title("Q Value Variation")
plt.xlabel("iteration")
plt.ylabel("difference in Q values")
plt.legend()
plt.savefig("sum_q_value_curves_initial")
plt.clf()

#print_policy_dif(hs[0], greens_all_grids[0], reds_all_grids[0], np.argmax(all_Q[0], axis=1), pi_policies[0])
#print_policy_dif(hs[3], greens_all_grids[3], reds_all_grids[3], np.argmax(all_Q[3], axis=1), pi_policies[3])

# print_q_policy(hs[0], greens_all_grids[0], reds_all_grids[0], all_Q[0])
# print_q_policy(hs[3], greens_all_grids[3], reds_all_grids[3], all_Q[3])

