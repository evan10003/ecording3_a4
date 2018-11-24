import numpy as np
import mdptoolbox as tool
from all_grids import all_info

eps = 0.1

def print_policy(h, w, greens, reds, pol):
    for i in range(h):
        strin = ""
        for j in range(w):
            if [i,j] in greens:
                strin += " g "
            elif [i,j] in reds:
                strin += " r "
            else:
                p = pol[w*i+j]
                if p == 0:
                    strin += " ^ "
                elif p == 1:
                    strin += " v "
                elif p == 2:
                    strin += " < "
                else:
                    strin += " > "
        print(strin)

def policy_dif(h, greens, reds, pol1, pol2):
    sum = 0
    for i in range(h):
        for j in range(h):
            if [i,j] not in greens and [i,j] not in reds:
                p1 = pol1[h*i+j]
                p2 = pol2[h*i+j]
                if p1 != p2:
                    sum += 1
    return sum

hs = [10, 20, 30, 40]
transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids = all_info()

def pi_policy_info(hs, transitions_all_grids, rewards_all_grids):
    pi_policies = []
    for m in range(len(hs)):
        transitions = np.array(transitions_all_grids[m])
        rewards = np.array(rewards_all_grids[m])

        pi = tool.mdp.PolicyIterationModified(transitions, rewards, 0.9, epsilon=0.1, max_iter=1000)
        pi.run()
        pol = pi.policy
        pi_policies.append(pol)

    return pi_policies

def vi_policy_info(hs, transitions_all_grids, rewards_all_grids):
    vi_policies = []
    for m in range(len(hs)):
        transitions = np.array(transitions_all_grids[m])
        rewards = np.array(rewards_all_grids[m])

        vi = tool.mdp.ValueIteration(transitions, rewards, 0.9, epsilon=0.1, max_iter=1000)
        vi.run()
        pol = vi.policy
        vi_policies.append(pol)

    return vi_policies

def pi_vi_times_info(hs, transitions_all_grids, rewards_all_grids):
    pi_iter_times = []
    vi_times = []
    for m in range(len(hs)):
        transitions = np.array(transitions_all_grids[m])
        rewards = np.array(rewards_all_grids[m])

        pi = tool.mdp.PolicyIterationModified(transitions, rewards, 0.9, epsilon=0.1, max_iter=1000)
        pi.run()
        pi_iter_times.append(pi.time)

        vi = tool.mdp.ValueIteration(transitions, rewards, 0.9, epsilon=0.1, max_iter=1000)
        vi.run()
        vi_times.append(vi.time)

    return pi_iter_times, vi_times

if __name__ == "__main__":
    hs = [10, 20, 30, 40]
    transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids = all_info()
    pi_policies = []
    vi_policies = []
    pi_values = []
    vi_values = []
    pi_iterations = []
    vi_iterations = []
    for m in range(len(hs)):
        transitions = np.array(transitions_all_grids[m])
        rewards = np.array(rewards_all_grids[m])
        greens = np.array(greens_all_grids[m])
        reds = np.array(reds_all_grids[m])

        pi = tool.mdp.PolicyIterationModified(transitions, rewards, 0.9, epsilon=0.1, max_iter=1000)
        if m in [0,3]:
            pi.setVerbose()
        pi.run()
        pi_policies.append(pi.policy)
        pi_values.append(pi.V)
        pi_iterations.append(pi.iter)

        vi = tool.mdp.ValueIteration(transitions, rewards, 0.9, epsilon=0.1, max_iter=1000)
        if m in [0,3]:
            vi.setVerbose()
        vi.run()
        vi_policies.append(vi.policy)
        vi_values.append(vi.V)
        vi_iterations.append(vi.iter)

    print("policy iteration policy - small grid")
    print_policy(hs[0], hs[0], greens_all_grids[0], reds_all_grids[0], pi_policies[0])
    print("policy iteration policy - big grid")
    print_policy(hs[3], hs[3], greens_all_grids[3], reds_all_grids[3], pi_policies[3])

    print("number of iterations - policy iteration")
    print(pi_iterations)
    print("number of iterations - value iteration")
    print(vi_iterations)