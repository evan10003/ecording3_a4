import numpy as np
import mdptoolbox as tool
from all_grids import all_info

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

hs = [10, 24, 33, 40]
transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids = all_info()

def pi_policy_info(hs, transitions_all_grids, rewards_all_grids):
    pi_policies = []
    for m in range(len(hs)):
        transitions = np.array(transitions_all_grids[m])
        rewards = np.array(rewards_all_grids[m])

        pi = tool.mdp.PolicyIterationModified(transitions, rewards, 0.9, epsilon=0.0001)
        pi.run()
        pol = pi.policy
        pi_policies.append(pol)

    return pi_policies


def pi_vi_times_info(hs, transitions_all_grids, rewards_all_grids):
    pi_iter_times = []
    vi_times = []
    for m in range(len(hs)):
        transitions = np.array(transitions_all_grids[m])
        rewards = np.array(rewards_all_grids[m])

        print("policy iteration iterative")
        pi = tool.mdp.PolicyIterationModified(transitions, rewards, 0.9, epsilon=0.001)
        pi.run()
        pi_iter_times.append(pi.time)
        print(pi.iter)

        print("value iteration")
        vi = tool.mdp.ValueIteration(transitions, rewards, 0.9, epsilon=0.001)
        vi.run()
        vi_times.append(vi.time)
        print(vi.iter)

    return pi_iter_times, vi_times


def pi_vi_iter_info(hs, transitions_all_grids, rewards_all_grids):
    pi_iters = []
    vi_iters = []
    for m in range(len(hs)):
        transitions = np.array(transitions_all_grids[m])
        rewards = np.array(rewards_all_grids[m])

        pi = tool.mdp.PolicyIterationModified(transitions, rewards, 0.9, epsilon=0.001)
        pi.run()
        pi_iters.append(pi.iter)

        vi = tool.mdp.ValueIteration(transitions, rewards, 0.9, epsilon=0.001)
        vi.run()
        vi_iters.append(vi.iter)

    return pi_iters, vi_iters


hs = [10, 24, 33, 40]

def value_function_info():
    pi_V = []
    vi_V = []
    for m in range(len(hs)):
        transitions = np.array(transitions_all_grids[m])
        rewards = np.array(rewards_all_grids[m])
        #greens = np.array(greens_all_grids[m])
        #reds = np.array(reds_all_grids[m])

        pi = tool.mdp.PolicyIteration(transitions, rewards, 0.9, eval_type='iterative')
        pi.run()
        pol = pi.policy
        pi_V.append(pi.V)

        vi = tool.mdp.ValueIteration(transitions, rewards, 0.9, epsilon=0.001)
        vi.run()
        vi_pol = vi.policy
        vi_V.append(vi.V)

    return pi_V[0], vi_V[0]

hs = [10]

if __name__ == "__main__":
    hs = [10, 24, 33, 40]
    #transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids = all_info()
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

        pi = tool.mdp.PolicyIterationModified(transitions, rewards, 0.9, epsilon=0.001)
        #pi = tool.mdp.PolicyIteration(transitions, rewards, 0.9, eval_type='iterative')
        # if m == 0 or m == 3:
        #     pi.setVerbose()
        pi.run()
        pi_policies.append(pi.policy)
        pi_values.append(pi.V)
        pi_iterations.append(pi.iter)
        print("policy iteration iterations", pi.iter)

        vi = tool.mdp.ValueIteration(transitions, rewards, 0.9, epsilon=0.001)
        # if m == 0 or m == 3:
        #     vi.setVerbose()
        vi.run()
        vi_policies.append(vi.policy)
        vi_values.append(vi.V)
        vi_iterations.append(vi.iter)
        print("value iteration iterations", vi.iter)

    print("policy iteration policy - small grid")
    print_policy(hs[0], hs[0], greens_all_grids[0], reds_all_grids[0], pi_policies[0])
    print("value iteration policy - small grid")
    print_policy(hs[0], hs[0], greens_all_grids[0], reds_all_grids[0], vi_policies[0])
    print("policy iteration and value iteration grid difference - small grid")
    print(policy_dif(hs[0], greens_all_grids[0], reds_all_grids[0], pi_policies[0], vi_policies[0]))
    print("policy iteration and value iteration grid difference - big grid")
    print(policy_dif(hs[3], greens_all_grids[3], reds_all_grids[3], pi_policies[3], vi_policies[3]))

    print("policy iteration and value iteration values sse - small grid")
    print(np.sum(np.square(np.array(pi_values[0])-np.array(vi_values[0]))))
    print("policy iteration and value iteration values sse - big grid")
    print(np.sum(np.square(np.array(pi_values[3])-np.array(vi_values[3]))))

    print("number of iterations - policy iteration")
    print(pi_iterations)
    print("number of iterations - value iteration")
    print(vi_iterations)