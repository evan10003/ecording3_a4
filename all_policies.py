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


hs = [10, 24, 33, 40]
transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids = all_info()

def policy_info():
    pi_mod_times = []
    pi_iter_times = []
    pi_matrix_times = []
    vi_times = []
    pi_mod_iters = []
    pi_iter_iters = []
    pi_matrix_iters = []
    vi_iters = []
    pi_mod_policies = []
    pi_iter_policies = []
    pi_matrix_policies = []
    vi_policies = []
    for m in range(len(hs)):
        transitions = np.array(transitions_all_grids[m])
        rewards = np.array(rewards_all_grids[m])
        greens = np.array(greens_all_grids[m])
        reds = np.array(reds_all_grids[m])

        #print("policy iteration modified")
        # pi = tool.mdp.PolicyIterationModified(transitions, rewards, 0.9, epsilon=0.0001)
        # pi.run()
        # pol = pi.policy
        # pi_mod_times.append(pi.time)
        # pi_mod_iters.append(pi.iter)
        # pi_mod_policies.append(pol)
        #print("grid size", hs[m]**2)
        #print("policy iteration")
        #print("iterations", pi.iter)
        #print("time", pi.time)

        print("policy iteration iterative")
        pi = tool.mdp.PolicyIteration(transitions, rewards, 0.9, eval_type='iterative')
        pi.run()
        pol = pi.policy
        pi_iter_times.append(pi.time)
        pi_iter_iters.append(pi.iter)
        pi_iter_policies.append(pol)
        #print("grid size", hs[m]**2)
        #print("policy iteration")
        #print("iterations", pi.iter)
        #print("time", pi.time)

        #print("policy iteration matrix")
        # pi = tool.mdp.PolicyIteration(transitions, rewards, 0.9, eval_type='matrix')
        # pi.run()
        # pol = pi.policy
        # pi_matrix_times.append(pi.time)
        # pi_matrix_iters.append(pi.iter)
        # pi_matrix_policies.append(pol)
        #print("grid size", hs[m]**2)
        #print("policy iteration")
        #print("iterations", pi.iter)
        #print("time", pi.time)

        print("value iteration")
        vi = tool.mdp.ValueIteration(transitions, rewards, 0.9, epsilon=0.0001)
        vi.run()
        vi_pol = vi.policy
        vi_times.append(vi.time)
        vi_iters.append(vi.iter)
        vi_policies.append(vi_pol)
        #print("value iteration")
        #print("iterations", vi.iter)
        #print("time", vi.time)
        #print("same policies:", pol == vi_pol)

        #print_policy(h, w, greens, reds, pol)
        #print_policy(h, w, greens, reds, vi_pol)

    return pi_mod_times, pi_iter_times, pi_matrix_times, vi_times, pi_mod_iters, pi_iter_iters, pi_matrix_iters, vi_iters

hs = [10]

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

        vi = tool.mdp.ValueIteration(transitions, rewards, 0.9, epsilon=0.0001)
        vi.run()
        vi_pol = vi.policy
        vi_V.append(vi.V)

    return pi_V[0], vi_V[0]
