import numpy as np
import mdptoolbox as tool
from big_grid import big_info
from grid import info

transitions, rewards, h, w, greens, reds = info()
big_transitions, big_rewards, big_h, big_w, big_greens, big_reds = big_info()

big_transitions = np.array(big_transitions)
big_rewards = np.array(big_rewards)

big_pi = tool.mdp.PolicyIteration(big_transitions, big_rewards, 0.9, max_iter = 1000, eval_type='iterative')
big_pi.run()
big_pol = big_pi.policy

for i in range(big_h):
    strin = ""
    for j in range(big_w):
        if [i,j] in big_greens:
            strin += " g "
        elif [i,j] in big_reds:
            strin += " r "
        else:
            p = big_pol[big_w*i+j]
            if p == 0:
                strin += " ^ "
            elif p == 1:
                strin += " v "
            elif p == 2:
                strin += " < "
            else:
                strin += " > "
    print(strin)

transitions = np.array(transitions)
rewards = np.array(rewards)

pi = tool.mdp.PolicyIteration(transitions, rewards, 0.9, max_iter = 1000, eval_type='iterative')
pi.run()
pol = pi.policy

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
