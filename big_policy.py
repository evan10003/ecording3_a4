import numpy as np
import mdptoolbox as tool
from big_grid import info
import random

transitions, rewards, h, w, greens, reds = info()

# g_y = h//4
# g_x = 3*w//4
# r_y = 3*h//4
# r_x = 3*w//4
# g = g_y*w + g_x
# r = r_y*w + r_x

# print("green")
# print(g_y, g_x)
# print(g)
# print("red")
# print(r_y, r_x)
# print(r)

# for a in range(4):
#     transitions_per_states = []
#     for i in range(h*w):
#         for k in range(h):
#             transitions_per_states.append(transitions[a][i][k*w:(k+1)*w])
#         transitions_per_states.append([9]*w)
    #np.savetxt("transitions_check"+str(a), transitions_per_states)

# rewards_matrix = []
# for k in range(h):
#     rewards_matrix.append(rewards[k*w:(k+1)*w])

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

