import numpy as np
import mdptoolbox as tool
from grid import info
import random

transitions, rewards, h, w = info()

g_y = h//4
g_x = 3*w//4
r_y = 3*h//4
r_x = 3*w//4
g = g_y*w + g_x
r = r_y*w + r_x

transitions = np.array(transitions)
rewards = np.array(rewards)

pi = tool.mdp.PolicyIteration(transitions, rewards, 0.9, max_iter = 1000, eval_type='iterative')
pi.run()
pol = pi.policy

for i in range(h):
    strin = ""
    for j in range(w):
        if i == g_y and j == g_x:
            strin += " g "
        elif i == r_y and j == r_x:
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

