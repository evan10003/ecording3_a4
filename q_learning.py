import numpy as np
from all_grids import all_info
from all_policies import value_function_info
import mdptoolbox as tool
from all_policies import print_policy
import matplotlib.pyplot as plt

def print_q_policy(h, w, greens, reds, pol):
    for i in range(h):
        strin = ""
        for j in range(w):
            if [i,j] in greens:
                strin += " g "
            elif [i,j] in reds:
                strin += " r "
            else:
                #print(pol[w*i+j,:])
                p = np.argmax(pol[w*i+j,:])
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

transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids = all_info()

T = transitions_all_grids[0]
R = rewards_all_grids[0]
greens = greens_all_grids[0]
reds = reds_all_grids[0]
l = len(T[0])
h = int(np.sqrt(l))

#print(T)
#print(R)
#print(l)

def q_learning_iter(T, R, Q, s, gamma, alpha_t):
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
    if np.random.random() < 0.2:
        a = np.random.randint(0,4)
    s_new = np.random.choice(range(len(T[a][s])), 1, p=T[a][s])
    s_new = s_new[0]
    x = r + gamma*np.amax(Q[s_new])
    Q[s,a] = (1-alpha_t)*Q[s,a] + alpha_t*x
    return Q, s_new


Q = 2+np.random.random((l,4))
s = None
while True:
    s = np.random.randint(0,l)
    s_y = s//h
    s_x = s%h
    if [s_y,s_x] not in greens and [s_y,s_x] not in reds:
        break

sample_q_values = []
sample_q_values.append(np.sum(Q))
for n in range(15000):
    #alpha = 1/(n+1)
    #print(s)
    Q, s = q_learning_iter(T, R, Q, s, 0.9, 0.9)
    sample_q_values.append(np.sum(Q))

plt.plot(range(len(sample_q_values)), sample_q_values)
plt.title("sum of q values over iterations")
plt.xlabel("iteration")
plt.ylabel("q value")
plt.savefig("q_values")

print_q_policy(10, 10, greens, reds, Q)







pi_V, vi_V = value_function_info()
#pi_V = np.array(pi_V)
#vi_V = np.array(vi_V)
# print("policy iter V")
# print(pi_V)
# print("value iter V")
# print(vi_V)
# print("Q")
# ql_V = np.amax(Q, axis=1)
# c = pi_V[0]/ql_V[0]
# ql_V = c*ql_V
# print(ql_V)

ql = tool.mdp.QLearning(np.array(T), np.array(R), 0.9)
ql.run()
ql_pol = ql.policy

#print_policy(10, 10, greens, reds, ql_pol)