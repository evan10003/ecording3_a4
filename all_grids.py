import random

hs = [10, 20, 30, 40]
seeds = [236, 237, 238, 239]
transitions_all_grids = []
rewards_all_grids = []
greens_all_grids = []
reds_all_grids = []
intervals = [1, 2, 3, 4]

for m in range(len(hs)):
    seed = seeds[m]
    h = hs[m]
    w = hs[m]
    #print("...")
    #print(h,w)
    #print("...")
    random.seed(seed)
    interval = intervals[m]
    combinations = [(a,b) for a in range(interval) for b in range(interval)]

    #print(combinations)

    greens = []
    reds = []
    for p,q in combinations:
        g_y = random.randint(h*p//interval, (h*(p+1)//interval)-1)
        r_y = random.randint(h*p//interval, (h*(p+1)//interval)-1)
        g_x = random.randint(w*q//interval, (w*(q+1)//interval)-1)
        r_x = random.randint(w*q//interval, (w*(q+1)//interval)-1)
        greens.append([g_y, g_x])
        reds.append([r_y, r_x])
        #print(g_y,g_x)
        #print(r_y,r_x)

    transitions = []
    for _ in range(4):
        transitions_action = []
        for i in range(h*w):
            transitions_action.append([0]*(h*w))
        transitions.append(transitions_action)

    s = 0
    for j in range(h):
        for k in range(w):
            up = 0
            down = 0
            left = 0
            right = 0
            if j-1 >= 0:
                up = 1
            if j+1 < h:
                down = 1
            if k-1 >= 0:
                left = 1
            if k+1 < w:
                right = 1
            # end state probs order: up down left right same
            actions = []
            up_next_state_probs = [0, 0, 0, 0, 0]
            if up == 0:
                up_next_state_probs[4] += 0.8
            else:
                up_next_state_probs[0] += 0.8
            if left == 0:
                up_next_state_probs[4] += 0.1
            else:
                up_next_state_probs[2] += 0.1
            if right == 0:
                up_next_state_probs[4] += 0.1
            else:
                up_next_state_probs[3] += 0.1
            down_next_state_probs = [0, 0, 0, 0, 0]
            if down == 0:
                down_next_state_probs[4] += 0.8
            else:
                down_next_state_probs[1] += 0.8
            if left == 0:
                down_next_state_probs[4] += 0.1
            else:
                down_next_state_probs[2] += 0.1
            if right == 0:
                down_next_state_probs[4] += 0.1
            else:
                down_next_state_probs[3] += 0.1
            left_next_state_probs = [0, 0, 0, 0, 0]
            if left == 0:
                left_next_state_probs[4] += 0.8
            else:
                left_next_state_probs[2] += 0.8
            if up == 0:
                left_next_state_probs[4] += 0.1
            else:
                left_next_state_probs[0] += 0.1
            if down == 0:
                left_next_state_probs[4] += 0.1
            else:
                left_next_state_probs[1] += 0.1
            right_next_state_probs = [0, 0, 0, 0, 0]
            if right == 0:
                right_next_state_probs[4] += 0.8
            else:
                right_next_state_probs[3] += 0.8
            if up == 0:
                right_next_state_probs[4] += 0.1
            else:
                right_next_state_probs[0] += 0.1
            if down == 0:
                right_next_state_probs[4] += 0.1
            else:
                right_next_state_probs[1] += 0.1

            actions.append(up_next_state_probs)
            actions.append(down_next_state_probs)
            actions.append(left_next_state_probs)
            actions.append(right_next_state_probs)

            for a in range(4):
                if s-w >= 0:
                    transitions[a][s][s-w] = actions[a][0]
                if s+w < h*w:
                    transitions[a][s][s+w] = actions[a][1]
                if s-1 >= 0:
                    transitions[a][s][s-1] = actions[a][2]
                if s+1 < h*w:
                    transitions[a][s][s+1] = actions[a][3]
                transitions[a][s][s] = actions[a][4]
            s += 1

    rewards = [-0.04]*(h*w)
    for green,red in zip(greens, reds):
        g = green[0]*w+green[1]
        r = red[0]*w+red[1]
        for a in range(4):
            if g-w >= 0:
                transitions[a][g][g-w] = 0
            if g+w < h*w:
                transitions[a][g][g+w] = 0
            if g-1 >= 0:
                transitions[a][g][g-1] = 0
            if g+1 < h*w:
                transitions[a][g][g+1] = 0
            transitions[a][g][g] = 1
            if r-w >= 0:
                transitions[a][r][r-w] = 0
            if r+w < h*w:
                transitions[a][r][r+w] = 0
            if r-1 >= 0:
                transitions[a][r][r-1] = 0
            if r+1 < h*w:
                transitions[a][r][r+1] = 0
            transitions[a][r][r] = 1
        rewards[g] = 1
        rewards[r] = -1
    transitions_all_grids.append(transitions)
    rewards_all_grids.append(rewards)
    greens_all_grids.append(greens)
    reds_all_grids.append(reds)

def all_info():
    return transitions_all_grids, rewards_all_grids, greens_all_grids, reds_all_grids

