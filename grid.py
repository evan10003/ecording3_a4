
import random
import numpy as np
import copy
import mdptoolbox as tool

h = 40
w = 40

green = [int((1/4)*h), int((3/4)*w)]
red = [int((3/4)*h), int((3/4)*w)]

block_pieces = []
block_pieces.append([[0,0,0],
                    [0,1,1],
                    [0,1,1]])
block_pieces.append([[0],
                    [0],
                    [0]])
block_pieces.append([[0,0,0]])
block_pieces.append([[0,0],
                    [0,0]])
block_pieces.append([[0]])

grid = []
for i in range(h):
    grid.append([1]*w)
# print("...")
# for row in grid:
#     print(row)

for i in range(80):
    random.shuffle(block_pieces)
    piece = block_pieces[0]
    x = random.randint(0, w-1)
    y = random.randint(0, h-1)
    for j in range(len(piece)):
        for k in range(len(piece[0])):
            if y+j < h and x+k < w:
                grid[y+j][x+k] = piece[j][k]
    # print("...")
    # for row in grid:
    #     print(row)

grid[green[0]][green[1]] = 1
grid[red[0]][red[1]] = 1

print(np.sum(np.array(grid)))

np.savetxt("big_grid", grid)

# transitions = [[], [], [], []]
# transitions = [copy.deepcopy(grid), copy.deepcopy(grid), copy.deepcopy(grid), copy.deepcopy(grid)]
transitions = []
for _ in range(4):
    transitions_action = []
    for i in range(h*w):
        transitions_action.append([0]*(h*w))
    transitions.append(transitions_action)

s = 0
for j in range(h):
    for k in range(w):
        if grid[j][k] != 0:
            up = 0
            down = 0
            left = 0
            right = 0
            if j-1 >= 0:
                up = grid[j-1][k]
            if j+1 < h:
                down = grid[j+1][k]
            if k-1 >= 0:
                left = grid[j][k-1]
            if k+1 < w:
                right = grid[j][k+1]
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
        else:
            for a in range(4):
                if s-w >= 0:
                    transitions[a][s][s-w] = 0
                if s+w < h*w:
                    transitions[a][s][s+w] = 0
                if s-1 >= 0:
                    transitions[a][s][s-1] = 0
                if s+1 < h*w:
                    transitions[a][s][s+1] = 0
                transitions[a][s][s] = 1

        s += 1

rewards = [-0.04]*(h*w)
rewards[green[0]*w+green[1]] = 1
rewards[red[0]*w+red[1]] = -1

transitions = np.array(transitions)
rewards = np.array(rewards)

print(transitions.shape)
print(rewards.shape)

mdp_object = tool.mdp.MDP(transitions, rewards, 0.9, 0.001, 1000000)

print("YO!")
