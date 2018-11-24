MDPs

python3

graphs.py creates and saves plots for policy and value iteration. For a few of
the plots, it reads from txt files where verbose output (printed by
all_policies.py) was copied.

all_policies.py prints policy variation and value function variation per
iteration for policy iteration and value iteration respectively
(plotted in graphs.py). It prints policies for the small and large grids and
number of iterations. Uses https://pypi.org/project/pymdptoolbox/.

q_learning.py has QL implementations and creates and saves plots for a bunch of
QL stuff, mostly times and policy performance. It takes a very long time to run,
so I would recommend adjusting iterators and/or the variable n (number of runs)
if you want to run it.

all_grids.py builds the grids and creates accompanying transition and reward
structures for use in the other files.