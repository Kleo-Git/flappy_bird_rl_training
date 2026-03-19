<<<<<<< HEAD
Environment setup :-

Setup your dqn evn using Anaconda : conda create -n dqnenv

Activate your env : conda activate dqnenv

install python 3.11 : conda install python=3.11 

install flappy bird module using :- pip install flappy-bird-gymnasium

install Pytorch : pip install pytorch

install yaml


Debugging :-
Python


Interpreter :-
Python3.11(dqn)(Created Environment)


Training :-

Run the code in terminal.

To train Flappy Bird: python agent.py flappybird1 --train, or python reinforce_agent.py 'name' -- train

'flappybird1' can be the name of any defined hyperparameter blocks.

Testing / Watching the Agent Play : python agent.py flappybird1

Monitoring Progress (While Training): Get-Content ./runs/flappybird1.log -Wait



Playing in the environment :-

To play the game (human mode), run the following command: flappy_bird_gymnasium

To see a random agent playing, add an argument to the command: flappy_bird_gymnasium --mode random

To see a Deep Q Network agent playing, add an argument to the command: flappy_bird_gymnasium --mode dqn
=======
# flappy_bird_rl_training
Using a variety of RL approaches to train an agent how to play flappy bird
>>>>>>> 792e851d07eafda1099800a3bef14f00f2b9b5e2
