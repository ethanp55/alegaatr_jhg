----------------------------------------------------------------------------------------
Notes about the code: 
A. This code is part of the Supplementary Material for the following paper:

    J. Skaggs, M. Richards, M. Morris, M. A. Goodrich, and J. W. Crandall
    Fostering Collective Action in Complex Societies using Community-Based Agents
    Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), Jeju, South Korea, 2024

B. The code has been developed for Linux and Mac, but has not been tested with Windows
C. Python3 code was developed with Python 3.8.0
D. Java code (GUI through which human player can make moves) was developed with java 17 2021-09-14 LTS
E. If you have questions, feel free to contact me (Jacob Crandall) at crandall@cs.byu.edu.  While I may be a bit slow to answer your query, I'd love to help if I can
F. The CAB agents are primarily implemented in the files GeneAgent.h (C++) and geneagent3.py (Python3)
----------------------------------------------------------------------------------------

Code is provided to do the following:

    1a. Run a game between already-trained CAB agents and potentially other players (CATs and a Human) using C++
    1b. Run a game between already-trained CAB agents and potentially other players (CATs and a Human) using Python3
    2a. Train the CAB agents using the evolutionary simulation described in the paper using C++

According to the tests we conducted, the C++ and Python3 implementations behave equivalently

In addition to this supplied code, the online platform found at juniorhighgame.com allows people to play the JHG with other people and with CAB agents.

***********************************************
  C++ Code
***********************************************
1a. Instructions for running a game between already-trained CAB agents, a human, and potentially CATs using C++

To run the code on Linux of Mac, do the following:

A -- Start the C++ Bots

- Navigate in a terminal to the GeneSimulation_cpp folder
- Compile the code: g++ main.cpp -o jhgsim
- Run the code:
    a) with randomly selected CAB agents used in user study: ./jhgsim play ../ResultsStudy 60 3 199 10 30 best_agents equal 0 nondeterministic humanConfig
    b) with randomly selected CAB agents from other training: ./jhgsim play ../ResultsSaved/theGenerations 100 3 99 10 30 best_agents equal 0 nondeterministic humanConfig

Parameters:
0: executable (don't change)
1: code directive to play a single game (don't change)
2: Folder where parameters of trained CAB agent are stored.  ResultsSaved/theGenerations contains already trained agents for 100 generations.  Newly trained agents will be stored in the folder specified (see 2a)
3: Number of agents in the gene pool (must be the same as number of agents in available gene pool)
4: Number of sets of genes.  3 was used for the CAB agents in the paper.  1 is also useable, but must correspond to how the agents were trained
5: Generation number that you want to draw CAB agents from (199 for user study, 0-99 for other pretrained)
6: Number of CAB agents to put into the game
7: Number of rounds in the game
8: Select the best agents (best_agents) or random agents (rnd_agents) from the gene pool
9: Initial popularities of the players (possible values: equal, highlow, power, random, step)
10: Poverty line (see SM-1)
11: Exploration characteristics (don't change -- parameter was used for debugging)
12: Configuration file specifying which agents to include (use "humanConfig" to add 1 human player to the game, use "human1Assassins1Config" to add a human and 1 CAT agent, use "human1Assassins2Config" to add a human and 2 CAT agents)

Note: To have all agents use the hardcoded parameters (genes) specified in the function assassinDefender() instead of trained parameters, comment out line 556 and uncomment out line 555 (then recompile the code)




B -- Start the Human Player

- Navigate in a different terminal to the Human folder
- Compile the code: javac *.java
- Run the code: java Human

You can play the game through the GUI.  The GUI buttons may be off a little bit.  Different operating systems do different things with the GUI display.  The GUI is pretty getto
Use the +/- buttons to adjust token allocations to other players.  Click submit to finalize token allocations for the round
No network graph is shown, only the tornado and popularity graphs are provided in this interface




************************************************
2. Train the CAB agents using the evolutionary simulation described in the paper using C++

- Navigate in a terminal to the GeneSimulation_cpp folder
- Comple the code: g++ main.cpp -o jhgsim
- run the code: ./jhgsim evolve ../Results/theGenerations 100 3 0 100 100 10 30 0 basicConfig varied

Parameters:
0: executable (don't change)
1: code directive to evolve the population (don't change)
2: Folder where parameters of trained CAB agent will be stored
3: Number of agents in the gene pool (100 used here -- must be the same as was used in training)
4: Number of sets of genes.  3 was used for the CAB agents in the paper.  1 is also useable, but must correspond to how the agents were trained
5: Generation to start training (0 to start from scratch)
6: Generation to end training (100 will train up through generation 99, as it starts at 0)
7: Number of games to play per generation (agents from the gene pool are selected at random)
8: Number of agents per game
9: Number of rounds per game
10: Poverty line (see SM-1)
11: Configuration file specifying which agents to include (use "basicConfig" to just train with CAB agents, use "assassins1Config" to train with 1 CAT agent, use "assassins2Config" to train with 2 CAT agents)
12: Initial popularities ("even" for all agents to always start at popularity 100, "varied" to start agents at various popularity levels)




***********************************************
  Python3 Code
***********************************************
1b. Instructions for running a game between already-trained CAB agents and a human player (CAT agents are not provided in Python)

A -- Start the Python3 Bots

- Navigate in a terminal to the GeneSimulation_py folder
- Run the code: 
    a) with randomly selected CAB agents used in user study: python3 main.py play ../ResultsStudy 60 3 199 10 30 best_agents equal 0 nondeterministic humanConfig
    b) with randomly selected CAB agents from other training: python3 main.py play ../ResultsSaved/theGenerations 100 3 99 10 30 best_agents equal 0 nondeterministic humanConfig

Parameters:
0: main.py (start of the program)
1: code directive to play a single game (don't change)
2: Folder where parameters of trained CAB agent are stored.  ResultsSaved/theGenerations contains already trained agents for 100 generations.  Newly trained agents will be stored in the folder specified (see 2a)
3: Number of agents in the gene pool (100 used here -- must be the same as was used in training)
4: Number of sets of genes.  3 was used for the CAB agents in the paper.  1 is also useable, but must correspond to how the agents were trained
5: Generation number (0 to 99) that you want to draw CAB agents from
6: Number of CAB agents to put into the game
7: Number of rounds in the game
8: Select the best agents (best_agents) or random agents (rnd_agents) from the gene pool
9: Initial popularities of the players (possible values: equal, highlow, power, random, step)
10: Poverty line (see SM-1)
11: Exploration characteristics (don't change -- parameter was used for debugging)
12: Configuration file specifying which agents to include (use "humanConfig" to add 1 human player to the game, use "human1Assassins1Config" to add a human and 1 CAT agent, use "human1Assassins2Config" to add a human and 2 CAT agents)

Note: To have all agents use the hardcoded parameters (genes) specified in the function assassinDefender() instead of trained parameters, comment out line 556 and uncomment out line 555 (then recompile the code)


B -- Start the Human Player

- Navigate in a different terminal to the Human folder
- Compile the code: javac *.java
- Run the code: java Human

You can play the game through the GUI.  The GUI buttons may be off a little bit.  Different operating systems do different things with the GUI display.  The GUI is pretty getto
Use the +/- buttons to adjust token allocations to other players.  Click submit to finalize token allocations for the round
No network graph is shown, only the tornado and popularity graphs are provided in this interface



