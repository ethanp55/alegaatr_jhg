This data set is part of the Supplementary Material for the following paper:

    J. Skaggs, M. Richards, M. Morris, M. A. Goodrich, and J. W. Crandall
    Fostering Collective Action in Complex Societies using Community-Based Agents
    Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), Jeju, South Korea, 2024


************************************
A.  Description of the Data
************************************

This data set provides round-by-round token allocations, popularities, and influences for each of the 18 games of the JHG played in the user study described in the paper.

6 games were played in each of the three study conditions:

1.  Majority Human (6 humans and 2 CAB agents): Data for these games provided in the folder majority_human
2.  Majority Bot (2 humans and 6 CAB agents): Data for these games provided in the folder majority_bot
3.  Even (4 humans and 4 CAB agents): Data for these games provided in the folder half_human_bot

Visualizations for each of the games is given in Section 6 of the SM documentation


************************************
B.  Format of the Data Files
************************************

Each data file provides information about a single game as a .csv file

The first row of each data file is the header information.  
The second row provides information about the initial popularities of the players.
Each subsequent row provides information about:

(1) player popularities at the end of the round (columns labeled p0 through p7)
(2) token allocations of all players in that round (the column labeled i-T-j gives the tokens player i gave to player j in that round)
(3) influence values of all players after that round (the column labeled i-I-j gives the influence of player i on player j after that round)





