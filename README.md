# Monte-Carlo Tree Search

Group 13
--------
Monte Carlo tree search (MCTS) is a general game-playing algorithm to find the best move from any given game state of any game.

Environment Details:
--------------------
Whole code is in python 3 version(3.9.7 to be precise)

Instructions:
-------------
1. Be in `AI-team-project/` directory to execute commands that are mentioned below.
2. Install all dependencies listed in requirements.txt: \
   `pip install -r requirements.txt`
3. Run the follwing command to execute Monte-Carlo Tree Search Agent:\
   `python pacman.py -p MCTSAgent -l smallClassic`
4. To run this agent multiple times(here it is 10, change the value according to the requirements), follow below command:\
   `python pacman.py -p MCTSAgent -l smallClassic -n 10` 
5. To run MCTS Agent in quiet mode - use option **-q**\
   `python pacman.py -q -p MCTSAgent -l smallClassic`
   
Files:
------
1. ***multiAgents.py*** - contains Monte-Carlo Tree Search algorithm implementation in `MCTSAgent` class and `betterEvaluationFunction` function
2. ***t_test.py*** - performs t-Test to compare MCTS Agent with other agents - Minimax, AlphaBeta and Expectimax
3. ***newLayoutGenerator.py*** - generates new layouts
4. ***search.py*** and ***searchAgents.py*** - taken from project 1 to make use of A-Star search

Layouts:
--------
1. testClassic
2. smallClassic
3. mediumClassic
4. projectClassic1 (new generated layout)
5. projectClassic2 (new generated layout)
6. projectClassic3 (new generated layout)
7. bigClassic
8. openClassic
9. powerClassic
10. trappedClassic
11. originalClassic

Generated results and p-values obtained from t-Test can be found in `results` folder.
