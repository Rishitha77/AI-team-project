#multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from random import randint
from util import manhattanDistance as md

from model import commonModel, Model
from featureBasedGameState import FeatureBasedGameState
from math import sqrt, log

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        newFoodList = currentGameState.getFood().asList()

        for ghostState in newGhostStates :
            if ghostState.getPosition() == tuple(newPos) and ghostState.scaredTimer == 0:
                return -100000000000000

        if action == 'Stop':
            return -100000000000000

        min_distance = 1000000
        for food in newFoodList:
            x = abs(food[0] - newPos[0])
            y = abs(food[1] - newPos[1])
            current_distance = x + y
            if current_distance < min_distance:
                min_distance = current_distance
        return -1 * min_distance


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.
      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        action = self.minimax(gameState, 0, 0)
        return action

    def minimax(self, gameState, index, depth):

        utility_value, action = self.max_value(gameState, index, depth)
        return action

    def max_value(self, gameState, index, depth):

        if gameState.getLegalActions(index) == [] or depth == self.depth:
            return gameState.getScore(), ""

        max_value = -100000000
        max_action = ""
        possibleActions = gameState.getLegalActions(index)

        for action in possibleActions:
            successor_index = index + 1

            successor_index = successor_index % gameState.getNumAgents()
            if successor_index == 0:
                current_utility_value, current_action = self.max_value(gameState.generateSuccessor(index, action), successor_index, depth + 1)
            else:
                current_utility_value, current_action = self.min_value(gameState.generateSuccessor(index, action), successor_index, depth)

            if current_utility_value > max_value:
                max_value = current_utility_value
                max_action = action

        return max_value, max_action

    def min_value(self, gameState, index, depth):

        if gameState.getLegalActions(index) == [] or depth == self.depth:
            return gameState.getScore(), ""

        min_value = 100000000
        min_action = ""
        possibleActions = gameState.getLegalActions(index)

        for action in possibleActions:
            successor_index = index + 1

            successor_index = successor_index % gameState.getNumAgents()
            if successor_index == 0:
                current_utility_value, current_action = self.max_value(gameState.generateSuccessor(index, action),
                                                                       successor_index, depth + 1)
            else:
                current_utility_value, current_action = self.min_value(gameState.generateSuccessor(index, action),
                                                                       successor_index, depth)

            if current_utility_value < min_value:
                min_value = current_utility_value
                min_action = action

        return min_value, min_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action = self.alphaBetaSearch(gameState, 0, 0, -10000000, 10000000)
        return action

    def alphaBetaSearch(self, gameState, index, depth, alpha, beta):

        utility_value, action = self.max_value(gameState, index, depth, alpha, beta)
        return action

    def max_value(self, gameState, index, depth, alpha, beta):

        if gameState.getLegalActions(index) == [] or depth == self.depth:
            return gameState.getScore(), ""

        max_value = -100000000
        max_action = ""
        possibleActions = gameState.getLegalActions(index)

        for action in possibleActions:
            successor_index = index + 1

            successor_index = successor_index % gameState.getNumAgents()
            if successor_index == 0:
                current_utility_value, current_action = self.max_value(gameState.generateSuccessor(index, action), successor_index, depth + 1,
                                                                       alpha, beta)
            else:
                current_utility_value, current_action = self.min_value(gameState.generateSuccessor(index, action), successor_index, depth,
                                                                       alpha, beta)

            if current_utility_value > max_value:
                max_value = current_utility_value
                max_action = action

            if max_value > beta:
                return max_value, max_action

            alpha = max(alpha, max_value)

        return max_value, max_action

    def min_value(self, gameState, index, depth, alpha, beta):

        if gameState.getLegalActions(index) == [] or depth == self.depth:
            return gameState.getScore(), ""

        min_value = 100000000
        min_action = ""
        possibleActions = gameState.getLegalActions(index)

        for action in possibleActions:
            successor_index = index + 1

            successor_index = successor_index % gameState.getNumAgents()
            if successor_index == 0:
                current_utility_value, current_action = self.max_value(gameState.generateSuccessor(index, action), successor_index, depth + 1, alpha, beta)
            else:
                current_utility_value, current_action = self.min_value(gameState.generateSuccessor(index, action), successor_index, depth, alpha, beta)

            if current_utility_value < min_value:
                min_value = current_utility_value
                min_action = action

            if min_value < alpha:
                return min_value, min_action

            beta = min(beta, min_value)

        return min_value, min_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action = self.expectimax(gameState, 0, 0)
        return action

    def expectimax(self, gameState, index, depth):
        utility_value, action = self.max_value(gameState, index, depth)
        return action

    def max_value(self, gameState, index, depth):

        if gameState.getLegalActions(index) == [] or depth == self.depth:
            return gameState.getScore(), ""

        max_value = -100000000
        max_action = ""
        possibleActions = gameState.getLegalActions(index)

        for action in possibleActions:
            successor_index = index + 1
            successor_index = successor_index % gameState.getNumAgents()
            if successor_index == 0:
                current_utility_value, current_action = self.max_value(gameState.generateSuccessor(index, action), successor_index, depth + 1)
            else:
                current_utility_value, current_action = self.expected_value(gameState.generateSuccessor(index, action), successor_index, depth)

            if current_utility_value > max_value:
                max_value = current_utility_value
                max_action = action

        return max_value, max_action

    def expected_value(self, gameState, index, depth):

        if gameState.getLegalActions(index) == [] or depth == self.depth:
            return gameState.getScore(), ""

        exp_value = 0.0
        possibleActions = gameState.getLegalActions(index)
        probability = 1.0 / len(possibleActions)

        for action in possibleActions:
            successor_index = index + 1
            successor_index = successor_index % gameState.getNumAgents()
            if successor_index == 0:
                current_utility_value, current_action = self.max_value(gameState.generateSuccessor(index, action), successor_index, depth + 1)
            else:
                current_utility_value, current_action = self.expected_value(gameState.generateSuccessor(index, action), successor_index, depth)

            exp_value = exp_value + probability * current_utility_value

        return exp_value, ""

def newEvaluationFunction(currentGameState):

    pacmanPos = currentGameState.getPacmanPosition()
    foodMatrix = currentGameState.getFood()
    foodList = foodMatrix.asList()
    successorGameScore = currentGameState.getScore()

    numberOfRemainingFood = len(foodList)

    distanceFromFoods = [md(pacmanPos, newFoodPos) for newFoodPos in foodList]
    distanceFromClosestFood = 0 if (len(distanceFromFoods) == 0) else min(distanceFromFoods)

    finalScore = successorGameScore - (60 * numberOfRemainingFood) - (5 * distanceFromClosestFood)  + randint(0,1)
    return finalScore

class MCTSAgent(MultiAgentSearchAgent):
    def __init__(self, evalFn = 'newEvaluationFunction', numTraining = '0', isReal = False):
        self.games = 0
        self.numberOfTrainingGames = int(numTraining)
        
        #Probability for the pacman to make a good move. It is an additional "Exploitation factor".
        #Can change this parameter if required.
        self.guidance = 0.3

        # The exploitation-exploration factor.
        #Higher the value higher the exploitation.
        self.c = sqrt(2) + 0.5

    def registerInitialState(self, state):
        self.games += 1

    def getUCBValue(self, weight, n, N, c):
        return weight/(n+1.0) + c*sqrt(log(N+1.0)/(n+1.0))

    def UCBValues(self, gameState, model):
        weights = {}
        n = {}
        N = 0
        legalActions = gameState.oldGameState.getLegalActions()
        for action in legalActions:
            if (gameState, action) not in model.data:
                n[action] = 0
                weights[action] = 0
            else:
                n[action] = model.data[(gameState, action)].nSimulations
                weights[action] = model.data[(gameState, action)].nWins \
                
            N += n[action]
        ucbValues = []
        for action in legalActions:
            uctValue = self.getUCBValue(weights[action], n[action], N, self.c)
            ucbValues.append((uctValue, action))
        return ucbValues

    def getAction(self, state):
        gameState = FeatureBasedGameState(state)
        if self.games <= self.numberOfTrainingGames:
            if random.random() < self.guidance and not gameState.ghostNearFood:
                return gameState.closestFood
            ucbValues = self.UCBValues(gameState, commonModel)
            actionToReturn = max(ucbValues)[1]
            return actionToReturn
        else:
            return self.actualAction(gameState, commonModel)

    def actualAction(self, gameState, model):
        valueActionPairs = []
        for action in gameState.oldGameState.getLegalActions():
            value = None
            if (gameState, action) not in model.data:
                value = 0
            else:
                # select the action with max simulations
                value = model.data[(gameState, action)].nSimulations 
            valueActionPairs.append((value, action))
        return max(valueActionPairs)[1]
