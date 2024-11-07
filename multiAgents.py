# multiAgents.py
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
from operator import itemgetter

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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #print(f"New successor game state: {successorGameState}")
        #print(f"New pacman position: {newPos}")
        #print(f"New food position: {newFood}")
        #print(f"New ghost states: {newGhostStates}")
        #print(f"New scared times: {newScaredTimes}")

        foodScore = 0
        ghostScore = 0

        #Evaluate on getting food and avoiding ghosts
        #Eating food (find closest food pellet and incentivize going to it -> reciprocal of distance to closest pellet yields a high value)
        for food in newFood:
            foodDist = min([manhattanDistance(newPos, food)])
            foodScore += (1/foodDist)
            
        #Avoiding ghosts (find closest ghost, if pacman stays too close, penalize -> incentivizes pacman to stay away from ghost)
        #Could probably find a way to ensure pacman never reaches same position as ghost... but this eval function does surprisingly well 
        for ghost in newGhostStates:
            ghostDist = min([manhattanDistance(newPos, ghost.configuration.pos)])
            if ghostDist == 1 and any(newScaredTimes) == 0:
                ghostScore -= 1000
            
        return successorGameState.getScore() + foodScore + ghostScore
        
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    bestAction = None
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

        #print(self.depth)
        #print(self.index)
        return self.minimax(gameState, self.index, 0)[1]
        util.raiseNotDefined()
    
    def minimax(self, gameState, agentIndex, depth):
        result = []
        numAgents = gameState.getNumAgents()
        #Finished first depth, move on to the next
        if agentIndex >= numAgents:
            agentIndex = 0
            depth += 1
        
        #Reached specified depth or win or loss occurs = stop
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return [self.evaluationFunction(gameState), None] 

        #Max (pacman)
        if agentIndex == 0:
            #value = float("-inf")
            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                maxValue = self.minimax(successorGameState, agentIndex + 1, depth)[0] #Max value that pacman should take based on successors (how to get associated action?)
                result.append((maxValue, action))
            
            result = max(result, key = lambda x : x[0])
        #Min (ghost)
        else:
            #value = float("inf")
            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                #value = self.minimax(successorGameState, agentIndex + 1, depth)
                minValue = self.minimax(successorGameState, agentIndex + 1, depth)[0] #Min value that ghost should take based on successors
                result.append((minValue, action))
            
            result = min(result, key = lambda x : x[0])

        return result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphabeta(gameState, self.index, 0, float("-inf"), float("inf"))[1]
        util.raiseNotDefined()

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        result = tuple()

        numAgents = gameState.getNumAgents()
        if agentIndex >= numAgents:
            agentIndex = 0
            depth += 1
        
        #Reached specified depth or win or loss occurs = stop
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), None)
        
        if agentIndex == 0:
            value = tuple((float("-inf"), ""))
            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                maxValue = self.alphabeta(successorGameState, agentIndex + 1, depth, alpha, beta)[0] #Max value that pacman should take based on successors (how to get associated action?)
                
                result = tuple((maxValue, action))
                if result[0] > value[0]:
                    value = tuple((result[0], result[1]))
                #else:
                #    value = tuple((value[0], value[1]))

                if value[0] > beta:
                    return value
                alpha = max(alpha, value[0])

        else:
            value = (float("inf"), "")
            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                #value = self.minimax(successorGameState, agentIndex + 1, depth)
                minValue = self.alphabeta(successorGameState, agentIndex + 1, depth, alpha, beta)[0] #Min value that ghost should take based on successors

                result = tuple((minValue, action))
                if result[0] < value[0]:
                    value = tuple((result[0], result[1]))

                if value[0] < alpha:
                    return value
                beta = min(beta, value[0])

        return value
    

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
        return self.expectimax(gameState, self.index, 0)[1]
        util.raiseNotDefined()

    def expectimax(self, gameState, agentIndex, depth):
        result = []
        expValue = 0

        numAgents = gameState.getNumAgents()
        #Finished first depth, move on to the next
        if agentIndex >= numAgents:
            agentIndex = 0
            depth += 1
        
        #Reached specified depth or win or loss occurs = stop
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return [self.evaluationFunction(gameState), None] 

        #Max (pacman)
        if agentIndex == 0:
            #value = float("-inf")
            legalActions = gameState.getLegalActions(agentIndex)

            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                maxValue = self.expectimax(successorGameState, agentIndex + 1, depth)[0] #Max value that pacman should take based on successors (how to get associated action?)
                result.append((maxValue, action))
            
            result = max(result, key = lambda x : x[0])
        #Min (ghost)
        else:
            #value = float("inf")
            legalActions = gameState.getLegalActions(agentIndex)
            #Uniform random prob of choosing each action
            probability = 1 / len(legalActions)

            for action in legalActions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                #value = self.minimax(successorGameState, agentIndex + 1, depth)
                minValue = self.expectimax(successorGameState, agentIndex + 1, depth)[0] #Min value that ghost should take based on successors
                expValue += probability * minValue
            result = tuple((expValue, action))
            
            #result = min(result, key = lambda x : x[0])

        return result

