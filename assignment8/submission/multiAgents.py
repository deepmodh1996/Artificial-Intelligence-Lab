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

from game import Agent
numCapsules = 0
isChanged = False
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        # print "bbbb"
        # print scores
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        # print chosenIndex

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
        oldFood = currentGameState.getFood()
        newFood = successorGameState.getFood()
        oldPos = currentGameState.getPacmanPosition()
        oldGhostPosition = currentGameState.getGhostPositions()
        newGhostPosition = successorGameState.getGhostPositions()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print "aaaaaa"
        # print newPos
        # print newScaredTimes
        # print oldGhostPosition
        # print oldGhostPosition
        # print newPos
        if util.manhattanDistance(newPos, oldGhostPosition[0]) <= 1.0:
          return 0
        if oldFood[newPos[0]][newPos[1]] == True:
          return 101
        q = util.Queue() 
        q.push([newPos, 1])
        isVisited = [newPos]
        size = newFood.asList()
        xlen = size[-1][0]
        ylen = size[-1][1]
        while not q.isEmpty():
          v = q.pop()
          x = v[0][0]
          y = v[0][1]
          if newFood[x][y] == True:
            r = 100.0/v[1]
            return r
          if (x-1 > 0) and ((x-1, y) not in isVisited):
            isVisited.append((x-1, y))
            q.push([(x-1, y), v[1]+1])
          if (y-1 > 0) and ((x, y-1) not in isVisited):
            isVisited.append((x, y-1))
            q.push([(x, y-1), v[1]+1])
          if (x+1 < xlen) and ((x+1, y) not in isVisited):
            isVisited.append((x+1, y))
            q.push([(x+1, y), v[1]+1])
          if (y+1 < ylen) and ((x, y+1) not in isVisited):
            isVisited.append((x, y+1))
            q.push([(x, y+1), v[1]+1])
        return 1
        # return 1/util.manhattanDistance(newPos, nearestFood)
        # return successorGameState.getScore()

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

################################################################################
####                          MiniMax                                       ####
################################################################################


    def getGhostScoreMiniMax(self, ghostIndex, gameState, depth):
      ghostActions = gameState.getLegalActions(ghostIndex)
      bestScore = float("+inf")
      bestAction = None
      for action in ghostActions:
        nextState = gameState.generateSuccessor(ghostIndex, action)
        if ghostIndex + 1 < gameState.getNumAgents():
          score = self.getGhostScoreMiniMax(ghostIndex + 1, nextState, depth)
        else:
          score = self.getPacmanScoreMiniMax(nextState, depth + 1)
        score = score[0]
        if score < bestScore:
          bestScore = score
          bestAction = action
      if bestScore == float("inf"):
        bestScore = self.evaluationFunction(gameState)
        bestAction = 'Center'
      return [bestScore, bestAction]

    def getPacmanScoreMiniMax(self, gameState, depth):
      if (gameState.isWin() == True) or (depth > self.depth):
        bestScore = self.evaluationFunction(gameState)
        bestAction = 'Center'
        return [bestScore, bestAction]
      pacmanActions = gameState.getLegalActions(0)
      bestScore = float("-inf")
      bestAction = None
      for action in pacmanActions:
        nextState = gameState.generateSuccessor(0, action)
        score = self.getGhostScoreMiniMax(1, nextState, depth)
        score = score[0]
        if score > bestScore:
          bestScore = score
          bestAction = action
      if bestScore == float("-inf"):
        bestScore = self.evaluationFunction(gameState)
        bestAction = 'Center'
      return [bestScore, bestAction]


################################################################################
####                          ExpectiMax                                    ####
################################################################################


    def getGhostScoreExpectiMax(self, ghostIndex, gameState, depth):
      ghostActions = gameState.getLegalActions(ghostIndex)
      sumScore = 0.0
      for action in ghostActions:
        nextState = gameState.generateSuccessor(ghostIndex, action)
        if ghostIndex + 1 < gameState.getNumAgents():
          score = self.getGhostScoreExpectiMax(ghostIndex + 1, nextState, depth)
        else:
          score = self.getPacmanScoreExpectiMax(nextState, depth + 1)
        score = score[0]
        sumScore += score
      if len(ghostActions) == 0:
        bestScore = self.evaluationFunction(gameState)
      else:
        bestScore = sumScore*1.0/len(ghostActions)
      dummyAction = 'Center'
      return [bestScore, dummyAction]

    def getPacmanScoreExpectiMax(self, gameState, depth):
      if (gameState.isWin() == True) or (depth > self.depth):
        bestScore = self.evaluationFunction(gameState)
        bestAction = 'Center'
        return [bestScore, bestAction]
      pacmanActions = gameState.getLegalActions(0)
      bestScore = float("-inf")
      bestAction = None
      for action in pacmanActions:
        nextState = gameState.generateSuccessor(0, action)
        score = self.getGhostScoreExpectiMax(1, nextState, depth)
        score = score[0]
        if score >= bestScore:
          bestScore = score
          bestAction = action
      if bestScore == float("-inf"):
        bestScore = self.evaluationFunction(gameState)
        bestAction = 'Center'
      return [bestScore, bestAction]

################################################################################
####                          AlphaBeta                                     ####
################################################################################


    def getGhostScoreAlphaBeta(self, ghostIndex, gameState, depth, alpha, beta):
      ghostActions = gameState.getLegalActions(ghostIndex)
      value = float("+inf")
      for action in ghostActions:
        nextState = gameState.generateSuccessor(ghostIndex, action)
        beta = min(beta, value)
        if ghostIndex + 1 < gameState.getNumAgents():
          score = self.getGhostScoreAlphaBeta(ghostIndex + 1, nextState, depth, alpha, beta)
        else:
          score = self.getPacmanScoreAlphaBeta(nextState, depth + 1, alpha, beta)
        score = score[0]
        if score < value:
          value = score
          bestAction = action
          if (value < alpha):
            break # prune
      if len(ghostActions) == 0:
        value = self.evaluationFunction(gameState)
      dummyAction = 'Center'
      return [value, dummyAction]

    def getPacmanScoreAlphaBeta(self, gameState, depth, alpha, beta):
      if (gameState.isWin() == True) or (depth > self.depth):
        bestScore = self.evaluationFunction(gameState)
        bestAction = 'Center'
        return [bestScore, bestAction]
      pacmanActions = gameState.getLegalActions(0)
      value = float("-inf")
      bestAction = None
      for action in pacmanActions:
        nextState = gameState.generateSuccessor(0, action)
        alpha = max(alpha, value)
        score = self.getGhostScoreAlphaBeta(1, nextState, depth, alpha, beta)
        score = score[0]
        if score > value:
          value = score
          bestAction = action
          if (beta < value):
            break # prune
      if len(pacmanActions) == 0:
        value = self.evaluationFunction(gameState)
        bestAction = 'Center'
      return [value, bestAction]

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
        """
        "*** YOUR CODE HERE ***"
        best = self.getPacmanScoreMiniMax(gameState, 1)
        return best[1]

        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        best = self.getPacmanScoreAlphaBeta(gameState, 1, float("-inf"), float("inf"))
        return best[1]
        # util.raiseNotDefined()

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
        best = self.getPacmanScoreExpectiMax(gameState, 1)
        return best[1]
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")
    score = scoreEvaluationFunction(currentGameState)
    newFood = currentGameState.getFood()
    foodPos = newFood.asList()
    nearestFoodDistance = float("inf")
    for p in foodPos:
        currdist = util.manhattanDistance(p, currentGameState.getPacmanPosition())
        nearestFoodDistance = min(currdist, nearestFoodDistance)
    numghosts = currentGameState.getNumAgents() - 1
    nearestGhost = float("inf")
    for i in range(1, numghosts+1):
        nextdist = util.manhattanDistance(currentGameState.getPacmanPosition(), currentGameState.getGhostPosition(i))
        nearestGhost = min(nearestGhost, nextdist)
        # print currentGameState.getGhostState(i)
    capsule = currentGameState.getCapsules()
    score -= 3.5 * len(capsule)
    # score += max(nearestGhost, 4) * 2
    score -= nearestFoodDistance * 1.5
    score -= 5 * len(foodPos)
    # numCapsules = len(capsule)
    # print numCapsules
    # if numCapsules < len(capsule):
    #   isChanged = True
    # if isChanged == True:
    #   score += nearestGhost*5
    #   if nearestGhost <= 1:
    #     isChanged = False
    return score
    # pos = currentGameState.getPacmanPosition()
    # food = currentGameState.getFood()
    # ghostPosition = currentGameState.getGhostPositions()
    # ghostStates = currentGameState.getGhostStates()
    # scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # if util.manhattanDistance(pos, ghostPosition[0]) <= 1.0:
    #   print "zzz"
    #   return 0
    # if food[pos[0]][pos[1]] == True:
    #   print "bbb"
    #   return 10001
    # q = util.Queue() 
    # q.push([pos, 1])
    # isVisited = [pos]
    # size = food.asList()
    # xlen = size[-1][0]
    # ylen = size[-1][1]
    # while not q.isEmpty():
    #   v = q.pop()
    #   x = v[0][0]
    #   y = v[0][1]
    #   if food[x][y] == True:
    #     # if (v[1]>3):
    #     #   r = 10000.0 - v[1]
    #     # else:
    #     #   r = 10000.0 - currentGameState.getNumFood()
    #     # r = 10000.0 - v[1] - 10*currentGameState.getNumFood()
    #     r = currentGameState.getScore() + 10000.0 - v[1]
    #     print r
    #     return r
    #   if (x-1 >= 0) and ((x-1, y) not in isVisited) and (not currentGameState.hasWall(x-1, y)):
    #     isVisited.append((x-1, y))
    #     q.push([(x-1, y), v[1]+1])
    #   if (y-1 >= 0) and ((x, y-1) not in isVisited) and (not currentGameState.hasWall(x, y-1)):
    #     isVisited.append((x, y-1))
    #     q.push([(x, y-1), v[1]+1])
    #   if (x+1 < xlen) and ((x+1, y) not in isVisited) and (not currentGameState.hasWall(x+1, y)):
    #     isVisited.append((x+1, y))
    #     q.push([(x+1, y), v[1]+1])
    #   if (y+1 < ylen) and ((x, y+1) not in isVisited) and (not currentGameState.hasWall(x, y+1)):
    #     isVisited.append((x, y+1))
    #     q.push([(x, y+1), v[1]+1])
    # print "aaa"
    # return 1
    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

