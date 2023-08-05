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
import random
import util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        # print(newScaredTimes)

        # 后继状态已经赢或输直接返回得分
        if successorGameState.isWin() or successorGameState.isLose():
            return successorGameState.getScore()

        food_list = newFood.asList()
        # 最近的食物
        nearest_food = min([manhattanDistance(food, newPos) for food in food_list])
        # 最近的鬼
        nearest_ghost = min(
            [
                manhattanDistance(ghoststate.getPosition(), newPos)
                for ghoststate in newGhostStates
            ]
        )
        # 如果鬼被恐惧（吃豆人可以吃它，且得分是食物的二十倍）权值为200，食物权值为10
        if newScaredTimes[0]:
            answer =  200/nearest_ghost + 10/nearest_food + successorGameState.getScore()
        # 否则鬼权值为-10，食物权值为10
        else:
            answer = -10/nearest_ghost + 10/nearest_food + successorGameState.getScore()
        return answer
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        # 返回第一层确定的行动

        return self.MaxStep(gameState, 0, 1)
        util.raiseNotDefined()

    def MaxStep(self, state: GameState, agentIndex, depth):
        actions = state.getLegalActions(0)
        evascore = -float("inf")  # 评估分数

        # 当前状态赢或输或已经搜索到要求的最大深度，直接返回得分
        if state.isWin() or state.isLose() or depth == self.depth + 1:
            return self.evaluationFunction(state)

        # bestaction = ""
        for action in actions:
            successor = state.generateSuccessor(agentIndex, action)
            # 调用MinStep
            minstep = self.MinStep(successor, agentIndex + 1, depth)
            # 若大于当前评估分数，则更新评估分数和最佳行动
            if minstep > evascore:
                bestaction = action
                evascore = minstep
        # 深度为1返回行动
        if depth == 1:
            return bestaction
        # 否则返回分数
        else:
            return evascore

    def MinStep(self, state: GameState, agentIndex, depth):
        agentnums = state.getNumAgents()
        actions = state.getLegalActions(agentIndex)
        evascore = float("inf")  # 评估分数

        # 当前状态赢或输或已经搜索到要求的最大深度，直接返回得分
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        for action in actions:
            successor = state.generateSuccessor(agentIndex, action)
            # 若鬼智能体还没搜索完，继续搜索鬼
            if agentIndex < agentnums - 1:
                minstep = self.MinStep(successor, agentIndex + 1, depth)
                # 更新评估分数
                if minstep < evascore:
                    evascore = minstep
            # 否则进入下一层，搜索吃豆人的分数
            else:
                maxstep = self.MaxStep(successor, 0, depth + 1)
                # 更新评估分数
                if maxstep < evascore:
                    evascore = maxstep

        return evascore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        return self.MaxStep(gameState, 0, 1, -float("inf"), float("inf"))
        util.raiseNotDefined()

    def MaxStep(self, state: GameState, agentIndex, depth, maxscore, minscore):
        actions = state.getLegalActions(0)
        evascore = -float("inf")  # 评估分数

        # 当前状态赢或输或已经搜索到要求的最大深度，直接返回得分
        if state.isWin() or state.isLose() or depth == self.depth + 1:
            return self.evaluationFunction(state)

        # bestaction = ""
        for action in actions:
            successor = state.generateSuccessor(agentIndex, action)
            # 搜索min层
            minstep = self.MinStep(successor, agentIndex + 1, depth, maxscore, minscore)
            # 若min层返回的值大于评估值，更新
            if minstep > evascore:
                bestaction = action
                evascore = minstep
                # 若评估值大于minscore，beta剪枝
                if evascore > minscore:
                    return evascore
                # 若评估值大于maxscore，更新maxscore
                if evascore > maxscore:
                    maxscore = evascore

        # 深度为1返回行动
        if depth == 1:
            return bestaction
        # 否则返回分数
        else:
            return evascore

    def MinStep(self, state: GameState, agentIndex, depth, maxscore, minscore):
        agentnums = state.getNumAgents()
        actions = state.getLegalActions(agentIndex)
        evascore = float("inf")  # 评估分数

        # 当前状态赢或输或已经搜索到要求的最大深度，直接返回得分
        if state.isWin() or state.isLose() or depth == self.depth + 1:
            return self.evaluationFunction(state)

        for action in actions:
            successor = state.generateSuccessor(agentIndex, action)
            # 若鬼还未搜索完，继续搜索鬼
            if agentIndex < agentnums - 1:
                # 搜索min层
                minstep = self.MinStep(
                    successor, agentIndex + 1, depth, maxscore, minscore
                )
                if minstep < evascore:
                    evascore = minstep
                    # 若评估值小于maxscore，alpha剪枝
                    if evascore < maxscore:
                        return evascore
                    # 若评估值小于minscore，更新minscore
                    if evascore < minscore:
                        minscore = evascore
            # 否则进入下一层，搜索max层
            else:              
                maxstep = self.MaxStep(successor, 0, depth + 1, maxscore, minscore)
                if maxstep < evascore:
                    evascore = maxstep
                    # 若评估值小于maxscore，alpha剪枝
                    if evascore < maxscore:
                        return evascore
                    # 若评估值小于minscore，更新minscore
                    if evascore < minscore:
                        minscore = evascore

        return evascore


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        return self.Pacmanstep(gameState,1,0)
        util.raiseNotDefined()

    def Pacmanstep(self, state:GameState, depth, agentIndex = 0):
        actions = state.getLegalActions(agentIndex)

        # 当前状态赢或输或已经搜索到要求的最大深度，直接返回得分
        if depth == self.depth + 1 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        evascore = -float('inf')
        for action in actions:
            successor = state.generateSuccessor(agentIndex,action)
            # 搜索鬼的下一步
            ghoststep = self.Ghoststep(successor,depth,1)
            # 更新评估分数，最佳行动
            if ghoststep > evascore:
                bestaction = action
                evascore = ghoststep
        
        if depth == 1:
            return bestaction
        else:
            return evascore


    def Ghoststep(self, state:GameState, depth, agentIndex = 1):
        actions = state.getLegalActions(agentIndex)
        agentnums = state.getNumAgents()

        # 当前状态赢或输或已经搜索到要求的最大深度，直接返回得分
        if depth == self.depth + 1 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        evascore = 0
        for action in actions:
            successor = state.generateSuccessor(agentIndex,action)
            # 若鬼还未搜索完，继续搜索鬼
            if agentIndex < agentnums - 1:
                evascore += self.Ghoststep(successor,depth,agentIndex+1)
            # 否则进入下一层，搜索吃豆人
            else:
                evascore += self.Pacmanstep(successor,depth + 1, 0)

        # 返回得分期望
        return evascore / len(actions)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # 当前吃豆人的位置
    curPacmanpos = currentGameState.getPacmanPosition()
    # 当前食物
    curFood = currentGameState.getFood()
    # 当前鬼位置
    curGhostspos = currentGameState.getGhostPositions() 
    # 当前鬼被恐惧时间
    curScaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]

    if currentGameState.isWin() or currentGameState.isLose():
        return currentGameState.getScore()
    
    food_list = curFood.asList()
    # 最近的食物
    nearestfood = min(manhattanDistance(curPacmanpos,foodpos) for foodpos in food_list)
    # 最近的鬼
    nearestghost = min(manhattanDistance(curPacmanpos,curGhostpos) for curGhostpos in curGhostspos)

    score = 0 
    # 如果鬼被恐惧（吃豆人可以吃它，且得分是食物的二十倍）权值为200，食物权值为10
    if curScaredTimes[0]:
        score = 200 / nearestghost + 10 / nearestfood + currentGameState.getScore()
    # 否则鬼权值为-10
    else:
        score = - 10 / nearestghost + 10 / nearestfood + currentGameState.getScore()

    return score 
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
