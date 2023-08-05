# search.py
# ---------
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

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # 若开始位置及是目标位置，返回一个空列表
    if problem.isGoalState(problem.getStartState()) == True:
        return []

    st = util.Stack()  # 栈内存状态元组
    for position, action, cost in problem.getSuccessors(problem.getStartState()):
        st.push((position, action))
    answer = []  # 节省空间，只维护一个ans列表
    visit = [problem.getStartState()]

    while not st.isEmpty():
        # 获取头元素，我不知道为什么stack类没有top方法，如果只维护一个answer列表就不能直接pop栈顶
        now_position, now_action = st.list[-1]
        if now_position in visit:
            st.pop()
            if now_action == answer[-1]:  # 此点下的深搜已经完成，碰到自己再弹出ans
                answer.pop()
            continue

        answer.append(now_action)
        visit.append(now_position)
        if problem.isGoalState(now_position):
            break

        tag = 0  # tag 判断当前状态是否能继续搜索
        for next_position, action, cost in problem.getSuccessors(now_position):
            if not (next_position in visit):
                st.push((next_position, action))
                tag = 1

        # 若不能继续搜索，弹出栈和ans列表
        if tag == 0:
            st.pop()
            answer.pop()
            continue

    return answer
    util.raiseNotDefined()


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()) == True:
        return []
    qu = util.Queue()
    answer = []
    visit = [problem.getStartState()]

    for position, action, cost in problem.getSuccessors(problem.getStartState()):
        qu.push((position, [action]))  # 队列中存当前位置和路径

    while not qu.isEmpty():
        now_position, now_path = qu.pop()
        if now_position in visit:
            continue
        visit.append(now_position)

        if problem.isGoalState(now_position):
            answer = now_path
            break

        for next_position, action, cost in problem.getSuccessors(now_position):
            if not next_position in visit:
                qu.push((next_position, now_path+[action]))  # 路径更新

    return answer
    util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()) == True:
        return []

    myPriorityQueue = util.PriorityQueue()
    answer = []
    visit = [problem.getStartState()]

    for position, action, cost in problem.getSuccessors(problem.getStartState()):
        # 用cost做priority
        myPriorityQueue.push((position, [action], cost), cost)

    while not myPriorityQueue.isEmpty():
        now_position, now_path, now_cost = myPriorityQueue.pop()
        if now_position in visit:
            continue
        visit.append(now_position)

        if problem.isGoalState(now_position):
            answer = now_path
            break

        for next_position, action, cost in problem.getSuccessors(now_position):
            if not next_position in visit:
                # 这里可以用当前的cost+下一步的cost算action的cost，也可以直接调用problem.getCostOfActions()
                # 我认为用cost+下一步的cost更快速，但是会多花一部分空间存储每一步的cost
                myPriorityQueue.push(
                    (next_position, now_path+[action], now_cost+cost), now_cost+cost)

    return answer
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()) == True:
        return []
    myPriorityQueue = util.PriorityQueue()
    answer = []
    visit = [problem.getStartState()]

    for position, action, cost in problem.getSuccessors(problem.getStartState()):
        # 用cost+heuristic(position,problem)当作priority
        myPriorityQueue.push(
            (position, [action], cost), cost+heuristic(position, problem))

    while not myPriorityQueue.isEmpty():
        now_position, now_path, now_cost = myPriorityQueue.pop()
        if now_position in visit:
            continue
        visit.append(now_position)

        if problem.isGoalState(now_position):
            answer = now_path
            break

        for next_position, action, cost in problem.getSuccessors(now_position):
            if not next_position in visit:
                myPriorityQueue.push(
                    (next_position, now_path+[action], now_cost+cost), now_cost+cost+heuristic(next_position, problem))

    return answer
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
