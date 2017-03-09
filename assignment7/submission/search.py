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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    """
    "*** YOUR CODE HERE ***"
    parent = [] # (state, parent, direction(from parent to state))
    isVisited = set()
    start = problem.getStartState()
    s = util.Stack()
    s.push([start, start, 'None'])
    currState = -1
    while (s.isEmpty() == False):
        r = s.pop()
        u = r[0]
        if u not in isVisited:
            isVisited.add(u)
            parent.append(r)
            if (problem.isGoalState(u)):
                currState = u
                break
            for v in problem.getSuccessors(u):
                x = [v[0], u, v[1]]
                s.push(x)
    path = []
    while (currState != start):
        index = [x[0] for x in parent].index(currState)
        path.append(parent[index][2])
        currState = parent[index][1]
    # print path
    return list(reversed(path))
    # print "Start:", problem.getStartState()
    # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    # print "Start's successors:", problem.getSuccessors(problem.getStartState())
    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    parent = [] # (state, parent, direction(from parent to state))
    isVisited = set()
    start = problem.getStartState()
    q = util.Queue()
    q.push(start)
    isVisited.add(start)
    currState = -1
    while (q.isEmpty() == False):
        r = q.pop()
        if (problem.isGoalState(r)):
            currState = r
            break
        for succ in problem.getSuccessors(r):
            u = succ[0]
            if u in isVisited:
                continue
            parent.append([u, r, succ[1]])
            isVisited.add(u)
            q.push(u)
    path = []
    while (currState != start):
        index = [x[0] for x in parent].index(currState)
        path.append(parent[index][2])
        currState = parent[index][1]
    # print path
    return list(reversed(path))
    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    parent = []
    parentTemp = []
    isVisited = set()
    start = problem.getStartState()
    pq = util.PriorityQueue()
    pq.update([start, start, 'None', 0], 0)
    while (pq.isEmpty() == False):
        r = pq.pop()
        u = r[0]
        s = r[:]
        s.pop()
        if u not in isVisited:
            isVisited.add(u)
            parent.append(s)
            if (problem.isGoalState(u)):
                currState = u
                break
            for v in problem.getSuccessors(u):
                x = [v[0], u, v[1], v[2] + r[3]]
                pq.push(x, v[2] + r[3])
    path = []
    while (currState != start):
        index = [x[0] for x in parent].index(currState)
        path.append(parent[index][2])
        currState = parent[index][1]
    # print path
    return list(reversed(path))

    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    parent = []
    parentTemp = []
    isVisited = set()
    start = problem.getStartState()
    pq = util.PriorityQueue()
    pq.update([start, start, 'None', 0], 0)
    while (pq.isEmpty() == False):
        r = pq.pop()
        u = r[0]
        s = r[:]
        s.pop()
        if u not in isVisited:
            isVisited.add(u)
            parent.append(s)
            if (problem.isGoalState(u)):
                currState = u
                break
            for v in problem.getSuccessors(u):
                x = [v[0], u, v[1], v[2] + r[3]]
                pq.push(x, v[2] + r[3] + heuristic(v[0], problem))
    path = []
    while (currState != start):
        index = [x[0] for x in parent].index(currState)
        path.append(parent[index][2])
        currState = parent[index][1]
    # print path
    return list(reversed(path))
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
