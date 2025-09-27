import heapq
# You do not need any other imports

def astar_search(starting_state):
    '''
    Implementation of A* search algorithm

    Input:
        starting_state: an SearchState object

    Return:
        A path consisting of a list of SearchState states
        The first state should be starting_state
        The last state should have state.goal_test()() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search algorithm
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state) and obtain the path
    #   - keep track of the distance of each state from start node
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of SearchState is implemented
    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of SearchState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    # TODO(III): implement the rest of the A* search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.generate_successors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    # Your code here ---------------
    while frontier:
        current = heapq.heappop(frontier)
        if current.goal_test() == True:
            return reconstruct_path(visited_states, current)
        current_g = visited_states[current][1]
        for neighbor in current.generate_successors():
            g_new = current_g + neighbor.path_cost
            previous = visited_states.get(neighbor)
            if previous is None or g_new < previous[1]:
                visited_states[neighbor] = (current, g_new)

                f_new = g_new + neighbor.calculate_heuristic()
                heapq.heappush(frontier, (f_new, neighbor))
    # ------------------------------
    # if you do not find the goal return an empty list
    return []

# TODO(III): implement backtrack method, to be called by astar_search upon reaching goal_state
# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
def reconstruct_path(visited_states, goal_state):
    path = []
    # Your code here ---------------
    current = goal_state
    while current:
        path.insert(0,current)
        current = visited_states[current][0]
    # ------------------------------
    return path
