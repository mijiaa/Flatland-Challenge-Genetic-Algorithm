import heapq
from flatland.envs.rail_env import RailEnv
from flatland.core.transition_map import GridTransitionMap
import numpy as np
from Agent import AgentState

def return_path(current_node):

    path = []
    curr = current_node
    g = current_node.g
    actions_taken = []
    while curr is not None:
        path.append(curr.position)
        if curr.action is not None:
            actions_taken.append(curr.action)
        curr = curr.parent

    return path[::-1], actions_taken[::-1],g  # return reverse path


def get_action(dir_index, outgoing):
    dir_cor = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    if dir_index is 0:
        if outgoing == dir_cor[0] or outgoing == dir_cor[2]:
            return 2  # forward
        if outgoing == dir_cor[1]:
            return 3  # right
        if outgoing == dir_cor[3]:
            return 1  # left

    if dir_index is 1:
        if outgoing == dir_cor[1] or outgoing == dir_cor[3]:
            return 2
        if outgoing == dir_cor[2]:
            return 3
        if outgoing == dir_cor[0]:
            return 1

    if dir_index is 2:
        if outgoing == dir_cor[2] or outgoing == dir_cor[0]:
            return 2
        if outgoing == dir_cor[3]:
            return 3
        if outgoing == dir_cor[1]:
            return 1

    if dir_index is 3:
        if outgoing == dir_cor[3] or outgoing == dir_cor[1]:
            return 2
        if outgoing == dir_cor[0]:
            return 3
        if outgoing == dir_cor[2]:
            return 1


def surrounding(map,time_step, location,current_loc,valid_dir):


    if map[time_step+1][location] or map[time_step][location]:
        return False
    if map[time_step][current_loc]:
        return False

    # edge cases for some deadlock caused by considering the possible location in the next time step
    surrounding_loc = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)]) + location
    #
    for index,loc in enumerate(surrounding_loc):
        if valid_dir[index] == 1:
            if map[time_step + 2][(loc[0], loc[1])] == 1:
                return False

    return True



def get_possible_directions_(env, current_node,map):
    dir_cor = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    position = current_node.position
    incoming = current_node.incoming_dir
    valid_dir_bool = GridTransitionMap.get_transitions(env.rail, current_node.position[0], current_node.position[1],
                                                       incoming)
    valid_direction, dir_index = [], []
    stop = False
    for i in range(len(valid_dir_bool)):
        if valid_dir_bool[i]:
            next_loc = (position[0] + dir_cor[i][0], position[1] + dir_cor[i][1])
            next_valid_dir = GridTransitionMap.get_transitions(env.rail, next_loc[0], next_loc[1],i)
            #                                            i)
            if surrounding(map, current_node.time_step, next_loc,position,next_valid_dir):
                valid_direction.append(dir_cor[i])
                dir_index.append(i)

    if map[current_node.time_step][position] == 0:
        stop = True
    if len(valid_direction) == 0:
        return 0,0,stop

    return valid_direction, dir_index,stop


def a_star_modified(env, agent, id, start, end, map):

    open = []
    closed = []
    deadline = agent.deadline
    release_date = agent.release_date
    max_time_step = env._max_episode_steps
    # current_time_step = 0
    start_node = AgentState(id, start, agent.initial_direction)
    end_node = AgentState(id, end)

    heapq.heappush(open, (start_node.f,start_node.h,start_node))
    # expand
    while len(open) > 0:

        item= heapq.heappop(open)
        current_node = item[2]
        closed.append(current_node)

        # if node is goal then backtrack
        if current_node.position == end_node.position:
            return return_path(current_node)


        # indicate that current node has been expanded
        current_node.expanded= True

        # after picking a node, find children
        children = []

        # relaxation, restrict possible directions using grid information
        possible_directions, direction_index,stop = get_possible_directions_(env, current_node, map)

        if possible_directions is 0:

            wait_node = AgentState(current_node.agent_index, current_node.position, current_node.incoming_dir,
                                  current_node.time_step + 1, 4, True, current_node)

            children.append(wait_node)

        else:
            # create node for possible next locations
            for index, dir in enumerate(possible_directions):
                incoming = direction_index[index]
                new_position = (current_node.position[0] + dir[0], current_node.position[1] + dir[1])
                action = get_action(current_node.incoming_dir, dir)
                new_node = AgentState(current_node.agent_index, new_position,incoming, current_node.time_step + 1,
                                      action, True,  current_node)
                children.append(new_node)

            # if stop:
            #     new_node = AgentState(current_node.agent_index, current_node.position, current_node.incoming_dir,
            #                           current_node.time_step+1, 4, True, current_node)
            #     children.append(new_node)


        # do all the necessary checking and calculation
        for child in children:

            child.g = current_node.g + 1 + get_time_penalty(child.time_step, max_time_step, release_date, deadline,
                                                                 child.status)
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + \
                      ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h


            if child is not None and child not in closed and (child.f, child.h,child) not in open:
                heapq.heappush(open, (child.f, child.h, child))



def get_time_penalty(current_time_step, max_time_step,release_date, deadline, agent_status):

    # By default, no penalty.
    penalty = 0

    # # Penalty only applies if the agent is active.
    if agent_status:

        # Compute the number of steps the agent is outside bounds.
        steps_outside = 0

        if current_time_step <= release_date:
            steps_outside = 1 + release_date - current_time_step
        if current_time_step >= deadline:
            steps_outside = 1 + current_time_step - deadline

        # Compute the normalized penalty.
        penalty = -((steps_outside * steps_outside) / (max_time_step * max_time_step / 4))

    return penalty