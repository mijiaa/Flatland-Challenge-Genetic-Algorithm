from flatland.envs.rail_env import RailEnv
from astar import a_star_modified
from occupany_map import Map
from flatland.envs.rail_env import RailEnv


def search(agent_order, env: RailEnv):

    num_of_agents = env.number_of_agents

    max_steps = env._max_episode_steps
    agents_actions = [[] for _ in range(num_of_agents)]
    map = Map(num_of_agents,max_steps,env)
    occupancy_map = map.create_empty_map()
    g_lst = []
    ord = []

    for order in agent_order:
        agent = env.agents[order.agent_id]
        start = agent.initial_position
        end = agent.target

        agent_map = occupancy_map[order.agent_id]  # used to check free tiles and expand
        agent_path,agent_action, g_val = a_star_modified(env, agent,order.agent_id, start, end, agent_map)
        agents_actions[order.agent_id] = agent_action
        occupancy_map = map.update_map(order.agent_id, agent_path,len(agent_path),occupancy_map)
        g_lst.append(g_val)
        ord.append(order.agent_id)

    max_len = max([len(i) for i in agents_actions])
    schedule = []

    for i in range(len(agents_actions)):
        while len(agents_actions[i]) < max_len:
            agents_actions[i].append(4)

    for m in range(max_len):
        _actions = {}
        for n in range(num_of_agents):
            _actions[n] = agents_actions[n][m]
        schedule.append(_actions)

    return schedule, sum(g_lst)

