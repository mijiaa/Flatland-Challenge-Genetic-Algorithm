from flatland.envs.rail_env import RailEnv
import numpy as np

class Map:
    def __init__(self, num_of_agents,max_len, env:RailEnv):
        self.num_of_agents= num_of_agents
        self.max_len = max_len
        self.occ_map =[]
        self.env = env
    def create_empty_map(self):
        r = self.env.rail.height
        c = self.env.rail.width
        map = np.zeros((self.num_of_agents,self.max_len, r, c))
        self.occ_map =[]
        return map

    def recompute_map(self,n, map):
        diff = n - self.max_len
        r = self.env.rail.height
        c = self.env.rail.width
        time_step = np.zeros(r,c)
        for i in range(diff):
            for j in range(self.num_of_agents):
                agent_map = map[j]
                agent_map.append(time_step)

        self.max_len = n


    def update_map(self, agent_id, path,n,map):
        if n > self.max_len:
            self.recompute_map(n,map)
        # collision = 0
        for index, location in enumerate(path):
            for i in range(self.num_of_agents):
                if i == agent_id:
                    continue
                agent_map = map[i][index]
                try:
                    nex_loc = path[index+1]
                    # print(location,"loc",i)
                    # print(nex_loc,"next")
                    # if agent_map[nex_loc] ==1 or agent_map[location] ==1 :
                    #     collision +=1
                    #     continue
                    agent_map[nex_loc] =1
                    agent_map[location] =1
                except IndexError:
                    pass
        return map



