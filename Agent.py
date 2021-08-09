class AgentState:
    def __init__(self, agent_index, position, incoming_dir=None, time_step=0, action=2 ,status =False,parent=None,expanded= False):
        self.position = position
        self.agent_index = agent_index
        self.parent = parent
        self.incoming_dir = incoming_dir
        self.action = action
        self.time_step = time_step
        self.expanded = expanded
        self.status =status
        self.g = 0  # cost
        self.h = 0  # cost to goal node
        self.f = 0  # total cost

    def __lt__(self, other):
        """
        Heap can compare node with other nodes with error
        Can change to self.g
        """
        return self.h < other.h



