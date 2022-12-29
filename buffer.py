import numpy as np
from pettingzoo.mpe import simple_adversary_v2

class MultiAgentReplayBuffer:
    def __init__(self,
                 max_size,
                 critic_dims,
                 actor_dims,
                 n_actions,
                 n_agents,
                 batch_size):

        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

        self.states = np.array([])
        self.rewards = []
        self.dones = []

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                            np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                            np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs,
                               action,
                               reward,
                               done):

        index = self.mem_cntr // self.n_agents
        agent_idx = self.mem_cntr % self.n_agents
        self.actor_state_memory[agent_idx][index] = raw_obs

        self.actor_action_memory[agent_idx][index] = action
        self.states = np.concatenate([self.states, raw_obs])
        self.rewards += [reward]
        self.dones += [done]

        if len(self.dones) == 3 :
            # print("agent_idx " , agent_idx)
            # print("index " , agent_idx)
            # print("self.states ", self.states)
            # print("self.rewards ", self.rewards)
            # print("self.dones ", self.dones)

            self.state_memory[index] = self.states
            self.reward_memory[index] = self.rewards
            self.terminal_memory[index] = self.dones

            self.states = np.array([])
            self.rewards = []
            self.dones = []

        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr//self.n_agents, self.mem_size)
        print("self.batch_size self.mem_cntr, self.mem_size",self.batch_size, self.mem_cntr, self.mem_size)
        print("max_mem self.batch_size",max_mem, self.batch_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)


        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.state_memory[batch + 1]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_state_memory[agent_idx][batch + 1])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, \
               actor_new_states, states_, terminal

    def ready(self):
        if (self.mem_cntr//3) >= self.batch_size:
            return True
