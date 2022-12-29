from agent import Agent
import numpy as np
from pettingzoo.mpe import simple_adversary_v2


agents = {}
actor_dims = []


env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=False)
env.reset()
n_agents = env.num_agents
actor_dims = [8, 10, 10]
critic_dims = sum(actor_dims)
n_actions = 5
alpha=0.01
beta=0.01
fc1=64
fc2=64
gamma=0.99
tau=0.01
chkpt_dir = ''

for agent_idx in env.possible_agents:
    agents[agent_idx] = Agent(
                                actor_dims = actor_dims,
                                critic_dims = critic_dims,
                                n_actions = n_actions,
                                n_agents = n_agents,
                                agent_idx = agent_idx ,
                                alpha=alpha,
                                beta=beta,
                                fc1=fc1,
                                fc2=fc2,
                                gamma=gamma,
                                tau=tau,
                                chkpt_dir=chkpt_dir
                            )
print("agents ", agents, )
print("actor_dims ", actor_dims)

print("agents action ", agents[env.possible_agents[0]].choose_action(observation = [1,2,3,4]))
