import numpy as np
# from maddpg import MADDPG
# from buffer import MultiAgentReplayBuffer
# from make_env import make_env
from pettingzoo.mpe import simple_adversary_v2

env = simple_adversary_v2.env(N=2, max_cycles=1000000, continuous_actions=False, render_mode='human')
env.reset()
for agent in env.agent_iter():
    print(agent)
    observation, reward, termination, truncation, info = env.last()
    action = None if termination or truncation else env.action_space(agent).sample()  # this is where you would insert your policy
    env.step(action)
    env.render()
    print("possible_agents"  , env.possible_agents)
