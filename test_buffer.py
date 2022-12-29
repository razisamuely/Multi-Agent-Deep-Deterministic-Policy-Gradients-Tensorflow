from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_adversary_v2

env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=False, render_mode='human')
env.reset()


n_agents = env.num_agents
actor_dims = [8, 10, 10]
critic_dims = sum(actor_dims)
n_actions = 5
batch_size=1024
max_size = 1000000
batch_size = 1024

memory = MultiAgentReplayBuffer(max_size =  max_size,
                                critic_dims = critic_dims,
                                actor_dims = actor_dims,
                                n_actions = n_actions,
                                n_agents = n_agents,
                                batch_size=batch_size)

print("memory = ", memory)
