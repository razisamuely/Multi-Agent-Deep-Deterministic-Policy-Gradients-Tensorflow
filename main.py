
import numpy as np
from pettingzoo.mpe import simple_adversary_v2
from pettingzoo.mpe import simple_tag_v2
from agent import Agent
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer

if __name__ == '__main__':
    env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=True, render_mode='human')

    render=True
    episodes=100
    avg_score_interval = 100
    PRINT_INTERVAL = 5
    score_history = []
    evaluate = False
    total_steps= 0


    agents = {}
    actor_dims = []


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
    max_size = 1000000
    batch_size = 1024

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

    maddpg_agents = MADDPG( actor_dims = actor_dims,
                            critic_dims = critic_dims,
                            n_actions = n_actions,
                            n_agents = n_agents,
                            alpha = alpha,
                            beta = beta,
                            fc1 = fc1,
                            fc2 = fc2,
                            gamma = gamma,
                            tau = tau,
                            chkpt_dir = chkpt_dir)


    # Inint memory
    memory = MultiAgentReplayBuffer( max_size =  max_size,
                                     critic_dims = critic_dims,
                                     actor_dims = actor_dims,
                                     n_actions = n_actions,
                                     n_agents = n_agents,
                                     batch_size=batch_size)


    completed_episodes = 0

    while completed_episodes < episodes:
        env.reset()
        total_reward = 0
        for  agent in env.agent_iter():

            if render:
                env.render()

            obs, reward, termination, truncation, _ = env.last()
            done = False

            total_reward += reward

            if termination or truncation:
                done = True

            action = maddpg_agents.agents[agent].choose_action(obs)

            memory.store_transition(raw_obs = obs,
                                    action = action,
                                    reward = reward,
                                    done = done)

            action = None if done else action
            env.step(action)

            if total_steps % 300 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            total_reward += reward
            total_steps += 1

        score_history.append(total_reward)
        avg_score = np.mean(score_history[-avg_score_interval:])
        completed_episodes += 1
        if completed_episodes % PRINT_INTERVAL == 0 :
            print('episode', completed_episodes, 'average score {:.1f}'.format(avg_score))

    if render:
        env.close()
