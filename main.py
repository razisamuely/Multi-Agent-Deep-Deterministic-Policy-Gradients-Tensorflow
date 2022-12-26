
import numpy as np
from pettingzoo.mpe import simple_adversary_v2
from pettingzoo.mpe import simple_tag_v2

if __name__ == '__main__':
    render=True
    episodes=30
    avg_score_interval = 100
    PRINT_INTERVAL = 5
    score_history = []

    env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=False, render_mode='human')


    # Init agaents
    # maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
    #                        fc1=64, fc2=64,
    #                        alpha=0.01, beta=0.01, scenario=scenario,
    #                        chkpt_dir='tmp/maddpg/')

    # Inint memory
    # memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
    #                        n_actions, n_agents, batch_size=1024)

    total_reward = 0
    completed_episodes = 0

    while completed_episodes < episodes:
        env.reset()
        for agent in env.agent_iter():
            if render:
                env.render()

            obs, reward, termination, truncation, _ = env.last()
            total_reward += reward

            if termination or truncation:
                action = None

            else:
                # actions = maddpg_agents.choose_action(obs)

                action = env.action_space(agent).sample()

            env.step(action)
            # if total_steps % 100 == 0 and not evaluate:
            #     maddpg_agents.learn(memory)

        score_history.append(total_reward)
        avg_score = np.mean(score_history[-avg_score_interval:])
        completed_episodes += 1
        if completed_episodes % PRINT_INTERVAL == 0 :
            print('episode', completed_episodes, 'average score {:.1f}'.format(avg_score))

    if render:
        env.close()
