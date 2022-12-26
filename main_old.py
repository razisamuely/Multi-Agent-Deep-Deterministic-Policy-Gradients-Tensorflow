import numpy as np
# from maddpg import MADDPG
# from buffer import MultiAgentReplayBuffer
# from make_env import make_env
from pettingzoo.mpe import simple_adversary_v2


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    #scenario = 'simple'
    # scenario = 'simple_adversary'
    env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=False)
    # n_agents = env.N
    n_agents = 2 
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = env.action_space[0].n
    # maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
    #                        fc1=64, fc2=64,
    #                        alpha=0.01, beta=0.01, scenario=scenario,
    #                        chkpt_dir='tmp/maddpg/')
    #
    # memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
    #                     n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 500
    N_GAMES = 50000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0
    avg_score_interval = 100

    # if evaluate:
    #     maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                env.render()
                #time.sleep(0.1) # to slow down the action for the video
            # actions = maddpg_agents.choose_action(obs)
            actions =  [np.array([0.69479346, 0.32742965, 1.1195362 , 0.84051263, 0.05457191], dtype=np.float32),
                        np.array([0.7225561 , 0.8595222 , 0.78903913, 0.96785694, 1.8510532 ], dtype=np.float32),
                        np.array([0.9638283 , 0.3712235 , 1.3028334 , 0.23278804, 0.6143129 ], dtype=np.float32)]

            obs_, reward, done, info = env.step(actions)

            # state = obs_list_to_state_vector(obs)
            # state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = [True]*n_agents

            # memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            # if total_steps % 100 == 0 and not evaluate:
            #     maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-avg_score_interval:])
        if not evaluate:
            if avg_score > best_score:
                # maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))