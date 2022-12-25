class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.01, beta=0.01, fc1=64,
                    fc2=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha,
                                    actor_dims, 
                                    fc1,
                                    fc2,
                                    n_actions,
                                  chkpt_dir=chkpt_dir,
                                  name=self.agent_name+'_actor'
                                  )

        # self.critic = CriticNetwork(beta, critic_dims,
        #                     fc1, fc2, n_agents, n_actions,
        #                     chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        # self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
        #                                 chkpt_dir=chkpt_dir,
        #                                 name=self.agent_name+'_target_actor')
        # self.target_critic = CriticNetwork(beta, critic_dims,
        #                                     fc1, fc2, n_agents, n_actions,
        #                                     chkpt_dir=chkpt_dir,
        #                                     name=self.agent_name+'_target_critic')
