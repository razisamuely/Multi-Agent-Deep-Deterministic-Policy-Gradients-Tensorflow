# import torch as T
from networks import ActorNetwork, CriticNetwork
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents,
                       agent_idx, chkpt_dir,alpha, beta, fc1, fc2, gamma, tau):
        self.actor_dims = actor_dims
        self.critic_dims = critic_dims
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.agent_name = 'agent_%s' % agent_idx
        self.agent_idx = agent_idx
        self.chkpt_dir=chkpt_dir
        self.alpha=alpha
        self.beta=beta
        self.fc1=fc1
        self.fc2=fc2
        self.gamma=gamma
        self.tau=tau


        self.actor = ActorNetwork(
                                    # alpha,
                                    # actor_dims,
                                    fc1_dims = self.fc1,
                                    fc2_dims = self.fc2,
                                    n_actions = self.n_actions,
                                    chkpt_dir=self.chkpt_dir,
                                    name=self.agent_name+'_actor'
                                  )
        self.critic = CriticNetwork(
                                    # beta, critic_dims,
                                    fc1_dims = self.fc1,
                                    fc2_dims = self.fc2,
                                    # n_agents,
                                    # n_actions,
                                    # chkpt_dir=chkpt_dir, name=self.agent_name+'_critic'
                                    )

        self.target_actor = ActorNetwork(
                                         # alpha, actor_dims,
                                         fc1_dims = fc1,
                                         fc2_dims = fc2,
                                         n_actions = self.n_actions,
                                         chkpt_dir=self.chkpt_dir,
                                         name=self.agent_name+'_actor'
                                         # chkpt_dir=chkpt_dir,
                                         # name=self.agent_name+'_target_actor'
                                         )
        self.target_critic = CriticNetwork(
                                            # beta, critic_dims,
                                            fc1_dims = fc1,
                                            fc2_dims = fc2,
                                            # n_agents, n_actions,
                                            # chkpt_dir=chkpt_dir,
                                            # name=self.agent_name+'_target_critic'
                                            )

        # self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        prob = self.actor(np.array([observation]))
        # noise = tf.random.uniform(shape=prob.shape,minval=0,maxval = 1 )
        # prob += noise
        prob = prob.numpy()[0]
        return prob


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
