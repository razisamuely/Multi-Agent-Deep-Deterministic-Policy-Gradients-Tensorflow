from agent import Agent
import numpy as np
from pettingzoo.mpe import simple_adversary_v2
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError


env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=False)
env.reset()

class MADDPG:

    def __init__(self,
                 n_agents,
                 actor_dims,
                 critic_dims,
                 n_actions,
                 alpha,
                 beta,
                 fc1,
                 fc2,
                 gamma,
                 tau,
                 chkpt_dir):

        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.critic_dims = critic_dims
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.fc1 = fc1
        self.fc2 = fc2
        self.gamma = gamma
        self.tau = tau
        self.chkpt_dir = chkpt_dir
        self.agents = {}

        for agent_idx in env.possible_agents:
            self.agents[agent_idx] = Agent(
                                        actor_dims = self.actor_dims,
                                        critic_dims = self.critic_dims,
                                        n_actions = self.n_actions,
                                        n_agents = self.n_agents,
                                        agent_idx = agent_idx,
                                        alpha = self.alpha,
                                        beta = self.beta,
                                        fc1 = self.fc1,
                                        fc2 = self.fc2,
                                        gamma = self.gamma,
                                        tau = self.tau,
                                        chkpt_dir = self.chkpt_dir
                                    )

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()
        print("1 rewards.shape" , rewards.shape)

        states  = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)
        dones   = tf.convert_to_tensor(dones, dtype=tf.float32)
        print("2 rewards.shape" , rewards.shape)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, (agent_name, agent) in enumerate(self.agents.items()):
            new_states = tf.convert_to_tensor(actor_new_states[agent_idx], dtype=tf.float32)

            new_pi = agent.target_actor(new_states)
            all_agents_new_actions.append(new_pi)

            mu_states = tf.convert_to_tensor(actor_states[agent_idx], dtype=tf.float32)
            pi = agent.actor(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = tf.concat(all_agents_new_actions, axis=1)
        mu = tf.concat(all_agents_new_mu_actions, axis=1)
        old_actions = tf.concat(old_agents_actions, axis=1)


        for agent_idx, (agent_name, agent) in enumerate(self.agents.items()):
            critic_value_ = agent.target_critic(states_, new_actions)
            critic_value_ = critic_value_.numpy()
            critic_value_[dones[:,0].numpy().astype(int)] = 0.0

            critic_value = agent.critic(states, old_actions)
            print("rewards.shape" , rewards.shape)
            print("rewards[:,agent_idx]" , tf.reshape(rewards[:,agent_idx],[1024,1]).shape)
            print("critic_value_.shape" , critic_value_.shape)
            print("agent_idx" , agent_idx)

            target = tf.reshape(rewards[:,agent_idx],[1024,1]) + agent.gamma*critic_value_
            print("target.shape = ", target.shape)
            print("critic_value.shape = ", critic_value_.shape)

            print("target = ", type(target))
            print("critic_value = ", type(critic_value_))

            critic_loss = MeanSquaredError(target.numpy(), critic_value)

        #     agent.critic.optimizer.zero_grad()
        #     critic_loss.backward(retain_graph=True)
        #     agent.critic.optimizer.step()
        #
        #     actor_loss = agent.critic.forward(states, mu).flatten()
        #     actor_loss = -T.mean(actor_loss)
        #     agent.actor.optimizer.zero_grad()
        #     actor_loss.backward(retain_graph=True)
        #     agent.actor.optimizer.step()
        #
        #     agent.update_network_parameters()
