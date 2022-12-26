"""
Playing around with MPE PettingZoo Environments: https://www.pettingzoo.ml/mpe
"""

import numpy as np
from pettingzoo.utils import random_demo
from pettingzoo.mpe import simple_adversary_v2
from pettingzoo.mpe import simple_tag_v2
# from pettingzoo.magent import adversarial_pursuit_v3

def tag(render=True, episodes=10):
    env = simple_tag_v2.env(num_good=1, num_adversaries=1, num_obstacles=1, render_mode='human')
    """Runs an env object with random actions."""
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
            elif isinstance(obs, dict) and "action_mask" in obs:
                action = random.choice(np.flatnonzero(obs["action_mask"]))
            else:
                action = env.action_space(agent).sample()
            env.step(action)

        completed_episodes += 1

    if render:
        env.close()

def demo_adversary():
    env = simple_adversary_v2.env()
    random_demo(env, render=True, episodes=10)

def demo_adv_pursuit():
    env = adversarial_pursuit_v3.env(map_size=20)
    random_demo(env, render=True, episodes=10)

def demo_tag():
    env = simple_tag_v2.env(num_good=4, num_adversaries=4, num_obstacles=3, render_mode='human')
    random_demo(env, render=True, episodes=10)

if __name__ == '__main__':
    tag()
    #demo_adversary()
    #demo_adv_pursuit()
    # demo_tag()
