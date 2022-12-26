import numpy as np
from pettingzoo.utils import random_demo
# from pettingzoo.mpe import simple_adversary_v2
from pettingzoo.mpe import simple_tag_v2
# from pettingzoo.magent import adversarial_pursuit_v3

def tag():
    env = simple_tag_v2.env(num_good=2, num_adversaries=1, num_obstacles=1, render_mode='human')
    env.reset()
    for i in range (1000):
        x = 0
        while x < 1:
            for agent in env.agent_iter():
                observation, reward, done, info, _ = env.last()
                if done:
                    action = None
                    env.reset()
                    print("DONE>>>>")
                    break
                else:
                    action = env.action_space(agent).sample()
                    #print("ACTION = ", action)
                env.step(action)
                #env.step(0)
                print(agent, " >>> ",  len(observation), observation)
                env.render()
            x += 1

def demo_adversary():
    env = simple_adversary_v2.env()
    random_demo(env, render=True, episodes=10)

def demo_adv_pursuit():
    env = adversarial_pursuit_v3.env(map_size=20)
    random_demo(env, render=True, episodes=10)

def demo_tag():
    env = simple_tag_v2.env(num_good=4, num_adversaries=4, num_obstacles=3, render_mode='human')
    random_demo(env, render=True, episodes=100)

if __name__ == '__main__':
    tag()
  #  demo_adversary()
    #demo_adv_pursuit()
    #demo_tag()
