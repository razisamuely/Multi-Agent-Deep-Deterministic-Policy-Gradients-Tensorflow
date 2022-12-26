
![image](https://openai.com/content/images/2017/06/simple_adv_maddpg_notag.gif)
  
# Multi-Agent-Deep-Deterministic-Policy-Gradients-Tensorflow

Solving multiagent agent problem using tensroflow of a simple multi-agent particle world with a continuous observation and discrete action space, along with some basic simulated physics.

The repository heavily relies on philtabor [pytorch implementation](https://github.com/philtabor/Multi-Agent-Deep-Deterministic-Policy-Gradients) for [Multi Agent Actor Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)

1. clone [Multi Agent Particle Environment(MAPE)](https://github.com/openai/multiagent-particle-envs) as detailed. 

2. cd to multiagent-particle-envs

3. Create [virtual environment](https://docs.python.org/3/library/venv.html) and activate it 

4. Install [required dependecies](https://github.com/openai/multiagent-particle-envs#:~:text=Python%20(3.5.4)%2C%20OpenAI%20gym%20(0.10.5)%2C%20numpy%20(1.14.5))

## Game rules 
1 adversary (red), N good agents (green), N landmarks (usually N=2). All agents observe position of landmarks and other agents. One landmark is the ‘target landmark’ (colored green). Good agents rewarded based on how close one of them is to the target landmark, but negatively rewarded if the adversary is close to target landmark. Adversary is rewarded based on how close it is to the target, but it doesn’t know which landmark is the target landmark. So good agents have to learn to ‘split up’ and cover all landmarks to deceive the adversary.

# Using petting zoo

1. Create python virtual env

2. install [petting-zoo](https://github.com/Farama-Foundation/PettingZoo) ```pip install 'pettingzoo[all]'```
