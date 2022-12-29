from networks import ActorNetwork, CriticNetwork
import numpy as np


network  = CriticNetwork(fc1_dims = 10,
                         fc2_dims = 10)

obs = np.array([[-1.297303, 3346, -1.297303 ,1.2814946, -0.53843164]])

reaults = network(obs)
print("CriticNetwork reaults = ",reaults)



network  = ActorNetwork(fc1_dims = 10,
                        fc2_dims = 10,
                        n_actions = 2)

obs = np.array([[-1.297303, 1.2814946, -1.297303 ,1.2814946, -0.53843164]])

reaults = network(obs)
print("ActorNetwork reaults = ",reaults)
