TENNIS_ENVIRONMENT = "/home/jawhar/Desktop/udacity/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/Tennis.x86_64"
DEVICE = "cuda:0"

N_EPISODES = 4000


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
WEIGHT_DECAY = 0.0  # L2 weight decay
