from collections import defaultdict
import numpy as np

class Agent:
    def __init__(self, env, learning_rate, epsilon, final_epsilon, decay):
        self.env = env
        self.lr = learning_rate
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.decay = decay
        self.discount = 0.95 #value used in docs
        # create a q values table initialising q values for unseen states
        # n=2 for blackjack (stand/hit)
        self. q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def get_action(self, obs):
        #uses epsilon-greedy selection to balance exploration and exploitation
        if np.random.random()< self.epsilon:
            # exploration (to improve agent knowledge), so choose random action
            return self.env.action_space.sample()
        # explotation (chooses greedy action for most reward)
        return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        #new value (temporal difference) = reward + discount factor* estimate of optimal future value
        if terminated:
            temporal_difference = reward
        else:
            temporal_difference = reward + self.discount*np.max(self.q_values[next_obs])
        # Q(s,a) <- (1-α)Q(s,a) + α[r + γmaxQ(s', a)]
        self.q_values[obs][action] = (1-self.lr)*self.q_values[obs][action] + self.lr*temporal_difference

    def decay_epsilon(self):
        # linear epsilon decay to slowly reduce exploration to exploitation
        self.epsilon = max(self.final_epsilon, self.epsilon- self.decay)
