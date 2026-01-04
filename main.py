import gymnasium as gym
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# follow Sutton and Barto rules of blackjack
# in this version the deck is "infinite" and cards are drawn with replacement, meaning you cant count cards.
# The player is dealt 2 cards, and dealer shows 1 card and hides the other.
# Player can only hit or stand (provided not busted over 21 already).
# Dealer hits until sum is 17+, if dealer busts (over 21) the player wins.
# rewards are as follows, +1 for a win, -1 for a loss and 0 for a draw.
#
# natural is ignored (when blackjack in first 2 cards, natural usually gives player reward +1 (if dealer has no natural)

class Agent:
    def __init__(self, learning_rate, epsilon, final_epsilon, decay):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.decay = decay
        self.discount = 0.95 #value used in docs
        # create a q values table initialising q values for unseen states
        # n=2 for blackjack (stand/hit)
        self. q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, obs):
        #uses epsilon-greedy selection to balance exploration and exploitation
        if np.random.random()< self.epsilon:
            # exploration (to improve agent knowledge), so choose random action
            return env.action_space.sample()
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

def train(env):
    #params
    lr = 0.001
    episodes = 1000000
    start_eps = 1.0
    end_eps = 0.1
    eps_decay = start_eps/ (episodes/2)
    agent = Agent(learning_rate=lr, epsilon=start_eps, final_epsilon=end_eps,decay=eps_decay)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=episodes)
    for ep in tqdm(range(episodes)):
        obs,info = env.reset()
        done = False
        while not done:
            #choose next action
            action = agent.get_action(obs)
            #perform next action
            next_obs, reward, terminated, truncated, info = env.step(action)
            #update agent q values based on action
            agent.update(obs, action, reward, terminated, next_obs)
            #update if hand complete and the current obs
            done = terminated or truncated
            obs = next_obs
        agent.decay_epsilon()

    return agent

def create_policy_grid(agent):
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)
    # hand has no usable ace
    policy_noace = np.zeros((len(player_sums), len(dealer_cards)))
    # hand has usable ace
    policy_ace = np.zeros((len(player_sums), len(dealer_cards)))
    for i, player_sum in enumerate(player_sums):
        for j, dealer_card in enumerate(dealer_cards):
            # no usable ace
            state_noace = (player_sum, dealer_card, False)
            policy_noace[i, j] = np.argmax(agent.q_values[state_noace])
            state_ace = (player_sum, dealer_card, True)
            policy_ace[i, j] = np.argmax(agent.q_values[state_ace])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # no usuable ace policy
    im1 = ax1.imshow(policy_noace, cmap='Accent_r', vmin=0, vmax=1, aspect='auto')
    ax1.set_title('No usable ace', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dealer Showing')
    ax1.set_ylabel('Player Sum')
    ax1.set_xticks(range(len(dealer_cards)))
    ax1.set_xticklabels(dealer_cards)
    ax1.set_yticks(range(len(player_sums)))
    ax1.set_yticklabels(player_sums)
    # usable ace policu
    im2 = ax2.imshow(policy_ace, cmap='Accent_r', vmin=0, vmax=1, aspect='auto')
    ax2.set_title('Usable ace', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dealer Showing')
    ax2.set_ylabel('Player Sum')
    ax2.set_xticks(range(len(dealer_cards)))
    ax2.set_xticklabels(dealer_cards)
    ax2.set_yticks(range(len(player_sums)))
    ax2.set_yticklabels(player_sums)

    plt.show()


def evaluate_agent(env, agent, episodes):
    wins = 0
    total_episodes = 0
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            #To test the agent we no longer explore, only greedy!!
            action = int(np.argmax(agent.q_values[obs]))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if reward > 0:
            wins += 1
        total_episodes += 1
    win_rate = wins / total_episodes * 100
    print(f"Win rate over {episodes} episodes: {win_rate:.1f}%")

if __name__ == "__main__":
    env = gym.make("Blackjack-v1", sab=True)
    agent = train(env)
    evaluate_agent(env, agent, episodes=10000)
    create_policy_grid(agent)

