import gymnasium as gym
import tqdm
import matplotlib as plt
import numpy as np
from agent import Agent

def train(env, agent, episodes):
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

def create_policy(agent):
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
    return win_rate

