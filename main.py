from src.agent import Agent
from src.utils import train, evaluate_agent, create_policy
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("Blackjack-v1", sab=True)
    #params
    lr = 0.001
    episodes = 1000000
    start_eps = 1.0
    end_eps = 0.1
    eps_decay = start_eps/ (episodes/2)
    agent = Agent(learning_rate=lr, epsilon=start_eps, final_epsilon=end_eps, decay=eps_decay, episodes=episodes)
    train(env, agent)
    print(f"Win rate: {evaluate_agent(env, agent, episodes=10000):.1f}%")
    create_policy(agent)