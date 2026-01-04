# blackjack-Qlearning

## Installation
1. Clone the repository:
```
git clone https://github.com/edison1x/blackjack-Qlearning.git
```
2. Navigate to the directory the code is saved:
```
cd blackjack-QLearning
```
3. Install requirements
```
pip install gymnasium matplotlib numpy tqdm
```
4. Run the program
```
python main.py
```
## About the project
The main aim of this project was to learn more about reinforcement learning by training an agent to play blackjack using Gymnasium.

## Environment
There are some things to note about the Gymnasium environment.
- It follows the Sutton and Barto rules of blackjack.
- The deck is "infinite" and cards drawn with replacement (so cannot card count).
- Only actions are Hit(1) or Stand(0).
- Rewards are +1 for a win, -1 for a loss and 0 for a draw

## Agent performance
The parameters used to train the agent were as follows:
 - learning rate = 0.001
 - number of episodes - 1,000,000
 - initial epsilon = 1.0
 - final epsilon = 0.1
 - eps_decay = initial_eps / (episodes/2)
 - discount factor = 0.95

I decided to train and evaluate the agent (by letting the agent play 10,000 games) peformance 3 times and achieved win rates of
 - 43.4%
 - 42.7%
 - 42.6%

## Hit/Stand Table
Below are the tables displaying whether the agent will hit or stand.
There are some noticeable results that can be drawn from the agent training but there are also slight differences.
Case 1 (No usable ace)
- it will always stand on 17+.
- if dealer shows 7-10, agent will hit from 12-16
- if dealer shows 2-6, agent will likely stand on 13-16
- if dealer shows 1, agent will hit on 12-16
- There are some differences when player has 12/13 and what to do when dealer has 2-4

Case 2 (Usable ace)
- Always hit on 17 or below
- hit on 18 in few cases (dealer shows 1/10 maybe even 9?)
<img width="1400" height="600" alt="table3" src="https://github.com/user-attachments/assets/b9155ea0-1023-4917-9bca-32a689ab3e86" />
<img width="1400" height="600" alt="table2" src="https://github.com/user-attachments/assets/d8eca810-785e-4df9-9090-11b710da8c21" />
<img width="1400" height="600" alt="table1" src="https://github.com/user-attachments/assets/6e284cb4-d9a7-4abc-970e-26c058c856c8" />

## References
https://gymnasium.farama.org/environments/toy_text/blackjack/
