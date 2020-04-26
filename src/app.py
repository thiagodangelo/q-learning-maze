# coding: utf-8
from maze_env import Maze
from rl_agent import QLearningTable

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pylab as plt


EPISODE_COUNT = 100
episodes = range(EPISODE_COUNT)
rewards = []
movements = []

def run_experiment():
    for episode in episodes:
        print(f"Episode {episode}/{EPISODE_COUNT}")
        observation = env.reset(is_random=True)
        moves = 0
        while True:
            env.render()
            action = q_learning_agent.choose_action(str(observation))
            observation_, reward, done = env.get_state_reward(action)
            moves += 1
            
            q_learning_agent.learn(str(observation), action, reward, str(observation_))
            observation = observation_

            if done:
                movements.append(moves)
                rewards.append(reward)
                print(f"Reward: {reward}, Moves: {moves}")
                break

    print("Game Over!")
    plot_reward_movements()

def plot_reward_movements():
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(episodes, movements)
    plt.xlabel("Episode")
    plt.ylabel("#Movements")

    plt.subplot(2, 1, 2)
    plt.step(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("reward_moviments_qlearning.png")
    plt.show()

if __name__ == "__main__":
    env = Maze()
    q_learning_agent = QLearningTable(actions=list(range(env.n_action)))
    env.window.after(10, run_experiment)
    env.window.mainloop()
