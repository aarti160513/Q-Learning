import os
import platform
import time

import gymnasium as gym
import numpy as np
from gymnasium import Env
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from config import settings


class Qtable:

    def __init__(self, rows: int, columns: int):
        self._qtable = np.zeros((rows, columns))

    @property
    def table(self):
        return self._qtable

    def row(self, index):
        return self._qtable[index, :]

    def update(self, state, action, reward, new_state):

        # implement QLearning algorithm here
        # Q(s,a) = Q(s,a) + α * [ r + λ * max(Q(s',a')) - Q(s,a)]

        delta = (
            reward
            + settings.gamma * np.max(self._qtable[new_state, :])
            - self._qtable[state, action]
        )

        self._qtable[state, action] = (
            self._qtable[state, action] + settings.alpha * delta
        )


class Explorer:

    @staticmethod
    def learn(episodes: int, env: Env, qtable: Qtable):

        rng = np.random.default_rng(settings.seed)

        for episode in range(1, episodes + 1):
            # reset the environment in every episode
            state, info = env.reset()

            while True:
                explr_explt_trdof = rng.uniform(0, 1)

                # Exploration vs Exploitation
                if explr_explt_trdof < settings.epsilon:
                    # Explore
                    action = env.action_space.sample()
                else:
                    # Exploit
                    row = qtable.row(state)
                    # Break ties randomly
                    if np.all(row) == row[0]:
                        action = env.action_space.sample()
                    else:
                        action = np.argmax(row)

                new_state, reward, temrinated, truncated, info = env.step(action)

                if temrinated or truncated:
                    qtable.update(state, action, reward, new_state)
                    break

                qtable.update(state, action, reward, new_state)
                state = new_state

    @staticmethod
    def traverse(env: Env, qtable: Qtable):
        state, info = env.reset()
        clear_screen_cmd = "cls" if platform.system() == "Windows" else "clear"
        while True:

            # clear the screen to show agent trevrsal as continous frames
            os.system(clear_screen_cmd)
            print(env.render(), end="")
            row = qtable.row(state)
            action = np.argmax(row)
            new_state, reward, temrinated, truncated, info = env.step(action)
            if temrinated or truncated:
                time.sleep(1)
                os.system(clear_screen_cmd)
                print(env.render(), end="")
                break
            state = new_state
            time.sleep(1)


if __name__ == "__main__":
    # create FrozenLake Environment in ansi mode
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False,
        render_mode="ansi",
        desc=generate_random_map(size=settings.size, p=settings.p, seed=settings.seed),
    )

    # initialise Qtable and the FrozenLake environment
    qtable = Qtable(env.observation_space.n, env.action_space.n)
    env.reset()

    # Learn and display the traversed path
    Explorer.learn(settings.episodes, env, qtable)
    Explorer.traverse(env, qtable)
