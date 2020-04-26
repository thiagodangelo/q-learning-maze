# coding: utf-8
import numpy as np
import time
import sys

import tkinter as tk

UNIT = 40
MAZE_H = 6
MAZE_W = 6


class Maze:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Q-Learning Maze")
        self.window.geometry(f"{MAZE_W * UNIT}x{MAZE_H * UNIT}")
        self.action_space = ["u", "d", "r", "l"]
        self.n_action = len(self.action_space)
        self.build_maze()

    def build_maze(self):
        self.canvas = tk.Canvas(
            self.window, bg="white", width=MAZE_W * UNIT, height=MAZE_H * UNIT
        )

        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_W * UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([20, 20])

        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15,
            hell1_center[1] - 15,
            hell1_center[0] + 15,
            hell1_center[1] + 15,
            fill="black",
        )

        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15,
            hell2_center[1] - 15,
            hell2_center[0] + 15,
            hell2_center[1] + 15,
            fill="black",
        )

        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15,
            oval_center[1] - 15,
            oval_center[0] + 15,
            oval_center[1] + 15,
            fill="yellow",
        )

        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15, origin[0] + 15, origin[1] + 15, fill="red",
        )

        self.canvas.pack()

    def render(self):
        time.sleep(0.1)
        self.window.update()

    def reset(self, is_random=False):
        self.window.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        random = np.array([0, 0])
        if is_random:
            x = np.random.randint(0, MAZE_W)
            y = np.random.randint(0, MAZE_H)
            random = np.array([x * UNIT, y * UNIT])
        origin = np.array([20, 20]) + random
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15, origin[0] + 15, origin[1] + 15, fill="red",
        )
        return self.canvas.coords(self.rect)

    def get_state_reward(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])
        s_ = self.canvas.coords(self.rect)
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = "terminal"
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
            s_ = "terminal"
        else:
            reward = 0
            done = False
        return s_, reward, done


if __name__ == "__main__":
    maze = Maze()
    maze.build_maze()
    maze.window.mainloop()
