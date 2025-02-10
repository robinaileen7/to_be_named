import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_state(n):
    np.random.seed(10)
    angle = np.random.normal(0, 1, n)
    real = [math.cos(x) for x in angle]
    img = [math.sin(x) for x in angle]
    real_norm = [x / np.sqrt(n) for x in real]
    img_norm = [x / np.sqrt(n) for x in img]
    psi_ls = [complex(x, y) for x, y in zip(real_norm, img_norm)]
    return psi_ls

class record_states:
    def __init__(self, param):
        self.n = param[0]
        self.r = param[1]
        self.angle = param[2]
        self.step = param[3]

    def save(self, norm_adj):
        n = self.n
        r = self.r
        angle = self.angle
        step = self.step
        np.random.seed(0)
        state_init = generate_state(n)
        x_start = [x.real for x in state_init]
        y_start = [x.imag for x in state_init]
        x_y_save = [[x_start, y_start]]
        for i in range(step):
            rand = np.random.uniform(0, 1, n)
            x_end = [x + r * math.cos(angle) if z > 0.5 else x + r * math.cos(-angle) for x, z in zip(x_start, rand)]
            y_end = [y + r * math.sin(angle) if z > 0.5 else y + r * math.sin(-angle) for y, z in zip(y_start, rand)]
            if norm_adj:
                beta = [(1/n)/(x ** 2 + y ** 2) for x, y in zip(x_end, y_end)]
                x_end = [np.sqrt(b)*x for b, x in zip(beta, x_end)]
                y_end = [np.sqrt(b)*y for b, y in zip(beta, y_end)]
            x_y_save.append([x_end, y_end])
            x_start = x_end
            y_start = y_end
        return x_y_save

    @staticmethod
    def draw_states(norm_adj):
        data = record_states(param).save(norm_adj)
        fig, ax = plt.subplots()

        def init():
            ax.set_xlim(-1, 10)
            ax.set_ylim(-2.2, 1.5)
            plt.xlabel('Real')
            plt.ylabel('Img')

        def init_norm():
            ax.set_xlim(0, 0.15)
            ax.set_ylim(-0.15, 0.15)
            plt.xlabel('Real')
            plt.ylabel('Img')

        def update(i):
            use_init = True
            ax.clear()
            ax.scatter(data[i][0], data[i][1], s = 10)
            if use_init:
                if norm_adj:
                    init_norm()
                else:
                    init()

        animation = FuncAnimation(fig, update, frames=param[3] + 1, interval=200, repeat=False)
        plt.show()

    @staticmethod
    def draw_dir(norm_adj):
        data = record_states(param).save(norm_adj)
        data_1 = data[:-1]
        data_2 = data[1:]
        fig, ax = plt.subplots()

        def init():
            plt.xlabel('Position')
            ax.set_ylim(-3, 8)
            plt.legend()

        def init_norm():
            plt.xlabel('Position')
            ax.set_ylim(-0.2, 0.2)
            plt.legend()

        def update(i):
            use_init = True
            ax.clear()
            x_grid = np.arange(0, param[0], 1)
            ax.scatter(x_grid, data[i][0], s=2, color='red', label='Real')
            ax.scatter(x_grid, data[i][1], s=2, color='blue', label='Img')
            for j in range(len(x_grid)):
                ax.arrow(x_grid[j], data_1[i][0][j], 0, data_2[i][0][j]-data_1[i][0][j], head_width=0.5,
                         head_length=0.01, fc='red', ec='red')
                ax.arrow(x_grid[j], data_1[i][1][j], 0, data_2[i][1][j] - data_1[i][1][j], head_width=0.5,
                         head_length=0.01, fc='blue', ec='blue')
            plt.grid()
            if use_init:
                if norm_adj:
                    init_norm()
                else:
                    init()

        animation = FuncAnimation(fig, update, frames=param[3], interval=1000, repeat=False)
        plt.show()

if __name__ == "__main__":
    param = [50, 0.1, math.pi / 4, 100]

    obj = record_states(param)
    obj.draw_states(norm_adj=True)
    obj.draw_dir(norm_adj=True)