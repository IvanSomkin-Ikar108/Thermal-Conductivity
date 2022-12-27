import io
import math
import os
from datetime import datetime
from random import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_coef(m, l):
    matrix = np.array([[pow(-l + j, i) for j in range(m + l + 1)] for i in range(m + l + 1)])
    a = np.zeros(m + l + 1)
    a[1] = 1
    return np.linalg.solve(np.array(matrix), a)


class Area:

    def __init__(
            self,
            net_pos: (int, int),
            img_pos: (int, int),
    ):
        self.neighbors = []
        self.id = None
        self.net_pos = net_pos
        self.img_pos = img_pos
        self.temperature_history = [0, 0, 0, random()]

    def initialize_neighbors(self, areas: []):
        self.neighbors.clear()
        neighbor_positions: [] = self.calculate_neighbor_positions()
        for a in areas:
            if a.net_pos in neighbor_positions:
                self.neighbors.append(a)
        return self.neighbors

    def calculate_neighbor_positions(self) -> []:
        net_x, net_y = self.net_pos
        return [
            (net_x, net_y - 1),
            (net_x, net_y + 1),
            (net_x - 1, net_y),
            (net_x - 1, net_y)
        ]

    def reproduce_interact_pro(self, time: float, time_step: float, data_precision: int):
        t = time
        a = self.prey_reproduce_rate(t)
        b = self.predator_eating_rate
        c = self.predator_dying_rate
        dt = time_step
        x_im3 = self.prey_history[1]
        y_im3 = self.predator_history[1]
        x_im2 = self.prey_history[2]
        y_im2 = self.predator_history[2]
        x_im1 = self.prey
        y_im1 = self.predator
        x_i = (3 * x_im1 * dt * (a - y_im1) + (10 * x_im1 - 5 * x_im2 + x_im3)) / 6.
        y_i = (3 * y_im1 * dt * (b * x_im1 - c) + (10 * y_im1 - 5 * y_im2 + y_im3)) / 6
        if x_i < 0:
            x_i = 0
        if y_i < 0:
            y_i = 0
        self.prey = round(x_i, data_precision)
        self.predator = round(y_i, data_precision)
        self.prey_history.pop(0)
        self.prey_history.append(self.prey)
        self.predator_history.pop(0)
        self.predator_history.append(self.predator)


class ThermalConductivitySimulation:
    def __init__(self):
        self.render_step_period = None
        self.render = None
        self.steps = None

        self.net_height = None
        self.net_width = None
        self.output_directory = None
        self.areas: [Area] = []
        self.img_height = None
        self.img_width = None

    def initialize_areas_with_image(self, img_path):

        with Image.open(img_path) as im:
            img_bw = im.convert("1")
            self.img_width, self.img_height = im.size
            net_measure = 50
            self.net_width = math.ceil(self.img_width / net_measure) + 1
            self.net_height = math.ceil(self.img_height / net_measure) + 1

            self.areas = []
            for net_x in range(0, self.net_width, 1):
                for net_y in range(0, self.net_height, 1):
                    img_x = net_x * net_measure
                    img_y = net_y * net_measure
                    if (img_x < img_bw.width and img_y < img_bw.height
                            and self.is_pixel_black(img_bw, (img_x, img_y))):
                        self.areas.append(Area(net_pos=(net_x, net_y), img_pos=(img_x, img_y)))

            for a in self.areas:
                a.initialize_neighbors(self.areas)

    @staticmethod
    def is_pixel_black(img_bw, xy: (int, int)) -> bool:
        return img_bw.getpixel(xy) == 0

    def set_output_directory(self, output_directory):
        self.output_directory = output_directory

    def step(self):
        pass
        # temperatures = [a.temperature_history[-1] for a in self.areas]
        # prey_new = np.linalg.solve(self.prey_migration_matrix, prey_old)
        # predator_new = np.linalg.solve(self.predator_migration_matrix, predator_old)
        # for i in range(0, len(self.areas)):
        #     self.areas[i].prey = prey_new[i]
        #     self.areas[i].predator = predator_new[i]

    def add_run_result(self, a: Area, t: int):
        pass
        # self.run_result[a.id].predator.append(a.predator)
        # self.run_result[a.id].prey.append(a.prey)
        # self.run_result[a.id].t.append(t)

    def initialize_run_result(self):
        pass
        # self.run_result = dict()
        # for i in range(len(self.areas)):
        #     a = self.areas[i]
        #     a.id = i
        #     self.run_result[i] = PredatorPreyRunResult(a, time_precision, data_precision)

    def render_step(self, img_path: str):

        heatmap = np.zeros((self.net_width, self.net_height))

        for a in self.areas:
            heatmap[a.net_pos[1], a.net_pos[0]] = a.temperature_history[-1]

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(heatmap, cmap='hot', interpolation='nearest')
        fig.savefig(img_path)

    def run(self, steps: int, time_step: float, render: bool, render_step_period: int):

        self.steps = steps
        self.render = render
        self.render_step_period = render_step_period

        time = 0
        self.initialize_run_result()
        for a in self.areas:
            self.add_run_result(a, time)

        run_directory = "result"

        if render:
            run_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_directory = os.path.join(self.output_directory, f'run_{run_datetime}')
            os.mkdir(run_directory)

        for step in range(0, steps):
            self.step()

            time += time_step
            for a in self.areas:
                self.add_run_result(a, time)

            if render and step % render_step_period == 0:
                self.render_step(f'{run_directory}/step_{step}.jpg')


def main():
    simulation = ThermalConductivitySimulation()
    simulation.initialize_areas_with_image('./assets/brazil.png')
    simulation.set_output_directory('result')

    simulation.run(steps=1, time_step=0.001, render=True, render_step_period=1)


if __name__ == '__main__':
    main()
