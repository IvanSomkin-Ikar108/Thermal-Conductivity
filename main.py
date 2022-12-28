import io
import math
import os
from datetime import datetime
from random import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_coef(l, m):
    matrix = np.array([[pow(-l + j, i) for j in range(m + l + 1)]
                       for i in range(m + l + 1)])
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
        self.neighbors_2 = []
        self.id = None
        self.net_pos = net_pos
        self.img_pos = img_pos
        self.x = 0.5
        self.y = 0.5
        self.temperature_history = [1., 1., 1., 1.]

    def initialize_neighbors(self, areas: []):
        self.neighbors.clear()
        neighbor_positions: [] = self.calculate_neighbor_positions()
        for a in areas:
            for n in range(len(neighbor_positions)):
                if a.net_pos == neighbor_positions[n]:
                    self.neighbors.append(a)
                    # if n < 2:
                    #     a.y += 0.25
                    # else:
                    #     a.x += 0.25

        return self.neighbors

    def calculate_neighbours_2(self):
        self.neighbors_2.clear()
        for neighbor in self.neighbors:
            for neighbor_2 in neighbor.neighbors:
                if neighbor_2 is not self and neighbor_2 not in self.neighbors and neighbor_2 not in self.neighbors_2:
                    self.neighbors_2.append(neighbor_2)

    def calculate_neighbor_positions(self) -> []:
        net_x, net_y = self.net_pos
        return [
            (net_x, net_y - 1),
            (net_x, net_y + 1),
            (net_x - 1, net_y),
            (net_x + 1, net_y)
        ]


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
        self.transition_matrix = None
        self.period = 2. * np.pi
        self.h = 1.
        self.l_m = (3, 0)
        self.time = 0.
        self.time_step = 0.01

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
                        self.areas.append(
                            Area(net_pos=(net_x, net_y), img_pos=(img_x, img_y)))

            for a in self.areas:
                a.initialize_neighbors(self.areas)
            for a in self.areas:
                a.calculate_neighbours_2()

    @staticmethod
    def is_pixel_black(img_bw, xy: (int, int)) -> bool:
        return img_bw.getpixel(xy) == 0

    def set_output_directory(self, output_directory):
        self.output_directory = output_directory

    def termal_step_rate(self, a: Area):
        if len(a.neighbors) < 4:
            return 1 + np.cos(self.period * self.time)
        return 0

    def step(self):
        main_coef = get_coef(*(self.l_m))
        left_hand = [-sum([a.temperature_history[i+1] * main_coef[i]
                          for i in range(len(main_coef) - 1)]) +
                     self.time_step * self.termal_step_rate(a)
                     for a in self.areas]

        new_temp = np.linalg.solve(self.transition_matrix, left_hand)
        for a in self.areas:
            a.temperature_history.pop(0)
            a.temperature_history.append(new_temp[a.id])
        print(self.areas[4].temperature_history[-1])

    def add_run_result(self, a: Area):
        pass
        # self.run_result[a.id].predator.append(a.predator)
        # self.run_result[a.id].prey.append(a.prey)
        # self.run_result[a.id].t.append(t)

    def initialize_run_result(self):
        self.run_result = dict()
        for i in range(len(self.areas)):
            a = self.areas[i]
            a.id = i
            # self.run_result[i] = PredatorPreyRunResult(
            #     a, time_precision, data_precision)

    def initialize_transition_matrix(self):
        matrix = np.zeros((len(self.areas), len(self.areas)),
                          int).astype('float')
        main_coef = get_coef(*(self.l_m))[-1]
        def_coef = 2. * self.time_step / pow(self.h, 2)
        for area in self.areas:
            matrix[area.id][area.id] = main_coef + \
                                       len(area.neighbors) * (area.x + area.y) * def_coef / 2
            for neighbour in area.neighbors:
                if area.net_pos[0] == neighbour.net_pos[0]:
                    matrix[area.id][neighbour.id] = - area.x * def_coef
                else:
                    matrix[area.id][neighbour.id] = - area.y * def_coef
            # for neighbour_2 in area.neighbors_2:
            #     matrix[area.id][neighbour_2.id] =\
            #         -1. * migration_rate *
        self.transition_matrix = matrix

    def render_step(self, img_path: str):

        heatmap = np.ones((self.net_width, self.net_height))
        heatmap[0][0] *= 10
        for a in self.areas:
            heatmap[a.net_pos[1], a.net_pos[0]] = a.temperature_history[-1]

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(heatmap, cmap='hot', interpolation='nearest')
        fig.savefig(img_path)

    def run(self, steps: int, time_step: float, render: bool, render_step_period: int):

        self.steps = steps
        self.render = render
        self.render_step_period = render_step_period
        self.time_step = time_step

        self.time = 0
        self.initialize_run_result()
        self.initialize_transition_matrix()
        for a in self.areas:
            self.add_run_result(a)

        run_directory = "result"

        if render:
            run_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_directory = os.path.join(
                self.output_directory, f'run_{run_datetime}')
            os.mkdir(run_directory)

        for step in range(0, steps):
            self.step()
            self.time += self.time_step

            for a in self.areas:
                self.add_run_result(a)

            if render and step % render_step_period == 0:
                self.render_step(f'{run_directory}/step_{step}.jpg')


def main():
    simulation = ThermalConductivitySimulation()
    simulation.initialize_areas_with_image('./assets/brazil.png')
    simulation.set_output_directory('result')

    simulation.run(steps=1000, time_step=0.01,
                   render=True, render_step_period=100)


if __name__ == '__main__':
    main()
