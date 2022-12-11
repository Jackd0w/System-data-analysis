import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime


import math
import random

import math

import random

class Particle:
    def __init__(self, fitness_function, position_list=None, velocity_list=None):
        self.fitness = float('inf')
        self.fitness_best = float('inf')
        self.fitness_function = fitness_function

        self.position_list = [] if position_list is None else position_list
        self.velocity_list = [] if velocity_list is None else velocity_list
        self.personal_best_list = []

    def update_personal_best(self):
        self.fitness = self.fitness_function(self.position_list, self.velocity_list)

        if self.fitness < self.fitness_best:
            self.fitness_best = self.fitness
            self.personal_best_list = list(self.position_list)


    def set_velocity_list(self, new_velocities):
        self.velocity_list = list(new_velocities)

    def set_position_list(self, new_positions):
        self.position_list = list(new_positions)


class Global:
    def __init__(self, max_speed, c1_individuality_factor=2.05, c2_sociability_factor=2.05):
        self.c1_individuality_factor = c1_individuality_factor
        self.c2_sociability_factor = c2_sociability_factor
        self.max_speed = max_speed

    def get_new_velocity(self, inertia_coef, particle, swarm_global_best_pos, clerc=False):
        new_velocities_list = []
        for i in range(0, len(particle.position_list)):
            r1 = random.random()
            personal_component = self.c1_individuality_factor * r1 * (particle.personal_best_list[i] - particle.position_list[i])

            r2 = random.random()
            social_component = self.c2_sociability_factor * r2 * (swarm_global_best_pos[i] - particle.position_list[i])

            if clerc:
                new_velocity = particle.velocity_list[i] + personal_component + social_component
                new_velocity = inertia_coef * new_velocity
            else:
                actual_component = inertia_coef * particle.velocity_list[i]
                new_velocity = actual_component + personal_component + social_component

            if new_velocity > self.max_speed[1]:
                new_velocity = self.max_speed[1]
            elif new_velocity < self.max_speed[0]:
                new_velocity = self.max_speed[0]

            new_velocities_list.append(new_velocity)

        return new_velocities_list

    def __str__(self):
        return "Global Topology"


class Local:
    def __init__(self, max_speed, c1_individuality_factor=2.05, c2_sociability_factor=2.05):
        self.c1_individuality_factor = c1_individuality_factor
        self.c2_sociability_factor = c2_sociability_factor
        self.max_speed = max_speed

    def get_new_velocity(self, inertia_coef, particle, swarm, clerc=False):
        neighborhood_particle = self._get_nearest_neighborhood(particle, swarm)
        new_velocities_list = []
        for i in range(0, len(particle.position_list)):
            r1 = random.random()
            personal_component = self.c1_individuality_factor * r1 * (particle.personal_best_list[i] - particle.position_list[i])

            r2 = random.random()
            social_component = self.c2_sociability_factor * r2 * (neighborhood_particle.position_list[i] - particle.position_list[i])

            if clerc:
                new_velocity = particle.velocity_list[i] + personal_component + social_component
                new_velocity = inertia_coef * new_velocity
            else:
                actual_component = inertia_coef * particle.velocity_list[i]
                new_velocity = actual_component + personal_component + social_component

            if new_velocity > self.max_speed[1]:
                new_velocity = self.max_speed[1]
            elif new_velocity < self.max_speed[0]:
                new_velocity = self.max_speed[0]

            new_velocities_list.append(new_velocity)

        return new_velocities_list

    def _get_nearest_neighborhood(self, particle, swarm):
        index = swarm.index(particle)
        if index == len(swarm) - 1:
            return swarm[0]

        return swarm[index + 1]

        # implementation using euclidian distance
        # nearest_neighbor = float('inf')
        # other_particle = None

        # for neighbor_particle in swarm:
        #     value = self._calculate_euclidean_distance(particle, neighbor_particle)
        #     if value < nearest_neighbor:
        #         nearest_neighbor = value
        #         other_particle = neighbor_particle
        # return other_particle

    def _calculate_euclidean_distance(self, particle, particle2):
        some_of_terms = 0
        for i in range(0, len(particle.position_list)):
            some_of_terms += (particle.position_list[i] - particle2.position_list[i]) ** 2
        return (some_of_terms ** 0.5)

    def __str__(self):
        return "Local Topology"


def boot__function(x):
    sum_ = 0.0
    for i in range(1, len(x) - 1):
        sum_ += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
    return sum_

'''Fitness function'''
def sphere_function(position_list):
    fitness = 0
    for i in range(0, len(position_list)):
        fitness += position_list[i] ** 2
    return fitness


def rastrigin_function(x):
    f_x = [xi ** 2 - 10 * math.cos(2 * math.pi * xi) + 10 for xi in x]
    return sum(f_x)


def boot_function(x, y):
    f_x = (x + 2*y - 7) ** 2 + (2*x + y - 5)**2
    return (f_x)


def ackley_function(position_list):
    fitness = 0
    for i in range(0, len(position_list) - 1):
        part_1 = - 0.2 * math.sqrt(0.5 * (position_list[i] * position_list[i] + position_list[i + 1] * position_list[i + 1]))
        part_2 = 0.5 * (math.cos(2 * math.pi * position_list[i]) + math.cos(2 * math.pi * position_list[i + 1]))
        value_point = math.exp(1) + 20 - 20 * math.exp(part_1) - math.exp(part_2)

        fitness += value_point

    return fitness

def goldstein_price(x, y):
    # goal : (x, y) = (0., -1.0)
    term1 = 19 - 14 * x + 3 * x * x - 14 * y + 6 * x * y + 3 * y * y
    term2 = 1 + ((x + y + 1))**2 * term1
    term3 = 18 - 32 * x + 12 * x * x + 48 * y - 36 * x * y + 27 * y * y
    term4 = 30 + ((2 * x - 3 * y))**2 * term3
    z = term2 * term4
    return z



'''Algorithm ismplementation'''
class PSOAlgorithm:
    def __init__(
        self, topology, bound, max_speed, dimensions, num_particles, num_iterations,
        fitness_function, inertia_coef=0.9
    ):
        # Configurations
        self.bound = bound
        self.max_speed = max_speed
        self.dimensions = dimensions
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.fitness_function = fitness_function
        self.inertia_coef = inertia_coef

        # Internals
        self.topology = topology
        self.global_best = float('inf')
        self.global_best_position = []
        self.list_global_best_values = []
        self.swarm = []

        self._initialize_swarm()

    def _initialize_swarm(self):
        self.swarm = []
        for _ in range(self.num_particles):
            particle = Particle(
                position_list=[
                    random.uniform(self.bound[0], self.bound[1])
                    for i in range(self.dimensions)
                ],
                velocity_list=[
                    random.uniform(self.max_speed[0], self.max_speed[1])
                    for i in range(self.dimensions)
                ],
                fitness_function=self.fitness_function
            )
            self.swarm.append(particle)
        print(
            f"Swarm initialized with {self.num_particles} particles "
            f"and {self.dimensions} dimensions"
        )

    def updates_global_best(self, particle):
        new_best = particle.fitness

        if new_best < self.global_best:
            self.global_best = new_best
            self.global_best_position = list(particle.position_list)

    def updates_velocity(self, particle, inertia_type):
        if type(self.topology) == Global:
            new_velocities = self.topology.get_new_velocity(
                inertia_coef=self.inertia_coef,
                particle=particle,
                swarm_global_best_pos=self.global_best_position,
                clerc=(3 == inertia_type))
            particle.set_velocity_list(new_velocities)

        if type(self.topology) == Local:
            new_velocities = self.topology.get_new_velocity(
                inertia_coef=self.inertia_coef,
                particle=particle,
                swarm=self.swarm,
                clerc=(3 == inertia_type))
            particle.set_velocity_list(new_velocities)

    def updates_position(self, particle):
        new_positions = [
            particle.position_list[i] + particle.velocity_list[i]  # note that velocity list was already updated
            for i in range(0, len(particle.position_list))
        ]

        final_new_positions = []
        for pos in new_positions:
            if pos > self.bound[1]:
                final_new_positions.append(self.bound[1])
            elif pos < self.bound[0]:
                final_new_positions.append(self.bound[0])
            else:
                final_new_positions.append(pos)
        particle.set_position_list(final_new_positions)

    def updates_inertia_weight_if_necessary(self, inertia_type, i):
        if inertia_type == 1:
            self.inertia_coef = 0.8
        elif inertia_type == 2:
            self.inertia_coef = (0.9 - 0.4) * ((self.num_iterations - i)/self.num_iterations) + 0.4
        elif inertia_type == 3:
            greek_letter = self.topology.c1_individuality_factor + self.topology.c2_sociability_factor
            square_func_val = (greek_letter * greek_letter) - (4 * greek_letter)
            self.inertia_coef = 2 / abs(2 - greek_letter - math.sqrt(square_func_val))
        else:
            raise Exception()

    def search(self, inertia_type):

        for i in range(self.num_iterations):
            for particle in self.swarm:
                particle.update_personal_best()
                self.updates_global_best(particle)

            self.updates_inertia_weight_if_necessary(inertia_type, i)
            for particle in self.swarm:
                self.updates_velocity(particle, inertia_type)
                self.updates_position(particle)

            self.list_global_best_values.append(self.global_best)
            print(f"End of iteration {i}. Best result is {self.global_best}. Best position len is: {len(self.global_best_position)}")





def run_experiments_and_plot_graphs(fitness_function_name, fitness_function, topology):
    mpl.style.use('seaborn')

    list_global_best_values, runs = main(1, topology, fitness_function)
    description = f"Constant Inertia - {fitness_function_name} - {topology}"
    plot_boxplot(runs, fitness_function_name, description)

    list_global_best_values, runs = main(2, topology, fitness_function)
    description = f"Linear Inertia - {fitness_function_name} - {topology}"
    plot_boxplot(runs, fitness_function_name, description)

    list_global_best_values, runs = main(3, topology, fitness_function)
    description = f"Clerc - {fitness_function_name} - {topology}"
    plot_boxplot(runs, fitness_function_name, description)

    plot_covergence_graphs(fitness_function_name, fitness_function, topology)


def plot_covergence_graphs(fitness_function_name, fitness_function, topology):
    mpl.style.use('seaborn')
    fig, ax = plt.subplots()
    list_global_best_values, runs = main(1, topology, fitness_function)
    ax.plot(list(range(0, 50)), list_global_best_values, 'b', label=f"Constant - Best: {list_global_best_values[-1]:.2f}")
    ax.set_title("PSO Local Topology")
    ax.set_ylabel("Best Fitness")
    ax.set_xlabel("Iterations")

    list_global_best_values, runs = main(2, topology, fitness_function)
    ax.plot(list(range(0, 50)), list_global_best_values, 'r', label=f"Linear - Best: {list_global_best_values[-1]:.2f}")

    list_global_best_values, runs = main(3, topology, fitness_function)
    ax.plot(list(range(0, 50)), list_global_best_values, 'g', label=f"Clerc - Best: {list_global_best_values[-1]:.2f}")
    ax.legend()
    plt.savefig(f'PSO Convergence {fitness_function_name} {topology} Average 30 runs_2.png')


def plot_boxplot(best_fitness, function_name, description):
    fig1, ax1 = plt.subplots()
    ax1.set_title(f'BoxPlot Best Fitness for {function_name}: {description}')
    ax1.boxplot(best_fitness, patch_artist=True, showfliers=False)
    ax1.legend()
    plt.savefig(f'PSO {description} Boxplot {function_name}_2.png')


def save_global_best_values(inertia_type, pso_algorithm):
    now = datetime.now().strftime("%d-%m-%Y%H:%M:%S")
    filepath = f"results/{inertia_type}-{now}.txt"
    with open(filepath, "w") as f:
        f.write(f"[\n")
        for i in pso_algorithm.list_global_best_values:
            f.write(f"{str(i)},\n")
        f.write(f"]")


def main(inertia_type, topology, fitness_function):
    runs = []
    for _ in range(30):
        pso_algorithm = PSOAlgorithm(
            topology=topology,
            bound= [-10,10],
            dimensions=30,
            num_particles=500,
            num_iterations=50,
            max_speed=[-1, 1],
            fitness_function=fitness_function,
        )
        pso_algorithm.search(inertia_type)
        runs.append(pso_algorithm.list_global_best_values)

    return np.average(runs, axis=0), runs


topology_l = Local(max_speed=[-1, 1])
topology_g = Global(max_speed=[-1, 1])

# run_experiments_and_plot_graphs("Rastrigin", fitness_functions.rastrigin_function, topology_l)
#run_experiments_and_plot_graphs("Rastrigin", rastrigin_function, topology_g)
#run_experiments_and_plot_graphs("Rosenbrocks", boot_function, topology_l)
run_experiments_and_plot_graphs("Boot", boot_function, topology_g)
#run_experiments_and_plot_graphs("Goldstein_price", goldstein_price, topology_g)