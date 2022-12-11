import random
import numpy as np


class Graph():
    def __init__(self, nodes, distance, default_pheromone_level = None):
        self.nodes = nodes
        self.distance = distance
        assert distance.shape[1] == distance.shape[0]
        if default_pheromone_level:
            self.intensity = np.full_like(distance, default_pheromone_level).astype('float64')
        else:
            self.intensity = np.full_like(distance, self.distance.mean()*10).astype('float64')


    def __str__(self):
        return f'nodes: {str(self.nodes)}\n{self.distance}\n{self.intensity}'


test_graph = Graph(4, np.asarray([[0,10,15,20],[10,0,35,25],[15,35,0,30],[20,25,30,0]]).astype('float64'), )

best_so_far = [132, 133, 131, 130, 127, 128, 123, 129, 126, 125, 124, 0, 158, 1, 2, 3, 4, 152,
 151, 6, 5, 155, 156, 157, 153, 154, 150, 149, 148, 143, 142, 141, 139, 138, 140, 144, 145, 146,
  147, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17, 18, 19, 20, 22, 21, 23, 24, 26, 25, 27, 28, 29,
   30, 31, 32, 34, 33, 35, 36, 137, 136, 135, 134, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 73, 74, 75, 72, 70, 71, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53,
     52, 51, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
      106, 94, 95, 86, 87, 93, 88, 89, 90, 91, 92, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
       117, 118, 119, 120, 121, 122, 41]

def cycle_length(g, cycle):
    length = 0
    i = 0
    while i < len(cycle) -1:
        length += g.distance[cycle[i]][cycle[i+1]]
        i+=1
    length+= g.distance[cycle[i]][cycle[0]]
    return length

def add_artificial_good_cycle(g):
    size = g.distance.shape[0]

    for i in range(size-1):
        g.distance[i][i+1]/=10
    g.distance[size-1][0]/=10


def traverse(g, cycle):
    i = 0
    while i < len(cycle) -1:
        print([cycle[i], cycle[i+1]])
        print(g.distance[cycle[i]][cycle[i+1]])
        i+=1
    print([cycle[i], cycle[0]])
    print(g.distance[cycle[i]][cycle[0]])

def break_most_traversed_edge(g, constant):
    index = g.intensity.argmax()
    index = np.unravel_index(index, g.intensity.shape)
    g.distance[index]*=constant
    return index # for logging purposes
'''Создаим граф и зададим высокий уровень феромонов, чтобы стимулировать агентов активней исследовать.
Также зададим новый гиперпараметр - уровень деградации, на который будм умножать 
'''
def traverse_graph(g, source_node = 0):
    ALPHA = 0.9
    BETA = 1.5
    visited = np.asarray([1 for _ in range(g.nodes)]) #originally no nodes have been visited
    visited[source_node] = 0 # except the initial/source node.

    cycle = [source_node]
    steps = 0
    current = source_node
    total_length = 0
    while steps < g.nodes -1:

        jumps_neighbors = []
        jumps_values = []
        for node in range(g.nodes):
            if visited[node] != 0:
               pheromone_level = max(g.intensity[current][node], 1e-5) #constant added to encourage exploration
               v = (pheromone_level**ALPHA ) / (g.distance[current][node]**BETA) 
               jumps_neighbors.append(node)
               jumps_values.append(v)

        next_node = random.choices(jumps_neighbors, weights = jumps_values)[0] # weighted (normalized) choice
       
        visited[next_node] = 0
        current = next_node
        cycle.append(current)
        steps+=1

    total_length = cycle_length(g, cycle) # just adds all the distances
    assert len(list(set(cycle))) == len(cycle)
    return cycle, total_length

def ant_colony_optimization(g, verbose=True, iterations = 100, ants_per_iteration = 50, q = 10, degradation_factor = .9, use_inertia = False):
    best_cycle = best_so_far # can be pre set or set to None
    best_length = cycle_length(g, best_so_far) #hardcoded instance. Else use None
    if use_inertia: #this is adding pheromones everywhere if the process stagnates. This did not improve my results and is left off.
      old_best = None
      inertia = 0
      patience = 100

    for iteration in range(iterations):
        cycles = [traverse_graph(g, random.randint(0, g.nodes -1)) for _ in range(ants_per_iteration)] # could be trivially parallelized if not on Mac through multiprocessing
        cycles.sort(key = lambda x: x[1])
        cycles = cycles[: ants_per_iteration//2] #optionally keep best half.

        if best_cycle: #elitism
            cycles.append((best_cycle, best_length))
            if use_inertia:
                old_best = best_length

        for cycle, total_length in cycles: # pheromone update
            total_length = cycle_length(g, cycle)
            if total_length < best_length:
                best_length = total_length
                best_cycle = cycle

            q = 10
            delta = q/total_length
            i = 0
            while i < len(cycle) -1:
                g.intensity[cycle[i]][cycle[i+1]]+= delta
                i+=1
            g.intensity[cycle[i]][cycle[0]] += delta
            g.intensity *= degradation_factor
        
        if use_inertia and best_cycle:        
            if old_best == best_length:
                    inertia+=1
            else:
                inertia = 0
            if inertia > patience:
                g.intensity += g.intensity.mean() # applying shake

    return best_cycle