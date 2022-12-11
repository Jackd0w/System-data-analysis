import math
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


def graph_construction(k_iteration_max, count_of_vertices, minimum_temperature):
    graph = nx.Graph()
    mass_iteration = [0]
    mass_all_trail_weight = []
    mass_all_trail = []
    mass_temperature = [100]
    mass_changes = []
    mass_iteration_count = [0]

    table_graph = len_weight_write_in_graph(graph, count_of_vertices)
    trail_calculation(count_of_vertices, mass_all_trail)
    calculation_length_trail(mass_all_trail[-1], table_graph, mass_all_trail_weight)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=1)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.savefig('../static/1.png')
    while mass_temperature[-1] > minimum_temperature and k_iteration_max + 1 > len(mass_iteration):
        print()
        random_change_road(mass_all_trail, count_of_vertices, mass_changes)
        length_trail_new = calculation_length_trail_new(mass_all_trail[-1], table_graph)
        annealing(mass_all_trail_weight, length_trail_new, mass_iteration,
                  mass_temperature, mass_iteration_count, mass_all_trail)
    if mass_temperature[-1] <= minimum_temperature:
        print("Достижение минимальной температуры!!!")
    else:
        print("Выполнились все циклы!!!")
    print(f'Всего циклов выполнелось - {mass_iteration[-1]}')
    print(
        f'Финальный маршрут ', end='')
    for i in mass_all_trail[-1]:
        print(i, end=' -> ')
    print(f'\nS = {mass_all_trail_weight[-1]}, ответ найден на - {mass_iteration_count[-1]} шаге')
    print(f'{mass_iteration_count}  {mass_all_trail_weight}')
    graph_final_data(count_of_vertices, mass_all_trail[-1], table_graph)
    create_plots(mass_iteration_count, mass_all_trail_weight)


def create_plots(mass_iteration_count, mass_all_trail_weight):
    fig3, axs3 = plt.subplots(1)

    fig3 = pd.DataFrame({'voltage': mass_iteration_count, 'val': mass_all_trail_weight})
    # fig3 = plt.plot(mass_iteration_count, mass_all_trail_weight)
    axs3 = fig3.plot.bar(x='voltage', y='val', rot=0)
    axs3.set_xlabel('Итерации')
    axs3.set_ylabel('Длинна пути')
    axs3.set_title('График показывающий уменьшении пути по итерациям')
    plt.savefig('../static/3.png')


def len_weight_write_in_graph(graph, count_of_vertices):
    len_weight = random_len_calculations(count_of_vertices)
    for i in range(len(len_weight)):
        graph.add_edge(len_weight[i][0], len_weight[i][1], weight=len_weight[i][2])
    return len_weight


def random_len_calculations(count_of_vertices):
    # len_weight = []
    # for i in range(1, count_of_vertices + 1):
    #     for j in range(i + 1, count_of_vertices + 1):
    #         len_weight.append([i, j, random.randint(50, 101)])
    len_weight = [[1, 2, 52], [1, 3, 73], [1, 4, 87], [1, 5, 66], [1, 6, 89], [2, 3, 60], [2, 4, 59], [2, 5, 54],
                  [2, 6, 90], [3, 4, 100], [3, 5, 79], [3, 6, 79], [4, 5, 58], [4, 6, 93], [5, 6, 69]]
    return len_weight


def trail_calculation(count_of_vertices, mass_all_trail):
    trail = []
    # while len(trail) != count_of_vertices:
    #     cache = random.randint(1, count_of_vertices)
    #     if cache not in trail:
    #         trail.append(cache)
    trail = [1, 4, 3, 5, 2, 6]
    trail.append(trail[0])
    mass_all_trail.append(trail)


def calculation_length_trail(last_trail, table_graph, mass_all_trail_weight):
    length_original_route = 0
    for i in range(len(last_trail) - 1):
        for j in range(len(table_graph)):
            if last_trail[i] == table_graph[j][0] and last_trail[i + 1] == table_graph[j][1] or \
                    last_trail[i + 1] == table_graph[j][0] and last_trail[i] == table_graph[j][1]:
                length_original_route += table_graph[j][2]
    mass_all_trail_weight.append(length_original_route)
    # print(length_original_route)


def calculation_length_trail_new(last_trail, table_graph):
    length_original_route = 0
    for i in range(len(last_trail) - 1):
        for j in range(len(table_graph)):
            if last_trail[i] == table_graph[j][0] and last_trail[i + 1] == table_graph[j][1] or \
                    last_trail[i + 1] == table_graph[j][0] and last_trail[i] == table_graph[j][1]:
                # print(last_trail[i], '  ', table_graph[j][0])
                # print(last_trail[i + 1], '  ', table_graph[j][1])
                # print(last_trail[i + 1], '  ', table_graph[j][0])
                # print(last_trail[i], '  ', table_graph[j][1])
                length_original_route += table_graph[j][2]
    return length_original_route


def random_change_road(mass_all_trail, count_of_vertices, mass_changes):
    trail_change_mass = mass_all_trail[-1]
    mass_to_changes = False
    close_peaks = False
    while mass_to_changes is not True:
        first_number = round(random.randrange(1, count_of_vertices))
        second_number = round(random.randrange(1, count_of_vertices))
        if first_number != second_number:
            for i in range(1, len(trail_change_mass)-1):
                if trail_change_mass[i] == first_number and trail_change_mass[i + 1] == second_number \
                        or trail_change_mass[i + 1] == first_number and trail_change_mass[i] == second_number:
                    close_peaks = True
            if not close_peaks:
                for i in range(len(trail_change_mass)):
                    if trail_change_mass[i] == second_number:
                        trail_change_mass[i] = first_number
                    elif trail_change_mass[i] == first_number:
                        trail_change_mass[i] = second_number
                if trail_change_mass in mass_all_trail:
                    mass_changes.append([first_number, second_number])
                    mass_to_changes = True
                    mass_all_trail.append(trail_change_mass)
        close_peaks = False
    # print(f'Рассмотрим новый путь №{mass_iteration[-1]} = ', end='')
    # for i in trail_change_mass:
    #     print(i, end=' -> ')
    # print()


def annealing(mass_all_trail_weight, new_trail, mass_iteration, mass_temperature, mass_iteration_count, mass_all_trail):
    mass_iteration.append(mass_iteration[-1] + 1)
    print(f'Старая температура: {mass_temperature[-1]}')
    print(f'Старый путь S - {mass_all_trail_weight[-1]}')
    print(f'Новый путь S - {new_trail}')
    delta = new_trail - mass_all_trail_weight[-1]
    print(f'delta S = {new_trail}', '-', f'{mass_all_trail_weight[-1]} = {delta}')
    if delta < 0:
        print(f'{delta} < 0')
        print('Расчёт температуры:')
        print(f'{mass_temperature[0]} / {math.log(1 + mass_iteration[-1])}')
        new_temp = round(mass_temperature[0] / math.log(1 + mass_iteration[-1]), 3)
        mass_temperature.append(new_temp)
        print(f'Новая температура - {mass_temperature[-1]}')
        mass_all_trail_weight.append(new_trail)
        mass_iteration_count.append(mass_iteration[-1])
        print('Новый путь: ')
        for i in mass_all_trail[-1]:
            print(i, end=' -> ')
        print()
    elif delta > 0:
        print(f'{delta} > 0')
        # p = 100 * math.e ** (-new_trail/mass_temperature[-1])
        p = (mass_temperature[0] * math.e) ** (- delta / mass_temperature[-1])
        random_p = round((random.uniform(1, mass_temperature[-1])), 5)
        print(random_p)
        print(f'{p} =?= {random_p}')
        if p >= random_p:
            print(f'{p} >= {random_p}')
            mass_all_trail_weight.append(new_trail)
            print(mass_all_trail_weight)
            mass_iteration_count.append(mass_iteration[-1])
            print('Новый путь: ')
            for i in mass_all_trail[-1]:
                print(i, end=' -> ')
            print()
        else:
            print(f'{p} < {random_p}')
            print('Не принимаем маршрут!')
            print('Старый путь: ')
            for i in mass_all_trail[mass_iteration_count[-1]-1]:
                print(i, end=' -> ')
            print()
            # mass_all_trail_weight.append(mass_all_trail_weight[-1])
    print('=' * 100)


def graph_final_data(count_of_vertices, mass_all_trail, table_graph):
    graph_final = nx.Graph()
    fig2, axs2 = plt.subplots(1)
    for i in range(len(mass_all_trail) - 1):
        for j in range(len(table_graph)):
            if mass_all_trail[i] == table_graph[j][0] and mass_all_trail[i + 1] == table_graph[j][1] or \
                    mass_all_trail[i + 1] == table_graph[j][0] and mass_all_trail[i] == table_graph[j][1]:
                graph_final.add_edge(table_graph[j][0], table_graph[j][1], weight=table_graph[j][2])
    fig2 = nx.spring_layout(graph_final)
    nx.draw(graph_final, fig2, with_labels=1)
    edge_labelss = nx.get_edge_attributes(graph_final, 'weight')
    nx.draw_networkx_edge_labels(graph_final, fig2, edge_labels=edge_labelss)
    plt.savefig('../static/2.png')


if __name__ == '__main__':
    # number_of_vertices = int(input("Введите вершин графа: "))
    count_of_vertices = 6
    # k_iteration_max = int(input("Введите максимальное число итераций: "))
    k_iteration_max = 200
    # k_iteration_max = 400
    # k_iteration_max = int(input("Введите минимаотную температуру: "))
    minimum_temperature = 20

    graph_construction(k_iteration_max, count_of_vertices, minimum_temperature)
    # plt.show()