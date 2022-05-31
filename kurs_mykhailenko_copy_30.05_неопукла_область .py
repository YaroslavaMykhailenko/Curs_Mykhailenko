from turtle import color
import numpy as np
import copy
import math
from numpy import sqrt, sum, hstack, ones, zeros, array, fill_diagonal, argsort, sin, cos
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.animation as animation
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from numpy.linalg import norm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


################# Параметры ##################

def PowFunction(x):
    PowFunction.counter += 1
    return pow((10 * pow((x[0] - x[1]), 2) + pow((x[0] - 1), 2)), 4)


PowFunction.counter = 0  # количество вызовов функции

x0 = np.array([-1.2, 0])  # начальная вершина
L = 1  # длина ребра

alpha = 1
gamma = 2.
betta_inside = -0.5
betta_outside = 0.5

tolerance = 1e-35  # точность, для критерия остановы
maxNumIter = 200  # максимальное количество итераций

M = 4  # количество зацикливаний вершины


# дані для опуклої області
limit_cond = 1  # значение радиуса для выпуклой обл
x_center = -1
y_center = 0

# дані для опуклої області

limit_cond1 = 1.2 # значение радиуса для невыпуклой обл (радиус большего круга)
x_center_circle1 = -0.5
y_center_circle1 = 0.5

limit_cond2 = 0.3  # значение радиуса для невыпуклой обл (радиус большего круга)
x_center_circle2 = 0
y_center_circle2 = -0.3

##############################################


def makeInitialSimplex(x0, L):
    dimension = len(x0)
    initialSimplex = np.zeros((dimension + 1, dimension), dtype=float)
    initialSimplex[0, :] = x0

    r1 = (L * (math.sqrt(dimension + 1) + dimension - 1)) / (dimension * math.sqrt(2))
    r2 = (L * (math.sqrt(dimension + 1) - 1)) / (dimension * math.sqrt(2))

    for j in range(dimension):
        initialSimplex[j + 1, :] = x0 + r2
    for i in range(dimension):
        initialSimplex[i + 1, i] += (r1 - r2)

    return initialSimplex


# Функция сортировки по значениям функции
def find_besties(simplex):
    return np.argsort(simplex['func_values'], axis=0)


# Функция подсчета цетроида
def calculate_centroid(points):
    return np.mean(points, axis=0)


# Функция критерия остановы
def criterion(function, simplex, tol):
    n = len(simplex['vertices'][0])
    xc = calculate_centroid(simplex['vertices'])

    return np.sqrt(1 / (n + 1) * sum(np.square((simplex['func_values'] - function(xc))))) <= tol


# criterion(PowFunction, iterations_history[0], tolerance)

# Функция отображения вершины
def MoveVertex(centroid, worst_vertex, function, theta):  # theta = alfa, gamma, betta_inside,  betta_outside
    x_displayed = centroid + theta * (centroid - worst_vertex)
    return x_displayed, function(x_displayed)


# Функция редукции
def check_cycling(history, iter_index, cycle_amount):
    if len(history) > cycle_amount:

        for point in history[iter_index]['vertices']:
            cycle_counter = 0
            for i in range(M):
                if (point in history[iter_index - i]['vertices']):
                    cycle_counter += 1

            if cycle_counter == cycle_amount:
                print('РЕДУКЦИЯ')
                return True

    return False


def reduce(function, history, iteration, low_vertex_index):
    temp_iteration = copy.deepcopy(history[iteration])

    for i in range(len(temp_iteration['vertices'])):
        if i == low_vertex_index:
            continue

        temp_iteration['vertices'][i] = temp_iteration['vertices'][low_vertex_index] + (
                temp_iteration['vertices'][i] - temp_iteration['vertices'][low_vertex_index]) / 2
        temp_iteration['func_values'][i] = function(temp_iteration['vertices'][i])
    # print(temp_iteration)
    return temp_iteration


# Опукла область(круг)

def IsOnArea(point, limit_cond, x_center, y_center):
    if (point[0] - x_center) ** 2 + (point[1] - y_center)  ** 2 <= limit_cond:
        return True
    else:
        return False

# Неопукла область(бублик)

def IsOnNonConvexArea(point, limit_cond1, limit_cond2, x_center_circle1, y_center_circle1, x_center_circle2, y_center_circle2):
    if ((point[0] - x_center_circle1 ) ** 2 + (point[1] - y_center_circle1) ** 2 <= limit_cond1) and ((point[0] - x_center_circle2)**2 + (point[1] - y_center_circle2)**2 > limit_cond2):
        return True
    else:
        return False



def NelderMeadRealization(function, x0, max_iter, alpha, gamma, betta_inside, betta_outside, M=4, use_modiffication=True, use_area=False):
    iterations_history = \
        {
            1: {
                'vertices': makeInitialSimplex(x0, L),
                'func_values': np.apply_along_axis(func1d=PowFunction, axis=1, arr=makeInitialSimplex(x0, L))
            }
        }

    deformations_counter = \
        {
            'Reflection': 0,
            'Expand': 0,
            'Inside': 0,
            'Outside': 0,
            'Shrink': 0
        }

    iteration_index = 1

    while True:
        low_index, good_index, high_index = find_besties(iterations_history[iteration_index])

        x_best, f_best = iterations_history[iteration_index]['vertices'][low_index], \
                         iterations_history[iteration_index]['func_values'][low_index]
        x_good, f_good = iterations_history[iteration_index]['vertices'][good_index], \
                         iterations_history[iteration_index]['func_values'][good_index]
        x_worst, f_worst = iterations_history[iteration_index]['vertices'][high_index], \
                           iterations_history[iteration_index]['func_values'][high_index]

        print(f"\n-----------------------------------Итерация № {iteration_index}---------------------------------\n")
        print('X(best):', x_best, '                 F(x_best):', f_best)
        print('X(good):', x_good, '                 F(x_good):', f_good)
        print('X(worst):', x_worst, '                F(x_worst):', f_worst)

        if iteration_index > 20:
            if iteration_index >= max_iter or criterion(PowFunction, iterations_history[iteration_index], tolerance):
                print(
                    f'---------------------------------------------------------------------------------                Вызовы функций: {function.counter}')
                print('Количество вызовов каждой деформации:\n', deformations_counter)
                print('')

                return iterations_history, x_best, function.counter, deformations_counter

        if check_cycling(iterations_history, iteration_index, M):
            iterations_history[iteration_index + 1] = reduce(function, iterations_history, iteration_index, low_index)
            deformations_counter['Shrink'] += 1
            iteration_index += 1

            continue

        x_centroid = calculate_centroid(np.delete(iterations_history[iteration_index]['vertices'], high_index, axis=0))

        x_reflected = x_centroid + alpha * (x_centroid - x_worst)

        # Условная оптимизация(выпуклая область) IsOnArea
        if use_area == 'Convex':

            if IsOnArea(x_reflected, limit_cond, x_center, y_center):
                x_reflected, f_reflected = MoveVertex(x_centroid, x_worst, function, alpha)
                print('\nПробное симметричное отображение:', x_reflected, '\nЗнач функции в точке:', f_reflected)
            else:
                f_reflected = f_worst * 1000  # заглушка

        # Условная оптимизация(невыпуклая область) IsOnNonConvexArea     
        elif use_area == 'NonConvex':

            if IsOnNonConvexArea(x_reflected, limit_cond1, limit_cond2, x_center_circle1, y_center_circle1, x_center_circle2, y_center_circle2):
                x_reflected, f_reflected = MoveVertex(x_centroid, x_worst, function, alpha)
                # print("Точка в области;", x_reflected)

                print('\nПробное симметричное отображение:', x_reflected, '\nЗнач функции в точке:', f_reflected)
            else:
                f_reflected = f_worst * 1000  # заглушка
                # print("Точка не в области;", x_reflected)

        else:
            x_reflected, f_reflected = MoveVertex(x_centroid, x_worst, function, alpha)


        if f_reflected <= f_best:
            new_point = x_centroid + gamma * (x_centroid - x_worst)

            # Условная оптимизация(выпуклая область) IsOnArea
            if use_area == 'Convex':
                if IsOnArea(new_point, limit_cond, x_center, y_center):
                    new_point, f_new_point = MoveVertex(x_centroid, x_worst, function, gamma)

                    if use_modiffication and f_reflected < f_new_point:
                        new_point, f_new_point = x_reflected, f_reflected
                        deformations_counter['Reflection'] += 1
                    else:
                        deformations_counter['Expand'] += 1
                else:
                    new_point, f_new_point = x_reflected, f_reflected
                    deformations_counter['Reflection'] += 1

            # Условная оптимизация(невыпуклая область) IsOnNonConvexArea            
            elif use_area == 'NonConvex':
                if IsOnNonConvexArea(new_point, limit_cond1, limit_cond2, x_center_circle1, y_center_circle1, x_center_circle2, y_center_circle2):
                    new_point, f_new_point = MoveVertex(x_centroid, x_worst, function, gamma)

                    if use_modiffication and f_reflected < f_new_point:
                        new_point, f_new_point = x_reflected, f_reflected
                        deformations_counter['Reflection'] += 1
                    else:
                        deformations_counter['Expand'] += 1
                else:
                    new_point, f_new_point = x_reflected, f_reflected
                    deformations_counter['Reflection'] += 1

            else:

                if use_modiffication and f_reflected < f_new_point:
                    new_point, f_new_point = x_reflected, f_reflected
                    deformations_counter['Reflection'] += 1
                else:
                    deformations_counter['Expand'] += 1

        if f_best < f_reflected < f_good:
            new_point, f_new_point = x_reflected, f_reflected
            deformations_counter['Reflection'] += 1

        if f_good <= f_reflected < f_worst:
            new_point, f_new_point = MoveVertex(x_centroid, x_worst, function, betta_outside)

            if use_modiffication and f_reflected < f_new_point:
                new_point, f_new_point = x_reflected, f_reflected
                deformations_counter['Reflection'] += 1
            else:
                deformations_counter['Outside'] += 1

        if f_reflected >= f_worst:
            new_point, f_new_point = MoveVertex(x_centroid, x_worst, function, betta_inside)
            if use_modiffication and f_reflected < f_new_point:
                new_point, f_new_point = x_reflected, f_reflected
                deformations_counter['Reflection'] += 1
            else:
                deformations_counter['Inside'] += 1

        iterations_history[iteration_index + 1] = copy.deepcopy(iterations_history[iteration_index])
        iterations_history[iteration_index + 1]["vertices"][high_index] = new_point
        iterations_history[iteration_index + 1]["func_values"][high_index] = f_new_point

        iteration_index += 1


#NelderMeadRealization(PowFunction, x0, maxNumIter, alpha, gamma, betta_inside, betta_outside, M)


def DrawModelFunction(x0, function):
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X, Y = np.meshgrid(x, y)
    Z = function((X, Y))

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='plasma', edgecolor='none')
    ax.view_init(10, -100)
    ax.set_title('3D model of PowFuncrion')

    # відображення початкової точки
    ax.scatter(x0[0], x0[1], function(x0), c='orange', s=30)
    ax.text(x0[0] - 1.1, x0[1] + .06, function(x0) + .25, 'Початкова точка', fontsize=12)
    plt.show()


# DrawModelFunction(x0,PowFunction, min_point)


# Лінії рівня функції
def LineLevelGraph(function):
    x = np.linspace(-0.5, 2, 1000)
    y = np.linspace(-0.5, 2, 1000)
    X, Y = np.meshgrid(x, y)

    plt.contour(function((X, Y)), levels=100)
    plt.title("Level lines")

    plt.show()


# LineLevelGraph(PowFunction)


def DrawDiffArea(history, min_point, function, limit_cond, limit_cond1, limit_cond2, x_center, y_center, \
     x_center_circle1, y_center_circle1, x_center_circle2, y_center_circle2, use_area='NonConvex'):

    points = [point for point in [history[i]['vertices'] for i in history]]


    xs, ys = [], []
    for triangle in points:
        for point in triangle:
            xs.append(point[0])
            ys.append(point[1])

        xs.append(triangle[0][0])
        ys.append(triangle[0][1])
        xs.append(triangle[2][0])
        ys.append(triangle[2][1])

    fig, axes = plt.subplots()

    if use_area == 'Convex':
        circle = plt.Circle((x_center, y_center), limit_cond, fill=False)
        axes.add_artist(circle)

    else:
        circle1 = plt.Circle((x_center_circle1, y_center_circle1), limit_cond1, fill=True, edgecolor='black', alpha = 0.4) 
        axes.add_artist(circle1)
        circle2 = plt.Circle((x_center_circle2, y_center_circle2), limit_cond2, fill=True, edgecolor='black', color='white', alpha = 0.9)
        axes.add_artist(circle2)


    worst_xs, worst_ys = [], []
    for simplex_index in history:
        if history[simplex_index]['func_values'][find_besties(history[simplex_index])[2]] > 10000:
            worst_point = history[simplex_index]['vertices'][find_besties(history[simplex_index])[2]]

            worst_xs.append(worst_point[0])
            worst_ys.append(worst_point[1])

    

    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-2, 2, 1000)
    X, Y = np.meshgrid(x, y)

    # # відображення мінімума точки
    axes.scatter(min_point[0], min_point[1], color='black', s=30)
    axes.scatter(1, 1, c='r', s=35, marker='*')
    axes.text(1 + 0.1, 1, 'Min point', fontsize=9)

    plt.plot(xs, ys, color='Crimson')
    plt.contour(X, Y, function((X, Y)), levels=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5, 1e+6, 1e+7, 1e+8, 1e+9], zorder=0, linewidths=0.5)
    plt.title('Схема области поиска')
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.show()


history, min_point = NelderMeadRealization(PowFunction, x0, maxNumIter, alpha, gamma, betta_inside, betta_outside, M, use_area='NonConvex')[:2]

DrawDiffArea(history, min_point, PowFunction, limit_cond, limit_cond1, limit_cond2, x_center, y_center, \
     x_center_circle1, y_center_circle1, x_center_circle2, y_center_circle2, use_area='NonConvex')

