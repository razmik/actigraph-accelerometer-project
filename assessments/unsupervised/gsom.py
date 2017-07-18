import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import sys


class Utilities:
    def get_distance(self, vector_1, vector_2):
        return np.linalg.norm(vector_1 - vector_2)

    def generate_index(self, x, y):
        return str(x) + ':' + str(y)

    def select_winner(self, nodemap, input_vector):
        min_dist = float("inf")
        winner = None

        for node in nodemap:
            curr_dist = Utilities().get_distance(nodemap[node].weights, input_vector)
            if curr_dist < min_dist:
                min_dist = curr_dist
                winner = nodemap[node]

        return winner


class GSOMParameters:
    def __init__(self, spread_factor, learning_itr, smooth_itr, max_neighbourhood_radius=4, start_learning_rate=0.3,
                 smooth_neighbourhood_radius_factor=0.5, smooth_learning_factor=0.5, distance='euclidean', fd=0.1,
                 alpha=0.9, r=3.8):
        self.SPREAD_FACTOR = spread_factor
        self.LEARNING_ITERATIONS = learning_itr
        self.SMOOTHING_ITERATIONS = smooth_itr
        self.MAX_NEIGHBOURHOOD_RADIUS = max_neighbourhood_radius
        self.START_LEARNING_RATE = start_learning_rate
        self.SMOOTHING_LEARNING_RATE_FACTOR = smooth_learning_factor
        self.SMOOTHING_NEIGHBOURHOOD_RADIUS_FACTOR = smooth_neighbourhood_radius_factor
        self.FD = fd
        self.R = r
        self.ALPHA = alpha
        self.DISTANCE = distance

    def get_gt(self, dimensions):
        return -dimensions * math.log(self.SPREAD_FACTOR)


class GSOMNode:
    R = random.Random()

    def __init__(self, x, y, weights):
        self.weights = weights
        self.x, self.y = x, y

        # Remember the error occuring at this particular node
        self.error = 0.0

        # To be used to map labels and classes after GSOM phases are completed
        self.mappedLabels = []
        self.mappedClasses = []
        self.data = []

    def adjust_weights(self, target, influence, learn_rate):
        self.weights += influence * learn_rate * (target - self.weights)

    def cal_and_update_error(self, input_vector):
        self.error += Utilities().get_distance(self.weights, input_vector)
        return

    def map_label(self, input_label):
        self.mappedLabels.append(input_label)

    def map_class(self, input_class):
        self.mappedClasses.append(input_class)

    def map_data(self, input_data):
        self.data.append(input_data)


class GrowthHandler:
    def grow_nodes(self, node_map, winner):
        x = winner.x
        y = winner.y

        self._grow_individual_node(x - 1, y, winner, node_map)
        self._grow_individual_node(x + 1, y, winner, node_map)
        self._grow_individual_node(x, y - 1, winner, node_map)
        self._grow_individual_node(x, y + 1, winner, node_map)

    def _grow_individual_node(self, x, y, winner, node_map):

        newNodeIndex = Utilities().generate_index(x, y)

        if newNodeIndex not in node_map:
            node_map[Utilities().generate_index(x, y)] = GSOMNode(x, y,
                                                                  self._generate_new_node_weights(node_map, winner, x,
                                                                                                  y))

    def _generate_new_node_weights(self, node_map, winner, x, y):

        new_weights = np.random.rand(len(winner.weights))

        if winner.y == y:

            # W1 is the winner in following cases
            # W1 - W(new)
            if x == winner.x + 1:

                next_node_str = Utilities().generate_index(x + 1, y)
                other_side_node_str = Utilities().generate_index(x - 2, y)
                top_node_srt = Utilities().generate_index(winner.x, y + 1)
                bottom_node_str = Utilities().generate_index(winner.x, y - 1)

                """
                 * 1. W1 - W(new) - W2
                 * 2. W2 - W1 - W(new)
                 * 3. W2
                 *    |
                 *    W1 - W(new)
                 * 4. W1 - W(new)
                 *    |
                 *    W2
                 * 5. W1 - W(new)
                """
                new_weights = self._get_new_node_weights_in_xy_axis(node_map, winner, next_node_str,
                                                                    other_side_node_str, top_node_srt, bottom_node_str)

            # W(new) - W1
            elif x == winner.x - 1:

                next_node_str = Utilities().generate_index(x - 1, y)
                other_side_node_str = Utilities().generate_index(x + 2, y)
                top_node_srt = Utilities().generate_index(winner.x, y + 1)
                bottom_node_str = Utilities().generate_index(winner.x, y - 1)

                """
                 * 1. W2 - W(new) - W1
                 * 2. W(new) - W1 - W2
                 * 3.          W2
                 *             |
                 *    W(new) - W1
                 * 4. W(new) - W1
                 *              |
                 *              W2
                 * 5. W(new) - W1
                """

                new_weights = self._get_new_node_weights_in_xy_axis(node_map, winner, next_node_str,
                                                                    other_side_node_str, top_node_srt, bottom_node_str)

        elif winner.x == x:

            """            
            * W(new)
            * |
            * W1
            """
            if y == winner.y + 1:

                next_node_str = Utilities().generate_index(x, y + 1)
                other_side_node_str = Utilities().generate_index(x, y - 2)
                left_node_srt = Utilities().generate_index(x - 1, winner.y)
                right_node_str = Utilities().generate_index(x + 1, winner.y)

                new_weights = self._get_new_node_weights_in_xy_axis(node_map, winner, next_node_str,
                                                                    other_side_node_str, left_node_srt, right_node_str)

            elif y == winner.y - 1:

                next_node_str = Utilities().generate_index(x, y - 1)
                other_side_node_str = Utilities().generate_index(x, y + 2)
                left_node_srt = Utilities().generate_index(x - 1, winner.y)
                right_node_str = Utilities().generate_index(x + 1, winner.y)

                new_weights = self._get_new_node_weights_in_xy_axis(node_map, winner, next_node_str,
                                                                    other_side_node_str, left_node_srt, right_node_str)

        new_weights[new_weights < 0] = 0.0
        new_weights[new_weights > 1] = 1.0

        return new_weights

    def _get_new_node_weights_in_xy_axis(self, node_map, winner, next_node_str, other_side_node_str,
                                         top_or_left_node_srt, bottom_or_right_node_str):

        if next_node_str in node_map:
            new_weights = self._new_weights_for_new_node_in_middle(node_map, winner, next_node_str)
        elif other_side_node_str in node_map:
            new_weights = self._new_weights_for_new_node_on_one_side(node_map, winner, other_side_node_str)
        elif top_or_left_node_srt in node_map:
            new_weights = self._new_weights_for_new_node_on_one_side(node_map, winner, top_or_left_node_srt)
        elif bottom_or_right_node_str in node_map:
            new_weights = self._new_weights_for_new_node_on_one_side(node_map, winner, bottom_or_right_node_str)
        else:
            new_weights = self._new_weights_for_new_node_one_older_neighbour(winner)

        return new_weights

    def _new_weights_for_new_node_in_middle(self, node_map, winner, next_node_str):
        return (winner.weights + node_map[next_node_str].weights) * 0.5

    def _new_weights_for_new_node_on_one_side(self, node_map, winner, next_node_str):
        return (winner.weights * 2) - node_map[next_node_str].weights

    def _new_weights_for_new_node_one_older_neighbour(self, winner):
        return np.full(len(winner.weights), (max(winner.weights) + min(winner.weights)) / 2)


class GSOM:
    map = {}

    def __init__(self, params, input_vectors):
        self.parameters = params
        self.inputs = input_vectors
        self.dimensions = input_vectors.shape[1]
        self.growth_handler = GrowthHandler()

    def grow(self):
        self._initialize_network(self.dimensions)

        learning_rate = self.parameters.START_LEARNING_RATE
        print('Growing GSOM for ', self.parameters.LEARNING_ITERATIONS, 'iterations...')
        start = time.time()
        for i in range(0, self.parameters.LEARNING_ITERATIONS):

            if i != 0:
                learning_rate = self._get_learning_rate(self.parameters, learning_rate, len(self.map))

            neighbourhood_radius = self._get_neighbourhood_radius(self.parameters.LEARNING_ITERATIONS, i,
                                                                  self.parameters.MAX_NEIGHBOURHOOD_RADIUS)

            for k in range(0, len(self.inputs)):
                self._grow_for_single_iteration_and_single_input(self.inputs[k], learning_rate, neighbourhood_radius)

        print('Growing cost -', (time.time() - start), 's')

                # print('Iteration', i, '/', self.parameters.LEARNING_ITERATIONS)
        # print('Growing completed.')

        return self.map

    def smooth(self):

        learning_rate = self.parameters.START_LEARNING_RATE * self.parameters.SMOOTHING_LEARNING_RATE_FACTOR
        reduced_neighbourhood_radius = self.parameters.MAX_NEIGHBOURHOOD_RADIUS * self.parameters.SMOOTHING_NEIGHBOURHOOD_RADIUS_FACTOR

        print('Smoothing GSOM for ', self.parameters.SMOOTHING_ITERATIONS, 'iterations...')
        start = time.time()
        for i in range(0, self.parameters.SMOOTHING_ITERATIONS):

            if i != 0:
                learning_rate = self._get_learning_rate(self.parameters, learning_rate, len(self.map))

            neighbourhood_radius = self._get_neighbourhood_radius(self.parameters.SMOOTHING_ITERATIONS, i,
                                                                  reduced_neighbourhood_radius)

            for k in range(0, len(self.inputs)):
                self._smooth_for_single_iteration_and_single_input(self.inputs[k], learning_rate, neighbourhood_radius)
                # print('Iteration', i, '/', self.parameters.SMOOTHING_ITERATIONS)

        print('Smoothing cost -', (time.time() - start), 's')

        # print('Smoothing completed.')

        return self.map

    def _smooth_for_single_iteration_and_single_input(self, input_vector, learning_rate, neigh_radius):

        winner = Utilities().select_winner(self.map, input_vector)

        left = Utilities().generate_index(winner.x - 1, winner.y)
        right = Utilities().generate_index(winner.x + 1, winner.y)
        top = Utilities().generate_index(winner.x, winner.y + 1)
        bottom = Utilities().generate_index(winner.x, winner.y - 1)

        if left in self.map:
            self._adjust_weights_for_neighbours(self.map[left], winner, input_vector, neigh_radius, learning_rate)
        elif right in self.map:
            self._adjust_weights_for_neighbours(self.map[right], winner, input_vector, neigh_radius, learning_rate)
        elif top in self.map:
            self._adjust_weights_for_neighbours(self.map[top], winner, input_vector, neigh_radius, learning_rate)
        elif bottom in self.map:
            self._adjust_weights_for_neighbours(self.map[bottom], winner, input_vector, neigh_radius, learning_rate)

    def _grow_for_single_iteration_and_single_input(self, input_vector, learning_rate, neigh_radius):

        winner = Utilities().select_winner(self.map, input_vector)

        # Update the error value of the winner node
        winner.cal_and_update_error(input_vector)

        # Weight adaptation for winner's neighborhood
        for node_id in list(self.map):
            self._adjust_weights_for_neighbours(self.map[node_id], winner, input_vector, neigh_radius, learning_rate)

            # Evaluate winner's weights and grow network it it's above Growth Threshold (GT)
            if winner.error > self.parameters.get_gt(len(input_vector)):
                self._adjust_winner_error(winner, len(input_vector))

    def _adjust_winner_error(self, winner, dimensions):

        left = Utilities().generate_index(winner.x - 1, winner.y)
        right = Utilities().generate_index(winner.x + 1, winner.y)
        top = Utilities().generate_index(winner.x, winner.y + 1)
        bottom = Utilities().generate_index(winner.x, winner.y - 1)

        if left in self.map and right in self.map and top in self.map and bottom in self.map:
            self._distribute_error_to_neighbours(winner, left, right, top, bottom, dimensions)
        else:
            self.growth_handler.grow_nodes(self.map, winner)

    def _distribute_error_to_neighbours(self, winner, left, right, top, bottom, dimensions):

        winner.error = self.parameters.get_gt(dimensions)
        self.map[left].error = self._calc_error_for_neighbours(self.map[left])
        self.map[right].error = self._calc_error_for_neighbours(self.map[right])
        self.map[top].error = self._calc_error_for_neighbours(self.map[top])
        self.map[bottom].error = self._calc_error_for_neighbours(self.map[bottom])

    def _calc_error_for_neighbours(self, node):
        return node.error * (1 + self.parameters.FD)

    def _adjust_weights_for_neighbours(self, node, winner, input_vector, neigh_radius, learning_rate):

        node_dist_sqr = math.pow(winner.x - node.x, 2) + math.pow(winner.y - node.y, 2)
        neigh_radius_sqr = neigh_radius * neigh_radius

        if node_dist_sqr < neigh_radius_sqr:
            influence = math.exp(- node_dist_sqr / (2 * neigh_radius_sqr))
            node.adjust_weights(input_vector, influence, learning_rate)

    def _initialize_network(self, dimensions):
        self.map = {
            '0:0': GSOMNode(0, 0, np.random.rand(dimensions)),
            '0:1': GSOMNode(0, 1, np.random.rand(dimensions)),
            '1:0': GSOMNode(1, 0, np.random.rand(dimensions)),
            '1:1': GSOMNode(1, 1, np.random.rand(dimensions)),
        }

    def _get_learning_rate(self, parameters, prev_learning_rate, nodemap_size):
        return parameters.ALPHA * (1 - parameters.R / nodemap_size) * prev_learning_rate

    def _get_neighbourhood_radius(self, total_iteration, iteration, max_neighbourhood_radius):
        time_constant = total_iteration / math.log(max_neighbourhood_radius)
        return max_neighbourhood_radius * math.exp(- iteration / time_constant)


class AnnotateGSOM:
    def label_map(self, nodemap, input_vectors, input_labels):

        # conduct the operation for each input vector
        if len(input_vectors) != len(input_labels):
            print('Error: Input vector length and label length differs.')
            return

        # Create a new map for test results
        test_result_map_labels = {}

        for i in range(0, len(input_vectors)):
            input_vector = input_vectors[i]
            input_vectors_label = input_labels[i]

            self._map_single_input_label(nodemap, input_vector, test_result_map_labels, input_vectors_label)

        return test_result_map_labels

    def _map_single_input_label(self, nodemap, input_vector, test_result_map_labels, input_vector_label):

        winner = Utilities().select_winner(nodemap, input_vector)
        winner.map_label(input_vector_label)
        winner.map_data(input_vector)

        winner_index = Utilities().generate_index(winner.x, winner.y)

        if winner_index not in test_result_map_labels:
            test_result_map_labels[winner_index] = str(input_vector_label)
        else:
            test_result_map_labels[winner_index] += ',' + str(input_vector_label)


def run_gsom_for_zoo_data(sf):
    data = pd.read_csv(
        'D:\Accelerometer Data\Processed\LSM2\Week 1\Wednesday/filtered/LSM203_(2016-11-02)_row_4442_to_8233.csv',
        nrows=1000, skiprows=100, usecols=[16, 18, 19, 20])
    data.columns = ['actilife_waist_intensity', 'Y', 'X', 'Z']

    classes = data['actilife_waist_intensity'].tolist()
    inputs = data.as_matrix(['Y', 'X', 'Z'])

    print('Dataset length -', len(inputs))

    gsom_params = GSOMParameters(sf, 70, 70)

    gsom = GSOM(gsom_params, inputs)

    gsom_map = gsom.grow()
    print('GSOM growing phase completed generating', len(gsom_map), 'nodes')

    gsom_map = gsom.smooth()
    print('GSOM smoothing phase completed for', len(gsom_map), 'nodes')

    with open('nodemap.pickle', 'wb') as handle:
        pickle.dump(gsom_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # gsom_map = pickle.load(open("nodemap.pickle", "rb"))

    mapped_classes = AnnotateGSOM().label_map(gsom_map, inputs, classes)
    print('Labeled the gsom map hit nodes')

    plt.figure(1)
    plt.title('GSOM - Accelerometer with Intensity with SF ' + str(sf))

    for key, value in mapped_classes.items():
        key_split = key.split(':')
        x = int(key_split[0])
        y = int(key_split[1])
        plt.plot(x, y, 'bo')
        plt.text(x, y+0.1, value, fontsize=12)


if __name__ == '__main__':

    """
    Test for,
    1. Input - raw x, y, z | class - intensity
    2. Input - vm, sdvm | class - intensity
    3. Input - high-correlated features | class - intensity
    """

    run_gsom_for_zoo_data(0.25)

    plt.show()
