from snake_game import SnakeGame
from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter

#
# these functions need to be declared above the actual processes
#

    # This function is used to create the basics of the neural network
def NeuralNetwork_Creation():
    created_network_model = input_data(shape=[None, 5, 1], name='input')
    # The hidden layer has 25 neurons and uses the activation function ReLU which gives 0
    # for negative numbers and returns the value inputted if its positive
    created_network_model = fully_connected(created_network_model, 25, activation='relu')
    # this last one creates the final layer, which is the output with a different activation function
    created_network_model = fully_connected(created_network_model, 1, activation='linear')

    # the Backwards-propagation of the Neural network that updates the weights based
    # on the derivative of the activation function
    #   The 0.2 is the learning rate of the neural network
    #   The optimizer and loss both relate to the error function of the Backwards propagation
    created_network_model = regression(created_network_model, optimizer='adam', learning_rate = 0.2, loss='mean_square', name='target')

    # DNN is just a file output for the created neural network
    return tflearn.DNN(created_network_model, tensorboard_dir='log')
    

def test_model(model):
    steps_arr = []
    scores_arr = []
    # how many games we'll be playing
    for _ in range(100):
        steps = 0
        game_memory = []
        game = SnakeGame(gui = True)
        _, score, snake, food = game.start()
        prev_observation = generate_observation(snake, food)
        # how many steps we are aiming for in the game
        for _ in range(2000):
            predictions = []
            for action in range(-1, 2):
               predictions.append(model.predict(add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
            action = np.argmax(np.array(predictions))
            game_action = get_game_action(snake, action - 1)
            done, score, snake, food  = game.step(game_action)
            game_memory.append([prev_observation, action])
            if done:
                print('-----')
                print(steps)
                print(snake)
                print(food)
                print(prev_observation)
                print(predictions)
                break
            else:
                prev_observation = generate_observation(snake, food)
                steps += 1
        steps_arr.append(steps)
        scores_arr.append(score)
    print('Average steps:',mean(steps_arr))
    print(Counter(steps_arr))
    print('Average score:',mean(scores_arr))
    print(Counter(scores_arr))

def get_game_action(snake, action):
    snake_direction = get_snake_direction_vector(snake)
    new_direction = snake_direction
    # these are for all the possible moves, like WASD, or up down left right
    vectors_and_keys = [[[-1, 0], 0], [[0, 1], 1], [[1, 0], 2], [[0, -1], 3]]

    if action == -1:
        new_direction = turn_vector_to_the_left(snake_direction)
    elif action == 1:
        new_direction = turn_vector_to_the_right(snake_direction)
    for pair in vectors_and_keys:
        if pair[0] == new_direction.tolist():
            game_action = pair[1]
    return game_action

def generate_observation(snake, food):
    snake_direction = get_snake_direction_vector(snake)
    food_direction = get_food_direction_vector(snake, food)
    barrier_left = is_direction_blocked(snake, turn_vector_to_the_left(snake_direction))
    barrier_front = is_direction_blocked(snake, snake_direction)
    barrier_right = is_direction_blocked(snake, turn_vector_to_the_right(snake_direction))
    angle = get_angle(snake_direction, food_direction)
    return np.array([int(barrier_left), int(barrier_front), int(barrier_right), angle])

def add_action_to_observation(observation, action):
    return np.append([action], observation)

def get_snake_direction_vector(snake):
    return np.array(snake[0]) - np.array(snake[1])

def get_food_direction_vector(snake, food):
    return np.array(food) - np.array(snake[0])

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

def get_food_distance(snake, food):
    return np.linalg.norm(get_food_direction_vector(snake, food))

def is_direction_blocked(snake, direction):
    point = np.array(snake[0]) + np.array(direction)
    return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

def turn_vector_to_the_left(vector):
    return np.array([-vector[1], vector[0]])

def turn_vector_to_the_right(vector):
    return np.array([vector[1], -vector[0]])

def get_angle(a, b):
    a = normalize_vector(a)
    b = normalize_vector(b)

    # atan2 is arctan(y,x)
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


tempmodel = NeuralNetwork_Creation()
tempmodel.load('snake_nn_0.tflearn')
test_model(tempmodel)