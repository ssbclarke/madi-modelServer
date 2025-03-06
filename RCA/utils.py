import numpy as np
from collections import deque
import torch
import argparse
import random

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e',
                        type=int,
                        default=1000,
                        help='Number of episodes')
    parser.add_argument('-b',
                        type=int,
                        default=100000,
                        help='Buffer size')
    parser.add_argument('-s',
                        type=int,
                        default=1,
                        help='Seed')
    parser.add_argument('-x',
                        type=int,
                        default=10,
                        help='Reporting the results every x episodes')
    parser.add_argument('-m',
                        type=int,
                        default=32,
                        help='Mini-batch size')
    parser.add_argument('-t',
                        type=int,
                        default=10,
                        help='Number of episodes for target update')
    parser.add_argument('-g',
                        type=float,
                        default=0.95,
                        help='Discount factor for estimating Q-values')
    parser.add_argument('-a',
                        type=float,
                        default=5.0,
                        help='Alpha, the weight of the CQL loss')
    parser.add_argument('-l',
                        type=float,
                        default=1e-3,
                        help='Learning rate for the optimizier')
    
    

    args, unknown = parser.parse_known_args()
    return args


def get_unique_state(state, states_dict, hour_unique, wind_dir_unique, wind_spd_unique, weather_unique):
    
    hour_value = hour_unique[np.where(state[np.arange(0, 24, 1)]==1)]
    dir_value = wind_dir_unique[np.where(state[np.arange(24, 32, 1)]==1)]
    spd_value = wind_spd_unique[np.where(state[np.arange(32, 37, 1)]==1)]
    weather_value = weather_unique[np.where(state[np.arange(37, 39, 1)]==1)]
    
    state_index = np.where((states_dict[:, 0] == str(hour_value[0])) &
                           (states_dict[:, 1] == dir_value[0]) &
                           (states_dict[:, 2] == str(spd_value[0])) &
                           (states_dict[:, 3] == weather_value[0]))[0]
    
    return state_index, dir_value, spd_value


def get_unique_state_updated(state, states_dict, hour_unique, wind_dir_unique,
                             wind_spd_unique, weather_unique, rw_config_unique):
    
    hour_value = hour_unique[np.where(state[np.arange(0, 24, 1)]==1)]
    dir_value = wind_dir_unique[np.where(state[np.arange(24, 32, 1)]==1)]
    spd_value = wind_spd_unique[np.where(state[np.arange(32, 38, 1)]==1)]
    weather_value = weather_unique[np.where(state[np.arange(38, 40, 1)]==1)]
    config_value = rw_config_unique[np.where(state[np.arange(40, 51, 1)]==1)]
    
    state_index = np.where((states_dict[:, 0] == str(hour_value[0])) &
                           (states_dict[:, 1] == dir_value[0]) &
                           (states_dict[:, 2] == str(spd_value[0])) &
                           (states_dict[:, 3] == weather_value[0]) & 
                           (states_dict[:, 4] == config_value[0]))[0]
    
    return state_index, dir_value, spd_value, weather_value, config_value


def get_unique_state_noflow(state, states_dict, hour_unique, wind_dir_unique,
                            wind_spd_unique, weather_unique):
    
    hour_value = hour_unique[np.where(state[np.arange(0, 24, 1)]==1)]
    dir_value = wind_dir_unique[np.where(state[np.arange(24, 32, 1)]==1)]
    spd_value = wind_spd_unique[np.where(state[np.arange(32, 38, 1)]==1)]
    weather_value = weather_unique[np.where(state[np.arange(38, 40, 1)]==1)]
    
    state_index = np.where((states_dict[:, 0] == str(hour_value[0])) &
                           (states_dict[:, 1] == dir_value[0]) &
                           (states_dict[:, 2] == str(spd_value[0])) &
                           (states_dict[:, 3] == weather_value[0]))[0]
    
    return state_index, dir_value, spd_value, weather_value