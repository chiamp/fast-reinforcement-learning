import torch
from torch import nn

import numpy as np

import pickle
import time

from classes import GridWorld,QModel


def run_q_learning(config):
    """
    Learn an optimal Q-function, via Q-learning.
    
    Args:
        config (dict): A dictionary specifying parameter configurations

    Returns: None
    """
    
    print('='*10 + ' Learning Q-model using hand-crafted feature model and hand-crafted task vectors ' + '='*10)
    
    q_model = QModel(config)
    
    env = GridWorld(config)
    evaluation_data = {'num_sample_transitions':[],'average_sum_rewards':[]} # every config['q_learning']['interval'], evaluate q_model with no exploration
    start_time = time.time()
    for episode_num in range(1,int(config['q_learning']['num_episodes'])+1):

        current_state = env.reset()
        total_transition_features = np.zeros( len(config['q_learning']['task']) ) # to be used when calculating total undiscounted return during print updates
        for timestep in range( int(config['q_learning']['episode_length']) ):

            # select action
            if np.random.uniform() < config['q_learning']['exploration_threshold']: action_index = np.random.choice( list( env.action_mapping.keys() ) )
            else: action_index = get_argmax_action_index( q_model , current_state , env.action_mapping.keys() )

            # apply action to environment
            new_state,reward = env.apply_action(action_index)
            best_future_action_index = get_argmax_action_index( q_model , new_state , env.action_mapping.keys() )

            # update model parameters
            td_estimate = reward + config['q_learning']['discount_factor']*q_model(new_state,best_future_action_index) # bootstrapped TD prediction
            q_model.update_weights(current_state,action_index,td_estimate)

            # update new state and total_transition_features
            total_transition_features += get_transition_features(current_state,action_index,new_state)
            current_state = new_state

        if episode_num % config['q_learning']['interval'] == 0:

            # evaluate q_model, and update evaluation_data statistics
            evaluation_data['num_sample_transitions'].append( episode_num * config['q_learning']['episode_length'] )
            evaluation_data['average_sum_rewards'].append( evaluate(q_model,config) )

            # save q_model and evaluation_data
            with open('models/q_model.pkl','wb') as file: pickle.dump(q_model,file)
            with open('evaluation_data/classic_q_learning.pkl','wb') as file: pickle.dump(evaluation_data,file)

            # print an update
            print(f"Episode: {episode_num}\tTotal transition features: {total_transition_features}\tTask: {config['q_learning']['task']}\t" + \
                  f"Total undiscounted return: {np.dot(total_transition_features,config['q_learning']['task'])}\tFinished in {time.time()-start_time} seconds")

            start_time = time.time()
            
def get_argmax_action_index(q_model,state,action_space):
    """
    Given q_model, state and action_space, return the action index that maximizes the action value.
    
    Args:
        q_model (QModel): The learned Q-function
        state (numpy.ndarray): The egocentric representation of the state
        action_space (iterable): An iterable over every possible action index in the GridWorld environment's action space

    Returns:
        action_index (int): Represents an action in the game's action space
    """
    
    return np.argmax( [ q_model(state,action_index).detach().numpy() for action_index in action_space ] )

def get_transition_features(state,action_index,new_state):
    """
    Given state, action_index and new_state, return the transition features (equivalent to the phi feature function in the paper)
    
    Args:
        state (numpy.ndarray): The egocentric representation of the state
        action_index (int): Represents an action in the game's action space
        new_state (numpy.ndarray): The egocentric representation of the resulting state, after applying action action_index to state

    Returns:
        transition_features (numpy.ndarray): The transition features
    """

    # look at the adjacent cell next to our agent (at (0,0) due to using egocentric states), and see if the corresponding action leads to picking up an object
    # return a vector denoting 1 if that corresponding object is picked up, 0 otherwise
    
    if action_index == 0: return state[-1,0,:-1] # up
    elif action_index == 1: return state[1,0,:-1] # down
    elif action_index == 2: return state[0,-1,:-1] # left
    elif action_index == 3: return state[0,1,:-1] # right
    else: raise Exception(f'Invalid action_index: {action_index}')

def evaluate(q_model,config):
    """
    Evaluate how well q_model performs with no exploration (i.e. always selecting the greedy action).
    
    Args:
        q_model (QModel): The learned Q-function
        config (dict): A dictionary specifying parameter configurations

    Returns:
        average_return (float): The average return the agent received using q_model, over a number of episodes specified in config['evaluate']['num_episodes']
    """
    
    env = GridWorld(config)

    total_undiscounted_return = 0 # keep track of all accumulated rewards throughout all episodes
    for episode_num in range(1,int(config['evaluate']['num_episodes'])+1):
        
        current_state = env.reset()
        total_transition_features = np.zeros( len(config['evaluate']['task']) ) # to be used when calculating total undiscounted return
        for timestep in range( int(config['evaluate']['episode_length']) ):

            # select action
            action_index = get_argmax_action_index( q_model , current_state , env.action_mapping.keys() )

            # apply action
            new_state,reward = env.apply_action(action_index)

            # update new state and total_transition_features
            total_transition_features += get_transition_features(current_state,action_index,new_state)
            current_state = new_state

        # add this episode's undiscounted return to total_undiscounted_return
        total_undiscounted_return += np.dot(total_transition_features,config['evaluate']['task'])

    return total_undiscounted_return / config['evaluate']['num_episodes']

