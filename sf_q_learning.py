import torch
from torch import nn

import numpy as np

import pickle
import time

from classes import GridWorld,SFModel


def run_sf_q_learning(config):
    """
    Learn a set of successor feature models, via Q-learning.
    
    Args:
        config (dict): A dictionary specifying parameter configurations

    Returns: None
    """
    
    print('='*10 + ' Learning successor feature models using hand-crafted feature model and hand-crafted task vectors ' + '='*10)
    
    # list of successor feature models
    sf_list = [ SFModel(config) for _ in range( len(config['sf_q_learning']['tasks'][0]) ) ] # a model for each feature dimension; it needs to match with the dimension of the tasks
    
    env = GridWorld(config)
    start_time = time.time()
    for episode_num in range(1,int(config['sf_q_learning']['num_episodes'])+1):
        policy_index = np.random.choice( range( len( config['sf_q_learning']['tasks'] ) ) ) # we have a policy for each task

        current_state = env.reset()
        total_transition_features = np.zeros( len(config['sf_q_learning']['tasks'][0]) )  # to be used when calculating total undiscounted return during print updates
        for timestep in range( int(config['sf_q_learning']['episode_length']) ):

            # select action
            if np.random.uniform() < config['sf_q_learning']['exploration_threshold']: action_index = np.random.choice( list( env.action_mapping.keys() ) )
            else: action_index = get_argmax_action_index( sf_list , current_state , policy_index , env.action_mapping.keys() , config['sf_q_learning']['tasks'] )

            # apply action to environment
            new_state,_ = env.apply_action(action_index)
            best_future_action_index = get_argmax_action_index( sf_list , new_state , policy_index , env.action_mapping.keys() , config['sf_q_learning']['tasks'] )

            # update model parameters
            transition_features = get_transition_features(current_state,action_index,new_state)
            for feature_index,sf_model in enumerate(sf_list):
                td_estimate = transition_features[feature_index] + config['sf_q_learning']['discount_factor']*sf_model(new_state,best_future_action_index)[policy_index] # bootstrapped TD prediction
                sf_model.update_weights(current_state,action_index,td_estimate,policy_index)

            # update new state and total_transition_features
            total_transition_features += transition_features
            current_state = new_state

        if episode_num % config['sf_q_learning']['interval'] == 0:
            
            # save sf_list models
            with open('models/sf_list.pkl','wb') as file: pickle.dump(sf_list,file)

            # print an update
            print(f"Episode: {episode_num}\tTotal transition features: {total_transition_features}\tTask: {config['sf_q_learning']['tasks'][policy_index]}\t" + \
                  f"Total undiscounted return: {np.dot(total_transition_features,config['sf_q_learning']['tasks'][policy_index])}\tFinished in {time.time()-start_time} seconds")
            
            start_time = time.time()
            
def get_argmax_action_index(sf_list,state,policy_index,action_space,task_vectors): # return the action that maximizes the action value
    """
    Given sf_list, state and action_space, return the action index that maximizes the action value.
    
    Args:
        sf_list (list[SFModel]): A list of successor feature models, where each model contains the output for one feature index of the transition features, for each policy/task
        state (numpy.ndarray): The egocentric representation of the state
        policy_index (int): Index indicates which task we're given for this episode, and therefore which output index to use from the successor feature models in sf_list
        action_space (iterable): An iterable over every possible action index in the GridWorld environment's action space
        task_vectors (list[ list[float] ]): The list of task vectors that we're training our successor feature models on

    Returns:
        action_index (int): Represents an action in the game's action space
    """
    
    # use sf_list, state, policy_index and action_space to compute expected discounted feature returns, for every action
    # then dot product the expected discounted feature returns with the task_vector corresponding to policy_index, to get the action values for each action
    # then return the argmax of the action values
    
    sf_output_features = []
    for action_index in action_space: # for each action, get the expected discounted feature returns for following policy policy_index at the current state until termination
        sf_output_features.append( [ sf_model(state,action_index).detach().numpy()[policy_index] for sf_model in sf_list ] )
    sf_output_features = np.array(sf_output_features) # action x feature
    # the (a,d) entry is the expected discounted return for feature d, if we were to take action a at the current state, and follow policy policy_index for the remainder of the episode
    
    return np.dot(sf_output_features,task_vectors[policy_index]).argmax() # dot product with corresponding task vector to get the action value, and then take the action that maximizes action value

def get_transition_features(state,action_index,new_state):
    """
    Given state, action_index and new_state, return the transition features (equivalent to the phi feature function in the paper)
    
    Args:
        state (numpy.ndarray): The egocentric representation of the state
        action_index (int): Represents an action in the game's action space
        new_state (numpy.ndarray): The egocentric representation of the resulting state, after applying action action_index to state

    Returns:
        transition_features (numpy.ndarray): The transition features for applying action action_index to state and resulting in new_state
    """
    
    # look at the adjacent cell next to our agent (at (0,0) due to using egocentric states), and see if the corresponding action leads to picking up an object
    # return a vector denoting 1 if that corresponding object is picked up, 0 otherwise
    
    if action_index == 0: return state[-1,0,:-1] # up
    elif action_index == 1: return state[1,0,:-1] # down
    elif action_index == 2: return state[0,-1,:-1] # left
    elif action_index == 3: return state[0,1,:-1] # right
    else: raise Exception(f'Invalid action_index: {action_index}')

def generalized_policy(sf_list,state,action_space,task_vector): 
    """
    Given a list of trained successor feature models sf_list, state, action_space and task_vector, return the action that maximizes the action value at the current state.
    
    Args:
        sf_list (list[SFModel]): A list of successor feature models, where each model contains the output for one feature index of the transition features, for each policy/task
        state (numpy.ndarray): The egocentric representation of the state
        action_space (iterable): An iterable over every possible action index in the GridWorld environment's action space
        task_vector (list[float]): The task used for this episode, which will determine the transition rewards received

    Returns:
        action_index (int): Represents an action in the game's action space
    """
    
    # use sf_list, state and action_space to compute expected discounted feature returns, for every action, for every policy
    # then dot product the expected discounted feature returns with the task_vector we want to maximize for, to get the action values for each action, for each policy
    # then take the argmax across all policies, so we have the maximum action value for each action
    # then return the argmax of the maximum action values
    
    sf_output_matrix = []
    for action_index in action_space: # for each action, get the expected discounted feature returns for following every policy at the current state until termination
        sf_output_matrix.append( [ sf_model(state,action_index).detach().numpy() for sf_model in sf_list ] )

    sf_output_matrix = np.array(sf_output_matrix) # action x feature x policy
    # the (a,d,n) entry is the expected discounted return for feature d, if we were to take action a at our current state, and follow policy n for the remainder of the episode
    sf_output_matrix = sf_output_matrix.swapaxes(1,2) # swap feature and policy axis, so now the dimensions are: action x policy x feature

    # dot product with the task_vector we're looking to maximize for
    action_values = np.dot(sf_output_matrix,task_vector) # action x policy
    # the (a,n) entry is the action value for taking action a at our current state, and then following policy n for the remainder of the episode

    return action_values.max(axis=1).argmax() # take the maximum action value across all policies, then take the argmax action out of these maximum action values
