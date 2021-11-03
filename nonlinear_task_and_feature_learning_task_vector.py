import torch
from torch import nn

import autograd.numpy as np
from autograd import grad

import pickle
import time

from classes import GridWorld,RewardModel,TFModel,SFModel
from sf_q_learning import generalized_policy,get_argmax_action_index


def transition_feature_learning(config):
    """
    Learn the transition feature function, via back propagation, while learning task_vectors from environment transition rewards, via linear regression.
    
    Args:
        config (dict): A dictionary specifying parameter configurations

    Returns:
        tf_model (TFModel): The learned transition feature function (equivalent to the phi_tilde function in the paper)
    """
    
    tf_model = TFModel(config)
    
    learned_task_vectors = [ np.random.rand(config['task_and_feature_learning']['num_task_features']) for _ in range( len( config['task_and_feature_learning']['tasks'] ) ) ]
    print(f'Initial task vector values: {learned_task_vectors}')
    grad_loss = grad(loss,0) # function that gives the gradient w.r.t the learned_task_vector[i] values

    env = GridWorld(config)
    evaluation_data = {'num_sample_transitions':[],'average_sum_rewards':[]}
    start_time = time.time()
    for episode_num in range(1,int(config['task_and_feature_learning']['num_episodes'])+1):

        current_state = env.reset()
        for timestep in range( int(config['task_and_feature_learning']['episode_length']) ):

            # select action
            action_index = np.random.choice( list( env.action_mapping.keys() ) )

            # apply action to environment
            new_state,_ = env.apply_action(action_index) # for each time step, we calculate the transition_rewards according to config['task_feature_learning']['tasks']

            # update tf_model parameters and the values of each task_vector in learned_task_vectors
            for i,transition_reward_function in enumerate( config['nonlinear_task_and_feature_learning']['tasks'] ): # calculate actual transition rewards for each task
                transition_reward = transition_reward_function(current_state,action_index,new_state)
                
                learned_task_vector_gradient = grad_loss( learned_task_vectors[i], tf_model,current_state,action_index,new_state, transition_reward ) # calculate the learned_task_vector_gradient before updating tf_model weights
                
                tf_model.update_weights(current_state,action_index,new_state,learned_task_vectors[i],transition_reward)
                learned_task_vectors[i] -= config['task_and_feature_learning']['task_vector_learning_rate'] * learned_task_vector_gradient

            # update new state
            current_state = new_state

        if episode_num % config['task_and_feature_learning']['interval'] == 0:

            # save tf_model
            with open('models/nonlinear_tf_model_task_vector.pkl','wb') as file: pickle.dump(tf_model,file)

            # print an update
            # ***NOTE***: the transition feature function may not learn the same representation as the hand-coded phi function, which means the learned_task_vectors may be represented differently as well
            print(f"Episode: {episode_num}\tTotal MSE: {tf_model.total_mse}\tFinished in {time.time()-start_time} seconds")
            tf_model.total_mse = 0
            
            start_time = time.time()

    return tf_model

def sf_q_learning(tf_model,task_vectors,config):
    """
    Learn a set of successor feature models, via Q-learning, using the learned transition function model.
    
    Args:
        tf_model (TFModel): The learned transition feature model (equivalent to the phi_tilde function in the paper)
        task_vectors (list[numpy.ndarray]): A list of task vectors for the successor feature to train on
        config (dict): A dictionary specifying parameter configurations

    Returns:
        sf_list (list[SFModel]): A list of successor feature models, where each model contains the output for one feature index of the transition features, for each policy/task
    """
    
    # list of successor feature models
    sf_list = [ SFModel(config) for _ in range( task_vectors[0].size ) ] # a model for each feature dimension; it needs to match with the dimension of the tasks
    
    env = GridWorld(config)
    start_time = time.time()
    for episode_num in range(1,int(config['sf_q_learning']['num_episodes'])+1):
        policy_index = np.random.choice( range( len( task_vectors ) ) ) # we have a policy for each task

        current_state = env.reset()
        for timestep in range( int(config['sf_q_learning']['episode_length']) ):

            # select action
            if np.random.uniform() < config['sf_q_learning']['exploration_threshold']: action_index = np.random.choice( list( env.action_mapping.keys() ) )
            else: action_index = get_argmax_action_index( sf_list , current_state , policy_index , env.action_mapping.keys() , task_vectors )

            # apply action to environment
            new_state,_ = env.apply_action(action_index)
            best_future_action_index = get_argmax_action_index( sf_list , new_state , policy_index , env.action_mapping.keys() , task_vectors )

            # update sf_list model parameters
            learned_transition_features = tf_model(current_state,action_index,new_state).detach().numpy() # need to detach this from computational graph, as it won't be used when updating successor feature model sf_model
            for feature_index,sf_model in enumerate(sf_list):
                td_estimate = learned_transition_features[feature_index] + config['sf_q_learning']['discount_factor']*sf_model(new_state,best_future_action_index)[policy_index] # bootstrapped TD prediction
                sf_model.update_weights(current_state,action_index,td_estimate,policy_index)

            # update new state
            current_state = new_state

        if episode_num % config['sf_q_learning']['interval'] == 0:

            # save sf_list models, trained using a learned transition feature function
            with open(f"models/nonlinear_sf_list_with_learned_transition_features_of_dimension_{config['task_and_feature_learning']['num_task_features']}_task_vector.pkl",'wb') as file: pickle.dump(sf_list,file)
            
            # print an update
            sf_losses = [sf_model.total_mse for sf_model in sf_list]
            print(f"Episode: {episode_num}\tTotal MSE: {sum(sf_losses)}\tFinished in {time.time()-start_time} seconds")
            for sf_model in sf_list: sf_model.total_mse = 0
            
            start_time = time.time()
                
    return sf_list

def task_learning_linear(sf_list,tf_model,config):
    """
    Using a learned set of successor feature models and learned transition function, learn a task vector from the environment transition rewards, that the successor feature models haven't seen before.
    
    Args:
        sf_list (list[SFModel]): A list of successor feature models, where each model contains the output for one feature index of the transition features, for each policy/task
        tf_model (TFModel): The learned transition feature model (equivalent to the phi_tilde function in the paper)
        config (dict): A dictionary specifying parameter configurations

    Returns: None
    """

    task_vector = np.random.rand(config['task_and_feature_learning']['num_task_features']) # task_vector to be learned from environment transition rewards
    print(f'Initial task vector values: {task_vector}')
    grad_loss = grad(loss,0) # function that gives the gradient w.r.t the task_vector values

    env = GridWorld(config)
    evaluation_data = {'num_sample_transitions':[],'average_sum_rewards':[]}
    start_time = time.time()
    for episode_num in range(1,int(config['task_learning']['num_episodes'])+1):

        current_state = env.reset()
        for timestep in range( int(config['task_learning']['episode_length']) ):
            # select action
            action_index = np.random.choice( list( env.action_mapping.keys() ) )

            # apply action to environment
            new_state,_ = env.apply_action(action_index)

            transition_reward = config['nonlinear_task_and_feature_learning']['task_to_learn'](current_state,action_index,new_state)

            # update task_vector values
            task_vector -= config['task_learning']['learning_rate'] * grad_loss(task_vector,tf_model,current_state,action_index,new_state,transition_reward)

            # update new state
            current_state = new_state

        if episode_num % config['task_learning']['interval'] == 0:

            # update evaluation_data statistics
            evaluation_data['num_sample_transitions'].append( episode_num * config['task_learning']['episode_length'] )
            evaluation_data['average_sum_rewards'].append( evaluate(sf_list,task_vector,config) )

            # save evaluation_data
            with open(f"evaluation_data/nonlinear_task_learning_with_learned_transition_features_of_dimension_{config['task_and_feature_learning']['num_task_features']}_task_vector.pkl",'wb') as file: pickle.dump(evaluation_data,file)
            
            # print an update
            print(f"Episode: {episode_num}\tLearned task: {task_vector}\tFinished in {time.time()-start_time} seconds")
            
            start_time = time.time()
            
def loss(task_vector,tf_model,current_state,action_index,new_state,reward):
    """
    Calculate the squared error of the predicted transition reward and the actual environment transition reward,
        using the learned transition function (equivalent to the phi_tilde function in the paper) to get the transition features.
    
    Args:
        task_vector (list[float]): The task used for this episode, which will determine the transition rewards received
        tf_model (TFModel): The learned transition feature function (equivalent to the phi_tilde function in the paper)
        current_state (numpy.ndarray): The egocentric representation of the current state
        action_index (int): Represents an action in the game's action space
        new_state (numpy.ndarray): The egocentric representation of the resulting state, after applying action action_index to current_state
        reward (float): The transition reward

    Returns:
        squared_error (float): The squared error of the predicted transition reward and the actual transition reward
    """
    
    return ( np.dot( tf_model(current_state,action_index,new_state).detach().numpy() , task_vector ) - reward )**2 # we use the learned transition function model and learned task_vector to get the predicted transition reward
    
def evaluate(sf_list,task_vector,config):
    """
    Evaluate how well sf_list performs with no exploration (i.e. always selecting the greedy action), using its generalized policy (learned from previous tasks),
        on a new task it hasn't seen before, learned from the environment transition rewards.
    
    Args:
        sf_list (list[SFModel]): A list of successor feature models, where each model contains the output for one feature index of the transition features, for each policy/task
        task_vector (list[float]): The learned task vector, learned from the environment transition rewards of a task the successor feature models haven't seen before
        config (dict): A dictionary specifying parameter configurations

    Returns:
        average_return (float): The average return the agent received using sf_list, over a number of episodes specified in config['evaluate']['num_episodes']
    """
    
    env = GridWorld(config)

    total_return = 0
    for episode_num in range(1,int(config['evaluate']['num_episodes'])+1):
        
        current_state = env.reset()
        total_transition_features = np.zeros( len(config['evaluate']['task']) )
        for timestep in range( int(config['evaluate']['episode_length']) ):

            # select action
            action_index = generalized_policy( sf_list , current_state , env.action_mapping.keys() , task_vector )

            # apply action
            new_state,_ = env.apply_action(action_index)

            # update new state and total_transition_features
            total_return += config['nonlinear_task_and_feature_learning']['task_to_learn'](current_state,action_index,new_state)
            current_state = new_state

    average_return = total_return / config['evaluate']['num_episodes']
    print(f"Avg undiscounted return: {average_return}")

    return average_return


def run_nonlinear_task_and_feature_learning_task_vector(config):
    """
    Learn a transition feature model and successor feature model using a set of nonlinear tasks, and then test it on a new nonlinear task it hasn't seen before.
    Use task vectors that are dot producted with the transition features to approximate the transition reward.
    
    Args:
        config (dict): A dictionary specifying parameter configurations

    Returns: None
    """
    
    # learn transition feature model from environment rewards
    print('\n' + '='*10 + ' Learning transition feature model and task vectors ' + '='*10)
    tf_model = transition_feature_learning(config)

    # learn successor feature model using learned transition feature function
    print('\n' + '='*10 + ' Learning successor feature models using learned transition feature model and task vectors ' + '='*10)
    sf_list = sf_q_learning( tf_model,
                             [ np.array( [ 1 if i==j else 0 for j in range(config['task_and_feature_learning']['num_task_features']) ] ) for i in range(config['task_and_feature_learning']['num_task_features']) ],
                             config ) # use the unit basis tasks for training the successor feature models (regardless of what the learned transition features represents)

    # learn new task vector using learned successor feature model
    print('\n' + '='*10 + ' Learning new task vector using using learned successor feature and transition feature model ' + '='*10)
    task_learning_linear(sf_list,tf_model,config)
