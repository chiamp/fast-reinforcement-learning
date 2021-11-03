import torch
from torch import nn

import autograd.numpy as np
from autograd import grad

import pickle
import time

from classes import GridWorld,RewardModel,TFModel,SFModel
from sf_q_learning import get_argmax_action_index


def transition_feature_learning(config):
    """
    Learn the transition feature function, via back propagation, while learning RewardModel parameters from environment transition rewards.
    
    Args:
        config (dict): A dictionary specifying parameter configurations

    Returns:
        tf_model (TFModel): The learned transition feature function (equivalent to the phi_tilde function in the paper)
    """
    
    tf_model = TFModel(config)
    reward_models = [ RewardModel(tf_model,config) for _ in range( len(config['nonlinear_task_and_feature_learning']['tasks']) ) ]

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

            # update tf_model parameters and the parameters of all reward_models
            for i in range( len(config['nonlinear_task_and_feature_learning']['tasks']) ): # calculate actual transition rewards for each task
                reward_models[i].update_weights( current_state,action_index,new_state,
                                                 config['nonlinear_task_and_feature_learning']['tasks'][i](current_state,action_index,new_state),
                                                 full=True )

            # update new state
            current_state = new_state

        if episode_num % config['task_and_feature_learning']['interval'] == 0:

            # save tf_model
            with open('models/nonlinear_tf_model_reward_model.pkl','wb') as file: pickle.dump(tf_model,file)

            # print an update
            print(f"Episode: {episode_num}\tRewardModel MSEs: {[reward_model.total_mse for reward_model in reward_models]}\tFinished in {time.time()-start_time} seconds")
            for reward_model in reward_models: reward_model.total_mse = 0
            
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
            with open(f"models/nonlinear_sf_list_with_learned_transition_features_of_dimension_{config['task_and_feature_learning']['num_task_features']}_reward_model.pkl",'wb') as file: pickle.dump(sf_list,file)
            
            # print an update
            sf_losses = [sf_model.total_mse for sf_model in sf_list]
            print(f"Episode: {episode_num}\tTotal MSE: {sum(sf_losses)}\tFinished in {time.time()-start_time} seconds")
            for sf_model in sf_list: sf_model.total_mse = 0
            
            start_time = time.time()
                
    return sf_list

def task_learning_nonlinear(sf_list,tf_model,config):
    """
    Using a learned set of successor feature models and learned transition function, learn a RewardModel from the environment transition rewards, of a nonlinear task that the successor feature models haven't seen before.
    
    Args:
        sf_list (list[SFModel]): A list of successor feature models, where each model contains the output for one feature index of the transition features, for each policy/task
        tf_model (TFModel): The learned transition feature model (equivalent to the phi_tilde function in the paper)
        config (dict): A dictionary specifying parameter configurations

    Returns: None
    """

    reward_model = RewardModel(tf_model,config)

    env = GridWorld(config)
    evaluation_data = {'num_sample_transitions':[],'average_sum_rewards':[]}
    start_time = time.time()
    for episode_num in range(1,int(config['task_learning']['num_episodes'])+1):

        current_state = env.reset()
        for timestep in range( int(config['task_learning']['episode_length']) ):
            # select action
            action_index = np.random.choice( list( env.action_mapping.keys() ) )

            # apply action to environment
            new_state,reward = env.apply_action(action_index)

            # update task_vector values
            reward_model.update_weights( current_state,action_index,new_state,
                                         config['nonlinear_task_and_feature_learning']['task_to_learn'](current_state,action_index,new_state),
                                         full=False ) # update only the RewardModel; the tf_model weights don't change during this update

            # update new state
            current_state = new_state

        if episode_num % config['task_learning']['interval'] == 0:

            # update evaluation_data statistics
            evaluation_data['num_sample_transitions'].append( episode_num * config['task_learning']['episode_length'] )
            evaluation_data['average_sum_rewards'].append( evaluate(sf_list,reward_model,config) )

            # save evaluation_data
            with open(f"evaluation_data/nonlinear_task_learning_with_learned_transition_features_of_dimension_{config['task_and_feature_learning']['num_task_features']}_reward_model.pkl",'wb') as file: pickle.dump(evaluation_data,file)
            
            # print an update
            print(f"Episode: {episode_num}\tRewardModel MSE: {reward_model.total_mse}\tFinished in {time.time()-start_time} seconds")
            reward_model.total_mse = 0
            
            start_time = time.time()

def generalized_policy(sf_list,state,action_space,reward_model): 
    """
    Given a list of trained successor feature models sf_list, state, action_space and reward_model, return the action that maximizes the action value at the current state.
    
    Args:
        sf_list (list[SFModel]): A list of successor feature models, where each model contains the output for one feature index of the transition features, for each policy/task
        state (numpy.ndarray): The egocentric representation of the state
        action_space (iterable): An iterable over every possible action index in the GridWorld environment's action space
        reward_model (RewardModel): The reward model for the corresponding nonlinear task used for this episode, which will determine the transition rewards received

    Returns:
        action_index (int): Represents an action in the game's action space
    """
    
    # use sf_list, state and action_space to compute expected discounted feature returns, for every action, for every policy
    # then feed the discounted feature returns into the reward_model, to get the outputted action values for each action, for each policy
    # then take the argmax across all policies, so we have the maximum action value for each action
    # then return the argmax of the maximum action values
    
    sf_output_matrix = []
    for action_index in action_space: # for each action, get the expected discounted feature returns for following every policy at the current state until termination
        sf_output_matrix.append( [ sf_model(state,action_index).detach().numpy() for sf_model in sf_list ] )

    sf_output_matrix = np.array(sf_output_matrix) # action x feature x policy
    # the (a,d,n) entry is the expected discounted return for feature d, if we were to take action a at our current state, and follow policy n for the remainder of the episode
    sf_output_matrix = sf_output_matrix.swapaxes(1,2) # swap feature and policy axis, so now the dimensions are: action x policy x feature

    # feed discounted feature returns into reward_model
    action_values = reward_model( torch.tensor(sf_output_matrix).float() )[:,:,0] # action x policy x action_value; since reward is 1 feature in length, remove the reward dimension so that action_values is dimension (action x policy)
    action_values = action_values.detach().numpy() # convert to numpy
    # the (a,n) entry is the action value for taking action a at our current state, and then following policy n for the remainder of the episode

    return action_values.max(axis=1).argmax() # take the maximum action value across all policies, then take the argmax action out of these maximum action values

def evaluate(sf_list,reward_model,config): # evaluate how well sf_list performs with a new task it hasn't seen before, learned from the environment transition rewards
    """
    Evaluate how well sf_list performs with no exploration (i.e. always selecting the greedy action), using its generalized policy (learned from previous tasks),
        on a new task it hasn't seen before, learned from the environment transition rewards.
    
    Args:
        sf_list (list[SFModel]): A list of successor feature models, where each model contains the output for one feature index of the transition features, for each policy/task
        reward_model (RewardModel): The learned reward model, learned from the environment transition rewards of a task the successor feature models haven't seen before
        config (dict): A dictionary specifying parameter configurations

    Returns:
        average_return (float): The average return the agent received using sf_list, over a number of episodes specified in config['evaluate']['num_episodes']
    """
    
    env = GridWorld(config)

    total_return = 0
    for episode_num in range(1,int(config['evaluate']['num_episodes'])+1):
        
        current_state = env.reset()
        for timestep in range( int(config['evaluate']['episode_length']) ):

            # select action
            action_index = generalized_policy( sf_list , current_state , env.action_mapping.keys() , reward_model )

            # apply action
            new_state,_ = env.apply_action(action_index)

            # update new state and total_transition_features
            total_return += config['nonlinear_task_and_feature_learning']['task_to_learn'](current_state,action_index,new_state)
            current_state = new_state

    average_return = total_return / config['evaluate']['num_episodes']
    print(f"Avg undiscounted return: {average_return}")
    
    return average_return

def run_nonlinear_task_and_feature_learning_reward_model(config):
    """
    Learn a transition feature model and successor feature model using a set of nonlinear tasks, and then test it on a new nonlinear task it hasn't seen before.
    Use RewardModels to approximate the transition reward.
    
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
                             config) # use the unit basis tasks for training the successor feature models (regardless of what the learned transition features represents)

    # learn a RewardModel on a new task using learned successor feature model
    print('\n' + '='*10 + ' Learning new task vector using using learned successor feature and transition feature model ' + '='*10)
    task_learning_nonlinear(sf_list,tf_model,config)


