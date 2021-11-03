import autograd.numpy as np
from autograd import grad

import pickle
import time

from classes import GridWorld
from sf_q_learning import get_transition_features,generalized_policy


def run_task_learning(config):
    """
    Using a learned set of successor feature models, learn a task vector from the environment transition rewards, that the successor feature models haven't seen before.
    
    Args:
        config (dict): A dictionary specifying parameter configurations

    Returns: None
    """
    
    print('='*10 + ' Learning task vector using hand-crafted feature model ' + '='*10)
    
    with open('models/sf_list.pkl','rb') as file: sf_list = pickle.load(file) # load the list of pre-trained successor feature models (they will be used to evaluate the accuracy of our learned task_vector)

    task_vector = np.random.rand(config['task_learning']['num_task_features']) # task_vector to be learned from environment transition rewards
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
            new_state,reward = env.apply_action(action_index)

            # update task_vector values
            task_vector -= config['task_learning']['learning_rate'] * grad_loss( task_vector , get_transition_features(current_state,action_index,new_state) , reward )

            # update new state
            current_state = new_state

        if episode_num % config['task_learning']['interval'] == 0:

            # update evaluation_data statistics
            evaluation_data['num_sample_transitions'].append( episode_num * config['task_learning']['episode_length'] )
            evaluation_data['average_sum_rewards'].append( evaluate(sf_list,task_vector,config) )

            # save evaluation_data
            with open('evaluation_data/task_learning.pkl','wb') as file: pickle.dump(evaluation_data,file)

            # print an update
            print(f"Episode: {episode_num}\tLearned task: {task_vector}\tFinished in {time.time()-start_time} seconds")
            
            start_time = time.time()

def loss(task_vector,transition_features,reward):
    """
    Calculate the squared error of the predicted transition reward and the actual environment transition reward
    
    Args:
        task_vector (list[float]): The task used for this episode, which will determine the transition rewards received
        transition_features (numpy.ndarray): The transition features that resulted in receiving the transition reward
        reward (float): The transition reward

    Returns:
        squared_error (float): The squared error of the predicted transition reward and the actual transition reward
    """
    
    return ( np.dot( transition_features , task_vector ) - reward )**2

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

    total_undiscounted_return = 0 # keep track of all accumulated rewards throughout all episodes
    for episode_num in range(1,int(config['evaluate']['num_episodes'])+1):
        
        current_state = env.reset()
        total_transition_features = np.zeros( len(config['evaluate']['task']) )
        for timestep in range( int(config['evaluate']['episode_length']) ):

            # select action
            action_index = generalized_policy( sf_list , current_state , env.action_mapping.keys() , task_vector )

            # apply action
            new_state,reward = env.apply_action(action_index)

            # update new state and total_transition_features
            total_transition_features += get_transition_features(current_state,action_index,new_state)
            current_state = new_state

        # add this episode's undiscounted return to total_undiscounted_return
        total_undiscounted_return += np.dot(total_transition_features,config['evaluate']['task'])

    return total_undiscounted_return / config['evaluate']['num_episodes']
