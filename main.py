import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt

import pickle
import time

from sf_q_learning import run_sf_q_learning
from task_learning import run_task_learning

from q_learning import run_q_learning

from task_and_feature_learning import run_task_and_feature_learning

from nonlinear_task_and_feature_learning_reward_model import run_nonlinear_task_and_feature_learning_reward_model
from nonlinear_task_and_feature_learning_task_vector import run_nonlinear_task_and_feature_learning_task_vector

import nonlinear_reward_functions


def run_linear_experiments(config):
    """
    Train a successor feature model using hand-crafted transition features and hand-crafted task vectors, and evaluate its performance on a task it hasn't seen before.
    Train a successor feature model using learned transition features and learned task vectors, and evaluate its performance on a task it hasn't seen before.
    Train a Q-learning model to compare how fast it converges relative to the successor feature models.
    All tasks are linear.
    
    Args:
        config (dict): A dictionary specifying parameter configurations

    Returns: None
    """
    

    start_time = time.time()

    # learn successor feature model from learned task vectors and a learned transition feature function
    run_task_and_feature_learning(config)
    
    # learn successor feature model from hand-crafted task vectors and a hand-crafted transition feature function
    run_sf_q_learning(config)
    run_task_learning(config)

    # learn Q-function from hand-crafted task vectors and a hand-crafted transition feature function
    run_q_learning(config)

    print(f'Finished training all models for linear experiments in {(time.time()-start_time)/3600} hours')

    # compare the performance of:
        # the learned successor feature model
        # the hand-crafted successor feature model
        # the Q-function
    # based on the average return of the agent using these models as its policy, and how fast it converges
    
    evaluation_data = {}
    with open('evaluation_data/classic_q_learning.pkl','rb') as file: evaluation_data['q_learning'] = pickle.load(file)
    with open('evaluation_data/task_learning.pkl','rb') as file: evaluation_data['task_learning'] = pickle.load(file)
    with open(f"evaluation_data/task_learning_with_learned_transition_features_of_dimension_{config['task_and_feature_learning']['num_task_features']}.pkl",'rb') as file:
        evaluation_data['task_and_feature_learning'] = pickle.load(file)

    for learning_protocol in evaluation_data: plt.plot(evaluation_data[learning_protocol]['num_sample_transitions'],evaluation_data[learning_protocol]['average_sum_rewards'],label=learning_protocol)
    plt.legend()
    plt.xlabel('Number of sample transitions')
    plt.ylabel('Average sum of rewards over 100 episodes')
    plt.show()
    plt.clf()

def run_nonlinear_experiments(config):
    """
    Train a successor feature model using learned transition features and learned task vectors, and evaluate its performance on a task it hasn't seen before.
    Train a successor feature model using learned transition features and learned reward models, and evaluate its performance on a task it hasn't seen before.
    Compare the two successor feature models in terms of the average return per episode.
    All tasks are nonlinear.
    
    Args:
        config (dict): A dictionary specifying parameter configurations

    Returns: None
    """

    start_time = time.time()

    # learn successor feature model from learned task vectors and a learned transition feature function, using a task vector that will be dot producted with the transition features
    # this is identical to the training process in run_task_and_feature_learning(), except we're using nonlinear tasks
    run_nonlinear_task_and_feature_learning_task_vector(config)

    # learn successor feature model from learned task vectors and a learned transition feature function, using a reward neural network
    run_nonlinear_task_and_feature_learning_reward_model(config)

    print(f'Finished training all models for nonlinear experiments in {(time.time()-start_time)/3600} hours')

    # compare the performance of:
        # the learned successor feature model, using task vectors
        # the learned successor feature model, using a reward neural network
    # based on the average return of the agent using these models as its policy, and how fast it converges

    evaluation_data = {}
    with open(f"evaluation_data/nonlinear_task_learning_with_learned_transition_features_of_dimension_{config['task_and_feature_learning']['num_task_features']}_task_vector.pkl",'rb') as file:
        evaluation_data['nonlinear_task_learning_task_vector'] = pickle.load(file)
    with open(f"evaluation_data/nonlinear_task_learning_with_learned_transition_features_of_dimension_{config['task_and_feature_learning']['num_task_features']}_reward_model.pkl",'rb') as file:
        evaluation_data['nonlinear_task_and_feature_learning_reward_model'] = pickle.load(file)

    for learning_protocol in evaluation_data: plt.plot(evaluation_data[learning_protocol]['num_sample_transitions'],evaluation_data[learning_protocol]['average_sum_rewards'],label=learning_protocol)
    plt.legend()
    plt.xlabel('Number of sample transitions')
    plt.ylabel('Average sum of rewards over 100 episodes')
    plt.show()
    plt.clf()
    
    
if __name__ == '__main__':
    config = { 'gridworld': { 'height': 10,
                              'width': 10,
                              'num_objects': 10,
                              'num_unique_objects': 2,
                              'action_space': range(4),
                              'reward_dynamics': [1,-1] }, # reward for the agent collecting the corresponding object (e.g. [1,-1,0] would mean +1 reward for collecting the first object, -1 and 0 for collecting the others, respectively)
               'sf_q_learning': { 'tasks': [ [1,0], # corresponds to the w-vectors in the paper
                                             [0,1] ], # there will be one policy for each corresponding task
                                  'learning_rate': 1e-4,
                                 'discount_factor': 0.9,
                                 'exploration_threshold': 0.1, # refers to the probability of sampling a random move, as opposed to following the agent's policy
                                 'num_episodes': 25000, # number of episodes to self-play and train
                                 'episode_length': 40, # length of episodes
                                 'interval': 1e2 }, # the interval to save the models and print an update
               'task_learning': { 'num_task_features': 2, # the number of features in the task vector
                                  'learning_rate': 1e-2,
                                  'num_episodes': 25000,
                                  'episode_length': 40,
                                  'interval': 1e2 },
               'q_learning': { 'task': [1,-1], # single task for classical q-learning
                               'learning_rate': 1e-4,
                               'discount_factor': 0.9,
                               'exploration_threshold': 0.1,
                               'num_episodes': 25000,
                               'episode_length': 40,
                               'interval': 1e2 },
               'task_and_feature_learning': { 'num_task_features': 2, # the number of features in the task vector (will also be used to determine phi_tilde function output dimension)
                                              'tf_model_learning_rate': 1e-4,
                                              'task_vector_learning_rate': 1e-3,
                                              'tasks': [ [1,0], # tasks to be used, to learn the transition feature model (equivalent to the phi_tilde function in the paper)
                                                         [0,1], # should match with the sf_q_learning tasks
                                                         [-1,1],
                                                         [1,1]
                                                         ],
                                              'num_episodes': 60000,
                                              'episode_length': 50,
                                              'interval': 1e2,
                                              },
               'nonlinear_task_and_feature_learning': { 'reward_model_learning_rate': 1e-4,
                                                        'tasks': [ nonlinear_reward_functions.pickup_even, # nonlinear task functions used to train the transition feature model (equivalent to the phi_tilde function in the paper)
                                                                   nonlinear_reward_functions.pickup_odd,
                                                                   nonlinear_reward_functions.pickup_vertical,
                                                                   nonlinear_reward_functions.pickup_horizontal ],
                                                        'task_to_learn': nonlinear_reward_functions.pickup_abundant }, # nonlinear task function used to evaluate how well the learned successor feature model can learn a new task
               'model': { 'hidden_units': [64,128], # number of hidden units for each hidden layer respectively
                          'activation_function': nn.ReLU() },
               'evaluate': { 'task': [1,-1], # task to evaluate the generalized policy
                             'num_episodes': 1e2, # number of episodes to average results over
                             'episode_length': 40 }, # length of episodes
               'seed': 1
               }

    # set random seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # run experiments
    run_linear_experiments(config)
    run_nonlinear_experiments(config)
