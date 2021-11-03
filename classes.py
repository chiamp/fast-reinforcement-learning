import torch
from torch import nn
from torch.optim import Adam

import numpy as np


class GridWorld:
    """
    An implementation of the toy environment used in the paper.
    The environment is a grid that contains the agent and different objects. The agent can move up, down, left or right to move within the grid.
    When an agent moves into a cell that contains an object, the agent picks up the object, and a new object is randomly generated on an empty cell in the grid.
    """
    
    def __init__(self,config):
        """
        Constructor method for the GridWorld class.
        
        Args:
            config (dict): A dictionary specifying parameter configurations
            
        Attributes:
            height (int): The height of the grid
            width (int): The width of the grid
            num_objects (int): The number of objects present at a given time in the grid
            num_unique_objects (int): The number of unique objects that can be generated in this instance of GridWorld
            reward_dynamics (list[float]): A vector that contains a float for each unique object, that represents the transition reward for picking up that corresponding object

            action_mapping (dict{ int : numpy.ndarray }): A dictionary that maps action indices to the corresponding translation vector that would be applied to the agent's position

            state (numpy.ndarray): A binary matrix with dimensions (self.height) x (self.width) x (self.num_unique_objects+1), where the last feature dimension represents the agent's position
            agent_position (tuple[int]): A tuple that keeps track of the current (height_index,width_index) position of the agent in the grid
        """
        
        self.height = config['gridworld']['height']
        self.width = config['gridworld']['width']
        self.num_objects = config['gridworld']['num_objects'] # initially start with num_objects in the grid, and everytime once is taken by the agent, replace it with a random object at a random open position
        self.num_unique_objects = config['gridworld']['num_unique_objects'] # for every unique object, we add an additional slice on the feature dimension of self.state
        self.reward_dynamics = np.array( config['gridworld']['reward_dynamics'] ) # a vector that determines the transition reward based on the dynamics of the environment
        # given a transition_vector that tells you how many of each unique object was picked up given the last state transition (state,action_index,new_state),
            # the transition reward is equal to the dot product of the transition_vector and self.reward_dynamics

        # self.action_mapping: dict{ action_index : translation_vector applied to the agent's position }
        self.action_mapping = { 0: np.array( (-1,0) ), # up
                                1: np.array( (1,0) ), # down
                                2: np.array( (0,-1) ), # left
                                3: np.array( (0,1) ) } # right

        self.state = None # A binary matrix with dimensions (self.height) x (self.width) x (self.num_unique_objects+1), where the last feature dimension represents the agent's position
        self.agent_position = None # (height,width) coordinate

        self.reset()
    def reset(self):
        """
        Reset GridWorld environment, and return the egocentric representation of its initial state.
        
        Args: None

        Returns:
            egocentric_state (numpy.ndarray): The egocentric representation of the initial state
        """
        
        self.state = np.zeros( ( self.height, self.width, self.num_unique_objects+1 ) ) # height x width x (num_unique_objects + agent)

        self.agent_position = self.sample_open_position()
        self.state[self.agent_position][-1] = 1 # populate agent position
        
        for _ in range(self.num_objects): self.state[self.sample_open_position()][np.random.choice(range(self.num_unique_objects))] = 1 # populate object positions with uniform probability (across positions and types of object)
        
        return self.get_state() # return egocentric representation of self.state
    def get_state(self):
        """
        Get the egocentric representation of self.state (details can be found in the supplementary information in the paper).

        Egocentric state properties:
            - matrix of dimensions: (self.height+1) x (self.width+1) x (self.num_unique_objects+1)
            - agent is always on the top left corner of egocentric state
            - each feature dimension (EXCEPT the last feature dimension) represents a binary matrix indicating whether an object of that type exists or not
            - last feature dimension represents a binary matrix indicating whether there's a wall or not (an agent feature dimension is not needed anymore, as its representation is implicit since its always in the top left corner)

        We construct the egocentric state by dividing self.state into 4 pieces, the split point being self.agent_position:
            [ d , b,
              c , a ]
        Since the agent is in the top left corner of section a, we piece together the matrix as such, so that the agent position is on the top left corner of the egocentric state
            [ a , c,
              b , d ]
        We add a horizontal wall between sections [a,c] and [b,d]
        We add a vertical wall between sections [a,b] and [c,d]
        
        Args: None

        Returns:
            egocentric_state (numpy.ndarray): The egocentric representation of self.state
        """

        egocentric_state = np.zeros( (self.height+1,self.width+1,self.num_unique_objects+1) )

        egocentric_state[ :(self.height-self.agent_position[0]) ,
                          :(self.width-self.agent_position[1]) ,
                          :self.num_unique_objects ] = self.state[ self.agent_position[0]: , self.agent_position[1]: , :self.num_unique_objects ] # bottom right (section a)

        egocentric_state[ egocentric_state.shape[0]-self.agent_position[0]: ,
                          :(self.width-self.agent_position[1]) ,
                          :self.num_unique_objects ] = self.state[ :self.agent_position[0] , self.agent_position[1]: , :self.num_unique_objects ] # top right (section b)
        
        egocentric_state[ :(self.height-self.agent_position[0]) ,
                          egocentric_state.shape[1]-self.agent_position[1]: ,
                          :self.num_unique_objects ] = self.state[ self.agent_position[0]: , :self.agent_position[1] , :self.num_unique_objects ] # bottom left (section c)
        
        egocentric_state[ egocentric_state.shape[0]-self.agent_position[0]: ,
                          egocentric_state.shape[1]-self.agent_position[1]: ,
                          :self.num_unique_objects ] = self.state[ :self.agent_position[0] , :self.agent_position[1] , :self.num_unique_objects ] # top left (section d)

        egocentric_state[(self.height-self.agent_position[0]),:,-1] = 1 # add horizontal wall
        egocentric_state[:,(self.width-self.agent_position[1]),-1] = 1 # add vertical wall

        return egocentric_state
    def sample_open_position(self):
        """
        Randomly sample an open/empty (height_index,width_index) coordinate from the grid (i.e. does not contain an object or agent).
        
        Args: None

        Returns:
            coordinates (tuple[int]): A tuple that denotes a randomly sampled empty (height_index,width_index) position in the grid
        """
        
        open_height_indices,open_width_indices = np.where( self.state.max(axis=2)==0 ) # collapse the feature dimension, and find all (height,width) coordinates that are empty
        sample_index = np.random.choice(range(len(open_height_indices)))
        return ( open_height_indices[sample_index] , open_width_indices[sample_index] )
    def apply_action(self,action_index): # given an action_index corresponding to {up,down,left,right}, return the resulting egocentric state and transition reward
        """
        Apply the action_index to the GridWorld environment and return the resulting egocentric state representation and transition reward.
        
        Args:
            action_index (int): Represents an action in the game's action space

        Returns:
            egocentric_state (numpy.ndarray): The egocentric representation of the resulting self.state after applying action action_index to the environment
            transition_reward (float): The transition reward received from applying action action_index to the environment
        """
        
        new_agent_position = np.array(self.agent_position) + self.action_mapping[action_index] # calculate new agent position
        if (new_agent_position < 0).any() or (new_agent_position[0] >= self.height) or (new_agent_position[1] >= self.width): return self.get_state(),0 # illegal move, nothing changes
        new_agent_position = tuple(new_agent_position)

        if self.state[new_agent_position].sum() > 0: object_in_new_position = True # check if the new agent position contains an object
        else: object_in_new_position = False

        transition_reward = ( self.reward_dynamics * self.state[new_agent_position][:-1] ).sum() # exclude the last feature index, as that's the agent slice

        self.state[new_agent_position] = [ 1 if i==(self.state.shape[2]-1) else 0 for i in range(self.state.shape[2]) ] # update self.state with new_agent_position
        self.state[:,:,-1][self.agent_position] = 0 # delete the old agent position in self.state

        self.agent_position = tuple(new_agent_position) # update self.agent_position attribute with new_agent_position

        if object_in_new_position: self.state[self.sample_open_position()][np.random.choice(range(self.num_unique_objects))] = 1 # replace the object consumed by the agent with a new object placed randomly in self.state
        
        return self.get_state(),transition_reward # return egocentric state and transition reward
    def __str__(self):
        """
        Return a human-readable string representation of self.state.
        The agent is represented by -1, while objects are represented by (1,2,...), for every unique object type.
        
        Args: None

        Returns:
            string_representation (str): The string representation of self.state
        """
        
        aggregated_state = self.state.copy()
        for i in range(self.num_unique_objects): # convert objects in feature_i to a number equal to (i+1)
            aggregated_state[:,:,i][ np.where( aggregated_state[:,:,i]==1 ) ] = i+1
        aggregated_state[:,:,-1][ np.where( aggregated_state[:,:,-1]==1 ) ] = -1 # convert agent location to a -1 value
        return str( aggregated_state.sum(axis=2) ) # aggregate all objects and the agent into a single
    def __repr__(self): return str(self)

class QModel(nn.Module):
    """
    An implementation of a classic Q-action_value model.
    Given a state and action, predicts the action-value; i.e. the expected discounted return the agent will receive if we follow our current policy till termination.
    """
    
    def __init__(self,config):
        """
        Constructor method for the QModel class.
        
        Args:
            config (dict): A dictionary specifying parameter configurations
            
        Attributes:
            layers (list[ torch.nn.Linear / torch.nn.ReLU ]): A list of layers in the neural entwork
            optimizer (torch.optim.Adam): Adam optimizer used for updating network weights
            loss_function (nn.MSELoss): MSE loss function used as the objective function
            action_space (range): A range that denotes the number of actions in the GridWorld environment
        """
        
        super(QModel,self).__init__()
        
        assert len(config['model']['hidden_units']) >= 1
        layers = [ nn.Linear( (config['gridworld']['height']+1) * (config['gridworld']['width']+1) * (config['gridworld']['num_unique_objects']+1) + len(config['gridworld']['action_space']) ,
                              config['model']['hidden_units'][0] ) , config['model']['activation_function'] ] # input layer; takes in the concatenation of the flattened egocentric_state and action_space vector
        for i in range(len(config['model']['hidden_units'][:-1])): layers.extend( [ nn.Linear( config['model']['hidden_units'][i] , config['model']['hidden_units'][i+1] ) , config['model']['activation_function'] ] ) # hidden layers
        layers.extend( [ nn.Linear( config['model']['hidden_units'][-1] , 1 ) ] ) # final hidden layer to output layer; outputs a single scalar value representing the action-value
        self.layers = nn.Sequential(*layers)
        
        self.optimizer = Adam(self.parameters(),lr=config['q_learning']['learning_rate'])
        self.loss_function = nn.MSELoss()

        self.action_space = config['gridworld']['action_space']
    def forward(self,state,action_index):
        """
        Forward propagation method. Given a state and action_index, return the predicted Q action-value.
        
        Args:
            state (numpy.ndarray): An egocentric representation of the state
            action_index (int): Represents an action in the game's action space

        Returns:
            action_value (float): The predicted Q action-value
        """

        return self.layers( torch.cat( ( torch.flatten( torch.tensor(state).float() ) ,
                                         torch.tensor( [ 1 if i==action_index else 0 for i in self.action_space ] ).float() )
                                       )
                            ) # concatenate flattened state and action vector and feed it into self.layers
    def __call__(self,state,action_index): return self.forward(state,action_index)
    def update_weights(self,state,action_index,td_estimate): # update the parameters of this model
        """
        Given the state, action_index and td_estimate (calculated using bootstrapping), update the parameters of this model.
        
        Args:
            state (numpy.ndarray): An egocentric representation of the state
            action_index (int): Represents an action in the game's action space
            td_estimate (float): The "ground truth" value calculated from bootstrapping, using the Bellman equation

        Returns: None
        """
        
        pred = self.forward(state,action_index) # output neuron is a prediction on the expected discounted return, if we were to follow the policy defined by this model until termination
        
        self.optimizer.zero_grad() # zero gradients
        self.loss_function(pred,td_estimate).backward() # compute gradients
        self.optimizer.step() # apply gradient to model
        
class SFModel(nn.Module): # successor feature model
    """
    An implementation of the successor feature model outlined in the paper.
    Given a state and action_index, predicts the discounted transition feature returns for a specific feature index (equivalent to the psi_tilde function in the paper).
    Each output neuron j corresponds to the discounted feature return (for a specific feature index) for following a corresponding policy j
        i.e. each SFModel is feature specific; it outputs values for a specific feature, across all policies.
    To get the predicted return (i.e. discounted total reward) for a particular policy j, we get the jth output of all SFModels,
        concatenate them together to get the expected discounted feature vector and dot product with task vector w
    """
    
    def __init__(self,config):
        """
        Constructor method for the SFModel class.
        
        Args:
            config (dict): A dictionary specifying parameter configurations
            
        Attributes:
            layers (list[ torch.nn.Linear / torch.nn.ReLU ]): A list of layers in the neural entwork
            optimizer (torch.optim.Adam): Adam optimizer used for updating network weights
            loss_function (nn.MSELoss): MSE loss function used as the objective function
            action_space (range): A range that denotes the number of actions in the GridWorld environment
            total_mse (float): Keeps track of the total loss accumulated since the last interval of the training algorithm
        """
        
        super(SFModel,self).__init__()

        # construct neural network layers
        assert len(config['model']['hidden_units']) >= 1
        layers = [ nn.Linear( (config['gridworld']['height']+1) * (config['gridworld']['width']+1) * (config['gridworld']['num_unique_objects']+1) + len(config['gridworld']['action_space']) ,
                              config['model']['hidden_units'][0] ) , config['model']['activation_function'] ] # input layer; takes in the concatenation of the flattened egocentric_state and action_space vector
        for i in range(len(config['model']['hidden_units'][:-1])): layers.extend( [ nn.Linear( config['model']['hidden_units'][i] , config['model']['hidden_units'][i+1] ) , config['model']['activation_function'] ] ) # hidden layers
        layers.extend( [ nn.Linear( config['model']['hidden_units'][-1] , config['task_and_feature_learning']['num_task_features'] ) ] ) # final hidden layer to output layer; outputs a value for each policy (and there's one policy per task)
        self.layers = nn.Sequential(*layers)
        
        self.optimizer = Adam(self.parameters(),lr=config['sf_q_learning']['learning_rate'])
        self.loss_function = nn.MSELoss()

        self.action_space = config['gridworld']['action_space']

        self.total_mse = 0
    def forward(self,state,action_index):
        """
        Forward propagation method. Given a state and action_index, return the predicted discounted feature returns.
        
        Args:
            state (numpy.ndarray): An egocentric representation of the state
            action_index (int): Represents an action in the game's action space

        Returns:
            feature_returns (numpy.ndarray): The predicted discounted feature returns
        """
        
        return self.layers( torch.cat( ( torch.flatten( torch.tensor(state).float() ) ,
                                         torch.tensor( [ 1 if i==action_index else 0 for i in self.action_space ] ).float() )
                                       )
                            ) # concatenate flattened state and action vector and feed it into self.layers
    def __call__(self,state,action_index): return self.forward(state,action_index)
    def update_weights(self,state,action_index,td_estimate,policy_index): # update the parameters of this model
        """
        Given the state, action_index, td_estimate (calculated using bootstrapping) and policy_index, update the parameters of this model.
        
        Args:
            state (numpy.ndarray): An egocentric representation of the state
            action_index (int): Represents an action in the game's action space
            td_estimate (float): The "ground truth" value calculated from bootstrapping, using the Bellman equation
            policy_index (int): The policy we're following for this current episode

        Returns: None
        """
        
        pred = self.forward(state,action_index) # each output neuron is a prediction on the expected discounted feature returns, if we were to follow each policy
        target = torch.clone(pred)
        target[policy_index] = td_estimate # td_estimate is equivalent to the transition feature (given by the phi function in the paper),
                                               # plus the maximum discounted feature return of the future state, if we take the maximizing action and follow policy policy_index until termination
        # we replace the policy_indexth entry (because we used the policy_indexth policy in this iteration of self-play) in the pred output with the td_estimate, because it's a more accurate estimate,
        # and then we train the network using this target
        
        self.optimizer.zero_grad() # zero gradients

        loss = self.loss_function( pred , target )
        self.total_mse += loss.detach().numpy()
        loss.backward()
        
        self.optimizer.step() # apply gradient to model

class TFModel(nn.Module):
    """
    An implementation of the transition feature model outlined in the paper.
    Given a state, action and resulting new_state, predicts the transition features (equivalent to the phi_tilde function in the paper).
    Each output neuron d corresponds to transition feature d.
    """
    
    def __init__(self,config):
        """
        Constructor method for the TFModel class.
        
        Args:
            config (dict): A dictionary specifying parameter configurations
            
        Attributes:
            layers (list[ torch.nn.Linear / torch.nn.ReLU ]): A list of layers in the neural entwork
            optimizer (torch.optim.Adam): Adam optimizer used for updating network weights
            loss_function (nn.MSELoss): MSE loss function used as the objective function
            action_space (range): A range that denotes the number of actions in the GridWorld environment
            total_mse (float): Keeps track of the total loss accumulated since the last interval of the training algorithm
        """
        
        super(TFModel,self).__init__()
        
        assert len(config['model']['hidden_units']) >= 1
        layers = [ nn.Linear( 2 * (config['gridworld']['height']+1) * (config['gridworld']['width']+1) * (config['gridworld']['num_unique_objects']+1) + len(config['gridworld']['action_space']) ,
                              config['model']['hidden_units'][0] ) , config['model']['activation_function'] ] # input layer; takes in the concatenation of the flattened egocentric_state, action_space vector and flattened future_egocentric_state
        for i in range(len(config['model']['hidden_units'][:-1])): layers.extend( [ nn.Linear( config['model']['hidden_units'][i] , config['model']['hidden_units'][i+1] ) , config['model']['activation_function'] ] ) # hidden layers
        layers.extend( [ nn.Linear( config['model']['hidden_units'][-1] , config['task_and_feature_learning']['num_task_features'] ) ] ) # final hidden layer to output layer; outputs the predicted transition features
        self.layers = nn.Sequential(*layers)
        
        self.optimizer = Adam(self.parameters(),lr=config['task_and_feature_learning']['tf_model_learning_rate'])
        self.loss_function = nn.MSELoss()

        self.action_space = config['gridworld']['action_space']

        self.total_mse = 0
    def forward(self,state,action_index,new_state):
        """
        Forward propagation method. Given a state, action_index and resulting new_state, return the predicted transition features.
        
        Args:
            state (numpy.ndarray): An egocentric representation of the state
            action_index (int): Represents an action in the game's action space
            new_state (numpy.ndarray): An egocentric representation of the resulting state, after applying action action_index to state

        Returns:
            transition_features (numpy.ndarray): The predicted transition features
        """
        
        return self.layers( torch.cat( ( torch.flatten( torch.tensor(state).float() ) ,
                                         torch.tensor( [ 1 if i==action_index else 0 for i in self.action_space ] ).float() ,
                                         torch.flatten( torch.tensor(new_state).float() ) )
                                       )
                            ) # concatenate flattened state, action vector and flattened new_state, and feed it into self.layers
    def __call__(self,state,action_index,new_state): return self.forward(state,action_index,new_state)
    def update_weights(self,state,action_index,new_state,learned_task_vector,transition_reward): # update the parameters of this model
        """
        Given the state, action_index, new_state, learned_task_vector and transition_reward, update the parameters of this model.
        The state, action_index and new_state are used to get the TFModel's predicted transition features.
        The predicted transition features are then dot producted with the learned_task_vector to get the predicted transition reward.
        The predicted transition reward is then matched against the "ground truth" transition_reward, and the parameters of this model are updated accordingly.
        
        Args:
            state (numpy.ndarray): An egocentric representation of the state
            action_index (int): Represents an action in the game's action space
            new_state (numpy.ndarray): An egocentric representation of the resulting state, after applying action action_index to state
            learned_task_vector (numpy.ndarray): The learned task vector used to calculate the transition_reward
            transition_reward (float): The reward received for the particular task learned_task_vector

        Returns: None
        """
        
        pred = torch.dot( self.forward(state,action_index,new_state) , torch.tensor(learned_task_vector).float() ) # pred is the predicted transition_reward, which is the dot product of this model's output and the learned_task_vector
        if pred == transition_reward: return # gradient will be zero if prediction matches target transition_reward, so no need to update weights for this call

        self.optimizer.zero_grad() # zero gradients
        
        loss = self.loss_function( pred , torch.tensor(transition_reward).float() )
        self.total_mse += loss.detach().numpy()
        loss.backward() # compute gradients; the target is the actual transition_reward received from the environment

        self.optimizer.step() # apply gradient to model

class RewardModel(nn.Module):
    """
    A neural network that replaces the task vector in the earlier linear experiments.
    Rather than dot producting a task vector with transition features to get the predicted transition reward, we feed in the transition features into a neural network.
    The neural network outputs the predicted transition reward.
    """
    
    def __init__(self,tf_model,config):
        """
        Constructor method for the RewardModel class.
        
        Args:
            tf_model (TFModel): The learned transition feature function (equivalent to the phi_tilde function in the paper). This will be needed for both forward and back propagation.
            config (dict): A dictionary specifying parameter configurations
            
        Attributes:
            layers (list[ torch.nn.Linear / torch.nn.ReLU ]): A list of layers in the neural entwork
            tf_model (TFModel): The transition feature model, whose predicted transition feature outputs will be used as inputs into the RewardModel's network
            
            reward_optimizer (torch.optim.Adam): Adam optimizer used for updating just the RewardModel weights
            full_optimizer (torch.optim.Adam): Adam optimizer used for updating both the RewardModel and tf_model weights
            
            loss_function (nn.MSELoss): MSE loss function used as the objective function
            action_space (range): A range that denotes the number of actions in the GridWorld environment
            total_mse (float): Keeps track of the total loss accumulated since the last interval of the training algorithm
        """
        
        super(RewardModel,self).__init__()
        
        assert len(config['model']['hidden_units']) >= 1
        layers = [ nn.Linear( config['task_and_feature_learning']['num_task_features'], config['model']['hidden_units'][0] ) ,
                   config['model']['activation_function'] ] # input layer; takes in the predicted transition features from the transition feature function, TFModel (equilvaent to the phi_tilde function in the paper)
        for i in range(len(config['model']['hidden_units'][:-1])): layers.extend( [ nn.Linear( config['model']['hidden_units'][i] , config['model']['hidden_units'][i+1] ) , config['model']['activation_function'] ] ) # hidden layers
        layers.extend( [ nn.Linear( config['model']['hidden_units'][-1] , 1 ) ] ) # final hidden layer to output layer; outputs the predicted transition reward
        self.layers = nn.Sequential(*layers)

        self.reward_optimizer = Adam(self.parameters(),lr=config['nonlinear_task_and_feature_learning']['reward_model_learning_rate']) # learnable parameters only include RewardModel's weights

        self.tf_model = tf_model # keep track of trannsition feature function, as when we update RewardModel weights, tf_model weights will also be updated since we use it's outputted transition features as our input for our network
        
        self.full_optimizer = Adam(self.parameters(),lr=config['nonlinear_task_and_feature_learning']['reward_model_learning_rate']) # learnable parameters include tf_model's parameters as well
        self.loss_function = nn.MSELoss()

        self.action_space = config['gridworld']['action_space']

        self.total_mse = 0
    def forward(self,transition_features):
        """
        Forward propagation method. Given the predicted transition features from self.tf_model, output the predicted transition reward.
        
        Args:
            transition_features (numpy.ndarray): The outputted predicted transition features from self.tf_model

        Returns:
            transition_reward (float): The predicted transition features
        """

        return self.layers(transition_features) # feed inputs to tf_model, get predicted transition features, and feed that into RewardModel's layers to get predictted transition reward
    def __call__(self,transition_features): return self.forward(transition_features)
    def update_weights(self,state,action_index,new_state,transition_reward,full=False): # update the parameters of this model
        """
        Given the state, action_index, new_state and transition_reward, update the parameters of this model.
        The state, action_index and new_state are used to get self.tf_model's predicted transition features.
        The predicted transition features are then fed into the RewardModel's network to get the predicted transition reward.
        The predicted transition reward is then matched against the "ground truth" transition_reward, and the parameters of this model are updated accordingly.
        
        Args:
            state (numpy.ndarray): An egocentric representation of the state
            action_index (int): Represents an action in the game's action space
            new_state (numpy.ndarray): An egocentric representation of the resulting state, after applying action action_index to state
            transition_reward (float): The reward received for the particular task learned_task_vector
            full (bool): True if you want to update both the RewardModel and self.tf_model weights, False if you want to update only the RewardModel weights

        Returns: None
        """
        
        pred = self.forward( self.tf_model(state,action_index,new_state) ) # pred is the predicted transition_reward
        if pred == transition_reward: return # gradient will be zero if prediction matches target transition_reward, so no need to update weights for this call

        if full: self.full_optimizer.zero_grad()
        else: self.reward_optimizer.zero_grad() # zero gradients
        
        loss = self.loss_function( pred , torch.tensor([transition_reward]).float() )
        self.total_mse += loss.detach().numpy()
        loss.backward() # compute gradients; the target is the actual transition_reward received from the environment

        if full: self.full_optimizer.step()
        else: self.reward_optimizer.step() # apply gradient to model
