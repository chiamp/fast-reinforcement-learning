import numpy as np


def pickup_abundant(state,action_index,new_state):
    """
    +1 reward for picking up the object that's the most abundant.
    -1 reward for picking up the object that's not the most abundant.
    If no object is picked up, 0 reward.
    
    Args:
        state (numpy.ndarray): An egocentric representation of the state
        action_index (int): Represents an action in the game's action space
        new_state (numpy.ndarray): An egocentric representation of the resulting state, after applying action action_index to state

    Returns:
        reward (int): The transition reward
    """
    
    if action_index == 0: destination_cell = state[-1,0,:-1] # up
    elif action_index == 1: destination_cell = state[1,0,:-1] # down
    elif action_index == 2: destination_cell = state[0,-1,:-1] # left
    elif action_index == 3: destination_cell = state[0,1,:-1] # right
    else: raise Exception(f'Invalid action_index: {action_index}')

    if np.sum(destination_cell) == 0: return 0 # agent didn't pick up anything

    num_objects = [ np.sum(state[:,:,i]) for i in range(state.shape[2]-1) ] # list[ num_type_0_objects, ... , num_type_n_objects ]
    max_number = max(num_objects) # the highest count for any unique object currently in state
    valid_object_pickup = [ i for i in range(len(num_objects)) if num_objects[i]==max_number ] # the unique object index that is a valid pickup since it has the maximum count

    if destination_cell.argmax() in valid_object_pickup: return 1
    else: return -1

def pickup_even(state,action_index,new_state):
    """
    If even number of objects for each unique object, +1 reward for picking up object 0, -1 reward for picking up object 1.
    If odd number of objects for each unique object, -1 reward for picking up object 0, +1 reward for picking up object 1.
    If no object is picked up, 0 reward.
    
    Args:
        state (numpy.ndarray): An egocentric representation of the state
        action_index (int): Represents an action in the game's action space
        new_state (numpy.ndarray): An egocentric representation of the resulting state, after applying action action_index to state

    Returns:
        reward (int): The transition reward
    """
    
    
    if action_index == 0: destination_cell = state[-1,0,:-1] # up
    elif action_index == 1: destination_cell = state[1,0,:-1] # down
    elif action_index == 2: destination_cell = state[0,-1,:-1] # left
    elif action_index == 3: destination_cell = state[0,1,:-1] # right
    else: raise Exception(f'Invalid action_index: {action_index}')

    if np.sum(destination_cell) == 0: return 0 # agent didn't pick up anything

    num_objects = [ np.sum(state[:,:,i]) for i in range(state.shape[2]-1) ] # list[ num_type_0_objects, ... , num_type_n_objects ]
    if num_objects[0] % 2 == 0: # even
        if destination_cell[0] == 1: return 1
        else: return -1
    else: # odd
        if destination_cell[1] == 1: return 1
        else: return -1

def pickup_odd(state,action_index,new_state):
    """
    If odd number of objects for each unique object, +1 reward for picking up object 0, -1 reward for picking up object 1.
    If even number of objects for each unique object, -1 reward for picking up object 0, +1 reward for picking up object 1.
    If no object is picked up, 0 reward.
    
    Args:
        state (numpy.ndarray): An egocentric representation of the state
        action_index (int): Represents an action in the game's action space
        new_state (numpy.ndarray): An egocentric representation of the resulting state, after applying action action_index to state

    Returns:
        reward (int): The transition reward
    """
    
    if action_index == 0: destination_cell = state[-1,0,:-1] # up
    elif action_index == 1: destination_cell = state[1,0,:-1] # down
    elif action_index == 2: destination_cell = state[0,-1,:-1] # left
    elif action_index == 3: destination_cell = state[0,1,:-1] # right
    else: raise Exception(f'Invalid action_index: {action_index}')

    if np.sum(destination_cell) == 0: return 0 # agent didn't pick up anything

    num_objects = [ np.sum(state[:,:,i]) for i in range(state.shape[2]-1) ] # list[ num_type_0_objects, ... , num_type_n_objects ]
    if num_objects[0] % 2 == 1: # odd
        if destination_cell[0] == 1: return 1
        else: return -1
    else: # even
        if destination_cell[1] == 1: return 1
        else: return -1

def pickup_vertical(state,action_index,new_state):
    """
    Picking up objects that are vertically in line with other objects gives a +1 reward.
    Picking up objects that aren't vertically in line with any other object gives a +0.1 reward.
    If no object is picked up, 0 reward.
    
    Args:
        state (numpy.ndarray): An egocentric representation of the state
        action_index (int): Represents an action in the game's action space
        new_state (numpy.ndarray): An egocentric representation of the resulting state, after applying action action_index to state

    Returns:
        reward (float): The transition reward
    """
    
    
    if action_index == 0:
        destination_cell = state[-1,0,:-1] # up
        destination_column = state[:,0,:-1]
    elif action_index == 1:
        destination_cell = state[1,0,:-1] # down
        destination_column = state[:,0,:-1]
    elif action_index == 2:
        destination_cell = state[0,-1,:-1] # left
        destination_column = state[:,-1,:-1]
    elif action_index == 3:
        destination_cell = state[0,1,:-1] # right
        destination_column = state[:,1,:-1]
    else: raise Exception(f'Invalid action_index: {action_index}')

    if np.sum(destination_cell) == 0: return 0 # agent didn't pick up anything
    elif np.sum(destination_column) == 1: return 0.1 # agent picked up an object and it was the only object in that vertical column
    else: return 1 # agent picked up an object and there's at least another object in that vertical column

def pickup_horizontal(state,action_index,new_state):
    """
    Picking up objects that are horizontally in line with other objects gives a +1 reward
    Picking up objects that aren't horizontally in line with any other object gives a +0.1 reward
    If no object is picked up, 0 reward.
    
    Args:
        state (numpy.ndarray): An egocentric representation of the state
        action_index (int): Represents an action in the game's action space
        new_state (numpy.ndarray): An egocentric representation of the resulting state, after applying action action_index to state

    Returns:
        reward (float): The transition reward
    """
    
    
    if action_index == 0:
        destination_cell = state[-1,0,:-1] # up
        destination_row = state[-1,:,:-1]
    elif action_index == 1:
        destination_cell = state[1,0,:-1] # down
        destination_row = state[1,:,:-1]
    elif action_index == 2:
        destination_cell = state[0,-1,:-1] # left
        destination_row = state[0,:,:-1]
    elif action_index == 3:
        destination_cell = state[0,1,:-1] # right
        destination_row = state[0,:,:-1]
    else: raise Exception(f'Invalid action_index: {action_index}')

    if np.sum(destination_cell) == 0: return 0 # agent didn't pick up anything
    elif np.sum(destination_row) == 1: return 0.1 # agent picked up an object and it was the only object in that horizontal column
    else: return 1 # agent picked up an object and there's at least another object in that horizontal column

