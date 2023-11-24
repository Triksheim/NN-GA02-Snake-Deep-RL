from replay_buffer import ReplayBufferNumpy
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary



class Agent():
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent

    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _board_grid : Numpy array
        A square filled with values from 0 to board size **2,
        Useful when converting between row, col and int representation
    _version : str
        model version string
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._n_frames, self._board_size, self._board_size)
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2).reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer 
        
        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
        """Get the current buffer size
        
        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None, return_buffer=False):
        """Load the buffer from disk
        
        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)
            print(f'Buffer {file_path}/buffer_{iteration} loaded')
        if return_buffer:
            return self._buffer

    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row*self._board_size + col






class DeepQLearningAgent(Agent):
    """
    This agent class implements the Q-learning algorithm using Deep Learning with PyTorch framework.
    """

    def __init__(self, board_size=10, frames=4, buffer_size=10000,
             gamma=0.99, n_actions=3, use_target_net=True,
             version=''):
        super().__init__(board_size, frames, buffer_size,
                        gamma, n_actions, use_target_net, version)
        
        # Determine the device to use (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize the model and transfer it to the appropriate device
        self._model = self._agent_model().to(self.device)
        if self._use_target_net:
            self._target_net = self._agent_model().to(self.device)
            self.update_target_net()

        # Set up the optimizer and the loss function for training. 
        self._optimizer = optim.RMSprop(self._model.parameters(), lr=0.0005)
        self._loss_function = nn.SmoothL1Loss() # Equals Huberloss
        

    def reset_models(self):
        """ Reset all the models by creating new graphs"""
        self._model = self._agent_model()
        if(self._use_target_net):
            self._target_net = self._agent_model()
            self.update_target_net()


    def _prepare_input(self, board):
        """
        Prepares the input data for the network. This includes converting the data to a tensor, 
        ensuring it's on the correct device, adjusting its dimensions and normalizing.
        """
        # Convert the board to a PyTorch tensor first if it's not already one
        if not isinstance(board, torch.Tensor):
            board = torch.tensor(board, dtype=torch.float32)

        # Move the tensor to the specified device (GPU or CPU)
        board = board.to(self.device)

        # Check if board is a single sample or a batch
        if board.ndim == 3:  # Assuming board is a single sample
            board = board.permute(2, 0, 1).unsqueeze(0)  # channel-first format and add batch dimension
        elif board.ndim == 4:  # Assuming board is already a batch
            board = board.permute(0, 3, 1, 2)  # channel-first format

        board = self._normalize_board(board)
        return board

    def _normalize_board(self, board):
        # Normalize the board before input to the network
        return board / 4.0


    def _get_model_outputs(self, board, model=None):
        """
        Passes the input through the neural network (forward pass) to get the output, 
        which in the context of Q-learning, represents the Q-values for each action.
        """
        board = self._prepare_input(board)
        if model is None:
            model = self._model
        model_outputs = model(board)    # Model forward
        return model_outputs

    
    def move(self, board, legal_moves, value=None):
        """
        Determines the best action to take based on the current state (board).
        It masks out illegal moves and selects the action with the highest Q-value.
        """
        model_outputs = self._get_model_outputs(board, self._model).to(self.device)
        # Ensure legal_moves_tensor is on the same device as model_outputs
        legal_moves_tensor = torch.tensor(legal_moves).to(self.device)
        # Create a tensor with -np.inf on the same device as model_outputs
        inf_tensor = torch.full_like(model_outputs, -np.inf)

        # Make sure output is on CPU since replay buffer uses Numpy
        return torch.argmax(torch.where(legal_moves_tensor == 1, model_outputs, inf_tensor), axis=1).cpu().numpy()


    def _agent_model(self):
        """
        Defines the the neural network model.
        Input: channels = self._n_frames, dim = board_size*board_size
        Output: channels = self._n_actions
        
        Summary: n_frames=2, n_actions=4, dim=10x10
        ==========================================================================================
        Layer (type:depth-idx)                   Output Shape              Param #
        ==========================================================================================
        ├─Conv2d: 1-1                            [-1, 16, 10, 10]          304
        ├─ReLU: 1-2                              [-1, 16, 10, 10]          --
        ├─Conv2d: 1-3                            [-1, 32, 8, 8]            4,640
        ├─ReLU: 1-4                              [-1, 32, 8, 8]            --
        ├─Conv2d: 1-5                            [-1, 64, 4, 4]            51,264
        ├─ReLU: 1-6                              [-1, 64, 4, 4]            --
        ├─Flatten: 1-7                           [-1, 1024]                --
        ├─Linear: 1-8                            [-1, 64]                  65,600
        ├─ReLU: 1-9                              [-1, 64]                  --
        ├─Linear: 1-10                           [-1, 4]                   260
        ==========================================================================================
        Total params: 122,068
        Trainable params: 122,068
        """
        
        model = nn.Sequential(                                              
            nn.Conv2d(self._n_frames, 16, kernel_size=3, padding=1, ),      # 2ch in 10x10, 16ch out 10x10
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3 ),                              # 16ch in 10x10, 32ch out 8x8
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=5 ),                              # 32ch in 8x8, 64ch out 4x4
            nn.ReLU(),

            nn.Flatten(),                                                   # flatten to 1 dim,  64*4*4 = 1024ch out

            nn.Linear(64 * (self._board_size - 6) * (self._board_size - 6), 64),   # 1024ch in, 64ch out
            nn.ReLU(),
            nn.Linear(64, self._n_actions)                                  # 64ch in, 4ch out
        )
        return model


    # def set_weights_trainable(self):
    #     """Set selected layers to non trainable and compile the model"""
    #     for layer in self._model.layers:
    #         layer.trainable = False
    #     # the last dense layers should be trainable
    #     for s in ['action_prev_dense', 'action_values']:
    #         self._model.get_layer(s).trainable = True
    #     self._model.compile(optimizer = self._model.optimizer, 
    #                         loss = self._model.loss)


    def save_model(self, file_path='', iteration=None):
        """Save the current models to disk using PyTorch
        inbuilt save model function (saves in pth format)
        
        Parameters
        ----------
        file_path : str, optional
            Path where to save the file
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        
        torch.save(self._model.state_dict(), "{}/model_{:04d}.pth".format(file_path, iteration) )
        if(self._use_target_net):
            torch.save(self._target_net.state_dict(), "{}/model_{:04d}_target.pth".format(file_path, iteration) )
        
    def load_model(self, file_path='', iteration=None):
        """ load any existing models, if available """
        """Load models from disk using pytorch's
        inbuilt load model function (model saved in pth format)
        
        Parameters
        ----------
        file_path : str, optional
            Path where to find the file
        iteration : int, optional
            Iteration number the file is tagged with, if None, iteration is 0

        Raises
        ------
        FileNotFoundError
            The file is not loaded if not found and an error message is printed,
            this error does not affect the functioning of the program
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        

        if torch.cuda.is_available():
            self._model.load_state_dict(torch.load("{}/model_{:04d}.pth".format(file_path, iteration)))
            if(self._use_target_net):
                self._target_net.load_state_dict(torch.load("{}/model_{:04d}_target.pth".format(file_path, iteration)))
        else:
            model_state_dict = torch.load("{}/model_{:04d}.pth".format(file_path, iteration), map_location=torch.device('cpu'))
            self._model.load_state_dict(model_state_dict)
            if(self._use_target_net):
                target_net_state_dict = torch.load("{}/model_{:04d}_target.pth".format(file_path, iteration), map_location=torch.device('cpu'))
                self._target_net.load_state_dict(target_net_state_dict)

        

    def print_models(self):
        """Print the model using torch summary method"""
        summary(self._model, (2, 10, 10))


    def train_agent(self, batch_size=64, num_games=1, reward_clip=False):
        """
        Trains the agent using samples from the replay buffer.

        """
        # Sample a batch of experiences from the replay buffer.
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)

        if reward_clip:
            r = np.sign(r)
        
        # Convert the samples to PyTorch tensors and move them to the specified device.
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        next_s = torch.tensor(next_s, dtype=torch.float32).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.int8).to(self.device)
        a = torch.tensor(a, dtype=torch.float32).to(self.device)
        legal_moves = torch.tensor(legal_moves, dtype=torch.int8).to(self.device)

        # Set model to training mode
        self._model.train()

        # Use the target network for prediction if enabled, else use main model.
        target_model = self._target_net if self._use_target_net else self._model

        # Compute Q values for the next states without gradient computation.
        with torch.no_grad():  
            next_model_outputs = self._get_model_outputs(next_s, target_model)
        
        # Compute the discounted reward; using a mask for invalid moves
        inf_tensor = torch.tensor(-np.inf).to(next_model_outputs.device)
        discounted_reward = r + self._gamma * torch.max(torch.where(legal_moves == 1, next_model_outputs, inf_tensor), dim=1).values.reshape(-1, 1) * (1 - done)

        # Calculate the target Q-values
        target = self._get_model_outputs(s, target_model)
        target = (1 - a) * target + a * discounted_reward

        # Calculate predicted Q-values
        prediction = self._get_model_outputs(s)

        # Compute loss between predicted Q-values and target Q-values
        loss = self._loss_function(prediction, target)

        # Backpropagation: compute gradients and update model parameters
        self._optimizer.zero_grad() # Reset gradients to zero
        loss.backward()             # Compute gradients
        self._optimizer.step()      # Update model parameters

        return loss.item()


    def update_target_net(self):
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())
