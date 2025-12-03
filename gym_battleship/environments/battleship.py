import gymnasium
import numpy as np
from gymnasium import spaces
from copy import deepcopy
from typing import Union
from typing import Tuple
from typing import Optional
from collections import namedtuple
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
        
Ship = namedtuple('Ship', ['min_x', 'max_x', 'min_y', 'max_y'])
Action = namedtuple('Action', ['x', 'y'])

class CHANNEL_MAP(Enum):
    MISSED = 0 # 0 = not missed, 1 = missed
    HIT = 1 # 0 = not hit, 1 = hit
    LEGAL_MOVE = 2 # 0 = legal move/unknown cell, 1 = illegal move/revealed cell

def is_notebook():
    """Helper used to change the way the environment in rendered"""
    from IPython import get_ipython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        else:
            return False  # Terminal running IPython or other types
    except NameError:
        return False  # Probably standard Python interpreter


class BattleshipEnv(gymnasium.Env):
    def __init__(self,
                 board_size: Tuple = None,
                 ship_sizes: dict = None,
                 episode_steps: int = 110,
                 reward_dictionary: Optional[dict] = None):

        self.ship_sizes = ship_sizes or {5: 1, 4: 1, 3: 2, 2: 1}
        self.board_size = board_size or (10, 10)

        self.board = None  # Hidden state updated throughout the game
        self.board_generated = None  # Hidden state generated and left not updated (for debugging purposes)
        self.observation = None  # the observation is a (3, n, m) matrix
        self.NUM_CHANNELS = 3
        
        self.done = None
        self.step_count = None
        self.episode_steps = episode_steps

        reward_dictionary = {} if reward_dictionary is None else reward_dictionary
        default_reward_dictionary = reward_dictionary or {  # todo further tuning of the rewards required
            'win': 100,
            'lose': -30,
            'missed': -0.2,
            'hit': 5,
            'proximal_hit': 20,
            'repeat_missed': -20,
            'repeat_hit': -3,
        }
        
        self.reward_dictionary = {key: reward_dictionary.get(key, default_reward_dictionary[key]) for key in default_reward_dictionary.keys()}
        self.action_space = spaces.Discrete(self.board_size[0] * self.board_size[1])
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.NUM_CHANNELS, self.board_size[0], self.board_size[1]))
            # three 10x10 matrices stacked together for the convolutional neural network
    
    def get_original_board(self) -> np.ndarray:
        return self.board_generated  
    
    def _in_horizontal_bounds(self, x: int) -> bool:
        return 0 <= x < self.board_size[0]
    
    def _in_vertical_bounds(self, y: int) -> bool:
        return 0 <= y < self.board_size[1]
    
    def _check_proximal_hit(self, action: tuple[int, int]) -> bool: 
        return (
            (self._in_horizontal_bounds(action.x - 1) and self.observation[CHANNEL_MAP.HIT.value, action.x - 1, action.y] == 1) or
            (self._in_horizontal_bounds(action.x + 1) and self.observation[CHANNEL_MAP.HIT.value, action.x + 1, action.y] == 1) or
            (self._in_vertical_bounds(action.y - 1) and self.observation[CHANNEL_MAP.HIT.value, action.x, action.y - 1] == 1) or
            (self._in_vertical_bounds(action.y + 1) and self.observation[CHANNEL_MAP.HIT.value, action.x, action.y + 1] == 1)
        )
    
    def step(self, raw_action: Union[int, tuple]) -> Tuple[np.ndarray, int, bool, dict]:
        if isinstance(raw_action, int) or isinstance(raw_action, np.int64):
            assert (0 <= raw_action < self.board_size[0]*self.board_size[1]),\
                "Invalid action (The encoded action is outside of the limits)"
            # action = Action(x=raw_action % self.board_size[0], y=raw_action // self.board_size[0])
            action = Action(x=raw_action // self.board_size[1], y=raw_action % self.board_size[1])

        elif isinstance(raw_action, tuple):
            assert (0 <= raw_action[0] < self.board_size[0] and 0 <= raw_action[1] < self.board_size[1]),\
                "Invalid action (The action is outside the board)"
            action = Action(x=raw_action[0], y=raw_action[1])

        else:
            raise AssertionError("Invalid action (Unsupported raw_action type)")

        self.step_count += 1

        truncated = False

        # Check if the game is done (if true, the current step is the "last step")
        if self.step_count >= self.episode_steps:
            self.done = False
            truncated = True

        
        
        if self.board[action.x, action.y] == 1: # hit ship
            self.board[action.x, action.y] = 0
            self.observation[CHANNEL_MAP.HIT.value, action.x, action.y] = 1
            self.observation[CHANNEL_MAP.LEGAL_MOVE.value, action.x, action.y] = 1
            
            if truncated:
                return self.observation, self.reward_dictionary['lose'], self.done, truncated, {}
            # Win (No boat left)
            if not self.board.any():
                self.done = True
                return self.observation, self.reward_dictionary['win'], self.done, truncated, {}
            if self._check_proximal_hit(action):
                return self.observation, self.reward_dictionary['proximal_hit'], self.done, truncated, {}
            
            return self.observation, self.reward_dictionary['hit'], self.done, truncated, {}

        else:
            if truncated:
                return self.observation, self.reward_dictionary['lose'], self.done, truncated, {}
            
            if self.observation[CHANNEL_MAP.MISSED.value, action.x, action.y] == 1:
                return self.observation, self.reward_dictionary['repeat_missed'], self.done, truncated, {}

            # repeat cell marked as hit 
            elif self.observation[CHANNEL_MAP.HIT.value, action.x, action.y] == 1:
                return self.observation, self.reward_dictionary['repeat_hit'], self.done, truncated, {}

            # Missed (Action not repeated and boat(s) not touched)
            else:
                self.observation[CHANNEL_MAP.MISSED.value, action.x, action.y] = 1
                self.observation[CHANNEL_MAP.LEGAL_MOVE.value, action.x, action.y] = 1
                
                return self.observation, self.reward_dictionary['missed'], self.done, truncated, {}

    def reset(self, seed=None, options=None) -> np.ndarray:
        self._set_board()
        self.board_generated = deepcopy(self.board)
        self.observation = np.zeros((self.NUM_CHANNELS, *self.board_size), dtype=np.float32)
        self.step_count = 0
        self.done = False
        return self.observation, {}

    def _set_board(self) -> None:
        self.board = np.zeros(self.board_size, dtype=np.float32)
        for ship_size, ship_count in self.ship_sizes.items():
            for _ in range(ship_count):
                self._place_ship(ship_size)

    def _place_ship(self, ship_size: int) -> None:
        can_place_ship = False
        while not can_place_ship:  # todo add protection infinite loop
            ship = self._get_ship(ship_size, self.board_size)
            can_place_ship = self._is_place_empty(ship)
        self.board[ship.min_x:ship.max_x, ship.min_y:ship.max_y] = True

    @staticmethod
    def _get_ship(ship_size: int, board_size: tuple) -> Ship:
        if np.random.choice(('Horizontal', 'Vertical')) == 'Horizontal':
            min_x = np.random.randint(0, board_size[0] + 1 - ship_size)
            min_y = np.random.randint(0, board_size[1])
            return Ship(min_x=min_x, max_x=min_x + ship_size, min_y=min_y, max_y=min_y + 1)
        else:
            min_x = np.random.randint(0, board_size[0])
            min_y = np.random.randint(0, board_size[1] + 1 - ship_size)
            return Ship(min_x=min_x, max_x=min_x + 1, min_y=min_y, max_y=min_y + ship_size)

    def _is_place_empty(self, ship: Ship) -> bool:
        return np.count_nonzero(self.board[ship.min_x:ship.max_x, ship.min_y:ship.max_y]) == 0

    def render(self, mode='human'):
        board = np.empty(self.board_size, dtype=str)
        board[self.observation[CHANNEL_MAP.MISSED.value] != 0] = '⚫'
        board[self.observation[CHANNEL_MAP.HIT.value] != 0] = '❌'
        
        if mode == 'image':
            return self._render_image(board)
        
        self._render(board)


    def render_board_generated(self):
        board = np.empty(self.board_size, dtype=str)
        board[self.board_generated != 0] = '⬛'
        self._render(board)
        
    @staticmethod
    def _render_image(board, symbol='⬜'):
        num_rows, num_columns = board.shape
        fig, ax = plt.subplots(figsize=(8, 8))
        color_map = {'⬜': 'lightgray', '⚫': 'blue', '❌': 'red', '⬛': 'black'}
        
        for i in range(num_rows):
            for j in range(num_columns):
                cell_value = board[i, j] if board[i, j] else symbol
                color = color_map.get(cell_value, 'white')
                rect = mpatches.Rectangle((j, num_rows - i - 1), 1, 1, 
                                        linewidth=1, edgecolor='black', 
                                        facecolor=color)
                ax.add_patch(rect)
                
                ax.text(j + 0.5, num_rows - i - 0.5, cell_value,
                    ha='center', va='center', fontsize=16)
        
        ax.set_xlim(0, num_columns)
        ax.set_ylim(0, num_rows)
        ax.set_aspect('equal')
        
        ax.set_xticks(np.arange(num_columns) + 0.5)
        ax.set_yticks(np.arange(num_rows) + 0.5)
        ax.set_xticklabels([chr(i) for i in range(ord('A'), ord('A') + num_columns)])
        ax.set_yticklabels(range(num_rows, 0, -1))
        
        ax.tick_params(length=0)
        ax.set_title('Battleship Board')
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, :3]
        
        plt.close(fig)
        return image

    @staticmethod
    def _render(board, symbol='⬜'):
        import pandas as pd

        num_rows, num_columns = board.shape
        columns = [chr(i) for i in range(ord('A'), ord('A') + num_columns)]
        index = [i + 1 for i in range(num_rows)]

        dataframe = pd.DataFrame(board, columns=columns, index=index)
        dataframe = dataframe.replace([''], symbol)

        if is_notebook():
            from IPython.display import display
            display(dataframe)
        else:
            print(dataframe, end='\n')

        # todo maybe put the board generated on the right side
        #  https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
