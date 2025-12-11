
import random
import numpy as np
from gym_battleship.environments.battleship import BattleshipEnv, CHANNEL_MAP
import logging
import sys

class HeuristicAgent:
    def __init__(self, env: BattleshipEnv, skip_invalid_actions: bool = True, verbose=False):
        self.env = env
        self.skip_invalid_actions = skip_invalid_actions
        self.verbose = verbose
        self.taken_actions = set()
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
        self.logger.handlers.clear()
        
        if verbose:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.prev_obs = None
        self.curr_obs = np.zeros((self.env.unwrapped.NUM_CHANNELS, *self.env.unwrapped.board_size), dtype=np.float32)
        
        self.last_action = None
        self.last_hit = None
        self.current_hunt = None
        self.num_steps = 0
            # {
            #     origin: int, the original hit action that started the hunt
            #     direction: "(-1, 0) | (1, 0) | (0, -1) | (0, 1) | None", the direction we determine the ship to be
            #     directions_to_try: [(-1, 0) | (1, 0) | (0, -1) | (0, 1)] list of actions we need to try to determine the direction
            #     end1_reached: bool, whether we've reached one end of the ship in the direction we've discovered
            #     end2_reached: bool , whether we've reached the other end of the ship
            # }
        
        
    def _get_random_valid_action(self) -> int:
        valid_actions = [ i for i in range(100) if i not in self.taken_actions ]
        action = random.choice(valid_actions)
        self.taken_actions.add(action)
        
        return action, None
    
    def _last_action_hit(self):
        return self.curr_obs[CHANNEL_MAP.HIT.value, self.last_action // 10, self.last_action % 10] == 1
        
    def _valid_action(self, x: int, y: int) -> bool: 
        return 0 <= x < 10 and 0 <= y < 10 and (x * 10 + y) not in self.taken_actions
        
    def _determine_ship_direction(self, hunt: dict, last_hit_action: int):
        origin_x, origin_y = hunt['origin'] // 10, hunt['origin'] % 10
        last_x, last_y = last_hit_action // 10, last_hit_action % 10
        
        if last_x < origin_x:
            return (-1, 0)
        elif last_x > origin_x:
            return (1, 0)
        elif last_y < origin_y:
            return (0, -1)
        elif last_y > origin_y:
            return (0, 1)
        else:
            return None
        
    def _try_direction(self):
        if not self.current_hunt['directions_to_try']:
            return self._get_random_valid_action()
        
        direction = self.current_hunt['directions_to_try'].pop()
        last_x, last_y = self.last_hit // 10, self.last_hit % 10
        next_x, next_y = last_x + direction[0], last_y + direction[1]
        
        while not self._valid_action(next_x, next_y):
            if not self.current_hunt['directions_to_try']:
                return self._get_random_valid_action()
            direction = self.current_hunt['directions_to_try'].pop()
            next_x, next_y = last_x + direction[0], last_y + direction[1]
        
        action = next_x * 10 + next_y
        self.taken_actions.add(action)
        self.last_action = action
        return action, None
        
        
        
    def predict(self, obs) -> int:
        self.prev_obs = self.curr_obs
        self.curr_obs = obs
        self.num_steps += 1
        
        if self.last_action is None: # edge case for first move
            action, _ = self._get_random_valid_action()
            self.last_action = action
            return action, None
        
        if self._last_action_hit():
            self.logger.debug(f"Last action was a hit: {self.last_action}")
            self.last_hit = self.last_action
            if not self.current_hunt:
                self.current_hunt = {
                    "origin": self.last_hit,
                    "direction": None,
                    "directions_to_try": [(-1, 0), (1, 0), (0, -1), (0, 1)], # left, right, up, down
                    "end1_reached": False,
                    "end2_reached": False
                }
                self.logger.debug(f"Started hunt: {self.current_hunt}")
            else: 
                self.current_hunt['direction'] = self._determine_ship_direction(self.current_hunt, self.last_hit)
                
        
        if self.current_hunt:
            if self.current_hunt['direction'] is None: # determine the direction
                action, _ = self._try_direction()
                self.logger.debug(f"Current hunt: {self.current_hunt}")
                self.logger.debug(f"Choosing random action to determine direction: {action}")
                return action, None
            else: # we've determined the direction
                direction = self.current_hunt['direction']
                self.logger.debug(f"Direction for current hunt is: {direction}")
                last_x, last_y = self.last_hit // 10, self.last_hit % 10
                next_x, next_y = last_x + direction[0], last_y + direction[1]
                action = next_x * 10 + next_y
                
                if self._valid_action(next_x, next_y) and action not in self.taken_actions:
                    self.taken_actions.add(action)
                    self.last_action = action
                    self.logger.debug(f"Choosing action: {action}")
                    return action, None
                else:
                    if not self.current_hunt['end1_reached']:
                        self.current_hunt['end1_reached'] = True
                        self.logger.debug("End 1 reached for current hunt.")  
                        # try the opposite direction from origin
                        origin_x, origin_y = self.current_hunt['origin'] // 10, self.current_hunt['origin'] % 10
                        opp_x, opp_y = origin_x - direction[0], origin_y - direction[1]
                        if self._valid_action(opp_x, opp_y) and (opp_x * 10 + opp_y) not in self.taken_actions:
                            action = opp_x * 10 + opp_y
                            self.taken_actions.add(action)
                            self.last_action = action
                            self.logger.debug(f"Trying action in opposite direction: {action}")
                            return action, None
                        else:
                            self.current_hunt['end2_reached'] = True
                            self.logger.debug("End 2 reached for current hunt.")
                            self.current_hunt = None
                            action, _ = self._get_random_valid_action()
                            self.last_action = action
                            self.logger.debug(f"No valid moves left in current hunt, choosing random action: {action}")
                            return action, None
                    else:
                        self.current_hunt['end2_reached'] = True
                        self.current_hunt = None
                        self.logger.debug("End 2 reached for current hunt.")
                        action, _ = self._get_random_valid_action()
                        self.last_action = action
                        self.logger.debug(f"No valid moves left in current hunt, choosing random action: {action}")
                        return action, None 
        
        self.logger.debug("No current hunt, choosing random valid action...")
        self.current_hunt = None
        action, _ = self._get_random_valid_action()
        self.last_action = action
        return action, None

               
            





