
# randomly makes moves until it wins the game

import random

class RandomAgent:
    def __init__(self, skip_invalid_actions: bool = True):
        self.skip_invalid_actions = skip_invalid_actions
        self.taken_actions = set()
        
    def predict(self) -> int: 
        action = random.randint(0, 99)
        while self.skip_invalid_actions and action in self.taken_actions: 
            action = random.randint(0, 99)
        
        self.taken_actions.add(action)
        
        return action, None



