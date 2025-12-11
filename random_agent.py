
# randomly makes moves until it wins the game

import random

class RandomAgent:
    def __init__(self, skip_invalid_actions: bool = True):
        self.skip_invalid_actions = skip_invalid_actions
        self.taken_actions = set()
        
    def predict(self, obs) -> int: 
        legal_moves = [ i for i in range(100) if i not in self.taken_actions ]
        if self.skip_invalid_actions:
            action = random.choice(legal_moves)
        else:
            action = random.randint(0, 99)
        
        self.taken_actions.add(action)
        
        return action, None



