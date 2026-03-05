import random    
import json
from src.ml_gym.MLEnvs.MLEnv import MLEnv

class ObserveFleetStateEnv(MLEnv):
    def __init__(self, write_to_file=False, output_f=None):
        # this class is for generating training data for chenhao's offer-learning model 
        # it observes the fleet state after receiving status update and writes to file, without taking any action (for training data generation only)
        # another class has to be implemented for applying the trained model
        super().__init__(None, None)
        self.output_f = output_f
        self.write_to_file = write_to_file
        
    def _compute_action(self, observation):
        # only observe and write to file, no action -> for training
        if not observation.get("vehicles"):  # interval gating returns {}
            return None
        if self.write_to_file:
            with open(self.output_f, 'a') as f:
                f.write(json.dumps(observation, ensure_ascii=False, default=str) + '\n')
        return None