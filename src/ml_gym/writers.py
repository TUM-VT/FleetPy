from src.ml_gym.actors import AbstractActor
import json


class JSONWriter(AbstractActor):
    def __init__(self, output_f):
        # this class is for generating training data for chenhao's offer-learning model
        # it observes the fleet state after receiving status update and writes to file, without taking any action (for training data generation only)
        # another class has to be implemented for applying the trained model
        self.output_f = output_f

    def compute_action(self, observation):
        with open(self.output_f, 'a') as f:
            f.write(json.dumps(observation, ensure_ascii=False, default=str) + '\n')
        return None