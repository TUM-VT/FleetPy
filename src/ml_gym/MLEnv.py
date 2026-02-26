

class MLEnv():
    def __init__(self, fleetpy_in_queue=None, fleetpy_out_queue=None):
        self.fleetpy_in_queue = fleetpy_in_queue
        self.fleetpy_out_queue = fleetpy_out_queue
    
    def run(self):
        # main loop to run the ML environment
        while True:
            # 1) get observation from fleetpy
            observation = self._observe()
            # TODO: remove print statements after debugging
            print(f"\nMLEnv: get observation -  {observation}")
            if observation == 'SIMULATION_ENDED':
                break
            action = self._compute_action(observation)
            
            # 2) trigger action
            # TODO: remove print statements after debugging
            print(f"\nMLEnv: apply action -  {action}")
            self.apply_action(action)
            
    def _compute_action(self, observation):
        # compute action based on observation (e.g. with a trained RL agent)
        return None
    
    def apply_action(self, action):
        # send action to fleetpy via queue
        self.fleetpy_in_queue.put(action)
        
    def _observe(self):
        # get observation from fleetpy via queue
        observation = self.fleetpy_out_queue.get()
        return observation
    
    def receive_observation(self, event, observation):
        # receive observation from fleetpy via method call (if not using multiprocessing)
        self.observation = observation
        
    def get_action(self):
        return self._compute_action(self.observation)
    
    
class GreedyRepositioningEnv(MLEnv):
    def __init__(self):
        super().__init__(None, None)
        
    def _compute_action(self, observation):
        # compute greedy repositioning action based on observation
        print("\nGreedyRepositioningEnv: compute greedy repositioning action")
        print(f"Observation: {observation}")
        exit()