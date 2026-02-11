

class MLEnv():
    def __init__(self, fleetpy_in_queue, fleetpy_out_queue):
        self.fleetpy_in_queue = fleetpy_in_queue
        self.fleetpy_out_queue = fleetpy_out_queue
    
    def run(self):
        # main loop to run the ML environment
        while True:
            observation = self._observe()
            print(f"MLEnv: get observation {observation}")
            action = self._compute_action(observation)
            
            # 2) trigger action
            print(f"MLEnv: apply action {action}")
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