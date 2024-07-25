
import math
import numpy as np
from statistics import mean
from gym import spaces

'''
    Reinforcement learning environment for traffic light agent, this is environment 1
'''

class TrafficLight:
    def __init__(
        self,
        tls_id: str,
        simulation_time: int,
        traci
    ):
        self.conn = traci
        self.tls_id = tls_id
        self.update_time = True
        self.green_phase = 0 # first action
        self.yellow_phase = None
        self.end_time = 0
        self.all_phases = []
        #lanes id starts from top-left one, clockwise
        self.lanes_id = ["E3_0", "E3_1", "E3_2","-E2_0", "-E2_1", "-E2_2", "-E4_0", "-E4_1", "-E4_2", "E1_0", "E1_1", "E1_2"]
        self.dict_action_wait_num = []
        self.correct_action = 0
        self.dict_lane_veh = None
        self.simulation_time = simulation_time
        self.total_reward = 0
        self.lanes_density = []
        self.duration = 0
        self.max_indices = []

        self.all_green_phases = []
        for index in range(len(self.all_phases)):
            for phase in self.all_phases[index] :
                if 'G' in phase.state:
                    self.all_green_phases.append(phase)

        self.observation_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Discrete(1) #link to computeUpdate


    #action is the order of phase 'gyr'
    ##TODO: Change the way of doing action
    def doAction(self,action):
        self.max_indices = self.getMaxCongestIndices()
        if action == 1:
            phase = self.conn.trafficlight.getRedYellowGreenState(self.tls_id[0])
            
            #Change previous green state to red
            for state in self.max_indices:
                phase = ''.join(['r' if i != state and i % 3 != 0 else char for i, char in enumerate(phase)])
            print(phase)
            for state in self.max_indices:
                if state % 3 == 0:
                    new_phase = ''.join(['g' if i == state else char for i, char in enumerate(phase)])
                new_phase = ''.join(['G' if i == state else char for i, char in enumerate(phase)])
            self.conn.trafficlight.setRedYellowGreenState(self.tls_id[0],new_phase)
        return action
            
    ##TODO: Change a reward function
    def computeReward(self, action):
        arrive_vehicles = self.conn.simulation.getArrivedNumber()
        self.last_arrived_vehicles = self.conn.simulation.getArrivedNumber()
        rewards = arrive_vehicles + action
        self.total_reward += rewards
        return rewards
   
    '''
    ## If a traffic lane have max congestion and its traffic phase is red, then turn it to green, the lane may result in confliction should be turned to red. 
    ## Whether current green phases lane have the max occupancy, if 'yes' then no need to update; it no update it.
    ## Replaced by neural network
    def computeUpdate(self):
        self.lanes_density = self.getObservation()
        max_congest = max(self.lanes_density)
        self.max_indices = [index for index, value in enumerate(self.lanes_density) if value == max_congest]
        
        #Check the traffic state''
        state = [value for i, value in enumerate(self.conn.trafficlight.getRedYellowGreenState(self.tls_id[0])) if i in self.max_indices]
        if all(state) == 'G' :
            return False
        else:
            return True
    '''

    
    ##TODO: can be deleted, dont care about the phase duaration, but change the phase directly    
    def calculatePhaseDuration(self):
        return True

    
    def computeNextState(self):
        observation = self.getObservation()
        next_state = np.array(observation, dtype=np.float32)
        return next_state


    def computeState(self):
        observation = self.getObservation()
        state = np.array(observation, dtype=np.float32)
        return state
    
    
    def getMaxCongestIndices(self):
        
        self.lanes_density = self.getObservation()
        sorted_indices = sorted(enumerate(self.lanes_density), key=lambda x: x[1], reverse=True)
        top_6_indices = [index for index, value in sorted_indices[:6]]
        max_congest = max(self.lanes_density)
        max_indices = [index for index, value in enumerate(self.lanes_density) if value == max_congest]
        print(self.lanes_density)
        return top_6_indices
        

    def getObservation(self):
        obs = []
        lanes_density = {}
        for i,lane_id in enumerate(self.lanes_id):
            obs.append(self.conn.lane.getLastStepOccupancy(lane_id))
            lanes_density[lane_id] = self.conn.lane.getLastStepOccupancy(lane_id)
        self.lanes_density = lanes_density
        return obs
