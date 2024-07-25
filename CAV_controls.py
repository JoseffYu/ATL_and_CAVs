import numpy as np
from gym import spaces

class CAV:
    
    def __init__(self,id,tl,veh2,traci) -> None:
        self.id = id
        self.vel = 0
        self.acc = 0
        self.pos = 0
        self.tl = tl # 包含TL信息
        self.veh2 = veh2
        self.traci = traci
        
        
        self.observation_space = spaces.Dict({
            'velocity': spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32),
            'acceleration': spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32),
            'traffic light state': spaces.Text(min_length=12,seed=None),
            'front distance': spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        })
        self.action_space = spaces.Box(low=-3,high=3,dtype=np.float32)
        
    
    def getFrontDistance(self):
        return self.traci.vehicle.getLeader(self.id)
    
    
    def getTlState(self):
        return self.tl.traci.trafficlight.getRedYellowGreenState(self.tl.id) # 获取当前信号灯信息
    
    def getState(self):
        return {self.vel, self.acc, self.getState(), self.getFrontDistance()}
    
    
    def doAction(self,acc):
        vel = self.traci.vehicle.getSpeed(self.id)  # 获取当前速度，单位为 m/s

        # 设置车辆的目标速度
        target_speed = vel + acc  # 将速度增加3 m/s
        self.acc = acc

        # 将车辆速度逐渐调整为目标速度
        self.traci.vehicle.slowDown(self.id, target_speed, duration=2)
        self.vel = target_speed
        
        return
    