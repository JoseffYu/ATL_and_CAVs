import traci
import math

'''
    All methods in TraCI are tested here before applying in the Adaptive Traffic Light control system
'''

# Connect to SUMO
sumo_cfg = "/Users/yuyanlin/Desktop/AdaptiveTrafficLightCAVsControl/SUMO simulation/vehicleControlNet.sumocfg"
traci.start(["sumo-gui", "-b","0", "-e","999999", "-c", sumo_cfg], numRetries=20,verbose = True)

# get Scenario Information

#lanes = traci.lane.getIDList()

num_vehicle = traci.vehicle.getIDCount()
traci.vehicle.add('v_1',routeID="default",depart="now",typeID="DEFAULT_VEHTYPE")
traci.vehicle.add('v_2',routeID="r_1",depart="now",typeID="DEFAULT_VEHTYPE")

traci.route

rou_ids = traci.route.getIDList()
print(rou_ids)

# begin simulation
step = 0
while step <= 1000:
    
    vehicle_ids = traci.vehicle.getIDList()
    traffic_light_id = traci.trafficlight.getIDList()
    for vid in vehicle_ids:
        # 获取车辆位置信息
        veh_pos = traci.vehicle.getPosition(vid)

        # 获取车辆前方车辆信息（例如获取前方第一个车辆的位置）
        front_vehicle_id = traci.vehicle.getLeader(vid)
        current_speed = 0.5
        behind_vehicle_id = traci.vehicle.getFollower(vid)
        if front_vehicle_id:
            front_veh_pos = traci.vehicle.getPosition(front_vehicle_id[0])

            # 计算当前车辆与前车的距离（这里假设计算纵向距离）
            distance_to_front_vehicle_ver = front_veh_pos[1] - veh_pos[1]
            distance_to_front_vehicle_hor = front_veh_pos[0] - veh_pos[0]
            
            realtive_distance = math.sqrt(abs(math.pow(distance_to_front_vehicle_ver,2)-math.pow(distance_to_front_vehicle_hor,2)))
            
            current_speed = traci.vehicle.getSpeed(vid)
            if realtive_distance <= 2:
                traci.vehicle.slowDown(vid, current_speed, 1)
            else:
                traci.vehicle.slowDown(vid, current_speed+10, 5)
    
    
    traci.simulationStep()
    step += 1
traci.close()


class PID:
    
    def __init__(self):
        self.Kd = 0.8
        self.Kd = 0.2
        self.current_value = 0.0
        self.previous_error = 0.0
        self.set_point = 0 #前车车尾位置
    
    def update(self, current_value): 
            # calculate P_term and D_term
            error = self.set_point - current_value
            P_term = self.Kp * error
            D_term = self.Kd * (error - self.previous_error)
            self.previous_error = error 
            return P_term + D_term #更新出来的加速度

    def setPoint(self, set_point):
        self.set_point = set_point
        self.previous_error = 0

    def setPD(self, P=0.0, D=0.0):
        self.Kp = P
        self.Kd = D
