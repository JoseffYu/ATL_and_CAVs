
'''
Here is the main class for the whole model. hyperparameters are set here
'''
import numpy as np
import math
from statistics import mean
from SUMO_TL import SumoEnv
from DQN_PER_ReplayBuffer import DQN, ReplayBuffer_Per
import matplotlib.pyplot as plt

device = "cpu"
sumo_cfg = "/Users/yuyanlin/Desktop/AdaptiveTrafficLightCAVsControl/SUMO simulation/oneJuncScen.sumocfg"

all_total_reward = []
num_vehicles_pass = []
average_reward = []
average_vehicle_passed = []

def main():
    
    env = SumoEnv(sumo_cfg_file=sumo_cfg,
                  simulation_time=500
                  )
    input_dim = env.observation_space().shape[0]
    output_dim = env.action_space().n
    agent = DQN(env = env,mode="train",input_dim=input_dim,output_dim=output_dim,gamma=0.9,replay_size=20000,batch_size=256,eps_start=0.95,eps_end=0.05,eps_decay=30000,LR=1e-2,TAU=0.9)
    replay_buffer = ReplayBuffer_Per(agent.replay_size)
    
    for episode in range(200):
        
        total_reward = 0
        total_arrived_vehicle = 0
        env.reset()
        state = env.computeState()
        done = False
        
        while not done:
            
            # for a different environment
            '''
            if env.traffic_light.update_time:
                action = agent.selectAction(state, agent.learn_step_counter, invalid_action)
                state,next_state, reward, done, info = env.step(env.tl_id,action)
                total_reward += reward
            else:
                env.only_step()
                env.traffic_light.update_time = True
            '''
            for i in range (500):
                if i % 15 == 0:
                    action = agent.selectAction(state, agent.learn_step_counter)
                    state,next_state, reward, done, info = env.step(env.tl_id,action)
                    total_reward += reward
                    
                    total_arrived_vehicle += env.traffic_light.conn.simulation.getArrivedNumber()
                    
                    replay_buffer.push(state, next_state, reward, info['do_action'], done)
                    agent.memory = replay_buffer
                    state = next_state
                    
                    agent.learn()
                else:
                    total_reward += reward
                    total_arrived_vehicle += env.traffic_light.conn.simulation.getArrivedNumber()
                    state,next_state, reward, done, info = env.step(env.tl_id,0) #not change states
                    
            
            for param in agent.policy_net_state_dict:
                agent.target_net_state_dict[param] = agent.policy_net_state_dict[param]*agent.TAU + agent.target_net_state_dict[param]*(1-agent.TAU)
            agent.target_net.load_state_dict(agent.target_net_state_dict)
            agent.TAU = 0.05 + (agent.TAU - 0.05) * math.exp(-1. * agent.learn_step_counter / agent.eps_decay)
        all_total_reward.append(total_reward)
        
        env.close()
        
        print('episode_return: ',total_reward)
        print('arrived_vehicles: ',total_arrived_vehicle)
        num_vehicles_pass.append(total_arrived_vehicle)
        average_vehicle_passed.append(mean(num_vehicles_pass))
        print('i_episode:', episode)
        print('TAU:', agent.TAU)
        
    print(mean(num_vehicles_pass))

    p1 = plt
    p1.subplots()
    p1.plot(all_total_reward)
    p1.title('Returns for DDQN')
    p1.xlabel('Episodes')
    p1.ylabel('returns')
    p1.grid()
    p1.savefig('/Users/yuyanlin/Desktop/AdaptiveTrafficLightCAVsControl//Experiments/Returns_2.png')
    
    p2 = plt
    p2.subplots()
    x = np.arange(len(num_vehicles_pass))
    y = np.ones_like(x) * 300
    p2.plot(x,y)
    p2.plot(num_vehicles_pass)
    p2.plot(average_vehicle_passed)
    p2.title("Vehicles passed over time")
    p2.xlabel("Episode")
    p2.ylabel("number of vehicles arrived destination")
    p2.legend(['base line','number of vehicle pass','average vehicles pass'])
    p2.grid()
    p2.savefig('/Users/yuyanlin/Desktop/AdaptiveTrafficLightCAVsControl//Experiments/arrived_vehicles_2.png')
    
    p3 = plt
    p3.subplots()
    p3.plot(agent.losses)
    p3.title("Training Loss Over Time")
    p3.xlabel("Learn_steps")
    p3.ylabel("Loss")
    p3.grid()
    p3.savefig('/Users/yuyanlin/Desktop/AdaptiveTrafficLightCAVsControl//Experiments/loss_2.png')

    p4 = plt
    p4.subplots()
    p4.plot(agent.expected_values)
    p4.title("Expected Values Over Time")
    p4.xlabel("Episode")
    p4.ylabel("Expected Value")
    p4.grid()
    p4.savefig('/Users/yuyanlin/Desktop/AdaptiveTrafficLightCAVsControl//Experiments/expected_q_2.png')

print(main())
