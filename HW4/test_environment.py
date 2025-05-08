from pettingzoo.mpe import simple_spread_v3
from dataKeys import AGENT_ZERO,AGENT_ONE, AGENT_TWO
env = simple_spread_v3.parallel_env(render_mode='human')

from neural_networks import AgentNetwork

env.reset()
action = 0

agent_dict = {
    AGENT_ZERO: AgentNetwork(0),
    AGENT_ONE: AgentNetwork(1),
    AGENT_TWO: AgentNetwork(2)
}

prev_action_dict = {
    AGENT_ZERO: 0,
    AGENT_ONE: 0,
    AGENT_TWO: 0
}

actions_dict = {
    AGENT_ZERO: 0,
    AGENT_ONE: 0,
    AGENT_TWO: 0
}




cur_agent_id = 0
prev_action_arr = [0 for _ in range(env.num_agents)]
observation_arr = [0 for _ in range(env.num_agents)]

"""
during collection, we need to collect 
1. observation for all three agents
2. total reward at current step
3. Q_total generated from current model
4. next 
"""


if __name__ == "__main__":

    cur_state = []

    # observations is a dictionary og agent_id to their observations
    observation_dict, infos = env.reset()



    while env.agents:
        # this is where you would insert your policy
        # actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        cur_state.append(np.concatenate((array1, array2, array3)))

        for agent_id in actions_dict.keys():
            cur_action, cur_Q_value = agent_dict[agent_id].select_action(observation_dict[agent_id],
                                                                         [prev_action_dict[agent_id], ])



            actions_dict[agent_id] = cur_action


        prev_action_dict = actions_dict

        observations, rewards, terminations, truncations, infos = env.step(actions_dict)
    env.close()

    # # important, agent_iter() indicates what the env.last() returns with respect to which agent
    # for agent in env.agent_iter():
    #     # always return correspond to the agent in env.agent_iter()
    #     observation, reward, termination, truncation, info = env.last()
    #
    #     agent = agent_NN_arr[cur_agent_id]
    #     prev_action = prev_action_arr[cur_agent_id]
    #
    #     # one cycle is done, can pass it to the mixNetwork
    #     if agent == 0:
    #         pass
    #
    #     # record the observation as inputs for mix network
    #     observation_arr[cur_agent_id] = observation
    #
    #
    #
    #
    #
    #
    #     # if prev_action is not None and prev_agent is not None:
    #     #     print(f"agent {prev_agent} picked action {prev_action} and obtained reward {reward}")
    #
    #     # observation is a 18 dim np array, action
    #     if termination or truncation:
    #         action = None
    #     else:
    #         action, Q_value = agent.select_action(observation, [prev_action,]) # this is where you would insert your policy
    #         prev_action_arr[cur_agent_id] = action
    #
    #     # update to next agent
    #     cur_agent_id += 1
    #     if cur_agent_id > env.num_agents:
    #         cur_agent_id = 0
    #
    #     env.step(action)
    # env.close()