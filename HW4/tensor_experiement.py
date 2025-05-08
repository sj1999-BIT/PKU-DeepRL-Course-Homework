import torch

if __name__ == "__main__":
    np = [[0,0,2,0,1,2], [0,1,2,0,1,2], [1,0,2, 0,1,2], [1,1,2, 0,1,2]]

    tensor = torch.tensor(np, dtype=torch.float32)

    tensor = tensor.reshape(4, 3, 2)


    # Now you can iterate through the second dimension to pass to different agents
    for agent_idx in range(3):
        # Extract tensor for current agent (shape will be 25, 18)
        agent_tensor = tensor[:, agent_idx, :]

        for time_step in range(4):
            cur_tensor = agent_tensor[time_step, :]

            # Pass to the respective agent
            # agent_function(agent_tensor)  # Uncomment and replace with your agent function
            print(f"Passing tensor {cur_tensor} to agent {agent_idx} at time_step {time_step}")

    print(tensor)