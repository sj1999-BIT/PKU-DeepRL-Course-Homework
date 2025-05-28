import math
import numpy as np
from pettingzoo.mpe import simple_spread_v3
from dataKeys import AGENT_ZERO, AGENT_ONE, AGENT_TWO
from random import random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import PIL.Image as Image
import PIL.ImageTk as ImageTk

# Constants
COLLISION_THRESHOLD = 0.6
OVERLAP_THRESHOLD = 0.2


def get_dist(x, y):
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))


def break_local_observation_to_arr(local_observation_arr):
    """
    [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]
    :param local_observation_arr:
    :return:
    """
    dict = {}
    dict['cur_vel'] = local_observation_arr[:2]
    dict['cur_pos'] = local_observation_arr[2:4]
    dict['landmark_rel_positions'] = local_observation_arr[4:10]
    dict['other_agent_rel_positions'] = local_observation_arr[10:14]
    dict['communication'] = local_observation_arr[14:]
    return dict


def get_agent_dist_punish(observation_dict: dict):
    """
    Calculate punishment for agents being too close to each other
    """
    agent_pos = []
    for agent_id, local_observation in observation_dict.items():
        agent_pos.append(local_observation[2:4])
    agent_dist = []
    for i in range(3):
        for j in range(i + 1, 3):
            agent_dist.append(get_dist(agent_pos[i][0] - agent_pos[j][0], agent_pos[i][1] - agent_pos[j][1]))
    punish = 0
    print(f"agent dist {agent_dist}")
    for cur_dist in agent_dist:
        if cur_dist >= 1:
            continue
        punish += (1 - cur_dist) / 3
        if cur_dist < COLLISION_THRESHOLD:
            print(f"punish for collision at dist {cur_dist}")
            punish += 1
    return punish


def get_landmark_dist_arr(landmark_rel_pos_arr):
    result = []
    for i in range(3):
        x_index = i * 2
        cur_landmark_rel_pos = landmark_rel_pos_arr[x_index:x_index + 2]
        result.append(get_dist(cur_landmark_rel_pos[0], cur_landmark_rel_pos[1]))
    return result


def get_landmark_dist_reward(observation_dict: dict):
    dist_mat = []
    for agent_id, local_observation in observation_dict.items():
        dist_mat.append(get_landmark_dist_arr(local_observation[4:10]))
    dist_mat = np.array(dist_mat)
    reward = 0
    mini_dist_arr = []
    for i in range(3):
        cur_landmark_dist = dist_mat[:, i]
        mini_dist = np.min(cur_landmark_dist)
        mini_dist_arr.append(mini_dist)
        if mini_dist >= 1:
            continue
        reward += (1 - mini_dist) / 3
        if mini_dist < OVERLAP_THRESHOLD:
            print(f"reward for overlap at dist {mini_dist}")
            reward += 1
    return reward


def calculate_reward(observation_dict: dict):
    return get_landmark_dist_reward(observation_dict) - get_agent_dist_punish(observation_dict)


def generate_action_dict():
    return {
        AGENT_ZERO: int(random() * 5),
        AGENT_ONE: int(random() * 5),
        AGENT_TWO: int(random() * 5),
    }


def extract_landmark_distances(observation_dict):
    """Extract distances between agents and landmarks"""
    distances = {}
    for agent_id, obs in observation_dict.items():
        landmark_rel_positions = obs[4:10]
        agent_distances = []
        for i in range(3):
            x_index = i * 2
            landmark_rel_pos = landmark_rel_positions[x_index:x_index + 2]
            distance = get_dist(landmark_rel_pos[0], landmark_rel_pos[1])
            agent_distances.append(distance)
        distances[agent_id] = agent_distances
    return distances


if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    root.title("Multi-Agent Environment")
    root.geometry("1200x600")

    # Create a split window
    paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned_window.pack(fill=tk.BOTH, expand=True)

    # Create a frame for the rendered environment
    env_frame = ttk.Frame(paned_window)
    paned_window.add(env_frame, weight=1)

    # Create a frame for the rewards and other info
    info_frame = ttk.Frame(paned_window)
    paned_window.add(info_frame, weight=1)

    # Add a notebook for multiple plots
    notebook = ttk.Notebook(info_frame)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Create frames for different plots
    rewards_frame = ttk.Frame(notebook)
    distances_frame = ttk.Frame(notebook)
    stats_frame = ttk.Frame(notebook)

    notebook.add(rewards_frame, text="Rewards")
    notebook.add(distances_frame, text="Distances")
    notebook.add(stats_frame, text="Statistics")

    # Environment display label
    env_display = ttk.Label(env_frame)
    env_display.pack(fill=tk.BOTH, expand=True)

    # Create a label to display current reward
    current_reward_var = tk.StringVar(value="Current Reward: 0.0")
    current_reward_label = ttk.Label(env_frame, textvariable=current_reward_var, font=("Arial", 12))
    current_reward_label.pack(pady=5)

    # Create the rewards plot
    rewards_fig = Figure(figsize=(6, 4))
    rewards_ax = rewards_fig.add_subplot(111)
    rewards_ax.set_title("Rewards Over Time")
    rewards_ax.set_xlabel("Step")
    rewards_ax.set_ylabel("Reward")
    rewards_ax.grid(True)
    rewards_canvas = FigureCanvasTkAgg(rewards_fig, master=rewards_frame)
    rewards_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Create the distances plot
    distances_fig = Figure(figsize=(6, 4))
    distances_ax = distances_fig.add_subplot(111)
    distances_ax.set_title("Agent-Landmark Distances")
    distances_ax.set_xlabel("Step")
    distances_ax.set_ylabel("Distance")
    distances_ax.grid(True)
    distances_canvas = FigureCanvasTkAgg(distances_fig, master=distances_frame)
    distances_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Create the statistics plot
    stats_fig = Figure(figsize=(6, 4))
    stats_ax = stats_fig.add_subplot(111)
    stats_ax.set_title("Agent-Agent Distances")
    stats_ax.set_xlabel("Step")
    stats_ax.set_ylabel("Distance")
    stats_ax.grid(True)
    stats_canvas = FigureCanvasTkAgg(stats_fig, master=stats_frame)
    stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Initialize the environment with render mode
    env = simple_spread_v3.parallel_env(render_mode='rgb_array')

    # Store data for plotting
    all_rewards = []
    agent_landmark_distances = {
        agent_id: {landmark_id: [] for landmark_id in range(3)}
        for agent_id in [AGENT_ZERO, AGENT_ONE, AGENT_TWO]
    }
    agent_agent_distances = {
        (AGENT_ZERO, AGENT_ONE): [],
        (AGENT_ZERO, AGENT_TWO): [],
        (AGENT_ONE, AGENT_TWO): []
    }

    step_count = 0
    max_steps = 100  # Set a maximum number of steps

    # Agent and landmark colors
    agent_colors = {
        AGENT_ZERO: 'red',
        AGENT_ONE: 'blue',
        AGENT_TWO: 'green'
    }


    # Function to update the rewards plot
    def update_rewards_plot():
        rewards_ax.clear()
        rewards_ax.plot(all_rewards, 'b-')
        rewards_ax.set_title("Rewards Over Time")
        rewards_ax.set_xlabel("Step")
        rewards_ax.set_ylabel("Reward")
        rewards_ax.grid(True)
        rewards_canvas.draw()


    # Function to update the distances plot
    def update_distances_plot():
        distances_ax.clear()

        # Plot distances for each agent-landmark pair
        for agent_id in [AGENT_ZERO, AGENT_ONE, AGENT_TWO]:
            for landmark_id in range(3):
                distances = agent_landmark_distances[agent_id][landmark_id]
                distances_ax.plot(distances,
                                  label=f"Agent {agent_id} - Landmark {landmark_id}",
                                  color=agent_colors[agent_id],
                                  linestyle=['-', '--', '-.'][landmark_id])

        distances_ax.legend()
        distances_ax.set_title("Agent-Landmark Distances")
        distances_ax.set_xlabel("Step")
        distances_ax.set_ylabel("Distance")
        distances_ax.grid(True)
        distances_canvas.draw()


    # Function to update the statistics plot
    def update_stats_plot():
        stats_ax.clear()

        # Plot distances between agents
        for (agent1, agent2), distances in agent_agent_distances.items():
            stats_ax.plot(distances,
                          label=f"Agent {agent1} - Agent {agent2}",
                          linestyle='-')

        stats_ax.legend()
        stats_ax.set_title("Agent-Agent Distances")
        stats_ax.set_xlabel("Step")
        stats_ax.set_ylabel("Distance")
        stats_ax.grid(True)
        stats_canvas.draw()


    # Function to calculate distances between agents
    def calculate_agent_distances(observation_dict):
        agent_pos = {}
        for agent_id, obs in observation_dict.items():
            agent_pos[agent_id] = obs[2:4]

        # Calculate distances between agents
        for i, agent1 in enumerate([AGENT_ZERO, AGENT_ONE, AGENT_TWO]):
            for j, agent2 in enumerate([AGENT_ZERO, AGENT_ONE, AGENT_TWO]):
                if i < j:  # To avoid duplicates
                    dist = get_dist(
                        agent_pos[agent1][0] - agent_pos[agent2][0],
                        agent_pos[agent1][1] - agent_pos[agent2][1]
                    )
                    agent_agent_distances[(agent1, agent2)].append(dist)


    # Function to update landmark distances
    def update_landmark_distances(observation_dict):
        for agent_id, obs in observation_dict.items():
            landmark_rel_positions = obs[4:10]
            for i in range(3):
                x_index = i * 2
                landmark_rel_pos = landmark_rel_positions[x_index:x_index + 2]
                distance = get_dist(landmark_rel_pos[0], landmark_rel_pos[1])
                agent_landmark_distances[agent_id][i].append(distance)


    # Create controls
    controls_frame = ttk.Frame(env_frame)
    controls_frame.pack(pady=10)

    # Create pause/resume button
    paused = False


    def toggle_pause():
        nonlocal paused
        paused = not paused
        if paused:
            pause_button.config(text="Resume")
        else:
            pause_button.config(text="Pause")
            run_step()


    pause_button = ttk.Button(controls_frame, text="Pause", command=toggle_pause)
    pause_button.pack(side=tk.LEFT, padx=5)


    # Create reset button
    def reset_simulation():
        nonlocal step_count, all_rewards, agent_landmark_distances, agent_agent_distances

        # Reset counters and data
        step_count = 0
        all_rewards = []
        agent_landmark_distances = {
            agent_id: {landmark_id: [] for landmark_id in range(3)}
            for agent_id in [AGENT_ZERO, AGENT_ONE, AGENT_TWO]
        }
        agent_agent_distances = {
            (AGENT_ZERO, AGENT_ONE): [],
            (AGENT_ZERO, AGENT_TWO): [],
            (AGENT_ONE, AGENT_TWO): []
        }

        # Reset environment
        observations, infos = env.reset()

        # Get initial frame
        initial_frame = env.render()
        initial_image = Image.fromarray(initial_frame)
        initial_photo = ImageTk.PhotoImage(initial_image)
        env_display.config(image=initial_photo)
        env_display.image = initial_photo

        # Update plots
        update_rewards_plot()
        update_distances_plot()
        update_stats_plot()

        # Resume simulation if paused
        if paused:
            toggle_pause()


    reset_button = ttk.Button(controls_frame, text="Reset", command=reset_simulation)
    reset_button.pack(side=tk.LEFT, padx=5)

    # Create speed control
    speed_var = tk.DoubleVar(value=10.0)


    def update_speed(val):
        # Speed is inversely proportional to delay (higher speed = lower delay)
        pass  # We'll use speed_var.get() directly in run_step


    ttk.Label(controls_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
    speed_slider = ttk.Scale(controls_frame, from_=1, to=20, orient=tk.HORIZONTAL,
                             variable=speed_var, command=update_speed)
    speed_slider.pack(side=tk.LEFT, padx=5)


    # Function to run a simulation step
    def run_step():
        nonlocal step_count
        if not paused and step_count < max_steps and env.agents:
            # Generate random actions
            actions = generate_action_dict()

            # Step the environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Calculate custom reward
            custom_reward = calculate_reward(observations)
            all_rewards.append(custom_reward)

            # Update current reward display
            current_reward_var.set(f"Current Reward: {custom_reward:.4f}")

            # Update distances
            update_landmark_distances(observations)
            calculate_agent_distances(observations)

            # Update plots
            update_rewards_plot()
            update_distances_plot()
            update_stats_plot()

            # Render the environment
            frame = env.render()

            # Convert NumPy array to PIL Image
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)

            # Update label with new image
            env_display.config(image=photo)
            env_display.image = photo  # Keep reference to prevent garbage collection

            # Schedule the next step (use speed slider to adjust delay)
            step_count += 1
            delay = int(1000 / speed_var.get())  # Convert speed to delay (ms)
            root.after(delay, run_step)
        elif not paused and env.agents:
            # Simulation ended normally
            env.close()


    # Reset the environment and start the simulation
    observations, infos = env.reset()

    # Get initial frame
    initial_frame = env.render()
    initial_image = Image.fromarray(initial_frame)
    initial_photo = ImageTk.PhotoImage(initial_image)
    env_display.config(image=initial_photo)
    env_display.image = initial_photo

    # Initialize distances
    update_landmark_distances(observations)
    calculate_agent_distances(observations)

    # Start the simulation
    run_step()

    # Start the main event loop
    root.mainloop()