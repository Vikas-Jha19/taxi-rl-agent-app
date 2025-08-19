import streamlit as st
import gymnasium as gym
import numpy as np
import time
import PIL.Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Taxi-v3 RL Agent",
    page_icon="🚕",
    layout="wide"
)

# --- App Title and Description ---
st.title("🚕 Interactive Taxi-v3 Agent")
st.markdown("""
This app showcases a Reinforcement Learning agent solving the **Taxi-v3** environment from Gymnasium.
The agent was trained using **Intra-Option Q-Learning** and this simulation uses its learned Q-table to select the best actions.

**Controls:**
- **Run Full Simulation:** Runs a complete episode automatically.
- **Reset:** Starts a new episode.
- **Step Manually:** Executes one action based on the agent's policy.
""")

# --- Load Q-table and Environment ---
Q_TABLE_FILENAME = 'best_intra_alt_q_table_U.npy'

@st.cache_resource
def load_resources():
    """Load the Q-table and create the environment. Caching for performance."""
    try:
        q_table = np.load(Q_TABLE_FILENAME)
    except FileNotFoundError:
        st.error(f"Error: Q-table file '{Q_TABLE_FILENAME}' not found. Make sure it's in the GitHub repository.")
        return None, None
    
    # Use render_mode='rgb_array' to get frames for display
    env = gym.make('Taxi-v3', render_mode='rgb_array')
    return q_table, env

q_table, env = load_resources()

if q_table is None or env is None:
    st.stop() # Stop the app if resources failed to load

# --- Streamlit Session State Management ---
# We use session state to keep track of the simulation across button clicks
if 'state' not in st.session_state:
    st.session_state.state, st.session_state.info = env.reset()
    st.session_state.total_reward = 0
    st.session_state.steps = 0
    st.session_state.terminated = False
    st.session_state.truncated = False
    st.session_state.last_action_desc = "None"
    st.session_state.last_reward = 0

def reset_simulation():
    """Resets the environment and session state variables."""
    st.session_state.state, st.session_state.info = env.reset()
    st.session_state.total_reward = 0
    st.session_state.steps = 0
    st.session_state.terminated = False
    st.session_state.truncated = False
    st.session_state.last_action_desc = "None"
    st.session_state.last_reward = 0
    st.success("Environment Reset!")

# --- Helper Functions ---
ACTION_MAP = {0: "South", 1: "North", 2: "East", 3: "West", 4: "Pickup", 5: "Dropoff"}

def select_best_action(state, q_table_local):
    """Selects the action with the highest Q-value."""
    return np.argmax(q_table_local[int(state), :])

# --- UI Layout ---
col1, col2 = st.columns([1, 1.5]) # Create two columns for controls and simulation view

# Column 1: Controls
with col1:
    st.header("Controls")
    
    # Buttons for simulation control
    if st.button("Run Full Simulation", use_container_width=True, type="primary"):
        reset_simulation()
        # The main simulation loop will run below
    
    if st.button("Reset Environment", use_container_width=True):
        reset_simulation()
    
    if st.button("Step Manually", use_container_width=True):
        if not st.session_state.terminated and not st.session_state.truncated:
            action = select_best_action(st.session_state.state, q_table)
            st.session_state.state, reward, term, trunc, _ = env.step(action)
            st.session_state.total_reward += reward
            st.session_state.steps += 1
            st.session_state.terminated = term
            st.session_state.truncated = trunc
            st.session_state.last_action_desc = f"{ACTION_MAP[action]}"
            st.session_state.last_reward = reward
        else:
            st.warning("Episode finished. Please reset the environment.")

# Column 2: Simulation View and Stats
with col2:
    st.header("Simulation View")
    
    # Create a placeholder for the image
    frame_placeholder = st.empty()
    
    # Display statistics below the image
    stats_placeholder = st.empty()

def update_display():
    """Renders the environment and updates the display placeholders."""
    # Render the environment frame
    frame = env.render()
    frame_placeholder.image(frame, caption=f"Step: {st.session_state.steps}", use_column_width=True)
    
    # Update stats
    with stats_placeholder.container():
        st.write(f"**Step:** `{st.session_state.steps}`")
        st.write(f"**Last Action:** `{st.session_state.last_action_desc}` (Reward: `{st.session_state.last_reward}`)")
        st.write(f"**Total Reward:** `{st.session_state.total_reward}`")

        if st.session_state.terminated:
            st.success(f"Episode finished successfully in {st.session_state.steps} steps!")
        elif st.session_state.truncated:
            st.warning(f"Episode truncated after {st.session_state.steps} steps.")

# --- Main Simulation Loop (for "Run Full Simulation") ---
# This part runs only if the full simulation was started
if "Run Full Simulation" in st.session_state and st.session_state["Run Full Simulation"]:
    # Loop until the episode is done
    while not st.session_state.terminated and not st.session_state.truncated:
        action = select_best_action(st.session_state.state, q_table)
        st.session_state.state, reward, term, trunc, _ = env.step(action)
        st.session_state.total_reward += reward
        st.session_state.steps += 1
        st.session_state.terminated = term
        st.session_state.truncated = trunc
        st.session_state.last_action_desc = f"{ACTION_MAP[action]}"
        st.session_state.last_reward = reward
        
        # Update the display and add a delay for visualization
        update_display()
        time.sleep(0.2) # Control animation speed

    # Clear the session state flag after the loop is done
    st.session_state["Run Full Simulation"] = False

# Always update the display on any script rerun (e.g., after button clicks)
update_display()
