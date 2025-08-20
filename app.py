import streamlit as st
import gymnasium as gym
import numpy as np
import time

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
""")

# --- Load Q-table and Environment ---
Q_TABLE_FILENAME = 'best_intra_alt_q_table_U.npy'

@st.cache_resource
def load_resources():
    try:
        q_table = np.load(Q_TABLE_FILENAME)
    except FileNotFoundError:
        st.error(f"Error: Q-table file '{Q_TABLE_FILENAME}' not found. Make sure it's in the GitHub repository.")
        return None, None
    env = gym.make('Taxi-v3', render_mode='rgb_array')
    return q_table, env

q_table, env = load_resources()

if q_table is None or env is None:
    st.stop()

# --- Streamlit Session State Management ---
if 'state' not in st.session_state:
    st.session_state.state, st.session_state.info = env.reset()
    st.session_state.total_reward = 0
    st.session_state.steps = 0
    st.session_state.terminated = False
    st.session_state.truncated = False
    st.session_state.last_action_desc = "None"
    st.session_state.last_reward = 0
    st.session_state.running_full_sim = False

def reset_simulation():
    st.session_state.state, st.session_state.info = env.reset()
    st.session_state.total_reward = 0
    st.session_state.steps = 0
    st.session_state.terminated = False
    st.session_state.truncated = False
    st.session_state.last_action_desc = "None"
    st.session_state.last_reward = 0
    st.session_state.running_full_sim = False

# --- Helper Functions ---
ACTION_MAP = {0: "South", 1: "North", 2: "East", 3: "West", 4: "Pickup", 5: "Dropoff"}

def select_best_action(state, q_table_local):
    return np.argmax(q_table_local[int(state), :])

# --- UI Layout ---
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.header("Controls")
    
    if st.button("Run Full Simulation", use_container_width=True, type="primary"):
        # Don't reset if it's already a fresh episode
        if st.session_state.steps > 0:
            reset_simulation()
        st.session_state.running_full_sim = True
        st.rerun()

    if st.button("Reset Environment", use_container_width=True):
        reset_simulation()
        st.rerun()

    if st.button("Step Manually", use_container_width=True):
        st.session_state.running_full_sim = False # Stop any automatic simulation
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
        st.rerun()

with col2:
    st.header("Simulation View")
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

# --- Logic for displaying the frame and stats ---
def update_display():
    frame = env.render()
    frame_placeholder.image(frame, caption=f"Step: {st.session_state.steps}", use_container_width=True)
    
    with stats_placeholder.container():
        st.write(f"**Step:** `{st.session_state.steps}`")
        st.write(f"**Last Action:** `{st.session_state.last_action_desc}` (Reward: `{st.session_state.last_reward:.1f}`)")
        st.write(f"**Total Reward:** `{st.session_state.total_reward:.1f}`")

        if st.session_state.terminated:
            st.success(f"Episode finished successfully in {st.session_state.steps} steps!")
            st.session_state.running_full_sim = False
        elif st.session_state.truncated:
            st.warning(f"Episode truncated after {st.session_state.steps} steps.")
            st.session_state.running_full_sim = False

# --- THE NEW ANIMATION LOOP ---
# This block replaces the old `while` loop
if st.session_state.get("running_full_sim", False):
    if not st.session_state.terminated and not st.session_state.truncated:
        # Perform one step
        action = select_best_action(st.session_state.state, q_table)
        st.session_state.state, reward, term, trunc, _ = env.step(action)
        st.session_state.total_reward += reward
        st.session_state.steps += 1
        st.session_state.terminated = term
        st.session_state.truncated = trunc
        st.session_state.last_action_desc = f"{ACTION_MAP[action]}"
        st.session_state.last_reward = reward
        
        # Trigger the next run in the animation loop
        time.sleep(0.1) # Control animation speed
        st.rerun()

# Always update the display at the end of every script run
update_display()
