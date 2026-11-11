import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

# 1. Load your existing MJCF file
model = mujoco.MjModel.from_xml_path("fallingbody.mjcf.xml")
data = mujoco.MjData(model)

# 2. Initialize simulation
data.qvel[0] = 0.0  # Initial X velocity
data.qvel[1] = 0.0  # Initial Y velocity
data.qvel[2] = 0.0  # Initial Z velocity
mujoco.mj_forward(model, data)

print("Model loaded successfully!")
print("nq =", model.nq)
print("nv =", model.nv)

# 3. Run simulation
simulation_time = 5.0

time_history = []
position_history = []
velocity_history = []
contact_history = []

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Run simulation for 5 seconds
    while viewer.is_running() and data.time < simulation_time:

        step_start = time.time()

        mujoco.mj_step(model, data)

        time_history.append(data.time)
        position_history.append(data.qpos[:3].copy())
        velocity_history.append(data.qvel[:3].copy())
        contact_history.append(data.ncon)

        viewer.sync()

        remaining = model.opt.timestep - (time.time() - step_start)

        if remaining > 0:
            time.sleep(remaining)

    # Number of detected contacts
    print("Number of contacts:", data.ncon)

    # Contact information
    for i in range(data.ncon):
            contact = data.contact[i]

            print("Contact position:", contact.pos)
            print("Contact distance:", contact.dist)

    # Simulation has stopped, but keep the viewer open
    print("Simulation finished. Viewer remains open.")

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)

# 4. Convert results
time_history = np.array(time_history)
position_history = np.array(position_history)
velocity_history = np.array(velocity_history)
contact_history = np.array(contact_history)

# 5. Plot results
fig, axes = plt.subplots(2, 1, figsize=(9, 7))

axes[0].plot(time_history, position_history)
axes[0].set_ylabel("Position (m)")
axes[0].set_xlabel("Time (s)")
axes[0].legend(["X", "Y", "Z"])
axes[0].grid()

axes[1].plot(time_history, velocity_history)
axes[1].set_ylabel("Velocity (m/s)")
axes[1].set_xlabel("Time (s)")
axes[1].legend(["Vx", "Vy", "Vz"])
axes[1].grid()

plt.tight_layout()
plt.show()