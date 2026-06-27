"""
Phase 1 / Week 1 starter -- the canonical MuJoCo loop on a hand-written 2-DOF arm.

Run it:
    mamba activate robo        # make sure you're in (robo), not (base)
    python run.py

What you'll see: the arm swings down and double-pendulums around under gravity,
because nothing is holding it up yet. THAT SAG is the Phase 2 problem
("make it stop sagging" = PD control). Close the viewer window to stop.

Mental map (same shape as your MD loop):
    mjModel  = the force-field file   (static; never mutated mid-run)
    mjData   = the trajectory buffer  (qpos, qvel, ctrl, sensordata, ...)
    mj_step  = the integrator         (advances mjData by one timestep)
"""

import time
import mujoco
import mujoco.viewer

# ===== SETUP (runs once) ====================================================
# Load the static model description from your hand-written MJCF.
model = mujoco.MjModel.from_xml_path("arm.xml")
# Allocate the dynamic state that mj_step will advance.
data = mujoco.MjData(model)

# Inspect what you just built -- these counts come straight from the model.
print(f"nq (position DoF) : {model.nq}")          # 2  -> shoulder, elbow angles
print(f"nv (velocity DoF) : {model.nv}")          # 2
print(f"nu (actuators)    : {model.nu}")          # 2  motors
print(f"nsensordata       : {model.nsensordata}") # 7  -> 1+1+1+1+3

# Look up the tip sensor's address by NAME instead of hardcoding an index.
# This is the habit to build: mj_name2id + sensor_adr, not magic numbers.
def get_sensor_address(model, sensor_name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    return model.sensor_adr[sensor_id]

# Push the arm off equilibrium so the swing is interesting (radians).
data.qpos[:] = [0.0, -0.8]   # [shoulder, elbow]

# After writing qpos by hand, call mj_forward ONCE so all derived quantities
# (body frames, sensor values) are consistent before you read them. Skipping
# this is the classic "sensor reads zero" pitfall.
mujoco.mj_forward(model, data)

# ===== LOOP (runs every timestep) ===========================================
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():

        # 2. CONTROL (Week 2): write into data.ctrl to drive the motors.
        #    Left commented for Week 1 -> the arm is unpowered and just falls.
        data.ctrl[:] = [5.0, -3.0]

        # 3. STEP: advance the physics by one model.opt.timestep.
        mujoco.mj_step(model, data)

        # 1. SENSE: read state back out. Print every 200 steps so it's readable.
        if step % 200 == 0:
            shoulder_pos_adr = get_sensor_address(model, "shoulder_pos")
            elbow_pos_adr = get_sensor_address(model, "elbow_pos")
            shoulder_vel_adr = get_sensor_address(model, "shoulder_vel")
            elbow_vel_adr = get_sensor_address(model, "elbow_vel")
            tip_adr = get_sensor_address(model, "tip_pos")

            shoulder_pos = data.sensordata[shoulder_pos_adr]
            elbow_pos = data.sensordata[elbow_pos_adr]
            shoulder_vel = data.sensordata[shoulder_vel_adr]
            elbow_vel = data.sensordata[elbow_vel_adr]
            tip = data.sensordata[tip_adr:tip_adr + 3]   # world (x, y, z) of the tip
            print(
                f"t={data.time:6.2f}  "
                f"shoulder_pos={shoulder_pos:+.3f}  elbow_pos={elbow_pos:+.3f}  "
                f"shoulder_vel={shoulder_vel:+.3f}  elbow_vel={elbow_vel:+.3f}  "
                f"tip=({tip[0]:+.3f}, {tip[1]:+.3f}, {tip[2]:+.3f})"
            )

        # 4. RENDER: push the new state to the viewer window.
        viewer.sync()

        # Keep the loop near real time (optional -- nicer to watch).
        time.sleep(model.opt.timestep)
        step += 1
