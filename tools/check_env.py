"""Environment sanity check.

Prints the MuJoCo version and JAX devices, then renders the Menagerie Panda
at keyframe "home" to results/setup/home.png using headless EGL.

Usage: python tools/check_env.py
"""
import os

# Force headless EGL (overrides e.g. MUJOCO_GL=glfw from ~/.bashrc).
# Must be set before importing mujoco.
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import jax
import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
OUT = ROOT / "results" / "setup" / "home.png"


def main() -> None:
    print(f"MuJoCo version : {mujoco.__version__}")
    print(f"MUJOCO_GL      : {os.environ['MUJOCO_GL']}")
    print(f"JAX version    : {jax.__version__}")
    print(f"JAX devices    : {jax.devices()}")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)  # kinematics only, no time stepping
    print(f"Panda: nq={model.nq} nu={model.nu} timestep={model.opt.timestep}")
    print(f"home qpos: {data.qpos.round(4)}")

    # The scene defines no cameras, so use a free camera aimed at the arm.
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.3, 0.0, 0.4]
    cam.distance = 2.0
    cam.azimuth = 135.0
    cam.elevation = -20.0

    # 640x480 is the default offscreen framebuffer size for this model.
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(OUT)
    print(f"Saved {pixels.shape[1]}x{pixels.shape[0]} frame to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
