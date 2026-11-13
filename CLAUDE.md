# mujoco-learning
Learning MuJoCo + robotics (Modern Robotics, Lynch & Park) toward robotics simulation roles.
Background: FEM/continuum, MD/stat mech, optimization. The study plan is PLAN.md.
 
## Environment
- WSL2 Ubuntu 24.04, RTX 2000 Ada 8 GB. Conda env: robo. Headless rendering: MUJOCO_GL=egl.
- Versions are pinned: mujoco==3.9.0, mujoco-mjx==3.9.0, playground==0.2.0. Never upgrade them.
  A bare "pip install mujoco" or "pip install playground" installs MuJoCo 4 / Playground 1.0
  (PyTorch + Warp), which breaks this repo. Always pass explicit ==versions.
- Your knowledge of MuJoCo is older than this install. Verify every function signature with
  help() in the installed package before using it.
- Menagerie Panda facts: 9 DOF (7 arm + 2 finger), 8 actuators, no sites, keyframe "home",
  integrator implicitfast, dt 0.002. Use body "hand" for Jacobians unless I add a site.
- Never install NVIDIA drivers inside WSL2.
 
## How to help me (this is a learning project)
- Do NOT write code in models/, controllers/ or rewards/ unless I say "write it".
  Explain, give hints, or leave TODO stubs instead.
- You MAY write: tests/, tools/, sweeps/, envs/ (from my spec), configs, plots.
- For experiments with a known answer, ask for my prediction before you run anything.
- Debugging: state the root cause and the evidence before proposing a fix.
- Never silently change physics parameters (timestep, integrator, gains, solref/solimp,
  cone, impratio, friction, damping). Propose the change and explain why.
- Tests must assert against MuJoCo ground truth or an analytic result, never against my own code.
- Use FEM / stat-mech analogies when they genuinely help.
 
## Conventions
- Tests: pytest -q        Render: python tools/render.py <model.xml> <out>
- Each experiment saves a plot (+ mp4 where useful) to results/<phase>/
- Commit before every tuning pass and before any task that touches physics.
  Message format: "phaseN: <what works>"
- End each session by updating PROGRESS.md (done / next / open questions).
