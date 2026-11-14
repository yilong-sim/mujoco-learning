# Progress

Update at the end of every session: done / next / open questions.

## 2026-11-13 — one-time setup (PLAN.md "One-time setup")

### Done
- Setup step 2: CLAUDE.md committed (`dbab5e2`).
- Setup step 3: `MuJoCo_Study_Plan_v4.docx` converted to PLAN.md with pandoc
  (`1df9ef2`). The 25 code blocks were then rebuilt from the docx so Python
  indentation survives (`d3d78dd`); all Python blocks pass a syntax check.
  The docx is ignored by git (`839cd07`); PLAN.md is the version that's tracked.
- Installed `mujoco-mjx==3.9.0` and `playground==0.2.0` in `robo`.
  mujoco stayed at 3.9.0, jax/jaxlib at 0.10.1. Brax 0.14.2, Flax, Optax,
  Orbax and warp-lang 1.19.0 came in as dependencies.
- Installed pandoc 3.1.3 (apt).
- Setup step 5 (smoke test): `tools/check_env.py` (`6e85cf8`). It prints the
  MuJoCo/JAX versions and devices (`CudaDevice(id=0)`) and renders the Panda
  "home" keyframe headless with EGL to `results/setup/home.png`. The rendered
  pose matches the keyframe qpos: elbow −π/2, wrist +π/2, joint 7 −π/4,
  gripper open at 0.04 m per finger.
- Setup step 4: permissions in `.claude/settings.local.json` (git-ignored).
  Allow: pytest, python, git add, git commit. Ask: pip, mamba, rm, git push.
  Rules match the command prefix, so `conda run -n robo python ...` still
  prompts. Launch Claude Code from inside the activated `robo` env.

### Next
- Phase 1 close-out (PLAN.md "Phase 1 — close-out"):
  - `tests/test_phase1.py` from your assertion list: both models load,
    Panda nq/nv/nu = 9/9/8, 1,000 steps without NaN, sensors nonzero after
    `mj_forward`.
  - Theory-map exercise in tutor mode on the toy arm.
  - Commit "phase1: gate passing, theory map exercise".
- Then Phase 2, Session 2.1: torque mode and gravity compensation.

### Open questions
- `~/.bashrc:147` exports `MUJOCO_GL=glfw`. Any headless script that only
  defaults to EGL will use glfw. `check_env.py` forces EGL. Keep glfw as the
  shell default (for the viewer), or switch to egl and set glfw per viewer run?
- Repo layout: work so far is in `Phase1/` and `Fallingbody/`, but CLAUDE.md
  and PLAN.md assume `models/`, `controllers/`, `rewards/`, `tests/`,
  `sweeps/`, `envs/`, `results/`. Move `Phase1/arm.xml` to `models/`?
  (Your call. models/ is your folder.)
- `Phase1/MUJOCO_LOG.TXT` is committed. Keep it, or add it to .gitignore?
- PLAN.md says repo `~/mujoco-study`; the actual repo is
  `~/projects/mujoco-learning`. Harmless, but the paths in PLAN.md won't
  match.
