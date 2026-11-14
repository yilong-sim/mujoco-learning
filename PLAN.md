**MuJoCo Study Plan — v4**

*Theory-grounded roadmap for robotics simulation engineering, with a
Claude Code workflow for every phase*

**Pace:** about 5 hrs/week • **Remaining:** about 12 weeks from
mid-Phase 2 • **Robot:** Franka Emika Panda throughout • **Pinned
stack:** MuJoCo 3.9.0, MJX 3.9.0, Playground 0.2.0 • **Revised:** 12 Nov
2026

# What changed from v3, and why

v3 treated MuJoCo as a black box. You have now read the two Todorov
papers, so v4 turns the engine's internals into experiments you run and
check against known answers. The Franka-only scope, the you-write /
agent-builds split, the test gates and the sensitivity-analysis capstone
all stay.

| **Change**                                                | **What and why**                                                                                                                                                                                                                                                                      |
|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Stack pinned; MuJoCo 4 deferred**                       | MuJoCo 4.0 shipped on 6 Nov 2026 and MuJoCo Playground 1.0 was rewritten in PyTorch on MuJoCo Warp with no Franka task. A plain pip install mujoco or pip install playground now pulls that stack. v4 pins 3.9.0 until the capstone ships (details in the next section).              |
| **Contact explained correctly**                           | v3 described solref/solimp as penalty-contact springs. They set a reference acceleration and an impedance inside a convex optimization. Phase 2.5 now teaches that, with measurements.                                                                                                |
| **Phase 2.5 grows to 2 weeks**                            | Week A is a contact lab with analytic answers (slip angle, creep, penetration scaling, integrator stability). Week B is grasp-and-lift. The lab results become your Phase 4 parameter ranges.                                                                                         |
| **Inverse dynamics added**                                | mj_inverse is the central result of the ICRA 2014 paper and v3 never called it. It is now a Phase 2 gate and a Phase 2.5 force-measurement tool.                                                                                                                                      |
| **New Phase 2.75: model-based control**                   | One week of predictive sampling and finite-difference linearization. The IROS 2012 paper is titled "a physics engine for model-based control"; v3 covered RL only.                                                                                                                    |
| **Phase 3 cut to 2 weeks**                                | The reach/push RL pipeline is a commodity artifact. The week saved pays for Phase 2.75.                                                                                                                                                                                               |
| **Phase 4 reordered: sensitivity analysis first, on CPU** | Sobol sampling needs batched rollouts, not GPU training. mujoco.rollout gives about 100,000 steps/s on two threads for the Panda-plus-cube scene, so the full analysis runs on your laptop in minutes, in float64, before any MJX work. This removes the biggest schedule risk in v3. |
| **Phase 4 trains on a known baseline**                    | SB3 cannot use thousands of vmapped MJX environments. v4 trains with Brax PPO on Playground 0.2.0's PandaPickCube, and your contribution is the randomization function.                                                                                                               |
| **Equivalence test redesigned**                           | v3's 200-step CPU-vs-MJX comparison fails on a correct port once contacts are moving. v4 tests one step tightly, short horizons loosely, and statistics over rollouts.                                                                                                                |

> **How the code in this plan was checked.** Every Python block was run
> on MuJoCo 3.9.0 with the current Menagerie Panda, on CPU. The
> reference numbers quoted beside them come from those runs. Four things
> were not run: SB3 training, Brax training, anything on a GPU, and
> offscreen rendering. Those blocks are marked **not run here**; have
> Claude Code verify them on your machine first.

# Stack decision: stay on 3.9.0 until the capstone ships

The ecosystem moved last week. Decide once, write it into CLAUDE.md, and
do not revisit it mid-plan.

|              | **Pinned stack (this plan)**                                                                                                                                      | **New stack (after the plan)**                                                                      |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Engine       | MuJoCo 3.9.0, MJX 3.9.0 (JAX)                                                                                                                                     | MuJoCo 4.0.x, MuJoCo Warp                                                                           |
| Environments | Playground 0.2.0: PandaPickCube, PandaRobotiqPushCube, PandaOpenCabinet                                                                                           | Playground 1.0: SO-101, Trossen WXAI, YAM, Unitree G1. No Franka.                                   |
| Training     | Brax PPO (JAX)                                                                                                                                                    | RSL-RL (PyTorch)                                                                                    |
| Why          | Your Franka work, the Menagerie MJX model and every snippet here run on it. 4.0 has breaking changes and 4.0.1 fixed a Newton-solver bug five days after release. | Where DeepMind and NVIDIA are heading. Port the capstone to it afterwards as a portfolio follow-up. |

Protect the environment now:

```bash
mamba activate robo
python -c "import mujoco; print(mujoco.__version__)"      # must print 3.9.0
pip freeze > requirements.lock.txt  # snapshot before anything else
# From now on, always install with pins. Never run a bare "pip install -U".
pip install "mujoco==3.9.0" "mujoco-mjx==3.9.0" SALib gymnasium
```

Add these lines to the Environment section of CLAUDE.md (the complete
file is in the Working with Claude Code section):

```markdown
- Versions are pinned: mujoco==3.9.0, mujoco-mjx==3.9.0, playground==0.2.0. Never upgrade them.
  A bare "pip install mujoco" or "pip install playground" installs MuJoCo 4 / Playground 1.0
  (PyTorch + Warp), which breaks this repo. Always pass explicit ==versions.
- Menagerie Panda facts: 9 DOF (7 arm + 2 finger), 8 actuators, no sites, keyframe "home",
  integrator implicitfast, dt 0.002. Use body "hand" for Jacobians unless I add a site.
```

# Roadmap at a glance

Phase 1 is done and Phase 2 is under way, so the count starts from the
rest of Phase 2. About 12 weeks remain at 5 hours a week.

| **Phase** | **Weeks** | **Focus**                                                | **Exit milestone**                                                                              | **Claude Code mode** |
|-----------|-----------|----------------------------------------------------------|-------------------------------------------------------------------------------------------------|----------------------|
| 1         | done      | MJCF + core API                                          | One session: run the gate, do the theory-map exercise                                           | Tutor                |
| 2         | 1–2       | PD, FK, Jacobians, IK, computed torque, inverse dynamics | IK tracks a moving target; computed torque holds pose; forward-inverse round trip passes        | Pair                 |
| 2.5       | 3–4       | Contact lab, then grasp-and-lift                         | Four contact experiments match their analytic answers; stable lift with measured contact forces | Pair + instruments   |
| 2.75      | 5         | Model-based control                                      | Predictive sampling reaches a target; linearization checked                                     | Pair                 |
| 3         | 6–7       | Gymnasium + SB3                                          | Push policy trained, 100-episode success rate reported                                          | Delegate             |
| 4         | 8–12      | Sensitivity analysis → MJX → targeted DR                 | Capstone report: Sobol ranking, three training arms, held-out robustness                        | Orchestrate          |

# Working with Claude Code

Claude Code is your lab partner in this plan, not the author. One rule
keeps the learning with you: **if it is in a Modern Robotics chapter or
it is a physics decision, you write it; if it is plumbing, delegate
it.** Physics decisions are model structure, controllers, gains, contact
parameters, reward terms, parameter ranges and experiment design.
Plumbing is scaffolding, tests, sweeps, plots, logging, configs and
ports. For v4 there is one addition: for every experiment with a known
answer, you write down the prediction before the agent runs the
measurement.

<table>
<colgroup>
<col style="width: 16%" />
<col style="width: 37%" />
<col style="width: 46%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>Mode</strong></th>
<th><strong>What it looks like</strong></th>
<th><strong>Typical prompt</strong></th>
</tr>
<tr class="odd">
<th><p><strong>Tutor</strong></p>
<p>(Phase 1)</p></th>
<th>Claude explains, quizzes and reviews. You type all the code.</th>
<th>"Explain why this hinge has no effect. Don't edit the file."</th>
</tr>
<tr class="header">
<th><p><strong>Pair</strong></p>
<p>(Phases 2–2.75)</p></th>
<th>You write the core. Claude writes tests, instruments and sweeps, and
reviews your diffs.</th>
<th>"Write tests that check my ik_step() against MuJoCo. Don't touch
controllers/."</th>
</tr>
<tr class="odd">
<th><p><strong>Delegate</strong></p>
<p>(Phase 3)</p></th>
<th>You write a spec. Claude plans, you approve, Claude builds. You read
every diff.</th>
<th>"Read ENV_SPEC.md. Propose a layout in plan mode, no code yet."</th>
</tr>
<tr class="header">
<th><p><strong>Orchestrate</strong></p>
<p>(Phase 4)</p></th>
<th>Claude runs multi-step pipelines and reusable skills. You own the
design and the interpretation.</th>
<th>"Run /sa-run on config B, then have a fresh subagent review the
results."</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## One-time setup (about 30 minutes)

1.  Launch from the active environment so the agent's shell sees your
    MuJoCo and GPU:

```bash
mamba activate robo
cd ~/mujoco-study && claude
```

2.  Put the CLAUDE.md below in the repo root. Claude Code reads it at
    the start of every session. It is the v3 file plus the v4 version
    pins and Panda facts.

3.  Give the agent the plan. Export this document to markdown and commit
    it, so any session can start with "read PLAN.md, Session 2.3":

```bash
pandoc MuJoCo_Study_Plan_v4.docx -t gfm -o PLAN.md && git add PLAN.md && git commit -m "plan v4"
```

4.  Set permissions with /permissions: allow pytest, python and git add
    / git commit without prompting. Keep prompts on for pip, mamba, rm
    and git push.

5.  Smoke test, as your first prompt: "Write tools/check_env.py that
    prints the MuJoCo version and JAX devices, renders one frame of the
    Panda home pose to results/setup/home.png with MUJOCO_GL=egl, then
    open the image and describe it." This confirms the environment,
    headless rendering, and that the agent can see what it renders.

```markdown
# mujoco-study
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
```

Repo layout, with ownership visible in the folders:

```text
mujoco-study/
  CLAUDE.md  PLAN.md  PROGRESS.md
  models/        MJCF                          <- you
  controllers/   PD, IK, computed torque, MPC  <- you
  rewards/       reward functions              <- you
  docs/          ENV_SPEC.md, SA_DESIGN.md, DR_PROTOCOL.md   <- you
  envs/          Gymnasium wrappers            <- Claude, from your spec
  tools/         render, contact logging       <- Claude
  sweeps/        parameter sweeps              <- Claude
  tests/         ground-truth tests            <- Claude (you define what to assert)
  drills/        broken copies for diagnosis   <- Claude
  results/<phase>/                             plots, logs, mp4
  .claude/skills/sa-run/SKILL.md               <- Phase 4
```

## A 5-hour week

| **Block**       | **Time** | **What happens**                                                                                                                                                    |
|-----------------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Start           | 10 min   | /clear, then: "Read PROGRESS.md and PLAN.md Session X. Today's goal is Y. Don't write code in controllers/." Use plan mode for anything that touches several files. |
| Core build      | 3 h      | You write the phase's core code. Claude answers questions and reviews; it does not supply solutions.                                                                |
| Parallel block  | 1 h      | Claude builds tests, instruments or sweeps while you do the reading. This is where the agent saves the most time.                                                   |
| Review + commit | 40 min   | Read every diff, run pytest -q, commit the working state.                                                                                                           |
| Explain-back    | 10 min   | "Quiz me on today's concept with 3 questions, no hints. Then update PROGRESS.md." Gaps go into the open-questions list.                                             |

## Mechanics you will use

| **Feature**     | **How and when**                                                                                                                                                                                                                      |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Plan mode       | Shift+Tab cycles the permission modes until the status line shows plan mode; claude --permission-mode plan starts a session in it. The agent reads and proposes, and edits nothing until you approve. Use it for every Delegate task. |
| Fresh context   | /clear between tasks. Long sessions get vague. Continuity lives in PROGRESS.md, not in the conversation.                                                                                                                              |
| Resume          | claude --continue reopens the most recent session; claude --resume shows a picker.                                                                                                                                                    |
| Background runs | Ask it to start training or a long sweep in the background and to summarize the result when it finishes, with frames from the eval video.                                                                                             |
| Subagent review | "Use a subagent that hasn't seen our work to review X." A reviewer with no stake in the work finds more.                                                                                                                              |
| Project skills  | A repeated workflow becomes .claude/skills/\<name\>/SKILL.md and is invoked as /\<name\> \<args\>. Build /sa-run in Phase 4 after doing the workflow by hand twice.                                                                   |
| Headless        | claude -p "summarize results/run_17 into results/LOG.md" at the end of a cluster job script, so finished runs document themselves.                                                                                                    |

> **Four habits.** (1) Make it verify APIs with help() on the installed
> package; MuJoCo 4 will make its stale knowledge worse. (2) Evidence
> before fixes: root cause first, then the change. (3) Scope every
> prompt to a folder. (4) Commit before any agent task that touches
> physics, so git checkout is your undo. If you cannot explain a
> milestone without the agent, too much was delegated: rewrite that
> piece yourself before moving on.

# Theory map: from the two papers to MuJoCo 3.9

Use this table whenever a field name looks arbitrary. The right-hand
column is what you can print in a Python session.

| **Paper concept**                                                       | **Where it lives in MuJoCo 3.9**                            | **Check it yourself**                                            |
|-------------------------------------------------------------------------|-------------------------------------------------------------|------------------------------------------------------------------|
| Generalized coordinates, joint-space inertia M(q) (IROS12)              | data.qM (sparse), unpacked by mj_fullM(model, dst, data.qM) | M is symmetric; condition number about 33 at the Panda home pose |
| Bias forces c(q, v): Coriolis, centrifugal, gravity                     | data.qfrc_bias                                              | Applying it as torque holds the arm still                        |
| Constraint Jacobian J, inverse inertia in constraint space A = J M⁻¹ Jᵀ | data.efc_J, data.nefc                                       | Rows appear when contacts or limits activate                     |
| Reference acceleration a\* = −b·v − k·r (ICRA14)                        | data.efc_aref, set by solref                                | Changes when you change solref timeconst                         |
| Regularizer R, impedance d                                              | data.efc_R, data.efc_D, set by solimp                       | Larger d → stiffer, less creep                                   |
| Constraint force f from the convex problem                              | data.efc_force; per-contact wrench via mj_contactForce      | Normal forces on a resting box sum to m·g                        |
| Analytic inverse dynamics                                               | mj_inverse → data.qfrc_inverse                              | Round-trip error about 1e-15 without contact                     |
| Finite-difference derivatives for control                               | mjd_transitionFD(model, data, eps, centered, A, B, C, D)    | A is 18×18 and B is 18×8 for the Panda                           |

> **What is newer than the papers.** The default solver is now Newton
> (the papers predate it). Friction cones can be pyramidal or elliptic.
> implicitfast is the recommended integrator and the Menagerie Panda
> already uses it. noslip_iterations adds a post-solve pass that
> suppresses creep. Read the **Computation** chapter of the MuJoCo docs
> during a Phase 2.5 parallel block; it is the current version of both
> papers.

# Phase 1 — close-out (one session)

**Goal:** confirm the Phase 1 gate still passes and connect what you
built to the theory map.

1.  Run pytest tests/test_phase1.py -q. If the file does not exist yet,
    have Claude write it from your assertion list: both models load,
    nq/nv/nu match (Panda: 9/9/8), 1,000 steps run without NaN, sensors
    are nonzero after mj_forward.

2.  Theory-map exercise, in tutor mode: "For each row of the theory map,
    show me the field on my toy arm with one hinge limit active, and ask
    me what I expect before printing it."

3.  Commit: git commit -m "phase1: gate passing, theory map exercise".

### Claude Code in this phase — Tutor

**Skill: context engineering.** Steer with CLAUDE.md and scoped prompts
instead of re-explaining every session.

- “Copy models/arm.xml to drills/broken_1.xml and introduce one bug that
  still loads but behaves wrong. Don't tell me what it is.”

- “Walk me through panda.xml top-down. Map each section to its toy-arm
  equivalent and ask me one question after each section.”

# Phase 2 — Kinematics, control and inverse dynamics (weeks 1–2)

**Goal:** command the arm in joint space, then in Cartesian space, then
with model-based torque, and prove each with a ground-truth test.
**Mode: Pair.** You write every controller; the agent writes the tests
and sweeps.

## Session 2.1 — torque mode and gravity compensation

The Menagerie Panda's actuators are position servos: gainprm is kp and
biasprm is \[0, −kp, −kv\], with kp of 4500, 3500 and 2000 across the
joints. Your own torque controller must not fight them.

1.  Print what you are dealing with:

```bash
python - <<'EOF'
import mujoco
m = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
print("nq nv nu", m.nq, m.nv, m.nu, "dt", m.opt.timestep, "nsite", m.nsite)
print("kp", m.actuator_gainprm[:, 0]); print("bias", m.actuator_biasprm[:, :3])
EOF
```

2.  Choose how to get torque mode. Option A: copy panda.xml into
    models/panda_torque.xml and replace the seven arm actuators with
    \<motor\> elements whose ctrlrange equals the old forcerange (±87
    N·m for joints 1–4, ±12 N·m for 5–7). Option B: keep the file and
    zero the servo gains at load time, then write torques to
    data.qfrc_applied. Option A is the cleaner portfolio artifact;
    Option B is the faster check. The B route, verified:

```python
m.actuator_gainprm[:7, 0] = 0          # kp off
m.actuator_biasprm[:7, 1:3] = 0        # -kp, -kv off
mujoco.mj_resetDataKeyframe(m, d, 0)   # keyframe 0 is "home"
```

3.  Gravity compensation: each step, call mj_forward, then set
    data.qfrc_applied\[:7\] = data.qfrc_bias\[:7\], then mj_step. The
    arm should hang motionless. Then push it in the viewer (double-click
    a link, Ctrl + right-drag) and watch it float.

4.  Add PD on top: τ = qfrc_bias + kp·(q_des − q) − kv·q̇. Start with kp
    = 100, kv = 20 and commit before every tuning pass.

## Session 2.2 — integrator experiment (30 minutes, high value)

Before tuning gains by feel, see why the Panda ships with implicitfast.
On a small one-joint arm (0.2 kg, 0.3 m) with a servo of kp = 100 at dt
= 0.002, the measured result was:

| **Integrator** | **kv = 1** | **kv = 10** | **kv = 100** |
|----------------|------------|-------------|--------------|
| Euler          | stable     | unstable    | unstable     |
| RK4            | stable     | unstable    | unstable     |
| implicitfast   | stable     | stable      | stable       |

Velocity-dependent actuator forces are stiff, and only the implicit
integrators treat them implicitly. Passive joint damping is different:
Euler already integrates it implicitly, so a heavily damped passive
pendulum is stable under every integrator. For energy conservation with
no damping, RK4 wins by a wide margin (energy drift about 1e-7 over 5 s
at dt = 0.002, against about 1e-2 for Euler and implicitfast). Reproduce
both results on your toy arm; ask Claude for sweeps/integrators.py after
you have predicted the outcome.

> **Rule to keep.** When the sim explodes, the order of checks is:
> integrator (implicitfast), then gains, then timestep. v3 said to
> reduce dt first.

## Session 2.3 — forward kinematics and Jacobians

The Panda has no sites, so use the hand body, or add a site at the
fingertip midpoint in your model copy (an MJCF decision, so yours).
Jacobian call and a finite-difference check:

```python
import numpy as np, mujoco
hand = m.body("hand").id
jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
mujoco.mj_forward(m, d)
mujoco.mj_jacBody(m, d, jacp, jacr, hand)  # with a site: mj_jacSite(m, d, jacp, jacr, site_id)
 
# finite-difference check of jacp
eps, q0, x0 = 1e-6, d.qpos.copy(), d.xpos[hand].copy()
Jfd = np.zeros((3, m.nv))
for j in range(m.nv):
    q = q0.copy(); dq = np.zeros(m.nv); dq[j] = eps
    mujoco.mj_integratePos(m, q, dq, 1.0)          # correct on any joint type
    d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    Jfd[:, j] = (d.xpos[hand] - x0) / eps
d.qpos[:] = q0; mujoco.mj_forward(m, d)
print(np.abs(Jfd - jacp).max())                    # reference: about 3e-7
```

Write your own FK from the MR Chapter 4 product of exponentials for at
least the first three joints and compare with data.xpos. That is the one
place in this phase where hand-derivation pays off.

## Session 2.4 — differential IK with damped least squares

The update is Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e. Write controllers/ik.py
yourself. The core lines, for checking after you have tried:

```python
J = jacp[:, :7]
e = target - d.xpos[hand]
dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), e)
d.qpos[:7] += step * dq  # kinematic test; then mj_kinematics + mj_comPos
```

Reference: with λ = 0.05 and step 0.5, a 0.27 m offset converges to
machine precision within 200 iterations in a purely kinematic loop. Then
make it dynamic: feed q + Δq to the position servos as data.ctrl\[:7\]
and track a target moving on a circle. Experiments that are yours to
design: sweep λ near a singularity (elbow straight) and plot tracking
error against joint velocity; add orientation error using
mujoco.mju_subQuat and the full 6×7 Jacobian.

## Session 2.5 — computed torque and the inverse-dynamics gate

The textbook form is τ = M(q)·q̈\_des + qfrc_bias with q̈\_des = kp·(q_des
− q) − kv·q̇. mj_fullM takes the destination first:

```python
M = np.zeros((m.nv, m.nv))
mujoco.mj_forward(m, d)
mujoco.mj_fullM(m, M, d.qM)
qacc_des = np.zeros(m.nv)
qacc_des[:7] = kp * (q_des - d.qpos[:7]) - kv * d.qvel[:7]
d.qfrc_applied[:7] = (M @ qacc_des + d.qfrc_bias)[:7]
mujoco.mj_step(m, d)
```

Now the check that v3 lacked. MuJoCo can invert its own dynamics, so you
can ask it for the torque that produces a given acceleration and compare
with yours:

```python
mujoco.mj_forward(m, d)  # computes d.qacc from applied forces
applied = d.qfrc_applied + d.qfrc_actuator
mujoco.mj_inverse(m, d)  # computes d.qfrc_inverse from d.qacc
print(np.abs(d.qfrc_inverse - applied).max())       # reference: about 9e-16, no contact
```

Two uses. First, as a test of your controller: set d.qacc\[:\] =
qacc_des, call mj_inverse, and d.qfrc_inverse is the exact torque for
that acceleration. Do this and you will find that v3's formula is **off
by up to about 0.65 N·m** at moderate joint speeds. The Panda model has
joint damping of 1 N·m·s/rad on every joint, which lives in
qfrc_passive, not in qfrc_bias. The exact expression is:

```python
tau = M @ qacc_des + d.qfrc_bias - d.qfrc_passive - d.qfrc_constraint
# matches d.qfrc_inverse to about 4e-15 on the Panda (was 0.65 without the last two terms)
```

With the short formula the PD loop absorbs the error, which is why the
arm still settles (to about 1e-4 rad within 2 s after a 0.2 rad
disturbance at kp = 100, kv = 20). The mass matrix already contains the
model's 0.1 kg·m² joint armature. Second use, as a permanent health
check: enable mjENBL_FWDINV and MuJoCo reports the forward-inverse
discrepancy in d.solver_fwdinv every step.

```python
m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_FWDINV
```

> **Why this matters for you.** This is the claim of the ICRA 2014 paper
> made concrete: because constraints are soft and the problem is convex,
> inverse dynamics is well defined even with contact. You will use it in
> Phase 2.5 to read contact forces. Hard-contact engines cannot do this.

### Claude Code in this phase — Pair

**Skill: test-driven verification and code review.** You trust the
agent's work through tests built on MuJoCo ground truth, and you use it
as a reviewer of your work.

- “Before answering, confirm the signatures of mj_fullM, mj_jacBody and
  mj_inverse with help() in the installed mujoco, not from memory.”

- “Write tests/test_phase2.py: my Jacobian against finite differences
  (1e-5) and my computed torque against mj_inverse (1e-8). Don't modify
  controllers/. If a test fails, tell me which one and why, and don't
  fix my code.”

- “Review computed_torque.py like a senior robotics engineer: frames,
  signs, units, when qfrc_bias is valid relative to mj_step, and whether
  any force term is missing.”

- “I predict Euler goes unstable above kv = 10 on the toy arm and
  implicitfast does not. Write sweeps/integrators.py to test that and
  plot it.”

- “The arm diverges above kp = 400. Diagnose from qacc and an energy
  trace. State the root cause before you suggest any change.”

- “Commit the current state as 'phase2: PD stable' before you run the
  gain sweep.”

## Phase 2 split, reading, gate

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>You write</strong></th>
<th><strong>Claude Code builds</strong></th>
</tr>
<tr class="odd">
<th><p>controllers/pd.py, ik.py, computed_torque.py</p>
<p>The torque-mode model decision</p>
<p>Predictions before each experiment</p>
<p>Sweep ranges and the definition of "stable"</p></th>
<th><p>tests/test_phase2.py against MuJoCo ground truth</p>
<p>sweeps/integrators.py, sweeps/gains.py (λ × kp × kv heatmaps)</p>
<p>Moving-target generator</p>
<p>Review of your controllers for frames, signs and stale mj_forward
calls</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

**Reading:** MR 6.2 (skim), MR 8.1–8.3 (structure of M, c, g; skip the
recursive Newton-Euler derivation). MuJoCo docs, Computation chapter:
sections on general framework and actuation.

> **Gate.** pytest tests/test_phase2.py passes: Jacobian matches finite
> differences to 1e-5; your computed torque, including passive and
> constraint terms, matches mj_inverse to 1e-8; after a 0.2 rad step
> disturbance on every joint, computed torque settles to within 1e-3 rad
> in 2 s; IK tracking error on the moving target stays under a threshold
> you set (5 mm is a reasonable start); python sweeps/gains.py
> regenerates the heatmaps. Commit the two controllers separately.

# Phase 2.5 — Contact: lab first, then grasp (weeks 3–4)

**Goal:** understand MuJoCo's contact model well enough to predict its
behavior, then use that to make a stable grasp. **Mode: Pair.** You
design and interpret; the agent builds sweeps, logs and renders.

> **The correct mental model.** MuJoCo does not use penalty springs.
> Each step it solves a convex optimization for constraint forces.
> solref sets the reference acceleration the contact tries to achieve,
> a\* = −b·v − k·r, where r is penetration. solimp sets the impedance d
> in (0, 1), which controls how strongly the constraint is enforced
> against that reference. The spring-damper acts on the target
> acceleration, not on force. The closest analogy from your background
> is a regularized Lagrange-multiplier (compliant constraint)
> formulation, not penalty FEM. Keep solref timeconst at or above 2·dt.

## Week A — four experiments with known answers

Use one tiny model for all four: a 0.5 kg box on a plane, with gravity
tilted so that the plane acts as a ramp of angle θ. Write the XML
yourself in models/ramp.xml; it is ten lines. Tilting gravity instead of
the plane keeps the geometry trivial:

```xml
<mujoco>
  <option timestep="0.002" cone="pyramidal" impratio="1" noslip_iterations="0"
          gravity="3.355 0 -9.218"/>  <!-- 9.81*sin(20 deg), -9.81*cos(20 deg) -->
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1" friction="0.5 0.005 0.0001"/>
    <body name="box" pos="0 0 0.05"><freejoint/>
      <geom name="box" type="box" size=".05 .05 .05" mass="0.5" friction="0.5 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
```

### Experiment 1 — Coulomb slip angle

Prediction: the box slides when tan θ exceeds μ, so at μ = 0.5 the
threshold is 26.57°. Sweep θ from 20° to 30° and record the box velocity
after 2 s. Measured reference: at 26° the box moves at about 5 mm/s
(pyramidal) or 1.6 mm/s (elliptic); at 27° it moves at about 0.17–0.19
m/s. The threshold is right, and the sub-threshold motion leads to the
next experiment.

### Experiment 2 — creep below the slip angle

At 20° the box should be stationary. It is not, because the constraint
is soft. Measured creep velocities:

| **Setting**                     | **Creep at 20°** | **What it shows**            |
|---------------------------------|------------------|------------------------------|
| pyramidal, impratio 1           | 1.8e-3 m/s       | Baseline                     |
| elliptic, impratio 1            | 9.8e-4 m/s       | Cone type changes the answer |
| elliptic, impratio 10           | 9.8e-5 m/s       | Creep scales as 1/impratio   |
| elliptic, impratio 100          | 9.8e-6 m/s       | Same scaling                 |
| elliptic, noslip_iterations 5   | 1.5e-7 m/s       | No-slip pass removes it      |
| pyramidal, noslip_iterations 10 | 1.8e-3 m/s       | No effect in this setup      |

Two questions for you to answer in PROGRESS.md. Why does creep scale
inversely with impratio? (Hint: it scales the friction-direction
impedance relative to the normal direction.) Why did noslip do nothing
with the pyramidal cone here? Work the second one out with Claude in
tutor mode against the docs; I measured it but did not investigate the
cause.

### Experiment 3 — penetration against solref

Rest the box flat (θ = 0) and change the timeconst through
model.geom_solref\[:, 0\]. Measured penetration: 1.1e-6 m at 0.005,
1.7e-5 m at 0.02 (the default), 1.05e-4 m at 0.05. Penetration grows as
timeconst squared, as the reference-acceleration formula predicts
(stiffness k is proportional to 1/timeconst²). Then check force balance:

```python
f = np.zeros(6)
for i in range(d.ncon):
    mujoco.mj_contactForce(m, d, i, f)  # f[0] is the normal force in the contact frame
    print(i, d.contact[i].dist, f[0])
# reference: 4 contacts, 1.226 N each, sum = 4.905 N = m*g
```

### Experiment 4 — solimp and the impedance curve

solimp is (dmin, dmax, width, midpoint, power): impedance rises from
dmin to dmax as penetration grows across width. Sweep dmin from 0.5 to
0.99 at fixed solref and plot both penetration and creep. You should
find that a higher impedance reduces both, at the cost of a stiffer,
harder-to-solve problem. This is the experiment that tells you which
solimp range is physically reasonable for Phase 4.

> **Deliverable for Week A.** results/phase2.5/contact_lab.md: four
> plots, each with your prediction, the measurement and one sentence on
> the gap. This page becomes the appendix that justifies the parameter
> ranges in your capstone, so write it as you go.

## Week B — grasp-and-lift on the Panda

1.  Build models/panda_grasp.xml: include the Panda and add a free box
    (about 4 × 4 × 6 cm, 50–100 g) on a table or the floor. Apply Week
    A: choose the cone and impratio for the grasp deliberately. The
    MuJoCo docs recommend the elliptic cone with a raised impratio to
    stop slip in manipulation; confirm it with your own creep
    measurement on the finger pads.

2.  Sequence the grasp yourself: move above the box with your Phase 2
    IK, descend, close (data.ctrl\[7\] = 0; 255 is open), settle for 0.3
    s, lift 15 cm, hold 1 s.

3.  Instrument it. The agent writes tools/contact_log.py logging per
    step: ncon, minimum contact.dist, normal and tangential force per
    finger from mj_contactForce, and box-versus-finger relative
    velocity.

4.  Friction budget: the two pads must supply 2·μ·N ≥ m·g. Compute the
    minimum grip force by hand, then find the measured slip onset by
    lowering the gripper actuator's forcerange or the pad friction.
    Compare the two.

5.  Inverse-dynamics view: record qpos, qvel, qacc during the hold, call
    mj_inverse, and compare data.efc_force and data.qfrc_inverse with
    the forward run. Raise solver precision first (model.opt.tolerance =
    1e-10, model.opt.iterations = 200) and report the discrepancy you
    measure with contacts active.

6.  Explore condim: with condim 3 a grasped box can rotate about the
    contact normal; condim 4 adds torsional friction. Tilt the box in
    the grasp and compare.

### Claude Code in this phase — Pair, with the agent as instrumentation engineer

**Skill: give the agent eyes.** It cannot see your viewer, but it can
read the images it renders and the numbers it logs. Agents that debug
from evidence are far more reliable than agents that guess.

- “My prediction: slip starts at 26.6 degrees and creep scales as
  1/impratio. Write sweeps/ramp.py over 20–30 degrees for both cones and
  impratio 1, 10, 100. Plot it and tell me where my prediction is
  wrong.”

- “Explain solref and solimp in terms of the reference acceleration and
  impedance from the MuJoCo Computation docs, not as penalty springs.
  Then quiz me on what raising timeconst does to penetration.”

- “Write tools/contact_log.py: per step, ncon, minimum contact.dist,
  normal and tangential force per finger via mj_contactForce, and
  box-to-finger slip velocity. Save plots to results/phase2.5/.”

- “Render frames at t = 0.5, 1.0 and 1.5 s with contact points and
  forces visualized. Look at them and tell me whether the box is
  slipping, citing the logged slip velocity.”

- “noslip_iterations had no effect with the pyramidal cone on my ramp.
  Find out why from the docs and the source, and show me the evidence.”

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>You write</strong></th>
<th><strong>Claude Code builds</strong></th>
</tr>
<tr class="odd">
<th><p>models/ramp.xml, models/panda_grasp.xml</p>
<p>Predictions and interpretation for all experiments</p>
<p>Grasp sequencing and every contact parameter choice</p>
<p>results/phase2.5/contact_lab.md</p></th>
<th><p>sweeps/ramp.py, sweeps/solimp.py and their plots</p>
<p>tools/contact_log.py</p>
<p>Renders with contact points and force arrows (mjVIS_CONTACTPOINT,
mjVIS_CONTACTFORCE)</p>
<p>tests/test_phase2_5.py</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

**Reading:** MuJoCo docs, Computation chapter: Constraint model, Solver
parameters, Contact. Modeling chapter: the solref/solimp section. Reread
ICRA14 section III afterwards; it will read differently.

> **Gate.** Slip threshold within 1° of arctan μ; creep scales as
> 1/impratio within 10%; penetration scales as timeconst² within 20%;
> resting normal forces sum to m·g within 1%. Grasp: box rises more than
> 10 cm, relative slip under 2 mm during the hold, no NaN. Commit:
> phase2.5: contact lab + stable grasp.

# Phase 2.75 — Model-based control (week 5)

**Goal:** use the simulator as a model inside a controller, which is
what MuJoCo was built for. One week, two exercises. It also gives you a
concrete answer when an interviewer asks about planning, and it is the
direct analogue of your Basin-Hopping work: sample, evaluate, keep the
best, repeat from there.

## Exercise 1 — predictive sampling with mujoco.rollout

mujoco.rollout runs many open-loop rollouts in parallel threads on CPU.
The controller: sample K candidate setpoints around the current best,
roll each out for a short horizon, pick the lowest cost, apply it for a
few steps, repeat. Write the cost and the sampling scheme yourself. The
complete loop, verified to bring the hand within 8 mm of a target 0.29 m
away in 4 s of simulated time:

```python
import numpy as np, mujoco
from mujoco import rollout
m = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
hand = m.body("hand").id
target = d.xpos[hand] + np.array([0.0, 0.25, -0.15])
K, horizon, sigma = 64, 100, 0.1  # samples, sim steps (0.2 s), noise in rad
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS
rng = np.random.default_rng(0)
pool = [mujoco.MjData(m) for _ in range(8)]  # one MjData per thread; use your core count
scratch = mujoco.MjData(m)
lo, hi = m.actuator_ctrlrange.T
nominal = d.ctrl.copy()
s0 = np.zeros(mujoco.mj_stateSize(m, FULL))
 
def cost(final_states):
    c = np.zeros(len(final_states))
    for k, s in enumerate(final_states):
        mujoco.mj_setState(m, scratch, s, FULL); mujoco.mj_kinematics(m, scratch)
        c[k] = np.linalg.norm(scratch.xpos[hand] - target) + 0.05 * np.linalg.norm(scratch.qvel)
    return c
 
while d.time < 4.0:
    mujoco.mj_getState(m, d, s0, FULL)
    cand = nominal + rng.normal(0, sigma, (K, m.nu)); cand[0] = nominal  # keep the incumbent
    cand = np.clip(cand, lo, hi)
    ctrl = np.repeat(cand[:, None, :], horizon, axis=1)  # (K, horizon, nu)
    states, _ = rollout.rollout(m, pool, np.tile(s0, (K, 1)), ctrl)  # (K, horizon, nstate)
    nominal = cand[np.argmin(cost(states[:, -1]))]
    d.ctrl[:] = nominal
    for _ in range(5): mujoco.mj_step(m, d)  # replan at 100 Hz
print(np.linalg.norm(d.xpos[hand] - target))
```

A lesson from getting this to work: sampling independent noise at every
knot of a spline stalled 11 cm from the target, while sampling one held
setpoint per candidate converged. Reproduce that failure, then explain
it. Extensions that are yours: a spline parameterization done properly,
a cost with an obstacle, and pushing the Phase 2.5 box to a goal, where
the planner has to discover contact.

## Exercise 2 — linearize and check

```python
A = np.zeros((2*m.nv, 2*m.nv)); B = np.zeros((2*m.nv, m.nu))
mujoco.mjd_transitionFD(m, d, 1e-6, True, A, B, None, None)   # x_next ~ A x + B u
print(np.abs(np.linalg.eigvals(A)).max())
```

Compute A and B at the home pose, then (a) verify them by perturbing the
state and comparing one mj_step against the linear prediction, (b)
design an LQR gain with scipy.linalg.solve_discrete_are for your
torque-mode model and compare its disturbance rejection with your PD
controller, (c) repeat the linearization with the gripper holding the
box and look at how the contact changes A. Part (c) is the
local-sensitivity counterpart to the global Sobol analysis in Phase 4.

**Optional:** install MuJoCo MPC (MJPC) and run its Panda task to see
predictive sampling and iLQG at production quality. Treat it as a
reference, not as your implementation.

### Claude Code in this phase — Pair

**Skill: profiling and diagnosis.** The agent measures where time and
cost go in your planner; you decide what to change.

- “Profile my predictive-sampling loop. How much time is rollout, how
  much is my cost function? Don't optimize anything yet.”

- “Reproduce the stall I get with independent per-knot noise. Plot the
  cost distribution of the candidates at each iteration and tell me what
  you see.”

- “Write a test that perturbs the state by 1e-4, compares one mj_step
  against my A and B from mjd_transitionFD, and reports which state
  components dominate the residual.”

- “Plot LQR against PD disturbance rejection for the same 0.2 rad kick,
  same axes.”

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>You write</strong></th>
<th><strong>Claude Code builds</strong></th>
</tr>
<tr class="odd">
<th><p>controllers/mpc.py: cost, sampling scheme, horizon</p>
<p>The LQR weights</p>
<p>The explanation of the stall</p></th>
<th><p>Profiling, plots, the linearization test</p>
<p>A target generator and obstacle scene helpers</p>
<p>Review of your planner for off-by-one errors in state
indexing</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

> **Gate.** Predictive sampling reaches within 1 cm of three different
> targets; the residual between the linear prediction and one true step
> is measured and explained (reference: 2.7e-5 at the home pose for a
> 1e-4 perturbation; check whether the fingers resting at their joint
> limit account for it); LQR-versus-PD comparison plot saved. Commit:
> phase2.75: predictive sampling + LQR.

# Phase 3 — RL pipeline with Gymnasium and SB3 (weeks 6–7)

**Goal:** a working, reproducible RL pipeline on a push task. It proves
competence and gives you a closed-loop policy for the capstone. Do not
polish it. **Mode: Delegate.**

1.  Install with pins: pip install gymnasium
    "stable-baselines3\[extra\]", then confirm python -c "import mujoco;
    print(mujoco.\_\_version\_\_)" still prints 3.9.0.

2.  Write ENV_SPEC.md yourself (one page): observation, action scaling,
    control rate, reward terms, termination, episode length, reset
    distribution. Decide the action space deliberately: joint-position
    targets through the servos are the easy default; Cartesian deltas
    through your Phase 2 IK learn faster and reuse your work.

3.  The agent proposes a layout in plan mode, you approve, it
    implements. The skeleton below passed Gymnasium's check_env on the
    Menagerie scene; your env adds the box, the goal and your reward
    from rewards/push.py.

```python
class PandaReachEnv(gym.Env):
    def __init__(self, xml, frame_skip=10, max_steps=200):
        self.model = mujoco.MjModel.from_xml_path(xml); self.data = mujoco.MjData(self.model)
        self.frame_skip, self.max_steps = frame_skip, max_steps  # 10 x 0.002 s = 50 Hz control
        self.hand = self.model.body("hand").id
        self.lo, self.hi = self.model.actuator_ctrlrange[:7].T
        self.action_space = spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (20,), np.float64)
    def _obs(self):
        return np.concatenate([self.data.qpos[:7], self.data.qvel[:7],
                               self.data.xpos[self.hand], self.target])
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.target = np.array([0.55, 0.0, 0.45]) + self.np_random.uniform(-0.15, 0.15, 3)
        mujoco.mj_forward(self.model, self.data); self.t = 0
        return self._obs(), {}
    def step(self, action):
        a = np.clip(action, -1, 1)
        self.data.ctrl[:7] = self.lo + (a + 1) * 0.5 * (self.hi - self.lo)
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        self.t += 1
        dist = np.linalg.norm(self.data.xpos[self.hand] - self.target)
        return self._obs(), -dist, bool(dist < 0.02), self.t >= self.max_steps, {"dist": dist}
```

4.  Train. This block is standard SB3 usage but was **not run here**;
    have the agent verify it and write train.py / eval.py with YAML
    configs:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
make = lambda: PandaPushEnv("models/panda_push.xml")
venv = make_vec_env(make, n_envs=12, vec_env_cls=SubprocVecEnv)
model = PPO("MlpPolicy", venv, verbose=1, tensorboard_log="runs/", seed=0)
model.learn(total_timesteps=2_000_000); model.save("checkpoints/ppo_push")
```

5.  Reward engineering is where your hours go. Start
    sparse-plus-distance, watch rollouts early, and add terms only in
    response to a failure you observed. Then train SAC with the same
    reward and compare sample efficiency.

6.  Use your Phase 2.75 planner as a baseline on the same task: success
    rate and time-to-goal for predictive sampling against PPO. That one
    table is more interesting than the policy alone.

### Claude Code in this phase — Delegate

**Skill: spec → plan → implement, and long-running tasks.** Write the
intent, review the plan, let it build, verify the result.

- “Read docs/ENV_SPEC.md. In plan mode, propose the env and training
  layout and interfaces. No code until I approve.”

- “Implement the plan. rewards/push.py is mine; import it, don't edit
  it. Add a test that runs gymnasium's check_env and checks that a fixed
  seed reproduces the first-episode return.”

- “Start PPO training in the background with config A. When it finishes,
  summarize the reward curve and episode length, and show me 3 frames
  from the eval video.”

- “I suspect reward hacking. Pull the 5 highest-reward rollouts and
  describe what the arm is actually doing, frame by frame.”

- “Run my Phase 2.75 planner and the PPO policy on the same 100 start
  states and make one table: success rate and time to goal.”

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>You write</strong></th>
<th><strong>Claude Code builds</strong></th>
</tr>
<tr class="odd">
<th><p>docs/ENV_SPEC.md</p>
<p>rewards/push.py, every term</p>
<p>The choice of action space, and PPO versus SAC</p>
<p>Reading the curves and spotting reward hacking</p></th>
<th><p>envs/ from your spec, with the check_env test</p>
<p>train.py, eval.py, YAML configs, seeding, TensorBoard, eval
videos</p>
<p>Background training runs and their summaries</p>
<p>The planner-versus-policy comparison script</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

**Reading:** MR Chapter 10 (motion planning), full read, during the
training runs. Sutton and Barto chapter 13 (policy gradients) if PPO
feels opaque.

> **Gate.** check_env passes; a fixed seed reproduces the first-episode
> return; eval.py reports success over 100 episodes;
> planner-versus-policy table saved. Tag the commit v0.3-rl-pipeline.

# Phase 4 — Capstone: sensitivity-guided sim-to-real (weeks 8–12)

**Thesis:** most domain randomization spreads uniform noise over
hand-picked ranges. A variance-based sensitivity analysis shows which
physical parameters the task outcome depends on, and randomizing those
deliberately buys more robustness per training step. **Mode:
Orchestrate.** You own the design, the ranges, the interpretation and
the report.

The order differs from v3. The sensitivity analysis comes first and runs
on CPU; MJX and training come after you already have a ranking in hand.

## Stage A — global sensitivity analysis on CPU (week 8)

Sobol analysis needs N·(D + 2) rollouts. For D = 8 parameters and N =
1024 that is 10,240 rollouts. A scripted grasp-and-lift through the
position servos is an open-loop ctrl sequence, which is exactly what
mujoco.rollout runs in parallel. On two threads in my check it ran about
100,000 steps/s on the Panda-plus-cube scene; your 16-core laptop should
finish the whole analysis in a few minutes, in float64, on the reference
engine.

1.  Install: pip install SALib (1.6.0 was used here).

2.  Use Menagerie's mjx_single_cube.xml as the scene for everything in
    Phase 4. It loads in CPU MuJoCo as well as MJX, so the analysis and
    the training share one model. Note what it changes relative to
    panda.xml: dt 0.005, 5 solver iterations, 8 line-search iterations,
    simplified finger collision geometry, eulerdamp disabled. Rerun your
    Week A creep and penetration checks on its finger pads before
    trusting it.

3.  Write SA_DESIGN.md yourself. For each parameter: the model field,
    the range, and the physical justification, citing your contact-lab
    page. Express contact parameters in physical terms (time constant
    relative to dt, damping ratio, impedance) and never let timeconst
    fall below 2·dt.

4.  Define the quantity of interest. Use a continuous one (final box
    height, or peak slip distance) for the Sobol indices and report the
    success fraction separately; a binary output makes the variance
    decomposition noisy.

5.  The agent builds the pipeline. The reference below ran end to end
    here: the nominal grasp lifts the box to 0.305 m, and a small N = 64
    run (384 rollouts, 7 s on two threads) already ranked box mass (ST
    0.86 ± 0.26) and finger gain (ST 0.60 ± 0.27) far above friction and
    arm damping (ST 0.04 each) over these deliberately wide ranges.

```python
import copy, numpy as np, mujoco
from mujoco import rollout
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
 
m0 = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/mjx_single_cube.xml")
d0 = mujoco.MjData(m0); FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS
grip, home = m0.key("pickup").id, m0.key("home").id
mujoco.mj_resetDataKeyframe(m0, d0, grip); mujoco.mj_forward(m0, d0)
s0 = np.zeros(mujoco.mj_stateSize(m0, FULL)); mujoco.mj_getState(m0, d0, s0, FULL)
 
dt = m0.opt.timestep; n_close, n_lift, n_hold = int(.5/dt), int(1.5/dt), int(.5/dt)
c_grip, c_home = m0.key_ctrl[grip].copy(), m0.key_ctrl[home].copy()
ctrl = np.tile(c_grip, (n_close + n_lift + n_hold, 1)); ctrl[:, 7] = 0.0  # close gripper
a = np.linspace(0, 1, n_lift)[:, None]
ctrl[n_close:n_close+n_lift, :7] = (1 - a) * c_grip[:7] + a * c_home[:7]       # lift
ctrl[n_close+n_lift:, :7] = c_home[:7]
 
box_g, box_b = m0.geom("box").id, m0.body("box").id
qz = m0.jnt_qposadr[m0.body_jntadr[box_b]] + 2  # box z in qpos
problem = {"num_vars": 4,
           "names": ["box_friction", "box_mass_scale", "finger_kp_scale", "arm_damping_scale"],
           "bounds": [[0.2, 1.5], [0.5, 8.0], [0.3, 1.5], [0.5, 2.0]]}  # yours to justify
 
def make(x):                                   # one perturbed model per sample
    m = copy.copy(m0)
    m.geom_friction[box_g, 0] = x[0]
    m.body_mass[box_b] *= x[1]; m.body_inertia[box_b] *= x[1]
    m.actuator_gainprm[7, 0] *= x[2]; m.actuator_biasprm[7, 1:3] *= x[2]
    m.dof_damping[:7] *= x[3]
    return m
 
X = sobol_sample.sample(problem, 1024, calc_second_order=False)  # N*(D+2) rows
pool = [mujoco.MjData(m0) for _ in range(16)]                                  # threads
Y = np.zeros(len(X)); CH = 16  # chunk size
for i in range(0, len(X), CH):
    models = [make(x) for x in X[i:i+CH]]
    n = len(models)
    states, _ = rollout.rollout(models, pool, np.tile(s0, (n, 1)), np.tile(ctrl, (n, 1, 1)))
    Y[i:i+CH] = states[:, -1, 1 + qz]                     # state = [t, qpos, qvel]
S = sobol.analyze(problem, Y, calc_second_order=False, num_resamples=1000)
for name, s1, st, ci in zip(problem["names"], S["S1"], S["ST"], S["ST_conf"]):
    print(f"{name:20s} S1 {s1:5.2f}   ST {st:5.2f} +/- {ci:.2f}")
```

> **Memory trap.** Each copy of this model is about 40 MB because of the
> meshes. Building 128 copies at once exhausted 8 GB in my run. Keep the
> chunk small (16 copies is about 640 MB) as shown.

6.  Convergence: rerun at N = 256, 512, 1024, 2048 and plot each ST with
    its confidence interval against N. Stop when the ranking is stable
    and the intervals do not overlap between the groups you intend to
    separate.

7.  Sanity checks: each S1 ≤ ST within noise; the sum of S1 is at most
    about 1; a dummy parameter that changes nothing has ST near 0. A
    large gap between ST and S1 means interactions, which is itself a
    finding.

8.  Closed-loop version: repeat with your Phase 3 push policy as the
    controller. Step in chunks of one control period: batch policy
    inference, then rollout for frame_skip steps from the batch of
    states, and repeat. Compare the two rankings; whether a feedback
    policy changes which parameters matter is a result worth reporting.

## Stage B — MJX and a trained baseline (week 9)

1.  Install the JAX line of Playground with a pin, then confirm the
    engine did not move:

```bash
pip install "playground==0.2.0" "mujoco==3.9.0" "mujoco-mjx==3.9.0"
python -c "import mujoco, jax; print(mujoco.__version__, jax.devices())"  # 3.9.0, CudaDevice
```

2.  Learn the functional style on the raw API before using any
    framework. Verified on CPU JAX:

```python
import jax, mujoco
from mujoco import mjx
mx = mjx.put_model(m0); dx = mjx.put_data(m0, d0)
step = jax.jit(mjx.step)
dx1 = step(mx, dx)  # first call compiles (tens of seconds)
 
mus = jax.numpy.linspace(0.3, 1.2, 8)                        # batch over a model field
def one(mu):
    mm = mx.replace(geom_friction=mx.geom_friction.at[:, 0].set(mu))
    return mjx.step(mm, dx).qpos
out = jax.jit(jax.vmap(one))(mus)                            # shape (8, 16)
```

3.  Equivalence tests, written by the agent before anything else.
    Measured differences between MJX (float32) and CPU MuJoCo on this
    scene: 8e-8 after one step; 1.5e-7 after 200 steps with the cube at
    rest; but with the cube dropped spinning onto the floor, 7e-5 after
    10 steps and 1.8e-2 after 50. A correct port diverges once contact
    is dynamic. So assert: one step from 100 random states within 1e-5;
    10 steps of dynamic contact within 1e-3; and for long horizons
    compare distributions (final box height over 256 rollouts), not
    trajectories.

4.  Load the Playground baseline and inspect it. Verified: the
    environment loads on the pinned stack with 66 observations, 8
    actions, control dt 0.02 and sim dt 0.005, and its PPO config asks
    for 2,048 environments and 20 million steps.

```python
from mujoco_playground import registry
from mujoco_playground.config import manipulation_params
env = registry.load("PandaPickCube", config_overrides={"impl": "jax"})  # default impl is "warp"
cfg = manipulation_params.brax_ppo_config("PandaPickCube")
print(env.observation_size, env.action_size, env.dt, env.sim_dt, cfg.num_envs, cfg.num_timesteps)
```

5.  Train the unrandomized baseline with Brax PPO. Have the agent start
    from the official training script in the repository's rss2025 tree
    rather than writing one from memory, and verify it against the
    installed 0.2.0 package. **Not run here.** Time one full run before
    planning Stage C. If 2,048 environments do not fit in 8 GB, halve
    num_envs; if a run takes more than about 90 minutes, move training
    to the cluster now rather than in week 11.

6.  Read pick.py in the installed package line by line and write a
    half-page note on how its reset, reward and termination map to your
    ENV_SPEC.md. That reading replaces porting your own env.

## Stage C — naive against targeted randomization (weeks 10–11)

Playground 0.2.0 ships no dynamics randomization for the Panda, so the
randomization function is your contribution. Brax's ppo.train accepts a
randomization_fn (confirmed in the installed version). The pattern,
taken from the package's own LEAP-hand task:

```python
def domain_randomize(model: mjx.Model, rng: jax.Array):  # rng has shape (num_envs, 2)
    @jax.vmap
    def rand(key):
        k1, k2 = jax.random.split(key)
        mu = jax.random.uniform(k1, minval=LO, maxval=HI)
        friction = model.geom_friction.at[BOX_GEOM, 0].set(mu)
        scale = jax.random.uniform(k2, minval=0.5, maxval=2.0)
        mass = model.body_mass.at[BOX_BODY].set(model.body_mass[BOX_BODY] * scale)
        inertia = model.body_inertia.at[BOX_BODY].set(model.body_inertia[BOX_BODY] * scale)
        return friction, mass, inertia
    friction, mass, inertia = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({"geom_friction": 0, "body_mass": 0, "body_inertia": 0})
    model = model.tree_replace(
        {"geom_friction": friction, "body_mass": mass, "body_inertia": inertia})
    return model, in_axes
```

The experiment, which you specify in DR_PROTOCOL.md before any run:

| **Arm**            | **Randomization**                                           | **Purpose**                |
|--------------------|-------------------------------------------------------------|----------------------------|
| None               | Nominal parameters only                                     | Lower bound                |
| Uniform-all        | Every parameter in SA_DESIGN.md, full range                 | The common practice        |
| Targeted           | Only the top-k parameters by total-order index, same ranges | Your method                |
| Inverse (optional) | Only the bottom-k parameters                                | Control: should help least |

- Equal training budget per arm, at least 3 seeds each. That is 9 to 12
  runs, which is why Stage B timed one.

- Rerun the Sobol analysis on PandaPickCube itself with the baseline
  policy before choosing k. The scripted-grasp ranking from Stage A is a
  prediction; check whether it transfers to the learned policy.

- Fix the evaluation set before training: one Sobol or Latin-hypercube
  sample of 1,000 parameter vectors drawn from ranges 25% wider than
  training, stored in a file and never touched again.

- Guard against the obvious confound: targeted DR randomizes fewer
  things, so the task is easier. Report nominal-environment success for
  every arm alongside held-out success.

## Stage D — validation and report (week 12)

1.  Evaluate all arms on the held-out set in MJX, then spot-check 100 of
    those parameter vectors in CPU MuJoCo with the same policy. Float64
    CPU MuJoCo with full solver iterations is the closest thing you have
    to a different "reality"; if a policy's success drops there, say so.

2.  Plots, each regenerated by one command: Sobol bars with confidence
    intervals; convergence of ST against N; success against training
    steps for each arm with seed bands; held-out success as a function
    of the top two parameters (heat map).

3.  Write the report yourself, four pages: question, method, sensitivity
    results, robustness comparison, limitations. State plainly what is
    simulated-to-simulated and what would be needed for a hardware
    claim.

4.  Fresh-context review: a subagent that has not seen the work checks
    the report for unsupported claims, missing error bars and budget
    confounds.

5.  Package the repeated workflow as a project skill
    (.claude/skills/sa-run/SKILL.md) once you have run it by hand twice.

### Claude Code in this phase — Orchestrate

**Skills: equivalence tests for every port, project skills,
fresh-context review, headless runs.** These are the ones that read as
senior-level in interviews.

- “Read docs/SA_DESIGN.md. Build the SALib pipeline with mujoco.rollout:
  Sobol sampling, chunked model copies (at most 16 alive), bootstrap
  confidence intervals, and a convergence plot of ST against N. Add a
  dummy parameter and assert its ST is near zero.”

- “Before any training, write the CPU-versus-MJX equivalence tests at
  three levels: one step from 100 random states, 10 steps of dynamic
  contact, and the distribution of final box height over 256 rollouts.
  Don't continue until they pass.”

- “Start from the official Brax PPO training script for Playground
  0.2.0, not from memory. Verify every call against the installed
  package, then time one full PandaPickCube run and report GPU memory.”

- “Wire my domain_randomize function into training. Don't edit its
  ranges. Launch the three arms in docs/DR_PROTOCOL.md with 3 seeds each
  and equal budgets.”

- “We've now run sample → simulate → analyze → plot by hand twice.
  Package it as .claude/skills/sa-run/SKILL.md so that /sa-run
  \<config\> does the whole thing.”

- “Use a subagent that hasn't seen our work to review report.md for
  unsupported claims, missing error bars, and confounds between
  randomization strategy and training budget.”

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>You write</strong></th>
<th><strong>Claude Code builds</strong></th>
</tr>
<tr class="odd">
<th><p>SA_DESIGN.md, DR_PROTOCOL.md</p>
<p>domain_randomize and its ranges</p>
<p>Interpretation of every index and curve</p>
<p>The report</p></th>
<th><p>SALib pipeline with chunking and bootstrap intervals</p>
<p>Equivalence tests</p>
<p>Training launch scripts, seeds, logging, Slurm files</p>
<p>make figures</p>
<p>/sa-run skill and the fresh-context review</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

> **Gate.** Equivalence tests pass at the three levels above. Sobol
> indices have bootstrap intervals, pass the sanity checks and are
> converged in N. Every figure regenerates from one command. The report
> states the confound controls. The repo carries CLAUDE.md, the tests
> and the skill.

# Pitfall watchlist

## MuJoCo

| **Symptom**                                                                           | **Usual cause and fix**                                                                                            |
|---------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| pip install changed MuJoCo to 4.x, or pulled PyTorch and Warp                         | Unpinned install of mujoco or playground. Restore from requirements.lock.txt; always pass == versions.             |
| Sim explodes with a servo or PD controller                                            | Velocity gain is stiff under Euler or RK4. Switch to implicitfast before touching dt.                              |
| Computed torque is slightly off; test against mj_inverse fails by a fraction of a N·m | Missing qfrc_passive (joint damping) and qfrc_constraint.                                                          |
| Object creeps in a grasp or on a slope it should hold                                 | Soft friction constraint. Use the elliptic cone, raise impratio, or add noslip_iterations; measure, do not guess.  |
| Contacts feel spongy or the object sinks                                              | solref timeconst too large (penetration grows as its square) or solimp dmin too low.                               |
| Contact turns unstable after shortening timeconst                                     | timeconst below 2·dt.                                                                                              |
| Out of memory during the Sobol run                                                    | Too many model copies alive at once (about 40 MB each). Reduce the chunk size.                                     |
| MJX and CPU trajectories disagree after tens of steps                                 | Expected with dynamic contact and float32. Test one step tightly and compare statistics for long horizons.         |
| Predictive sampling stalls short of the goal                                          | Noise is averaging out across knots. Sample held setpoints or a low-dimensional spline; always keep the incumbent. |
| Sobol indices negative or S1 greater than ST                                          | N too small, or a binary output. Increase N, use a continuous quantity of interest.                                |

## Working with Claude Code

| **Symptom**                                               | **Usual cause and fix**                                                                                                    |
|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Agent uses a wrong or removed MuJoCo function             | Stale knowledge (old signatures, or MuJoCo 4 habits later). Make it confirm with help() on the installed 3.9.0 before use. |
| Agent runs pip install -U or installs an unpinned package | Pins not stated or prompt too loose. Keep install prompts on; restore from requirements.lock.txt.                          |
| Instability "fixed" by a smaller timestep or more damping | Agent tuned physics silently. Reject the diff and ask for the root cause.                                                  |
| Tests pass, physics still wrong                           | Tests checked your code against itself. Assert against MuJoCo ground truth or an analytic limit only.                      |
| Agent edited controllers/ or models/                      | Boundary not stated or prompt too broad. Revert with git, restate the rule, scope the prompt to a folder.                  |
| Agent ran the experiment before you predicted the answer  | You lost the learning. Add the prediction to the prompt itself: "I predict X; test it."                                    |
| Answers get vaguer over a long session                    | Context is full of old detail. /clear between tasks; continuity lives in PROGRESS.md.                                      |
| "It works, but I can't explain it"                        | Too much delegated. Do the explain-back quiz; rewrite that piece yourself before moving on.                                |
| Agent claims success without running anything             | No gate defined. Ask it to run the tests and show the output or the rendered frames.                                       |

# After this plan

- **Port the capstone to MuJoCo 4, MuJoCo Warp and Playground 1.0.** The
  new stack is PyTorch with RSL-RL, includes a system-identification
  module (mujoco.sysid) and sim-and-real task definitions, and is where
  DeepMind and NVIDIA are converging. Rerunning your study there, with
  the equivalence-test pattern, is a strong second artifact. Use a
  separate environment (robo4); do not upgrade robo.

- **System identification.** The natural sequel to a sensitivity study:
  fit the high-sensitivity parameters to trajectories, then randomize
  around the fit.

- **Diffusion Policy / LeRobot** if embodied-AI roles lead your list.
  **Isaac Lab** if NVIDIA roles do.

- **Claude Code in CI** on the capstone repo: tests and review on every
  pull request.

# Sources for the ecosystem changes

- MuJoCo 4 announcement (7 Nov 2026):
  github.com/google-deepmind/mujoco/discussions/3773

- MuJoCo releases, 4.0.0 and 4.0.1 notes:
  github.com/google-deepmind/mujoco/releases

- MuJoCo Playground 1.0: playground.mujoco.org and
  pypi.org/project/playground/1.0.1/

- MuJoCo Playground technical report (2025, the JAX/MJX line):
  arxiv.org/abs/2502.08844

- Package contents and dependencies of playground 0.2.0 and 1.0.1 were
  inspected directly from PyPI.
