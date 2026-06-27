# MuJoCo Learning — Phase 1

Phase 1–4 robotics-simulation roadmap on the Franka Emika Panda.
Environment: WSL2 / Ubuntu 24.04, MuJoCo 3.9, JAX (CUDA 12).

## Week 1 — MJCF mechanics on a toy arm

A hand-written 2-DOF planar arm to learn the model file and the canonical
loop before touching the production Franka model.

### Run it

```bash
mamba activate robo          # be in (robo), not (base)
python run.py
```

The arm swings down and double-pendulums under gravity — nothing holds it up
yet. That sag is the Phase 2 problem. Close the viewer window to stop.

### Files

- `arm.xml` — the model. Two bodies, two hinge joints, capsule geoms, plus
  actuators and sensors you'll use in Week 2. Read it top to bottom.
- `run.py` — the canonical loop: load model → make data → (sense → control →
  step → render) every timestep.

### Week 1 milestone — break it and know why

The skill isn't building the arm; it's diagnosing it when it breaks. Make each
change in `arm.xml`, run, read the error, then revert:

1. **Delete `link1_geom`.** → "mass and inertia must be positive." A moving
   body needs mass, and MuJoCo was inferring it from that geom.
2. **Set `timestep="0.05"`** in `<option>`. → the arm explodes / NaNs. dt too
   large for the integrator. This is pitfall #1.
3. **Nest `link2` outside `link1`** (move it up a level). → the kinematic chain
   breaks; the elbow no longer follows the shoulder.
4. **Set the shoulder `range="-10 10"`.** → the swing clamps at the limits.
5. **In `run.py`, read `data.sensordata` before `mj_forward`.** → it reads
   zeros. Sensors are only valid after a forward/step. Pitfall #4.

When you can predict each break before running it, Week 1 is done.

---

## Save your work to GitHub

Commit at every working state — it's a pinned principle, and load-bearing once
Phase 2 PD gains start blowing up the sim. Run the arm FIRST (above); none of
this is needed to run. Then preserve it.

### One-time machine setup (do once, ever — not per project)

**1. Git identity** (needed for any commit):

```bash
git config --global user.name "Yilong Zhou"
git config --global user.email "yilongz393@gmail.com"
git config --global init.defaultBranch main
git config --global core.editor nano
```

**2. SSH key** (needed to push to GitHub):

```bash
ssh-keygen -t ed25519 -C "yilongz393@gmail.com"   # press Enter 3x
cat ~/.ssh/id_ed25519.pub
```

Copy the full output (starts `ssh-ed25519`). In a browser:
GitHub → avatar → Settings → SSH and GPG keys → New SSH key → paste → save.
Then test:

```bash
ssh -T git@github.com        # type "yes" if prompted; a greeting = success
```

### First commit + push (this project)

**3.** Initialize the repo and add a `.gitignore` so junk and the Menagerie
never get committed:

```bash
cd ~/projects/mujoco-learning      # repo root (one level above phase1/)
git init
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.ipynb_checkpoints/
*.mp4
*.png
checkpoints/
runs/
logs/
mujoco_menagerie/
.DS_Store
EOF
```

**4.** Make the GitHub repo: GitHub → New repository → name it
`mujoco-learning` → leave it EMPTY (no README/gitignore, you have them) →
Create.

**5.** Commit and push:

```bash
git add .
git status                         # confirm: no mujoco_menagerie/ listed
git commit -m "phase1: hand-written 2-DOF arm + canonical loop"
git remote add origin git@github.com:yilong-sim/mujoco-learning.git
git branch -M main
git push -u origin main
```

Refresh the GitHub page — your files are there. Backed up.

### The loop from here

After the one-time setup, preserving work is three commands at every working
state:

```bash
git add .
git commit -m "describe what changed"
git push
```

Because your files live inside the WSL container, `git push` IS your real
backup. Commit at the end of each session and your work survives anything that
happens to the VM.

---

## Next (Week 2)

Uncomment the `data.ctrl` line in `run.py` and drive the motors. Add live
sensor reads. Then graduate to the Franka model in Week 3.
