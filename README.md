# PPO Caging Capture

Dynamic Cage reconstruction and hybrid Jacobian–PPO tracking of a moving object with a UR5e and an eye-in-hand RGB-D camera, implemented in MuJoCo on Ubuntu.

The final system acquires Cage-selection experience offline with EST dynamics simulation, then reuses a fixed normalized minor-arc position **α = 0.4375** online. The controller combines Jacobian approach with a curriculum-trained PPO policy for local joint-velocity tracking.

## Final workflow

1. **Offline experience:** generate smooth quasiplanar trajectories, construct speed-dependent minor-arc candidates, and evaluate candidates with repeated MuJoCo EST expansion under the real U-shaped gripper geometry. Retain candidate scores and the selected arc coordinate for audit.
2. **Policy training:** learn static TCP reaching through ten masked-position stages, ten observed-position stages, and ten joint-pose stages. Fifty logical slots share the UR5e model while maintaining independent states. Actor and Critic use independent Adam optimizers.
3. **Online reconstruction:** continuously acquire RGB-D images at 10 Hz. Five valid raw visual centers initialize position, velocity, acceleration, and covariance. Kalman correction, causal local spline fitting, and local-plane estimation then reconstruct the Cage using the fixed prior.
4. **Tracking:** Jacobian control approaches the reference; when TCP error is below 15 mm and 45°, PPO is latched for local tracking at 100 Hz. MuJoCo integrates at 500 Hz. Short visual misses use Kalman prediction; three consecutive misses terminate the task.

The final workflow uses simulator RGB-D measurements with perception errors. Generated future points are available only for offline evaluation and simulation setup; online Cage reconstruction consumes causal visual history.

## Installation

Use the existing `camera` environment, or create a compatible environment with Python 3.8 or later. Install the CUDA-enabled PyTorch build appropriate for your machine before training.

```bash
cd /home/sjr/PPO_caging_capture
conda activate camera
pip install -e '.[learning,simulation,visualization,test]'
```

The project contains no ROS runtime requirement. Preserve the UR5e MJCF directory and its relative mesh resources and third-party licenses.

## Offline EST experience

```bash
python -m cage_system.cli generate-trajectories \
    --config configs/ur5e_cage_state_arc_data.yaml \
    --count 3000 \
    --output artifacts/cage_state_arc_v1/trajectories.npz

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m cage_system.cli generate-labels \
    --config configs/ur5e_cage_state_arc_data.yaml \
    --input artifacts/cage_state_arc_v1/trajectories.npz \
    --output artifacts/cage_state_arc_v1/cage_state_arc_labels.npz \
    --audit artifacts/cage_state_arc_v1/cage_state_arc_audit.jsonl.gz \
    --workers 8
```

Label generation resumes per-track checkpoints by default. This geometry-only archive supports the fixed-prior audit; the final pipeline does not train a Cage-pose MLP or prepare a visual training dataset.

## Train the final PPO policy

Use a new output directory to keep the completed research run available.

```bash
python -m cage_system.cli generate-jailer-initial-pool \
    --config configs/ur5e_jailer_joint_v14_full.yaml \
    --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
    --tcp-site attachment_site \
    --count 50 \
    --output artifacts/final_training/initial_joint_pool.npz

python -m cage_system.cli train-jailer \
    --config configs/ur5e_jailer_joint_v14_full.yaml \
    --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
    --tcp-site attachment_site \
    --initial-pose-pool artifacts/final_training/initial_joint_pool.npz \
    --output artifacts/final_training/jailer_joint_ppo.pt \
    --device auto
```

The one-million-update limit is a safety ceiling; training stops when the final curriculum stage is mastered. Stage-promotion checkpoints, periodic checkpoints, `best.pt`, `latest.pt`, training history, and MOVE50/MOVE100 statistics are saved alongside the run.

To resume, repeat the training command and add:

```text
--resume-checkpoint artifacts/final_training/jailer_joint_ppo_checkpoints/latest.pt
```

## Visualize one validation trajectory

The model path below uses the `best.pt` produced by the training command above. Checkpoints and datasets are local artifacts and are not included in Git. To use the existing completed research run instead, replace the checkpoint path with `artifacts/jailer_joint_v14_reward_v2/jailer_joint_ppo_checkpoints/best.pt`.

```bash
python -m cage_system.cli evaluate-dynamic-cage \
    --config configs/ur5e_dynamic_cage_validation_10.yaml \
    --checkpoint artifacts/final_training/jailer_joint_ppo_checkpoints/best.pt \
    --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
    --tcp-site attachment_site \
    --stage joint_pose_j10_006_150mm_045deg \
    --episodes 1 \
    --seed 20261028 \
    --video-output artifacts/final_demo/seed_20261028/tracking.mp4 \
    --trajectory-output artifacts/final_demo/seed_20261028/trajectory.npz \
    --output artifacts/final_demo/seed_20261028/report.json \
    --video-fps 25 \
    --show-gripper \
    --show-window \
    --realtime-display \
    --device auto
```

Seeds **20261020–20261029** select ten profiles spanning spatial position, height, speed, curvature, and turning direction. Keep the same code and configuration to reproduce a seed. Omit the display options for headless evaluation and configure a supported MuJoCo rendering backend. Use `--hide-gripper` to hide only the gripper visualization.

## Code map

| Path | Responsibility |
| --- | --- |
| `cage_system/cli.py` | Seven final commands and their arguments |
| `cage_system/pipeline.py`, `parallel.py` | Offline candidate evaluation and resumable EST data generation |
| `cage_system/cage/` | Motion plane, minor arc, real gripper region, feasibility, EST scoring |
| `cage_system/trajectory/` | Trajectory generation, spline fitting, Kalman and causal state estimation |
| `cage_system/perception/eye_on_hand.py` | Rendered RGB-D detection and camera-to-Base measurement conversion |
| `cage_system/perception/observation_planner.py` | Simulator-only initial camera coverage checking |
| `cage_system/learning/train_jailer.py` | Curriculum, rollouts, training transitions and checkpoints |
| `cage_system/learning/jailer_policy.py`, `ppo2.py` | Policy, value network, normalization, GAE and independent PPO updates |
| `cage_system/sim/` | UR5e dynamics, logical slots, TCP, IK and collision checks |
| `cage_system/control/` | Constrained differential Jacobian controller |
| `cage_system/dynamic_cage_inference.py` | Final visual reconstruction and hybrid tracking loop |
| `configs/` | Five retained final/shared configurations |
| `docs/paper/` | Latest IEEE LaTeX manuscript and retained figures |
| `tests/` | Numerical core and final PPO optimizer checks |
| `artifacts/` | Local datasets, policy run and validation evidence; ignored by Git |

Shared core files still contain compatibility helpers needed by serialized checkpoints or offline array formats. They remain dependencies of the final commands.

## Lightweight checks

```bash
python -m cage_system.cli --help
python -m pytest tests -q
```

The implementation includes third-party UR5e model resources; retain their original attribution and license files. See [the code reading guide](docs/project_guide_cn.md) and [cleanup record](docs/cleanup_manifest.json).
