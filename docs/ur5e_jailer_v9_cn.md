# UR5e JAiLeR V9：完整任务排空与独立姿态残差

## 版本目标

V9 不覆盖 V8。它从已经完成的 P6 位置 checkpoint 中只导入位置 Actor
和状态归一化统计，冻结位置 Actor，重新初始化姿态残差 Actor、残差
`log_std`、Critic 和 Adam。V8 中在统一物理缩放下学到的姿态残差不会导入
V9。

V9 的动作关系为：

```text
归一化执行动作 = clip(确定性位置动作 + 姿态增益 × 随机姿态残差, -1, 1)
```

位置动作保持 P6 训练时的物理缩放 1.0。姿态增益在每个 100 Hz 控制步由
当前 TCP 最短旋转角误差确定，并在相邻锚点间线性插值：

| 角度误差 | 0° | 5° | 10° | 16° | 24° | 35° | 45°及以上 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 残差增益 | 0.10 | 0.10 | 0.15 | 0.25 | 0.40 | 0.65 | 1.00 |

原课程的 `action_velocity_scale` 在 V9 中只作为姿态残差增益上限，不再缩放
冻结的位置动作。PPO 保存和评价的是姿态残差动作的概率；位置动作与姿态增益
都是状态确定的控制映射，因此 PPO 的 `tanh` Jacobian 仍只作用于随机残差。

## Rollout 时间边界

每个阶段的 `rollout_steps_per_slot` 同时是名义采集边界和单任务最大控制步数。
V9 在名义边界以前允许成功或失败的槽位开始新任务；到达名义边界后停止创建
新任务，并让已经开始的任务继续运行到真实成功、真实超时、碰撞、越界或非法
状态。40 个槽位全部得到真实结果后才计算 GAE 并执行一次 PPO 更新。

因此 V9 满足：

- `rollout_truncated_task_count` 恒为 0；
- 每条被用于 PPO 的控制序列最终都有真实任务结果；
- 任务结束处同时停止 GAE 递推和 Critic bootstrap；
- rollout 的 transition 数量允许大于名义的“槽位数 × 阶段步数”；
- 不再用只统计提前完成任务的成功率推动课程晋级。

新增日志包括排空 transition 数、排空控制步、姿态残差增益均值/P95、位置动作
幅值、原始及缩放后姿态残差幅值、合成动作裁剪率和实际物理缩放。

## V8 O5/O6 固定任务验证

验证使用相同初始关节池、相同随机目标、确定性策略，并让每个首任务执行到真实
终点；每组 120 个任务。完整机器可读结果位于
`artifacts/jailer_v8/o5_vs_o6_fixed_tasks_before_v9.json`。

| 测试范围 | checkpoint | 成功率 | 位置均值 | 姿态均值 |
|---|---|---:|---:|---:|
| O1 | O5完成 | 25.8% | 2.90 mm | 1.60° |
| O1 | 最终O6 | 15.0% | 3.41 mm | 1.98° |
| O5 | O5完成 | 60.0% | 19.28 mm | 11.12° |
| O5 | 最终O6 | 67.5% | 9.00 mm | 6.69° |
| O6 | O5完成 | 54.2% | 66.42 mm | 28.25° |
| O6 | 最终O6 | 52.5% | 53.95 mm | 18.52° |

O5 对实际相关的 O1 小范围任务保留了更高精度；最终 O6 在 O5 范围更强，但
小范围精度下降。O6 宽范围中两者都有明显失败长尾，因此 V9 同时修正动作合成
和任务统计边界。

## 建议先运行100-update试验

```bash
cd /home/sjr/PPO_caging_capture
conda activate camera

python -m cage_system.cli train-jailer \
  --config configs/ur5e_jailer_curriculum_v9_pilot.yaml \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --initial-pose-pool artifacts/jailer_v1/initial_joint_pool.npz \
  --actor-warm-start-checkpoint artifacts/jailer_v6/jailer_ppo_checkpoints/stage_position_p06_006_150mm_complete.pt \
  --actor-warm-start-stage pose_bridge_b01_006_025mm_12deg \
  --output artifacts/jailer_v9_pilot/jailer_ppo.pt \
  --device auto
```

重点检查 `scale=1.00`、`gain≈0.10`、`cut=0`、`drain`、MOVE50、位置
P95和 `composed_action_clip_fraction`。这是新的动作动力学，不能只根据V8的结果
直接假定稳定。

## 正式训练

```bash
cd /home/sjr/PPO_caging_capture
conda activate camera

python -m cage_system.cli train-jailer \
  --config configs/ur5e_jailer_curriculum_v9.yaml \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --initial-pose-pool artifacts/jailer_v1/initial_joint_pool.npz \
  --actor-warm-start-checkpoint artifacts/jailer_v6/jailer_ppo_checkpoints/stage_position_p06_006_150mm_complete.pt \
  --actor-warm-start-stage pose_bridge_b01_006_025mm_12deg \
  --output artifacts/jailer_v9/jailer_ppo.pt \
  --device auto
```

V9 不应从 V8 的 O/B checkpoint 做完整 resume，因为动作和 rollout 契约均已变化。
V9 自己中断后，可以使用完全状态恢复：

```bash
python -m cage_system.cli train-jailer \
  --config configs/ur5e_jailer_curriculum_v9.yaml \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --initial-pose-pool artifacts/jailer_v1/initial_joint_pool.npz \
  --resume-checkpoint artifacts/jailer_v9/jailer_ppo_checkpoints/latest.pt \
  --output artifacts/jailer_v9/jailer_ppo.pt \
  --device auto
```

每次新训练或恢复都会在 checkpoint 目录的 `run_snapshots/start_*` 下保存解析后的
完整配置、关键训练源文件副本、输入文件 SHA-256、Python/PyTorch/NumPy版本和
运行契约。V1–V8 没有完整源码快照；该可复现机制从 V9 开始生效。
