# UR5e JAiLeR V10 完整三段训练

V10 从随机参数开始，在同一个进程和 checkpoint 链中自动完成三段课程：

1. `position_masked_p01` 至 `position_masked_p06`：姿态误差输入严格置零，只学习位置到达。
2. `position_observed_p01` 至 `position_observed_p06`：恢复真实18维输入。边界处保留位置 Actor，清零 Actor 第一层的姿态输入列，重置旧 Critic 和全部 Adam 状态，姿态输入标准差初始化为10度。
3. `pose_bridge_b01` 至 `pose_bridge_b06`，随后 `pose_o01` 至 `pose_o06`：冻结位置 Actor，只训练姿态残差。

第二轮 P6 进入 B1 时保留 Actor 和 Critic 权重，清空全部 Adam 历史，并使用5个完整 B1 rollout 只对齐 Critic。完成对齐后才开始 Actor/Critic 联合 PPO。B1 之后的所有课程晋级连续继承 Actor、Critic、Adam 和归一化统计。

## Rollout边界

P、B、O 全部使用同一排空契约：到达名义 rollout 边界后停止创建新任务，已经启动的任务继续运行至成功、任务超时、碰撞、工作空间越界或数值错误。每个进入 GAE 的任务序列都有真实结果，`rollout_truncated_task_count` 必须为0。

## 动作缩放

位置动作的物理缩放始终为1。姿态阶段的执行动作是冻结位置动作与姿态残差之和，只有姿态残差根据当前姿态误差动态缩放。误差/增益锚点为：0/5/10/16/24/35/45度对应0.10/0.10/0.15/0.25/0.40/0.65/1.00，中间连续线性插值，并受当前课程的上限约束。

## 奖励审计

每个控制步都记录：位置/姿态质量、位置前进奖励、姿态前进奖励、`quality-1`逐步时间成本、关节加速度惩罚、成功奖励和约束惩罚。位置阶段的位置前进奖励覆盖完整距离范围；姿态阶段的位置前进奖励只在目标附近启用，避免压制姿态修正。

成功时额外奖励固定为 `+1.0`；碰撞或工作空间越界每项惩罚为 `-2.0`。未成功的每一步通过 `quality-1` 累积非正时间成本，因此越早达到终止条件越好。

## MOVE50/MOVE100

MOVE50和MOVE100分别由真实MuJoCo正运动学位置在5个和10个100 Hz控制间隔内计算。训练历史记录总体均值和P95；SQLite按当前位置误差和姿态误差累计分箱，不保存海量原始transition。每100个update更新：

- `move50_move100_vs_error.csv.gz`
- `move50_move100_vs_error.pdf`

图中给出各训练阶段MOVE50/MOVE100均值及一个标准差随误差的变化。

## 课程阈值

两轮位置P1--P6使用完全相同的任务范围和阈值：

|阶段|初始位置范围|任务上限|最少update|单任务成功位置|晋级位置均值|最近10轮成功率|
|---|---:|---:|---:|---:|---:|---:|
|P1|6--20 mm|60|100|1.5 mm|2.5 mm|至少80%|
|P2|6--35 mm|60|100|2.0 mm|3.0 mm|至少80%|
|P3|6--55 mm|70|100|2.5 mm|3.5 mm|至少80%|
|P4|6--80 mm|80|100|3.0 mm|4.5 mm|至少80%|
|P5|6--110 mm|100|150|3.5 mm|5.5 mm|至少80%|
|P6|6--150 mm|130|150|4.0 mm|8.0 mm|严格大于80%|

|阶段|初始范围（位置/姿态）|任务上限|最少update|单任务成功（位置/姿态）|晋级均值（位置/姿态）|最近10轮成功率|
|---|---:|---:|---:|---:|---:|---:|
|B1|6--25 mm / 1--5°|80|100|6.0 mm / 12°|6.0 mm / 7°|>80%|
|B2|6--25 mm / 1--5°|90|125|5.0 mm / 8°|5.5 mm / 6°|>80%|
|B3|6--25 mm / 1--5°|100|150|4.0 mm / 4°|5.0 mm / 4.5°|>80%|
|B4|6--25 mm / 1--5°|110|150|4.0 mm / 3.5°|4.5 mm / 3.7°|>80%|
|B5|6--25 mm / 1--5°|120|150|3.5 mm / 3.0°|4.0 mm / 3.2°|>80%|
|B6|6--25 mm / 1--5°|130|175|2.8 mm / 2.2°|3.2 mm / 2.5°|>80%|
|O1|6--25 mm / 1--5°|80|150|2.0 mm / 1.5°|2.5 mm / 2.5°|>80%|
|O2|6--40 mm / 1--10°|90|150|2.5 mm / 2.5°|3.0 mm / 3.5°|>80%|
|O3|6--60 mm / 1--16°|110|175|3.0 mm / 3.5°|3.5 mm / 4.5°|>80%|
|O4|6--90 mm / 1--24°|130|175|3.5 mm / 4.5°|4.5 mm / 5.5°|>80%|
|O5|6--120 mm / 1--35°|150|200|4.0 mm / 5.5°|5.5 mm / 7.0°|>80%|
|O6|6--150 mm / 1--45°|180|200|4.5 mm / 7.0°|6.5 mm / 9.0°|>80%|

位置任务还要求TCP线速度不超过0.08 m/s；姿态任务同时要求线速度不超过0.08 m/s、角速度不超过0.30 rad/s。所有表中的任务上限单位均为100 Hz控制步。

## 从随机参数开始正式训练

```bash
cd /home/sjr/PPO_caging_capture
conda activate camera

python -m cage_system.cli train-jailer \
  --config configs/ur5e_jailer_curriculum_v10_full.yaml \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --initial-pose-pool artifacts/jailer_v1/initial_joint_pool.npz \
  --output artifacts/jailer_v10_full/jailer_ppo.pt \
  --device auto
```

命令不得添加 `--resume-checkpoint` 或 `--actor-warm-start-checkpoint`，因此 Actor、Critic 和 Adam 均从新随机参数开始。最大update为1,000,000，仅作为安全上限；最终O6满足连续10个update的晋级条件后自动保存并终止。

中断后才使用：

```bash
python -m cage_system.cli train-jailer \
  --config configs/ur5e_jailer_curriculum_v10_full.yaml \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --initial-pose-pool artifacts/jailer_v1/initial_joint_pool.npz \
  --resume-checkpoint artifacts/jailer_v10_full/jailer_ppo_checkpoints/latest.pt \
  --output artifacts/jailer_v10_full/jailer_ppo.pt \
  --device auto
```

## 旧模型推理视频

`render-jailer-video` 会根据旧checkpoint内保存的课程定义复现旧动作缩放，不使用V10动作解释旧模型。默认建议使用OSMesa无窗口渲染：

```bash
MUJOCO_GL=osmesa python -m cage_system.cli render-jailer-video \
  --config configs/ur5e_jailer_curriculum_8000.yaml \
  --checkpoint artifacts/jailer_v8/jailer_ppo_checkpoints/best.pt \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --initial-pose-pool artifacts/jailer_v1/initial_joint_pool.npz \
  --stage pose_o05_006_120mm_01_35deg \
  --slot-id 0 \
  --seed 20260901 \
  --control-steps 600 \
  --video-fps 25 \
  --output artifacts/jailer_v8/best_o5_inference_6s.mp4 \
  --device auto
```
