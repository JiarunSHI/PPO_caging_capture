# 曲面增强 Cage + PPO 端到端流程

当前网络输入仍为 20 维：严格位于当前时刻之前的五帧，每帧包含 Base 下目标中心 `xyz` 与线速度大小。漏检或被卡尔曼门控拒绝的测量使用因果预测态补齐；真实因果曲率或五帧总转角超限时不调用 Cage/PPO，继续积累到新的合格五帧。

标签生成只保留稳定正法向分支。IK 使用 ROS-Industrial UR5e 解析多分支作为真实 MJCF 数值修正的初值，并在必要时执行确定性多初值搜索。IK 和碰撞只作为硬可行性门；所有可行候选复用相同三组扰动，每组 640 个 EST 节点，最终严格选择三次内部节点比例均值最大的候选。交点只限制候选劣弧，不限制 EST 树的动力学状态空间。

Cage 网络训练前使用五帧历史自身构造因果局部参考系，去除无意义的全局平移和运动平面旋转变化；网络仍然是 20 维输入、6 维相对位姿输出，推理结果会还原为 Base 下六维 TCP。该参考系不读取当前观测或未来曲线。

## 已跑通的小规模集成流程

```bash
mkdir -p artifacts/e2e_curved_v2

python -m cage_system.cli generate-trajectories \
  --config configs/e2e_curved_cage.yaml \
  --count 20 \
  --output artifacts/e2e_curved_v2/cage_trajectories.npz

python -m cage_system.cli generate-labels \
  --config configs/e2e_curved_cage.yaml \
  --input artifacts/e2e_curved_v2/cage_trajectories.npz \
  --output artifacts/e2e_curved_v2/cage_labels_geometric.npz \
  --audit artifacts/e2e_curved_v2/cage_labels_audit.jsonl.gz \
  --workers 8

python -m cage_system.cli filter-est-empty-labels \
  --input artifacts/e2e_curved_v2/cage_labels_geometric.npz \
  --audit artifacts/e2e_curved_v2/cage_labels_audit.jsonl.gz \
  --output artifacts/e2e_curved_v2/cage_labels_geometric_nonzero.npz

python -m cage_system.cli generate-vision-dataset \
  --config configs/e2e_curved_cage.yaml \
  --trajectories artifacts/e2e_curved_v2/cage_trajectories.npz \
  --labels artifacts/e2e_curved_v2/cage_labels_geometric_nonzero.npz \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --output artifacts/e2e_curved_v2/cage_labels_vision.npz \
  --audit artifacts/e2e_curved_v2/vision_dataset_audit.jsonl.gz \
  --device auto

python -m cage_system.cli train-cage \
  --config configs/e2e_curved_cage.yaml \
  --dataset artifacts/e2e_curved_v2/cage_labels_vision.npz \
  --output artifacts/e2e_curved_v2/cage_point20.pt \
  --device auto

python -m cage_system.cli generate-trajectories \
  --config configs/e2e_curved_ppo.yaml \
  --count 12 \
  --output artifacts/e2e_curved_v2/ppo_trajectories.npz

python -m cage_system.cli train-ppo \
  --config configs/e2e_curved_ppo.yaml \
  --cage-checkpoint artifacts/e2e_curved_v2/cage_point20.pt \
  --trajectories artifacts/e2e_curved_v2/ppo_trajectories.npz \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --output artifacts/e2e_curved_v2/ppo2.pt \
  --device auto
```

## 生产规模

生产 Cage 和 PPO 必须使用不同随机种子和不同轨迹归档：

```bash
mkdir -p artifacts/production_curved

python -m cage_system.cli generate-trajectories \
  --config configs/production_curved_cage.yaml --count 1000 \
  --output artifacts/production_curved/cage_trajectories.npz

python -m cage_system.cli generate-labels \
  --config configs/production_curved_cage.yaml \
  --input artifacts/production_curved/cage_trajectories.npz \
  --output artifacts/production_curved/cage_labels_geometric.npz \
  --audit artifacts/production_curved/cage_labels_audit.jsonl.gz \
  --workers 16
```

中断后重复同一条标签命令会从逐轨检查点继续。完成标签后，按照上面的小规模流程依次执行视觉重建和 Cage 训练。随后用 `configs/production_curved_ppo.yaml` 另生成 500 条轨迹，并执行 PPO 训练。完整 PPO 日程为 2048 个 transition/rollout、500 次更新。

## 随机十标签可视化

```bash
python -m cage_system.cli visualize-random-cage-segment \
  --config configs/e2e_curved_cage.yaml \
  --trajectories artifacts/e2e_curved_v2/cage_trajectories.npz \
  --labels artifacts/e2e_curved_v2/cage_labels_geometric_nonzero.npz \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --track-id -1 --pose-count 10 --seed 20260824 \
  --output artifacts/e2e_curved_v2/random_track_10_cage_labels.png
```
