# Dynamic Cage + PPO2

本项目实现“离线生成 Cage 标签、用 eye-on-hand RGB-D 重放生成监督输入、冻结 Cage 后在 UR5e MuJoCo 视觉闭环中训练 PPO2”的两阶段系统。

当前固定接口：目标是中心点质点；Cage 输入为严格过去五帧 `[x,y,z,线速度大小]`，张量形状 `[5,4]`、展平 20 维；输出为当前时刻 Base 下的 Cage TCP 六维位姿。完整未来曲线只用于离线标签，不进入网络。PPO2 状态为 Cage 位姿 6 维、关节角 6 维、关节速度 6 维，动作是 6 维关节增量。

## 建议阅读顺序

1. [项目阅读与实现状态](docs/project_guide_cn.md)
2. [完整算法与数据流](docs/method_design_cn.md)
3. [环境、训练和 ROS 接入](docs/setup_cn.md)
4. [冻结基础策略与残差PPO](docs/residual_rl_pipeline_cn.md)
5. [统一配置](configs/default.yaml)
6. [命令行入口](cage_system/cli.py)

## 快速验证

```bash
conda activate camera
cd /home/sjr/PPO_caging_capture
python -m pytest tests -q
python -m cage_system.cli smoke \
  --config configs/default.yaml \
  --output-dir artifacts/smoke
```

带真实UR5e IK/碰撞和相机的快速测试可使用 `configs/ur5e_vision_smoke.yaml`。

## 完整流水线

`generate-trajectories` 生成的是整体近似平面、局部平面缓慢变化的准平面轨迹。每条轨迹保存前都会硬检查：所有连续五帧均有非零但受限的曲率变化，整条轨迹的局部平面变化非零且不超过配置上限，相邻局部平面不能跳变。命令结束时会输出本批数据的验收统计。

所有正式Cage样本还必须通过统一的历史质量门控：五个历史测量全部被因果滤波器接受，历史五帧到当前标签时刻的任一间隔不超过1.25个视觉周期，因果滚动平滑曲率不超过配置上限。PPO训练和ROS推理复用同一门控；异常窗口不调用Cage/PPO，机器人发布零增量/当前关节保持命令，直到重新积累五个正常历史帧。

```bash
python -m cage_system.cli generate-trajectories \
  --config configs/ur5e_vision_cage_data.yaml \
  --count 3000 \
  --output artifacts/quasiplanar_v1/trajectories.npz

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m cage_system.cli generate-labels \
  --config configs/ur5e_vision_cage_data.yaml \
  --input artifacts/quasiplanar_v1/trajectories.npz \
  --output artifacts/quasiplanar_v1/cage_labels.npz \
  --audit artifacts/quasiplanar_v1/cage_audit.jsonl.gz \
  --workers 8

python -m cage_system.cli generate-vision-dataset \
  --config configs/ur5e_vision_cage_data.yaml \
  --trajectories artifacts/quasiplanar_v1/trajectories.npz \
  --labels artifacts/quasiplanar_v1/cage_labels.npz \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --output artifacts/quasiplanar_v1/cage_labels_vision.npz \
  --audit artifacts/quasiplanar_v1/vision_dataset_audit.jsonl.gz \
  --device auto

python -m cage_system.cli train-cage \
  --config configs/ur5e_vision_cage_train.yaml \
  --dataset artifacts/quasiplanar_v1/cage_labels_vision.npz \
  --output artifacts/quasiplanar_v1/cage_point20.pt \
  --device auto

# PPO must not replay the trajectories used to supervise Cage.  Generate a
# disjoint training bank with a different seed/configuration.
python -m cage_system.cli generate-trajectories \
  --config configs/ur5e_vision_ppo_train.yaml \
  --count 5000 \
  --output artifacts/ppo_train_v1/trajectories.npz

python -m cage_system.cli train-ppo \
  --config configs/ur5e_vision_ppo_train.yaml \
  --cage-checkpoint artifacts/quasiplanar_v1/cage_point20.pt \
  --trajectories artifacts/ppo_train_v1/trajectories.npz \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --output artifacts/ppo_train_v1/ppo2.pt \
  --device auto

# Held-out inference/evaluation trajectories use a third seed.  Never pass
# this archive to train-cage or train-ppo.
python -m cage_system.cli generate-trajectories \
  --config configs/ur5e_vision_ppo_eval.yaml \
  --count 1000 \
  --output artifacts/ppo_eval_v1/trajectories.npz
```

训练完成后，在独立种子的轨迹上执行确定性 eye-on-hand → Cage → PPO
闭环推理，并保存 MuJoCo 快照、腕部相机 RGB 和逐帧误差曲线：

```bash
python -m cage_system.cli generate-trajectories \
  --config configs/production_curved_eval.yaml \
  --count 20 \
  --output artifacts/production_curved/eval_trajectories.npz

MUJOCO_GL=egl python -m cage_system.cli infer-visualize \
  --config configs/production_curved_eval.yaml \
  --cage-checkpoint artifacts/production_curved/cage_point20.pt \
  --ppo-checkpoint artifacts/production_curved/ppo2.pt \
  --trajectories artifacts/production_curved/eval_trajectories.npz \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --snapshot-count 6 \
  --output artifacts/production_curved/inference_visualization.png \
  --device auto
```

输出 PNG 中的机械臂、固定腕部相机、物理 Cage 和目标均来自闭环 MuJoCo
状态；红绿蓝坐标轴是动作时刻的 Cage 网络目标。完整轨迹只作为可视化审计
背景，不输入网络。相邻的同名 JSON 保存误差、奖励、视觉成功率、异常历史
比例、碰撞次数和动作饱和率。

Cage候选只生成一个姿态分支：抓子z轴严格等于经过Base参考方向定号、跨时间稳定化后的局部运动平面正法向。反法向分支不生成；正法向分支不可达或碰撞时样本直接无效，不允许回退。Cage训练显示逐epoch进度，并为每一轮记录验证集位置均值/P95（mm）、姿态均值/P95（度）以及25 mm、8度阈值下的联合达标率。最多训练300轮；验证损失连续30轮无改善时提前停止，停滞时自动降低学习率，检查点始终保存验证集最优轮次。最终测试集不参与逐轮调参。PPO训练显示总体update进度和每次rollout的有效转移采集进度；主进度条同步显示平均奖励、最大近似KL、Critic解释率和实际完成的策略更新轮数。

若已有的是旧版未过滤几何标签，先执行一次可重复过滤，再生成视觉数据：

```bash
python -m cage_system.cli filter-cage-labels \
  --config configs/ur5e_vision.yaml \
  --trajectories artifacts/quasiplanar_v1/trajectories.npz \
  --input artifacts/quasiplanar_v1/cage_labels_unfiltered_schema6_backup.npz \
  --output artifacts/quasiplanar_v1/cage_labels.npz
```

正式生成前可先把 `--count` 改为 `20` 做端到端试采。标签命令按轨迹原子保存到 `cage_labels.npz.parts`，默认自动续算；中断后直接重新执行同一条命令即可，不要加 `--no-resume`。全部轨迹合并成功后临时分片会自动删除；调试时可加 `--keep-parts`。`--workers` 是独立 MuJoCo 进程数，建议从8开始，根据CPU占用和内存调整，不要直接设成全部逻辑线程数。正式视觉配置要求 Cage 使用 `generate-vision-dataset` 的输出训练，不能直接把几何标签文件当成网络输入；这保证训练和在线推理使用同一套 eye-on-hand 观测链。Cage监督、PPO训练和最终评估必须使用三个不同随机种子的轨迹集合；评估集合不能传给任何训练命令。PPO命令保持不变；18维状态已经改为`TCP到当前Cage的Base相对位姿误差6 + q6 + qdot6`。默认生产配置先执行与轨迹无关的随机局部阻尼Jacobian示范和行为克隆，再从C7开始执行混合DAgger、Critic预热及纯Actor的on-policy PPO微调。完整契约见[`docs/expert_training_pipeline_cn.md`](docs/expert_training_pipeline_cn.md)。

可单独检查腕部相机：

```bash
python -m cage_system.cli render-eye \
  --config configs/default.yaml \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --output artifacts/visualizations/eye_on_hand.png
```

把一条实际有效Cage标签放回UR5e并联合查看全臂、固定腕部相机、小型待抓取Cage和视觉事件驱动时序：

```bash
python -m cage_system.cli visualize-system \
  --config configs/ur5e_vision_smoke.yaml \
  --trajectories artifacts/system_demo/trajectories.npz \
  --labels artifacts/system_demo/labels.npz \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --sample-index -1 \
  --output artifacts/visualizations/complete_ur5e_visible_standoff_event_driven.png
```

沿一条完整轨迹同时绘制多个Cage/物体位姿关系：

```bash
python -m cage_system.cli visualize-trajectory-cages \
  --config configs/ur5e_vision_smoke.yaml \
  --trajectories artifacts/system_demo/trajectories.npz \
  --labels artifacts/system_demo/labels.npz \
  --track-id -1 \
  --pose-count 8 \
  --output artifacts/visualizations/trajectory_multi_cage_relationship.png
```

若要同时查看每个时刻的完整 UR5e、刚性固定的抓子/相机、目标和整条轨迹：

```bash
MUJOCO_GL=osmesa python -m cage_system.cli visualize-trajectory-robot-cages \
  --config configs/ur5e_vision.yaml \
  --trajectories artifacts/trajectories.npz \
  --labels artifacts/cage_labels.npz \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --track-id -1 \
  --pose-count 6 \
  --output artifacts/visualizations/trajectory_full_ur5e_cage_sequence.png
```

随机选择一条具有可行IK的轨迹，并截取一段连续 Cage 标签，同时显示MuJoCo外部总览和同一状态下的真实腕部相机RGB画面：

```bash
MUJOCO_GL=osmesa MPLCONFIGDIR=/tmp/matplotlib-cage \
python -m cage_system.cli visualize-random-cage-segment \
  --config configs/ur5e_vision.yaml \
  --trajectories artifacts/quasiplanar_v1/trajectories.npz \
  --labels artifacts/quasiplanar_v1/cage_labels.npz \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --track-id -1 \
  --pose-count 4 \
  --seed 42 \
  --output artifacts/visualizations/random_cage_segment.png
```

`--track-id -1` 表示随机选择有可行标签的轨迹；修改 `--seed` 会换一个可复现的随机轨迹/片段。图中的粗洋红线是被选择的连续轨迹片段，青色和黄色细线仍显示完整轨迹的历史/未来部分。

已接入的 UR5e 来自 Google DeepMind MuJoCo Menagerie，并保留 BSD-3-Clause 许可证和来源提交。它是从公开 URDF 整理的高质量简化模型，不是某一台实体机械臂的标定数字孪生。派生场景增加了与当前内部尺寸一致的 Cage 抓子、腕部 RGB-D 相机和可沿轨迹移动的目标。

`default.yaml` 的 `workspace` 后端只适合流程自检；上面的正式命令使用 `ur5e_vision.yaml`，它强制启用同一UR5e场景的MuJoCo IK/碰撞筛选。标签生成使用CPU多进程，每个进程独占并复用自己的MuJoCo模型和 `MjData`；EGL只加速后续相机渲染，不加速这里的刚体动力学。`tests/assets/six_joint_contract.xml` 仍然只用于快速API测试。

本实现不使用抓子旋转生成的三维扫掠捕获体，也不存在第二个投影评价位姿。候选圆只是局部运动平面中的随动端点采样几何；同一个候选位姿同时接受 UR5e 可行性筛选、MuJoCo EST 动力学扩展和真实抓子内部区域统计。EST 根节点允许位于 Cage 外部，主分数是整棵树中落入真实内部区域的节点比例，网络标签就是该候选的六维 TCP 位姿。

## 独立的 100 Hz 控制对比分支

项目另外保留两条不改动现有 20 Hz Cage/PPO 数据流的控制分支：

- `train-jailer`：同一真实 UR5e MJCF 上的 40 个逻辑动力学槽，经过位置、六级姿态桥接和完整位姿阶段的课程 PPO；状态误差全部以 `attachment_site` TCP 为准，动作是 100 Hz 关节期望速度；近目标区域同时奖励完整位姿质量和相邻控制步的误差减小趋势；各课程使用独立的动态步数上限，采集边界处未完成的任务不计作失败且不会跨 rollout 续跑；
- `configs/ur5e_jailer_curriculum_v9.yaml`：冻结 P6 位置动作，只训练按当前角度误差连续缩放的随机姿态残差；名义 rollout 结束后停止开新任务并排空全部活动任务，确保用于 GAE/PPO 的每条序列最终都有真实结果。V9 的设计、O5/O6固定任务对比和正式指令见 [`docs/ur5e_jailer_v9_cn.md`](docs/ur5e_jailer_v9_cn.md)。
- `configs/ur5e_jailer_curriculum_v10_full.yaml`：从随机参数自动完成 masked P1–P6、observed P1–P6、B1–B6/O1–O6；P6→B1保留Critic权重、清Adam并先做Critic-only对齐，同时输出MOVE50/MOVE100随误差变化图。完整契约与命令见 [`docs/ur5e_jailer_v10_full_cn.md`](docs/ur5e_jailer_v10_full_cn.md)。
- `run-constrained-jacobian`：依据指定总时间生成笛卡尔直线参考，并在每个 10 ms 周期求解带关节位置、速度、加速度边界的微分逆运动学 QP。

两条分支都会统计实际位置伺服和 MuJoCo 动力学执行后，连续5个控制间隔（50 ms）与10个控制间隔（100 ms）的 TCP FK 净位移、P95与路径长度。完整方法、参数含义、初始姿态池和正式指令见 [`docs/ur5e_jailer_and_constrained_jacobian_cn.md`](docs/ur5e_jailer_and_constrained_jacobian_cn.md)。
