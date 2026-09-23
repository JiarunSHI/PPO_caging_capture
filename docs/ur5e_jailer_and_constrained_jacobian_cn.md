# UR5e 100 Hz 课程 PPO 与时间约束 Jacobian 分支

## 1. 统一的 UR5e/TCP 契约

两条控制分支都通过关节名称解析当前 MJCF，而不是假定 `qpos[0:6]`：

- 六关节顺序固定为 `shoulder_pan_joint`、`shoulder_lift_joint`、`elbow_joint`、`wrist_1_joint`、`wrist_2_joint`、`wrist_3_joint`；
- 控制与误差计算统一使用 `attachment_site`；
- 当前 MJCF 中 `attachment_site` 与 Cage/抓子根坐标系刚性重合；
- 其 x 轴经固定变换后沿 link6 的伸长方向；
- MuJoCo 物理步长为 2 ms，一个 100 Hz 控制周期严格推进 5 个物理步；
- 每个动作的目标是 6 关节期望速度，再按 10 ms 积分为位置伺服目标，由 MJCF 中的关节 PD 动力学实际执行。

启动时会保存并输出关节、qpos、dof、actuator、TCP 和时间步映射。模型、TCP 或关节顺序变化时，断点恢复会直接拒绝，防止旧权重误用于不同机械臂契约。

## 2. 40 个逻辑环境槽

40 个槽共享一份只读 `MjModel`，每个槽各自保存：

- 独立 `MjData`，即独立的 q、qdot、执行器和动力学状态；
- 上一周期的关节期望速度；
- 连续积分的 PD 关节位置目标；
- 当前 Base 坐标系目标 TCP 位姿；
- 最近 6 个控制边界处的真实 FK TCP 位置；
- 独立恢复次数和随机状态。

场景中仍然只有一台 UR5e 的几何模型，不会复制 40 台机械臂，也不会产生 40 倍渲染数据。每个 PPO update 开始时冻结当前策略，将 40 个初始 q6 重新打乱后一对一分配给 40 个槽，清零 qdot、PD 目标速度和 50 ms 历史，再分别推进 40 份动力学状态。每个课程阶段自行规定每槽的 100 Hz 控制步数上限，因此一个 update 的批量大小等于“本阶段步数上限乘以 40”，不再固定为 8,000。所有槽完成该采集窗口后才进行多个 PPO epoch，不能在采集中途更新策略。

一个局部目标达到成功阈值后立即形成真实终止边界；对应槽随后从初始池恢复并开始新任务，继续填充同一个 on-policy 批次。接触、越出工作空间、非法数值和完整用尽本阶段任务预算也属于真实终止。若后来启动的新任务只因本次 rollout 采集结束而被打断，则标记为 `rollout_truncated`，不计成功或失败，也不进入晋级误差均值。该截断会停止 GAE 向下一个批次递推，但 Critic 仍从截断后实际观测到的下一状态回举价值。下一次 rollout 会重新打乱初始池并重置全部 40 个槽，不恢复该任务的关节状态、目标或已执行步数。

## 3. 初始姿态池

初始池保存的是 40 组 UR5e 关节姿态 q6，而不只是 TCP pose。这样可以保留同一 TCP 位姿下不同肩、肘、腕运动学分支。

生成器围绕当前项目的 nominal q 随机采样，随后用同一个 MJCF 做以下硬检查：

- 六关节处于配置和 MJCF 关节范围的交集内，并留有关节安全边距；
- 当前状态没有 MuJoCo 接触；
- `attachment_site` 位于项目配置的 UR5e 工作空间；
- 任意两个已接受 q6 之间具有最小关节距离，避免 40 个槽退化成近似相同状态。

池文件保存 q6、对应 TCP pose6、关节名称、TCP 名称、MJCF 路径和随机种子。训练载入时会再次用所选 MJCF 逐行验证。

## 4. 位置到姿态的课程

Actor/Critic 始终保持同一个 18 维接口：

`目标TCP到实际TCP的Base坐标相对位置3 + 相对旋转向量3 + q6 + qdot6`

课程不会改变网络维度：

1. 六个位置阶段使用从 6--20 mm 逐步扩展到 6--150 mm 的累计误差包络；位置和姿态阶段始终输入真实旋转误差，不再把状态中的三个姿态维度置零；
2. P6之后先经过六个共享 6--25 mm、1--5 度目标分布的姿态桥接阶段，单任务姿态阈值依次为12、8、4、3.5、3.0和2.2度，再进入从 6--25 mm、1--5 度逐步扩展到 6--150 mm、1--45 度的六个正式位姿阶段；
3. 每个目标都由一个无碰撞 UR5e 目标关节构型正运动学得到，所以完整 pose 目标有已知可达解；
4. 每个阶段至少训练 100--200 个 update，并在最近连续 10 个 rollout 的真实已终止任务中同时满足平均误差与任务成功率后才晋级；`minimum_updates` 就是该阶段至少完成的 rollout/update 数，而每轮 transition 数随阶段动态变化；P1--P5成功率至少为80%，P6、全部桥接阶段及全部O阶段要求严格大于80%；
5. 一个目标成功后立即终止，不会继续在其附近采样；采集边界截断的任务不参与晋级统计，下一轮也不会接着执行；
6. 位置和位姿课程的动态上限、单任务成功阈值及晋级均值如下。速度辅助条件保持为TCP线速度不超过0.08 m/s；全部B/O位姿课程还要求角速度不超过0.30 rad/s：

| 阶段 | 每槽上限（步/秒） | 每轮transition | 单任务位置阈值 | 晋级位置均值 | 单任务角度阈值 | 晋级角度均值 |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 60 / 0.60 s | 2,400 | 1.5 mm | 2.5 mm | 不启用 | 不启用 |
| P2 | 60 / 0.60 s | 2,400 | 2.0 mm | 3.0 mm | 不启用 | 不启用 |
| P3 | 70 / 0.70 s | 2,800 | 2.5 mm | 3.5 mm | 不启用 | 不启用 |
| P4 | 80 / 0.80 s | 3,200 | 3.0 mm | 4.5 mm | 不启用 | 不启用 |
| P5 | 100 / 1.00 s | 4,000 | 3.5 mm | 5.5 mm | 不启用 | 不启用 |
| P6 | 130 / 1.30 s | 5,200 | 4.0 mm | 8.0 mm | 不启用 | 不启用 |
| B1 | 80 / 0.80 s | 3,200 | 6.0 mm | 6.0 mm | 12.0° | 7.0° |
| B2 | 90 / 0.90 s | 3,600 | 5.0 mm | 5.5 mm | 8.0° | 6.0° |
| B3 | 100 / 1.00 s | 4,000 | 4.0 mm | 5.0 mm | 4.0° | 4.5° |
| B4 | 110 / 1.10 s | 4,400 | 4.0 mm | 4.5 mm | 3.5° | 3.7° |
| B5 | 120 / 1.20 s | 4,800 | 3.5 mm | 4.0 mm | 3.0° | 3.2° |
| B6 | 130 / 1.30 s | 5,200 | 2.8 mm | 3.2 mm | 2.2° | 2.5° |
| O1 | 80 / 0.80 s | 3,200 | 2.0 mm | 2.5 mm | 1.5° | 2.5° |
| O2 | 90 / 0.90 s | 3,600 | 2.5 mm | 3.0 mm | 2.5° | 3.5° |
| O3 | 110 / 1.10 s | 4,400 | 3.0 mm | 3.5 mm | 3.5° | 4.5° |
| O4 | 130 / 1.30 s | 5,200 | 3.5 mm | 4.5 mm | 4.5° | 5.5° |
| O5 | 150 / 1.50 s | 6,000 | 4.0 mm | 5.5 mm | 5.5° | 7.0° |
| O6 | 180 / 1.80 s | 7,200 | 4.5 mm | 6.5 mm | 7.0° | 9.0° |

这些动态上限不是要求策略必须用满相应步数。成功仍会立即终止，表中上限只给较远或带姿态目标更多收敛时间，并减少近距离阶段的大量冗余采样。

7. 总训练长度为 4,000 update；最终阶段掌握后仍继续训练到总轮数，避免过早停止。

位置质量沿用 JAiLeR 的指数型 TCP 位置质量。每步奖励使用“质量减去 1”的非正到达代价，并在成功终止时加入正奖励，从而使更少步骤到达优于长期停留在阈值外。姿态质量保留论文建议的 TCP 三轴标记点，但目标与实际的三个 5 cm 虚拟点都锚定在同一计算原点，仅比较旋转，不再重复混入 TCP 平移；大姿态误差使用不会迅速饱和的反二次质量，并增加最短旋转角减小时的进展奖励。

在全部位姿阶段中还加入近目标位置进展项：比较相邻两个100 Hz控制步的位置误差，误差减小获得正奖励，误差增大得到对称惩罚。该项按5 mm误差变化归一化并截断，只在位置误差30 mm、姿态误差15度以内通过连续门控生效，越接近目标权重越高，边界外平滑降为零。因此它提供“继续向目标靠近”的方向信息，但不改变成功阈值，也不能让尚未满足位置、姿态和速度条件的状态提前成功。`training_history.jsonl`会记录位置进展、该项实际奖励和门控均值，便于判断奖励是否真正起作用。

课程采样、成功终止与报告仍使用目标相对当前 TCP 的最短旋转角。关节加速度只作为小权重控制努力项；线速度和角速度是成功终止门槛，目前不是独立奖励项。B1--B6保持相同的0.10动作缩放，细分课程只收紧终止要求，不通过额外降速改变控制问题。

## 5. 50 ms 与 100 ms 真实移动能力统计

每个槽在每次 10 ms 动作执行完后读取 MuJoCo 的实际 `attachment_site` FK 位置。历史缓冲严格在当前任务内部维护，任务恢复时清空，因此不会跨两个独立目标拼接位移。连续积累5个和10个控制间隔后分别统计：

- `tcp_50ms_net_displacement_mean_m`：相隔 50 ms 的实际 TCP 直线位移均值；
- `tcp_50ms_net_displacement_p95_m`：上述位移的 P95；
- `tcp_50ms_path_length_mean_m`：50 ms 内 5 段实际 TCP 路程之和的均值。
- `tcp_100ms_net_displacement_mean_m`：相隔 100 ms 的实际 TCP 直线位移均值；
- `tcp_100ms_net_displacement_p95_m`：上述位移的 P95；
- `tcp_100ms_path_length_mean_m`：100 ms 内 10 段实际 TCP 路程之和的均值。

这些值包含位置执行器 PD、惯性、关节限速和 MuJoCo 动力学的影响，不是根据动作命令估算的位移。训练进度条实时显示 `move50` 和 `move100`，完整均值、P95与路径长度写入每个update的 `training_history.jsonl`。

每个成功任务还会记录初始位置误差、初始姿态误差、逻辑槽、任务序号、到达步数和实际控制时间。记录使用可恢复的 SQLite 主表，并周期性导出：

- `successful_reaches.csv.gz`：全部成功案例的逐条明细；
- `successful_reach_steps_by_initial_error.pdf`：初始误差到平均/中位/P95到达步数曲线、各误差区间样本数和分课程曲线；
- 姿态课程开始后，PDF自动增加“初始姿态误差—到达步数”页面。

## 6. 时间约束的 Jacobian 分支

该分支对应用户提供的 `Jacobian_约束微分逆运动学_算法流程.pdf`：

1. 从控制开始时的真实 TCP pose 到 Base 坐标系目标 TCP pose 生成给定总时间的直线位置参考；
2. 姿态开启时，同步生成最短旋转参考；位置模式可用 `--position-only` 关闭姿态；
3. 每个 10 ms 读取最新 q/qdot 和实际 TCP，并重新计算 `attachment_site` Jacobian；
4. 参考前馈速度与参考轨迹误差反馈共同形成末端速度命令；
5. 直接求解带边界的最小二乘 QP，使 Jacobian 映射后的 TCP 速度接近命令；
6. QP 边界同时包含关节速度限制、下一周期预测关节位置限制和相邻命令的关节加速度限制；
7. 安全位置/速度边界优先级高于加速度边界，若两者冲突会审计具体放宽次数；
8. QP 输出关节速度；底层接受位置命令时按 10 ms 积分并由同一 UR5e PD 执行；
9. 在规定时间 T 同时检查 TCP 位置误差、姿态误差、线速度和角速度。若误差很小但仍高速运动，仍标记为“规定时间未成功”；可继续进入配置的短收敛段。

## 7. 指令

先生成可审计的 40 个初始关节姿态：

```bash
cd /home/sjr/PPO_caging_capture
conda activate camera

python -m cage_system.cli generate-jailer-initial-pool \
  --config configs/ur5e_jailer_curriculum.yaml \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --count 40 \
  --output artifacts/jailer_v1/initial_joint_pool.npz
```

完全随机初始化Actor时可直接启动课程：

```bash
python -m cage_system.cli train-jailer \
  --config configs/ur5e_jailer_curriculum.yaml \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --initial-pose-pool artifacts/jailer_v1/initial_joint_pool.npz \
  --output artifacts/jailer_v6_random/jailer_ppo.pt \
  --device auto
```

推荐从旧版训练结果只热启动Actor。动态rollout重新定义了时间截断和Critic回举语义，因此旧版Critic、Adam、课程窗口和未完成环境状态都不继承：

```bash
python -m cage_system.cli train-jailer \
  --config configs/ur5e_jailer_curriculum.yaml \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --initial-pose-pool artifacts/jailer_v1/initial_joint_pool.npz \
  --actor-warm-start-checkpoint artifacts/jailer_v4/jailer_ppo_checkpoints/stage_position_p06_090_150mm_complete.pt \
  --output artifacts/jailer_v6/jailer_ppo.pt \
  --device auto
```

只有同一动态rollout版本的新训练发生普通中断后，才使用完整恢复；Actor、Critic、优化器、归一化、课程和update编号都会接续：

```bash
python -m cage_system.cli train-jailer \
  --config configs/ur5e_jailer_curriculum.yaml \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --initial-pose-pool artifacts/jailer_v1/initial_joint_pool.npz \
  --resume-checkpoint artifacts/jailer_v6/jailer_ppo_checkpoints/latest.pt \
  --output artifacts/jailer_v6/jailer_ppo.pt \
  --device auto
```

当前推荐从已经晋级的B3完成点继续新增B4。该恢复保留B3的Actor、姿态残差、Critic、Adam动量和归一化统计，只清空旧课程晋级窗口并明确将入口映射为B4；不会执行Critic重置或额外预热：

```bash
python -m cage_system.cli train-jailer \
  --config configs/ur5e_jailer_curriculum.yaml \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --initial-pose-pool artifacts/jailer_v1/initial_joint_pool.npz \
  --resume-checkpoint artifacts/jailer_v6/jailer_ppo_checkpoints/stage_pose_bridge_b03_006_025mm_04deg_complete.pt \
  --resume-start-stage pose_bridge_b04_006_025mm_035deg \
  --output artifacts/jailer_v7/jailer_ppo.pt \
  --device auto
```

这里应使用B3的阶段完成点，而不是已经在旧O1上长期退化后的`latest.pt`。`--resume-start-stage`只改变课程游标和清空晋级窗口，不重建网络或优化器。

执行一次带明确时间约束的 Jacobian/QP 控制；以下用目标 q6 的 FK 构造保证可达的目标 TCP：

```bash
python -m cage_system.cli run-constrained-jacobian \
  --config configs/ur5e_jailer_curriculum.yaml \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --tcp-site attachment_site \
  --goal-joints 0.10 -1.50 1.50 -1.55 -1.50 0.10 \
  --duration 2.0 \
  --output artifacts/jailer_v1/constrained_jacobian_report.json
```

实际 Cage 目标可改用 `--goal-pose X Y Z RX RY RZ`。该 pose 必须是 Base 坐标系下 `attachment_site` 的 xyz+旋转向量，不能传入法兰、link6 原点或物体坐标。

## 8. 与动态 Cage 的后续连接

静态课程阶段不读取视觉或未来轨迹。接入现有动态系统时，20 Hz eye-on-hand/Cage 网络每 50 ms 产生一个新的 Base 坐标系目标 TCP pose；100 Hz 控制器在接下来的 5 个控制步中保持该目标，并用每个 10 ms 的最新 q/qdot 与 TCP FK 重新计算 18 维反馈状态。这样视觉目标频率和底层控制频率相互独立，又不会把离线未来曲线输入策略。
