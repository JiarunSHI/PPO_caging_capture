# 项目阅读与实现状态

## 1. 先理解一条主数据流

项目不是一个端到端同时训练的网络，而是两段串联、信息边界明确的流水线：

```text
随机准平面中心点轨迹（连续变曲率、局部平面缓慢变化）
  → 离线卡尔曼和完整三维样条
  → 可见随动候选圆、交点、劣弧和 TCP 候选
  → 几何/机械臂硬筛选
  → 同一个候选位姿上的低密度 MuJoCo EST 动力学评估
  → 最佳当前 Cage 六维标签
  → UR5e eye-on-hand RGB-D 重放历史观测
  → [五帧视觉 xyz+speed] 监督训练 Cage MLP
  → 冻结 Cage MLP
  → MuJoCo 相机继续因果产生动态 TCP 目标
  → 真实 UR5e MJCF 中训练 PPO2
  → ROS/Gazebo/ros_control 集成推理
```

最关键的时间顺序是：用 `t-5 ... t-1` 预测当前 `t`。当前观测 `t` 在预测完成后才进入历史缓存。离线未来曲线只能决定 `t` 的标签，不能出现在 20 维输入里。

## 2. 目录应该怎么看

```text
PPO_caging_capture/
├── configs/default.yaml              # 全系统唯一默认参数入口
├── cage_system/
│   ├── cli.py                         # 从命令开始追踪代码的总入口
│   ├── config.py                      # 5×4、点质点、PPO 18/6 硬校验
│   ├── types.py                       # 跨模块数据结构
│   ├── io.py                          # 随机轨迹 NPZ 格式版本
│   ├── trajectory/
│   │   ├── generator.py               # 五帧变曲率、局部平面缓变的准平面轨迹
│   │   ├── kalman.py                  # 位置/速度/加速度九状态滤波
│   │   ├── spline.py                  # 共享时间节点三维三次 B 样条
│   │   ├── offline.py                 # 完整离线曲线与解析导数
│   │   └── online.py                  # 因果滚动估计和短时连续延伸
│   ├── cage/
│   │   ├── plane.py                   # 局部运动平面和法向连续性
│   │   ├── candidates.py              # 速度半径、圆、交点、劣弧、位姿
│   │   ├── internal_region.py         # 抓子真实内部点质点判定
│   │   ├── feasibility.py             # 工作空间参考硬筛选
│   │   ├── est.py                     # MuJoCo 前向积分的低密度 EST
│   │   └── selector.py                # EST 优先的逐级最优选择
│   ├── pipeline.py                    # 把轨迹、候选、筛选、EST 串成标签
│   ├── parallel.py                    # 按轨迹多进程标签生成
│   ├── perception/
│   │   ├── eye_on_hand.py             # RGB/深度渲染、检测与三维反投影
│   │   └── vision_dataset.py           # 用同一视觉链重建监督历史输入
│   ├── learning/
│   │   ├── dataset.py                 # [N,5,4]→[N,6] 数据契约
│   │   ├── cage_network.py            # 20→6 MLP、训练、冻结推理
│   │   ├── device.py                  # auto/cpu/cuda 设备与随机种子
│   │   ├── online_interface.py        # 先预测 t、后缓存 t 的无泄漏接口
│   │   ├── ppo2.py                    # 18→6 Actor-Critic、GAE、RTG
│   │   └── train_ppo.py               # 完整 episode 采样与 PPO2 更新
│   ├── sim/
│   │   ├── target_playback.py         # 无视觉模式的兼容目标回放
│   │   ├── mujoco_env.py              # UR5e 视觉闭环与六关节 Gymnasium 环境
│   │   └── mujoco_feasibility.py      # MuJoCo IK/碰撞可行性后端
│   ├── ros1/moveit_backend.py          # MoveIt IK/状态有效性后端
│   ├── visualization.py               # 候选与 EST 三维审计图
│   ├── trajectory_cage_visualization.py # 完整轨迹上的多 Cage 几何关系
│   └── system_visualization.py        # 单帧及多帧完整 UR5e/Cage/相机渲染
├── assets/mujoco_menagerie/
│   └── universal_robots_ur5e/          # 有来源和许可证的 UR5e 相机场景
├── ros_ws/src/cage_ppo_ros/            # PointStamped + PPO + ros_control 桥
├── tests/                              # 算法、泄漏、网络和 MuJoCo 契约测试
└── docs/                               # 方法、环境和本阅读指南
```

## 3. 推荐的源码阅读顺序

如果你想先看“输入到底是什么”，依次读 `perception/eye_on_hand.py`、`types.py`、`learning/dataset.py`、`learning/online_interface.py`。这里能看到 RGB 检测、深度反投影、Base 变换，以及当前帧没有提前进入预测。

如果你想看“标签怎么来的”，从 `pipeline.py` 开始，随后进入 `trajectory/offline.py`、`cage/candidates.py`、`cage/internal_region.py`、`cage/est.py` 和 `cage/selector.py`。`pipeline.py` 是最适合打断点的地方，因为每个轨迹时刻的数据都在那里汇合。

如果你想看“两个网络怎么训练”，先读 `perception/vision_dataset.py` 和 `learning/cage_network.py`，再读 `sim/mujoco_env.py`、`learning/ppo2.py` 和 `learning/train_ppo.py`。视觉模式下动态 Cage 目标在 MuJoCo 步进过程中由实时 RGB-D 观测产生。

如果你想看“在线系统怎么运行”，读 `learning/history_quality.py`、`learning/online_interface.py` 和 `ros_ws/src/cage_ppo_ros/scripts/cage_ppo_controller.py`。系统只有在五个严格历史帧全部被接受、时间间隔正常且因果曲率不过限时才运行冻结 Cage 和 PPO2；异常窗口与漏检期间发布当前关节保持命令，直到重新积累完整正常窗口。随后还会执行陈旧保护、TCP目标速度检查和关节增量限幅。

## 4. 已经实现并验证的内容

### 轨迹和数据

- 曲率沿整条路径连续缓变的准平面轨迹，包含升曲率和降曲率分布。
- 生成时逐个验收所有连续五帧的曲率跨度，曲率变化太小或太大的轨迹会重新生成。
- 局部运动平面沿轨迹小幅连续变化，并对全程变化范围和相邻帧变化设置硬上限；全局平面只用于法向初始化，不覆盖逐时刻局部平面。
- 非均匀时间间隔、位置噪声、异常点和短时丢帧注入。
- 九状态卡尔曼、异常创新门控、缺帧传播。
- 共享节点三维三次样条，连续查询位置、速度和加速度。
- 离线完整未来曲线与在线因果短时连续延伸严格隔离。
- 轨迹归档与监督数据都有版本和 `point_mass` 元信息；轨迹归档保存曲率与局部平面验收诊断。

### Cage 标签

- 稳定局部运动平面和直线退化处理。
- 速度到可见随动半径的线性正比例映射、上下限、低通和变化率约束。
- 二维候选圆、前向切线交点、未来首次出圆交点和劣弧。
- 均匀采样、上一最佳附近密集采样、回投到候选圆/运动平面和去重。
- 平面约束的六维 E/TCP 姿态和正反法向分支。
- 同一候选位姿贯穿筛选、EST 和标签，以及基于抓子内表面、根部、上下界与开口的真实点质点内部区域。
- 工作空间参考筛选、MoveIt 后端、MuJoCo 数值 IK/碰撞后端接口。
- 根节点可在 Cage 外、优先扩展低密度区域的 MuJoCo EST；随机外力、力矩、作用时间和选择分位数在候选间公平复用。
- 3次EST整树内部节点比例均值优先，近似最优集合内只按六关节变化和上一TCP位姿连续性决胜，不使用其他余量、能量 Softmax 或 Cap 概率。
- 每时刻 JSONL 审计、无效原因和三维可视化。

### 网络和控制

- Cage MLP 输入从旧接口迁移到 `[5,4]`，展平 20 维，输出六维 TCP。
- 输入标准化、旋转几何损失、按整条轨迹拆分训练/验证。
- 检查点保存格式版本、输入形状、目标模型和完整训练历史。
- 冻结推理严格拒绝旧60维、旧大抓子几何、非link6对齐或非视觉事件驱动检查点。
- PyTorch 设备自动选择；显式 CUDA 不可用时给出清晰错误。
- PPO2使用`TCP到Cage的Base相对位姿误差6 + q6 + qdot6`的18维状态和6维动作硬校验，Actor/Critic独立主干，tanh变换后的联合动作对数密度、新旧策略联合密度比、GAE和对应λ-return。
- 轨迹无关的随机局部阻尼Jacobian示范、从C7开始的混合DAgger逐轮聚合监督训练、全Actor验证轮、全新Critic预热和纯Actor PPO微调。
- 动作与奖励严格使用状态中的同一个Cage位姿；周期末视觉更新只形成下一状态目标，不能反向改变本次动作奖励。
- 18维状态运行均值/方差归一化及冻结推理统计，tanh后策略蒙特卡洛熵、GAE λ-return、KL早停、截断率/饱和率/解释方差诊断。
- MuJoCo Gymnasium 环境、冻结 Cage 动态目标播放、关节速度/增量/位置三重限制。
- MuJoCo Menagerie UR5e、派生 Cage 抓子、腕部透视相机和动态目标场景。
- RGB、米制深度和仿真分割渲染；默认检测器只读取 RGB，仿真分割仅用于审计。
- RGB 掩码、深度前景聚类、相机反投影、Base 中心点、视觉丢帧和深度噪声。
- Ultralytics 本地 COCO 检测或分割检查点的可插拔接口。
- `generate-vision-dataset` 用严格上一标签的相机姿态重放监督输入，当前标签不影响当前图像。
- PPO2视觉环境由20 Hz渲染/检测事件触发，每帧只执行一次PPO并立即下发关节目标。
- ROS `PointStamped` 中心点接口、TF到Base、视觉回调内即时PPO；`ros_control`只承担底层随动。
- 全部三维坐标系为右手系；抓子和相机作为刚体固定在link6，Cage x轴沿link6伸长方向。

### 已执行的验证

- 24 个自动测试全部通过，包括准平面轨迹验收、卡尔曼、样条、局部平面、候选边界、MuJoCo EST、IK前快速硬筛选、均值/关节变化/连续性选择、无未来泄漏、RGB检测、深度反投影、UR5e相机场景、生产配置继承、网络和MuJoCo环境。
- 当前3次EST均值开发烟雾集生成6个有效 `[5,4]` 标签；生产UR5e IK/碰撞烟雾集生成1个有效标签，其3次内部比例为1.25%、2.5%和5%，均值为2.92%。
- 其中4条轨迹通过真实UR5e初始IK和相机可视性筛选，生成78个视觉监督标签；检测成功帧的中心平均误差约4.3 mm。
- 视觉Cage网络在 `cuda:0` 完成100轮训练，并重新进入UR5e视觉环境成功输出18维PPO状态。
- 当前生产配置 smoke test 使用同一个实际候选检查 UR5e MuJoCo IK/碰撞并执行正式动力学 EST；不再生成投影捕获分支。
- 在0.100--0.180 m随动半径、横纵各±25 mm误差的45个腕部相机测试点中全部检测到球体；20 Hz连续跟踪0.45 m/s目标时最大滞后约7.1 mm。

## 5. 仍需要标定或大规模训练才能完成的工作

- 已接入Menagerie UR5e，但它仍是从公开URDF整理的简化模型；实体部署需要质量、摩擦和执行器系统辨识。
- `workspace` 标签后端只做开发自检。正式数据应切换到与实际抓子、环境一致的 MoveIt 或 MuJoCo 后端。
- 当前抓子是按配置内部尺寸建立的几何代理；需要用最终CAD替换并标定TCP—E变换。
- 腕部相机当前是320×240、69度视场的理想针孔模型；实体相机需要手眼、内参、畸变、曝光和延迟标定。
- 红色质点代理已走通RGB-D链路；使用本地COCO权重时需要填写权重路径、类别编号并换成对应可渲染物体。
- 已验证小规模GPU闭环，但尚未完成大规模PPO2收敛、Gazebo sim-to-sim和实体机械臂迁移。

## 6. 每次实验重点看哪些文件

轨迹 NPZ 用于复现实验；Cage 标签 NPZ 用于训练；JSONL 审计用于解释每个时刻为什么有效或无效；Cage `.pt` 保存监督模型；PPO2 `.pt` 保存控制策略。大规模审计建议使用 `.jsonl.gz`，读写和可视化入口会自动识别压缩格式。建议每次实验额外保存所用 YAML 配置、随机种子、命令和模型来源。

查看标签是否合理时，先统计 JSONL 中的 `invalid_reason`，再挑选有效和无效样本运行 `visualize`。查看网络问题时，先确认 NPZ 输入是 `[N,5,4]`，再检查按轨迹划分和检查点中的 `input_contract`。查看 PPO2 问题时，先验证冻结 Cage 目标序列，再看 TCP 误差、碰撞、动作饱和和关节限位，而不是只看累计奖励。
