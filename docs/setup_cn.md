# 环境、训练和 ROS 接入

## 1. 当前 Python 环境

项目统一使用现有 Conda 环境 `camera`，不要再在 `mujoco_robot` 中安装深度学习依赖。

已核验的主要版本：

- Python 3.8.20
- PyTorch 2.4.1，CUDA 构建版本 12.4
- MuJoCo 3.2.3
- Gymnasium 0.29.1
- NumPy 1.23.4、SciPy 1.10.1
- pytest 8.3.5

主机上的 RTX 4060 Laptop GPU 已能被 `camera` 环境识别。检查命令：

```bash
conda activate camera
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
python -c "import mujoco, gymnasium; print(mujoco.__version__); print(gymnasium.__version__)"
```

训练命令默认使用 `--device auto`。若当前进程能访问 CUDA，则选择 `cuda:0`；否则回落 CPU。调试 GPU 映射时同时检查 `nvidia-smi` 和 `/dev/nvidia*`。

## 2. 资源分工

随机轨迹、样条、候选和标准 MuJoCo EST 主要消耗 CPU。批量标签可使用 `generate-labels --workers N` 按轨迹进程并行；每个进程只构建一次UR5e可行性模型和EST模型，随后跨轨迹复用。Cage MLP 和 PPO2 参数更新使用 GPU。每个标签 worker 和每个 MuJoCo 环境必须有独立 `MjData`，不要在多个线程之间共享仿真状态。

建议先用20条轨迹验证标签有效率，再扩大规模。默认EST每个候选320节点、重复3次，标签生成比网络训练耗时明显更长。低密度选择使用增量密度桶，明显越界或违反连续性的候选会在迭代IK前拒绝。正式命令建议以8个worker起步，并设置 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`，避免每个进程内部再次创建BLAS线程造成过度竞争。

标签生成每完成一条轨迹，就原子保存标签、压缩审计和完成标记到 `<输出文件>.parts/`。终端中断或服务器重启后，重新执行完全相同的命令会自动跳过完成轨迹并续算。只有明确希望丢弃检查点并从头计算时才使用 `--no-resume`；完整合并后分片默认自动删除，`--keep-parts` 可保留它们用于调试。

## 3. 已接入的 MuJoCo UR5e 与 eye-on-hand

项目已将 MuJoCo Menagerie 的 `universal_robots_ur5e` 模型放在 `assets/mujoco_menagerie/universal_robots_ur5e`，保留上游 BSD-3-Clause 许可证、README 和来源提交。派生文件 `ur5e_eye_on_hand.xml` 增加与当前配置内部尺寸一致的 Cage 几何、`attachment_site`、`cage_endpoint` 和腕部相机；`scene_eye_on_hand.xml` 增加光源、地面和轨迹控制的可渲染目标。

`configs/default.yaml` 保留工作空间近似后端供快速测试；正式数据和训练使用 `configs/ur5e_vision.yaml`，该配置继承默认参数并强制 `production_mode: true`、MuJoCo IK/碰撞后端和视觉监督输入来源检查。

Menagerie 自身把该模型描述为从公开URDF整理的简化UR5e，因此可以用于算法训练和sim-to-sim，但不能称为特定实体的精确数字孪生。正式训练和部署前仍需检查：

1. Base、TCP 和 ROS/Gazebo 定义一致。
2. 关节顺序、符号、限制和执行器控制方式一致。
3. 抓子质量、惯量和碰撞体已经加入。
4. 无动作时模型稳定，动作限幅后不违反关节速度。
5. TCP 位姿与独立正运动学结果一致。

腕部相机默认320×240、69度垂直视场。MuJoCo在GPU/EGL上分别渲染RGB、米制深度和几何分割。默认红色目标检测器只读取RGB；几何分割只计算审计IoU，不能进入网络输入。检测掩码经过深度前景聚类后反投影到Base，输出与ROS相同含义的三维中心点。

```bash
python -m cage_system.cli render-eye \
  --config configs/default.yaml \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --output artifacts/visualizations/eye_on_hand.png
```

要检查一个真实标签对应的完整机械臂、相机和Cage构型，使用：

```bash
python -m cage_system.cli visualize-system \
  --config configs/ur5e_vision_smoke.yaml \
  --trajectories artifacts/system_demo/trajectories.npz \
  --labels artifacts/system_demo/labels.npz \
  --mujoco-model assets/mujoco_menagerie/universal_robots_ur5e/scene_eye_on_hand.xml \
  --sample-index -1 \
  --output artifacts/visualizations/complete_ur5e_visible_standoff_event_driven.png
```

`--sample-index -1` 会自动选择从标称关节状态可获得无碰撞IK解的第一条标签。真实随动抓子通常位于球体外部并保留相机视线；候选审计图中的半透明区域就是同一个候选位姿下由左右爪内表面和后挡板围成的真实内部区。EST 根节点可以在该区域外，后续树节点按是否落入该区域统计；它不是正负30度旋转形成的 Cap 扫掠体。

## 4. ROS Noetic、Gazebo 和 ros_control

ROS 桥位于 `ros_ws/src/cage_ppo_ros`。构建前确保 ROS 环境能导入本项目和训练检查点依赖：

```bash
cd /home/sjr/PPO_caging_capture/ros_ws
catkin_make
source devel/setup.bash
```

目标检测节点发布 `geometry_msgs/PointStamped` 到默认话题 `/detector/object_point`。消息只包含目标中心点，必须带真实采集时间戳和可转换到 `base_link` 的 `frame_id`。桥接节点在每个约20 Hz观测回调中读取最新关节状态、更新Cage目标、执行一次PPO并立即发布`Float64MultiArray`位置命令；没有独立的100 Hz PPO定时器。`ros_control`内部只负责随动到最新关节目标。

```bash
roslaunch cage_ppo_ros ur5e_cage_gazebo.launch \
  config:=/home/sjr/PPO_caging_capture/configs/default.yaml \
  cage_checkpoint:=/absolute/path/cage_point20.pt \
  ppo_checkpoint:=/absolute/path/ppo2.pt \
  object_point_topic:=/detector/object_point
```

ROS 节点默认设备也是 `auto`。若 ROS 使用的 Python 与 `camera` 不同，应显式保证该解释器能导入 `torch` 和 `cage_system`；不要覆盖 ROS 系统 Python 包。

## 5. 视觉事件和机械臂速度匹配

20 Hz检测同时就是PPO决策事件频率。实际部署先测量相机曝光、检测、深度中心提取、TF、网络推理、ROS排队和底层伺服响应的总延迟，再标定以下参数：

在当前RTX 4060、320×240、理想MuJoCo RGB-D和红色RGB掩码条件下，预热后的实测中位耗时约为：RGB-D加检测1.48 ms，额外开启仿真分割审计后2.29 ms，Cage MLP单样本0.061 ms，PPO确定性动作0.187 ms。整条事件计算链明显小于50 ms帧间预算；该数字不包含实体相机曝光、ROS传输、TF，也不能替代本地COCO权重的单独测速。正式运行应记录P95/P99而不是只看平均值，训练时可关闭 `render_segmentation_for_audit` 节省第三次渲染。

使用当前调过增益的派生 UR5e 执行器，以0.45 m/s目标连续运行0.5 s得到的频率对比为：10 Hz最大滞后约76 mm且每步饱和，15 Hz最大滞后约17.4 mm且多数步骤饱和，20 Hz最大滞后约7.1 mm、最终约4.6 mm，25--30 Hz没有动作饱和但滞后约8.3--9.4 mm。因而当前保留20 Hz；这只是当前MuJoCo模型的系统辨识结果，实体相机、COCO推理和实体UR5e必须重新做相同扫频。

- 随机轨迹最大速度不得超过 UR5e 可靠 TCP 线速度除以安全系数。
- 速度到随动半径采用经过上下限、低通和变化率约束的线性正比例映射；半径受相机视场、目标表面间隙、工作空间和跟踪能力约束，不要求根节点初始位于小抓子内部。
- 每个视觉事件的关节增量同时受单事件配置增量与关节速度乘实际事件间隔限制。
- 两次感知之间由底层控制器继续跟踪最近关节目标，不重复执行PPO。
- Cage 新目标相对上一目标的线速度和角速度超过跟踪包络时拒绝该跳变。
- 在实际设备上至少扫频10、15、20、25和30 Hz，联合检查跟踪滞后、动作饱和、COCO P95/P99和陈旧保护。

## 6. COCO 中心点接入

本系统不需要目标姿态。当前红色球体是点质点的可渲染代理，默认使用RGB颜色检测，以便完整验证图像而不读取目标真值。`eye_on_hand.py` 同时支持本地Ultralytics COCO检测和分割权重：

```yaml
vision:
  detector_backend: ultralytics
  coco_weights_path: /absolute/path/to/local_weights.pt
  coco_class_id: 0
  coco_confidence_threshold: 0.35
```

分割权重直接提供掩码；检测权重使用框，再由深度前景聚类剔除框内背景。两种方式最终都只输出Base下中心点。要使用某个COCO类别，还需要在MJCF中放入具有相应外观的可渲染物体；红色球体本身不会被通用COCO权重可靠识别。

## 7. 生产前检查

- 输入数组严格是 `[N,5,4]`，每帧是 xyz 加线速度大小。
- 每个历史时间戳严格早于当前标签时间戳。
- Cage数据集格式版本为9、Cage检查点格式版本为10、PPO检查点格式版本为11；目标模型为 `point_mass`，PPO奖励目标契约为 `same_cage_pose_as_action_state`。PPO检查点还必须包含相对18维状态契约、独立Actor/Critic契约、初始化来源、状态运行统计、tanh后熵估计和GAE λ-return契约。
- 生产标签没有使用 `workspace` 近似后端。
- Cage训练输入已经通过 `generate-vision-dataset` 由同一腕部RGB-D链生成。
- Base、link6、Cage、TCP和相机三维坐标系全部为右手系；像素纵轴向下不参与三维旋转定义。
- 抓子刚体固定在link6，Cage x轴沿link6伸长方向；`link6←Cage`、`Cage←TCP`、`Cage←Camera`和抓子真实内部尺寸已标定。
- MuJoCo 和 Gazebo 的 Base、TCP、关节名、限制与控制周期一致。
- 冻结 Cage 参数不在 PPO2 优化器中。
- PPO2 保存实际动作及六关节联合旧对数密度。
- PPO状态中的Cage位姿、产生的动作和该转移奖励必须属于同一目标；下一视觉帧预测的Cage只能进入下一状态。
- 真实机器人另设急停、速度缩放、保护停止和独立工作空间监控。
