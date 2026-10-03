蓝队无 GPS 定位与导航实现方案
===========================

文档性质：面向编码和联调的设计稿；不表示下述模块已经实现。
适用项目：`/data-ssd/guoyu/uav/baseline/swarm-eod`。
前置约定：沿用 01 至 04 方案的 `PoseEstimate`、`frame_id` 和单步协调器。
本文件只设计允许的测量融合，不读取地图真值、无人机真值或物理 body。

1. 当前代码事实和问题

已核对的源码及结论：

- `solutions/my_drone_blue_basic.py` 的 `_remember_home_position()`、
  `_choose_remembered_bomb()`、`_control_to_remembered_bomb()`、
  `_record_breadcrumb()`、`_next_return_waypoint()`、`_control_carry()`、
  `_disposal_target()` 依赖即时 GPS/指南针或由其得到的世界点。
  `_point_from_sensor()` 只做有限值检查，不是定位器。
- `solutions/my_drone_rescue_example.py` 的 `_semantic_to_world()`、
  `_move_toward()`、`_within_radius()` 直接再次读取测量。
  `_find_closest_semantic()`、`_control_carry()`、`_control_carry_push()`
  会间接调用这些方法，迁移时必须考虑父类动态分派。
- `simulation/drone/drone_abstract.py` 提供 `measured_gps_position()`、
  `measured_compass_angle()`、`odometer_values()` 和禁用状态查询。
  GPS 返回世界像素坐标，指南针返回弧度；`measured_velocity()`
  由里程计和指南针推导，`measured_angular_velocity()` 由里程计推导，
  不是两份独立观测。真值 API 和直接状态属性不能用于控制。
- `simulation/drone/drone_sensors.py`：GPS 每轴平稳标准差 5 px，
  指南针平稳标准差 4 度，二者噪声均为 AR(1)，`rho=0.98`。
  里程计每步为 `[dist_travel, alpha, theta]`，独立高斯标准差
  分别为 0.2 px、8 度、1 度。`alpha` 是“本步位移方向减上步
  机体朝向”，`theta` 是本步朝向变化；二者均归一化到
  `[-pi, pi)`。首个传感器更新的真实位移为零，不能使用其
  无意义的 `alpha` 做朝向初始化。
- `simulation/drone/sensor.py`：禁用时观测接口返回 `None`；
  初始化默认值可能是 NaN。`simulation/utils/utils_noise.py`
  表明 AR(1) 的 5 px / 4 度是平稳标准差，并非每步白噪声。
  `simulation/utils/utils.py` 的 `normalize_angle()` 输出
  `[-pi, pi)`。
- `simulation/utils/constants.py`：通信半径 250 px；Lidar
  最大 300 px、语义传感器最大 200 px。运动比例常量不是
  可直接当作“每步最大像素位移”的严格物理上界。
- 提示词所列 `simulation/gui_map/zones.py` 在当前仓库不存在。
  实际实现位于 `simulation/elements/sensor_disablers.py`：
  `NoGpsZone` 同时禁用 `DroneGPS` 和 `DroneCompass`，不禁用
  里程计；`NoComZone` 仅禁用通信；`KillZone` 禁用全部设备。
  `simulation/drone/device.py` 的 `pre_step()` 每步清除禁用标记，
  当前步再次碰撞区域才继续禁用。当前实现是禁用，不是
  位置/角度随机重置；控制器仍应防御有限值跳点。
- `simulation/gui_map/closed_playground.py` 的世界坐标以中心为原点，
  合理范围约为 `[-width/2,width/2] x [-height/2,height/2]`，
  不是 `[0,width] x [0,height]`。边界检查必须留噪声余量。
- `simulation/gui_map/gui_sr.py` 在一个仿真步中先调用
  `define_message_for_all()`，后设置 `elapsed_timestep` 并调用
  `control()`；因此广播只能发送上一控制步的缓存位姿。
- `submission_check.py` 禁止 `true_position()`、`true_angle()`、
  `true_velocity()`、`true_angular_velocity()` 等真值调用，
  也不允许控制器使用已禁用的直接状态属性。

现状的核心问题是：GPS 一旦为 `None`，世界目标和面包屑失去
可用位姿；多个函数在同一轮重复读传感器，且没有方差、frame
或重定位事件。后续四模块无法区分“坐标存在”和“坐标可信”。

2. 目标与非目标

目标：每机独立维护一份 3 状态位姿和协方差；正常时融合 GPS、
指南针和里程计；进入无 GPS/无指南针区时连续传播；恢复时
拒绝跳点并一致地处理旧地图、目标和轨迹。每控制步只发布一个
不可变 `PoseEstimate`，让建图、探索、分配、返航与运动控制
消费同一快照。

非目标：不做完整视觉/Lidar SLAM；不利用地图 PNG/JSON、墙真值
或队友真值；不保证无限长 GPS 中断时绝对位置准确；不让
定位模块生成油门、转向或抓取命令。低置信度时安全降级比
维持世界坐标功能更重要。

首版选轻量 3 状态 EKF，理由是跨模块必须有可传播的方差、
GPS 与指南针会独立缺失、里程计运动模型非线性但 Jacobian
很小。互补滤波加手工方差传播可作为单元测试对照基线；其
位置-朝向交叉协方差和门控会比 3x3 EKF 更容易写错。

3. 完整目录树

```text
swarm-eod/
  src/swarm_rescue/solutions/
    my_drone_eval.py                     [保持不变：评测入口]
    my_drone_rescue_example.py           [保持不变：官方动作原语]
    my_drone_blue_basic.py               [修改：单次采样和接线]
    blue_team/
      __init__.py                        [共享新增：公开导出]
      contracts.py                       [共享新增：不可变数据契约]
      config.py                          [共享新增：参数集合]
      coordinator.py                     [共享新增：单步编排]
      localization.py                    [本任务新增：EKF 与 frame]
      occupancy_grid.py                  [第 04 模块]
      path_planner.py                    [第 04 模块]
      exploration.py                     [第 01 模块]
      task_allocation.py                 [第 02 模块]
      return_path.py                     [第 03 模块]
      message_codec.py                   [共享新增：通信边界]
  tests/solutions/blue_team/
    test_localization.py                 [本任务新增：确定性测试]
    test_localization_integration.py     [建议新增：控制器接线]
  doc/blue_team/
    01_系统化多机探索实现方案.md
    02_基于代价的目标分配实现方案.md
    03_返航轨迹压缩与安全捷径实现方案.md
    04_占据栅格与A星路径规划实现方案.md
    05_无GPS定位与导航实现方案.md       [本文件]
```

上述 Python 文件是实施目标，目前并非 baseline 已有实现。
本任务原则上只拥有 `localization.py` 和定位测试；共享文件
经团队冻结接口后再由集成负责人修改，避免五人冲突。

4. 每个文件职责

`blue_team/contracts.py`：只定义 `SensorFrame`、`PoseEstimate`、
`FrameTransform`、`LocalizationDiagnostics`、`LocalizationResult`
等不可变类型及状态字面值；不持有 NumPy 数组或滤波器状态。

`blue_team/config.py`：新增 `LocalizationConfig`，统一噪声、
门控和下游阈值；不记录某张地图的墙坐标、炸弹坐标或出发点。

`blue_team/localization.py`：唯一拥有 EKF 的 `x`、`P`、当前
`frame_id`、epoch、上次采纳观测步、GPS 恢复候选缓存；只接收
`SensorFrame`、配置及可选地标对应观测，输出 `LocalizationResult`。
公开接口建议如下：

```python
class LocalizationManager:
    def __init__(self, drone_id: int,
                 config: LocalizationConfig) -> None: ...
    def update(self, frame: SensorFrame) -> LocalizationResult: ...
    def latest(self) -> Optional[PoseEstimate]: ...
    def reset(self, epoch: Optional[int] = None) -> None: ...
```

`blue_team/coordinator.py`：控制步开始创建一次传感器快照，调用
定位器一次，处理 frame 事件，然后把同一 `pose` 交给其他模块。
不得私自重算 GPS 位姿。

`blue_team/occupancy_grid.py`：持有带 frame 的地图。收到
`FrameTransform` 时选择保守重采样或丢弃局部子图，并使地图
revision 递增；不是定位器内部状态。

`blue_team/path_planner.py`：收到 frame/revision 变化就作废旧
A* 路径，不负责坐标校正。

`blue_team/exploration.py`：用同 frame 位姿计算前沿代价；
定位不可信则回退到机载 Lidar 局部探索。

`blue_team/task_allocation.py`：只接受可信的 `world` 位姿参与
跨机竞价，不把 local frame 点广播为世界炸弹。

`blue_team/return_path.py`：面包屑带 frame、step、方差；frame
切换时分段或整体变换，并取消尚未验证的安全捷径。

`blue_team/message_codec.py`：只编码/解码可公开摘要，校验
`frame_id`、方差、step、TTL；不把队友坐标送入本机 EKF。

`my_drone_blue_basic.py`：持有本机 manager/coordinator 实例，
缓存同一步 `pose`；替换自身直接传感器读取，并覆盖父类仍
直接读取 GPS/指南针的三个方法。保留抓取、Lidar 避障及
投放动作原语。

`tests/solutions/blue_team/test_localization.py`：纯合成观测测试，
无需图形窗口；集成测试校验控制器接线及提交检查。

5. SensorFrame 和 PoseEstimate 契约

```python
from dataclasses import dataclass
from typing import Optional, Tuple

WorldPoint = Tuple[float, float]

@dataclass(frozen=True)
class SensorFrame:
    step: int
    gps_position: Optional[WorldPoint]
    compass_heading: Optional[float]
    odometry: Optional[Tuple[float, float, float]]
    measured_velocity: Optional[WorldPoint] = None
    measured_angular_velocity: Optional[float] = None
    gps_disabled: bool = False
    compass_disabled: bool = False
    odometer_disabled: bool = False
    world_size: Optional[WorldPoint] = None

@dataclass(frozen=True)
class PoseEstimate:
    position: Optional[WorldPoint]
    heading: Optional[float]
    position_variance: float
    heading_variance: float
    source: str
    frame_id: str
    step: int
    valid: bool
```

`measured_velocity` 和 `measured_angular_velocity` 默认 `None`；
即使提供，也仅供日志诊断，绝不再送 EKF 校正。原因是它们
分别复用同一里程计（前者还复用指南针），重复融合会错误地
压低协方差。控制周期的采样器应将 ndarray 立即复制成有限值
tuple；禁止保留指向传感器内部可变数组的引用。

`PoseEstimate.position_variance` 定义为 `P_xy` 最大特征值，
即任意水平方向的保守方差上界；`heading_variance=P[2,2]`。
未知/发散用 `float("inf")`。`valid=True` 只表示位置、朝向、
frame、step 均已定义，且 `position_variance<=1600 px^2`、
`heading_variance<=0.25 rad^2`；它不是“可以建图”的承诺。
`position=None` 但 `heading` 有值时为 `HEADING_ONLY`，
`position_variance=inf`、`valid=False`，仅可用于朝向相关的
局部避障，不能投影语义目标。

`source` 的允许值固定为：`UNINITIALIZED`、`GPS_FUSED`、
`ODOM_COMPASS_FUSED`、`DEAD_RECKONING`、`GPS_POSITION_ONLY`、
`HEADING_ONLY`、`RELOCALIZING`、`LOCAL_ONLY`、`LOST`。
`source` 表示本步输出的信息来源，诊断中的 `mode` 表示
持续状态；二者不要混用。`frame_id` 只能是 `world` 或
`local:d<ID>:e<EPOCH>`；不同无人机或不同 epoch 的 local 点
即使数值接近也不能直接相减。所有角度为弧度、`[-pi,pi)`。

6. 状态模型和噪声参数

内部状态 `x=[x,y,h]`，`P` 为 3x3 对称协方差；`h` 是机体
朝向，`x,y` 属于当前 frame。只用允许的测量初始化和更新。

原始传感器模型：

```text
GPS:     z_g = [x_world,y_world] + b_g
Compass: z_h = wrap(h_world + b_h)
Odom:    u = [d,alpha,delta_h] + independent Gaussian noise
b_t = 0.98*b_(t-1) + epsilon_t
std(b_g) = 5 px, std(b_h) = 4 deg
std(epsilon) = std(b)*sqrt(1 - 0.98^2)
```

首版不把 GPS/指南针偏置增广进状态；采用稀疏校正和有效
测量协方差。每隔至少 10 步才用一次常规 GPS 或指南针
校正，`R_eff = sigma_stationary^2*(1+rho^10)/(1-rho^10)`：
GPS 每轴约 250 px²，指南针约 0.049 rad²。初始化用
`Pxy=100 px²`、`Ph=(4 deg)^2`，而非把首次读数当真值。
这些是保守初值，不是理论最优；须用带 AR(1) 偏置的合成
轨迹检验真实 95% 误差是否落入宣称的不确定度。

里程计过程噪声基本值：`sigma_d=0.2 px`，
`sigma_alpha=0.13963 rad`，`sigma_theta=0.01745 rad`。
高速碰撞、推弹、低速滑移可额外放大 1.5 至 3 倍；仅靠
基本噪声可能低估未建模的接触运动。额外最小漂移地板：
平移每步 `q_xy=0.25 px²`、转角每步
`q_h=(0.5 deg)^2`，待评测校准。

互补滤波备选：`x_pred` 按里程计积分，再以小增益拉向 GPS，
朝向拉向指南针，显式加方差。这可快速原型，但没有
`P_xh/P_yh`，长距离转弯的定位置信度容易过乐观。因此
首版实现 EKF，而不是只凭经验增益。

7. 预测和校正公式

源码中 `alpha` 是本步位移相对“上一时刻”朝向的角；所以：

```text
phi = wrap(h_old + alpha)
x_pred = x_old + d*cos(phi)
y_pred = y_old + d*sin(phi)
h_pred = wrap(h_old + delta_h)
```

不能把 `delta_h/2` 再加到 `phi`，否则与本项目里程计定义
不一致。`d` 很小时 `alpha` 对位移没有意义，直接令
`d=0`，只传播 `delta_h`。

Jacobians 与过程噪声：

```text
F = [[1,0,-d*sin(phi)], [0,1,d*cos(phi)], [0,0,1]]
G = [[cos(phi),-d*sin(phi),0],
     [sin(phi), d*cos(phi),0],
     [0,        0,         1]]
Q_u = diag(sigma_d^2,sigma_alpha^2,sigma_theta^2)
Q_floor = diag(q_xy,q_xy,q_h)
P_pred = F*P_old*F.T + G*Q_u*G.T + Q_floor
```

GPS 有效且当前在 `world` frame 时：

```text
H_g = [[1,0,0],[0,1,0]]
r_g = z_g - H_g*x_pred
S_g = H_g*P_pred*H_g.T + R_g
K_g = P_pred*H_g.T*inverse(S_g)
x_new = x_pred + K_g*r_g
P_new = (I-K_g*H_g)*P_pred*(I-K_g*H_g).T + K_g*R_g*K_g.T
```

指南针有效时：

```text
H_h = [0,0,1]
r_h = wrap(z_h - h_pred)
S_h = H_h*P_pred*H_h.T + R_h
K_h = P_pred*H_h.T/S_h
x_new = x_pred + K_h*r_h; x_new[2] = wrap(x_new[2])
P_new = (I-K_h*H_h)*P_pred*(I-K_h*H_h).T + K_h*R_h*K_h.T
```

先指南针、后 GPS，二者可独立校正。没有 GPS 只预测
`x,y`；没有指南针仍可用 `theta` 预测朝向；没有里程计时
只增加过程方差，不根据控制命令假定已经移动，仍允许
GPS/指南针各自更新。更新后对称化 `P`，检查有限值和
最小特征值，异常则恢复上一份安全快照并进入降级。

8. 初始化和 frame 管理

情况 A：GPS 与指南针都有效，首帧设 `frame_id="world"`，
`x=[gps_x,gps_y,compass]`。首帧里程计不积分；GPS/指南针
协方差取上节初始化值。

情况 B：GPS 无效，但指南针和里程计可用，建立
`local:d<ID>:e0`，原点 `(0,0)`，朝向使用指南针绝对角。
此 local 坐标的轴方向初步与世界轴一致，但原点未知且
指南针有偏置；任何炸弹、地图、面包屑不能直接与队友世界
数据合并。若 GPS 和指南针都无效，里程计也不能确定世界
朝向；可建立任意朝向 0 的 local frame，但 `heading` 只
在该 local frame 有意义，`source=LOCAL_ONLY`，禁止跨机使用。

情况 C：有 GPS、无指南针。若已有可信朝向，GPS 初始化
世界平移，朝向由原状态/里程计维护。若冷启动朝向未知，
仅输出 `GPS_POSITION_ONLY`、`heading=None`、`valid=False`。
不能从单帧 GPS 得到机体朝向；无人机可侧移，世界位移
方向也不等于机体朝向。若里程计可用，先在任意方向的
local frame 积分轨迹，积累两组 GPS-局部位置对应；
基线距离至少 35 px，且大于 3 倍相对位置标准差，再用
两条位移向量的夹角估计 local 到 world 的旋转。静止时
保持朝向未知；只有 GPS 且无里程计/指南针时永远不能
可靠恢复机体朝向，不得伪造。

统一变换定义：

```text
p_world = R(rotation)*p_local + translation
h_world = wrap(h_local + rotation)
```

一个 GPS-局部位置对应点只确定平移，不确定旋转；如果
同时有可信的绝对指南针朝向，则该朝向约束旋转，一点
可形成候选变换。为抑制 AR 偏置和异常点，提交变换仍需
至少 3 个相隔 10 步的 GPS 观测，并通过一致性门控。
无可信指南针时，至少两组空间分离的对应点确定候选旋转
和平移，第三组独立点验证残差。`frame_id` 每次创建
local frame 时递增 epoch，不复用旧 local ID。

若短时 GPS 中断，保持 `world` frame 并让 `P` 增长。
一旦位置方差超过建图阈值，停止向世界地图写射线；
如还需要局部记录，则创建新 local frame 和子图，
发出 frame 变化事件，保留之前的世界轨迹为独立段。
这避免把后续高漂移局部点伪装成精确世界点。

9. GPS 丢失、恢复和重定位

状态机以可用观测、当前 frame 和置信度决定动作：

- `UNINITIALIZED`：尚无可用位置/朝向；仅局部 Lidar 动作。
  得到 GPS+指南针进入 `GPS_FUSED`；得到指南针+里程计
  可进入 local `DEAD_RECKONING`。
- `GPS_FUSED`：世界位姿可用；GPS 与指南针按低频门控
  校正，里程计每步预测。GPS 缺失转 `DEAD_RECKONING`。
- `DEAD_RECKONING`：GPS 缺失，里程计预测；指南针若可用
  则继续校正。初始可在 world 或 local frame；`P` 随步数
  增长，依阈值逐项停用世界功能。GPS 恢复转 `RELOCALIZING`。
- `HEADING_ONLY`：无可信位置但有朝向；`position=None`，
  只能旋转、Lidar 避障，不能建图/竞价；获得位置再初始化。
- `RELOCALIZING`：收集多帧 GPS、指南针/里程计对应，
  暂停世界地图写入、竞价和旧路径执行。候选一致后提交
  frame 变换或世界位姿渐进修正；失败继续局部传播。
- `DEGRADED`：位置和朝向可能存在，但方差高或里程计失效；
  只允许低速局部安全动作，必要时原地等传感器恢复。
- `LOST`：所有测量连续无效，或协方差达到上限；发布
  `valid=False`，只允许安全停止/原地慢速搜索。重新获得
  GPS+指南针后走候选重定位，不在一帧盲目切回。

恢复策略比较：立即把 GPS 值赋给位姿会导致地图、目标、
面包屑错位，拒绝；无门控的多步线性插值也可能被跳点
欺骗，拒绝。首选“候选门控 + Kalman 校正”；如果仍在
`world` frame，最多每步修正 8 px、5 度，并冻结地图更新
直到一致；如果在 local frame，先估计并验证刚体变换，
由协调器一次性处理所有持久数据，然后切换到 `world`。
重定位期间保留的局部地图只在变换不确定度足够低时保守
重采样；否则丢弃局部地图，而不是产生伪墙。炸弹 track
位置及其协方差同步旋转、平移、膨胀；面包屑整体变换并
标记转换段。旧路径、竞价租约和消息缓存全部作废。

10. 异常值拒绝

每步先验证 `step` 是非负严格递增整数、shape 正确、所有
数字有限。重复 step 直接返回同一缓存 `LocalizationResult`，
不能重复积分；小于上步则抛出数据契约错误并保持状态。
若 step 跳跃超过 1，里程计只代表最近一步，不能重复
使用它补齐漏掉的历史；先按缺失步数增加不确定度，
本步 odom 可用于最后一步，但当缺口很大时只用于诊断，
进入 `DEGRADED` 等待绝对观测。

门控规则建议：

```text
gps_nis = r_g.T*inverse(S_g)*r_g
accept_gps if gps_nis <= 11.83 and bounds_ok and cadence_ok
compass_nis = r_h*r_h/S_h
accept_compass if compass_nis <= 9.0 and abs(r_h) <= pi/2
accept_odom if finite(u) and -0.6 <= d <= 40 px/step
```

11.83 是二维卡方约 99.73% 门限，9 是一维 3-sigma；
40 px/step 是宽松工程保护值，必须用训练场景的最大
实测单步位移校准，不能从 `LINEAR_SPEED_RATIO` 当作
精确物理极限。`-0.6<=d<0` 可视作零距离测量噪声，
更负则拒绝。超出以地图中心为基准的边界超过 30 px
的 GPS 点拒绝；在边缘 30 px 内需结合方差，不机械截断。
指南针角度先归一化再比较，跨 `pi/-pi` 不应误判跳变。

GPS 缺失期间第一帧恢复不能直接接受大残差；连续 3 个
相隔至少 10 步的候选点满足相互速度/几何一致，才进入
渐进校正或 frame 切换。连续 5 次门控拒绝时，不无限
拒绝真实重定位：转 `RELOCALIZING`，扩大候选搜索窗，
但仍要求时空连续性和独立验证。若 GPS 与指南针出现
慢变 AR 偏置，10 步抽样与 `R_eff` 已抑制重复证据；
创新连续同向 30 步时增加诊断 `possible_bias`，临时
放大测量方差，不能在每步独立融合中把方差虚假降到零。

11. 置信度与下游使用规则

以下阈值是默认方案，不是已实测的比赛最优值：

```text
V_map = 400 px^2       # 约 20 px 单轴保守标准差
V_plan = 225 px^2      # 约 15 px
V_track = 225 px^2
V_breadcrumb = 400 px^2
V_safe = 1600 px^2     # 约 40 px
H_plan = 0.04 rad^2
H_track = 0.02 rad^2
ray_sigma_limit = 30 px
```

- 建图：`valid`、同 frame、步龄不超过 2、
  `position_variance<=V_map`，且每条射线
  `sqrt(V_position + range^2*V_heading)<=30 px` 才写图。
  该条件与 04 方案一致；不满足时保留 Lidar 即时避障，
  不更新持久世界地图。local frame 可有独立子图，
  绝不直接并入世界图。
- A*：同 frame 且 `V_position<=V_plan`、
  `V_heading<=H_plan`，地图 revision 新鲜；否则输出
  `NO_POSE` 或局部安全动作，不执行旧世界路径。
- 探索：可信时用位置/前沿距离；不可信时停止基于
  世界覆盖率的分工，转局部 Lidar 搜索。
- 目标竞价：只接受 `frame_id="world"`、
  `V_position<=V_track`、`V_heading<=H_track`；否则退出
  世界目标竞价并释放或到期本机租约。
- 面包屑：只在同 frame 且 `V_position<=V_breadcrumb`
  时记录；低置信度时开启“断段”标记，不能把断段
  两端直接连成安全捷径。local 面包屑允许供本机短距返航。
- 语义转世界点：要求同 frame、有效朝向，目标点方差
  按 `V_target >= V_position + distance^2*V_heading`
  保守估计；不达 `V_track` 则只保留“相对方位/距离”，
  不创建世界炸弹 track。
- 通信：只广播 `world` 且满足竞价阈值的点与低频
  位姿摘要；local 点留本机。低置信度位置不进入队友地图。
- 运动控制：`V_position>V_safe` 或 pose 无效时停止
  世界目标转向，保留 Lidar 防撞与限速，必要时悬停。

阈值调节依据应是离线误差覆盖率、地图伪墙、撞墙次数和
救援得分；不要仅凭单次轨迹“看起来顺”而降低门槛。

12. 输入契约

`SensorFrame.step` 用仿真逻辑步 `elapsed_timestep`，不是
walltime；由控制器本步开始采样并拥有。GPS 为长度 2
的世界像素 tuple 或 `None`；指南针为弧度 float 或 `None`；
里程计为长度 3 的 `(px,rad,rad)` 或 `None`。shape 错误、
NaN、inf、bool、禁用却非空的值一律视为无效，诊断记录
原因，不能传入矩阵运算。禁用状态分别调用允许的
`gps_is_disabled()`、`compass_is_disabled()`、
`odometer_is_disabled()` 一次；以状态与测量均有效为准。

`world_size` 从 `size_area` 或 `MiscData.size_area` 获得，
仅供边界合理性检查，单位 px。上步内部状态只由
`LocalizationManager` 自身持有，外部不能写入。可选地标
重观测必须带本机相对测量、同 frame 已有地标 track、
方差和时间戳；首版默认不启用，绝不能从地图文件读取
墙或回收中心真值当作观测。

13. 输出契约

```python
@dataclass(frozen=True)
class FrameTransform:
    source_frame: str
    target_frame: str
    translation: WorldPoint
    rotation: float
    confidence: float
    created_step: int
    revision: int = 0

@dataclass(frozen=True)
class LocalizationDiagnostics:
    mode: str
    accepted_gps: bool
    accepted_compass: bool
    accepted_odometry: bool
    gps_innovation: float
    rejected_reason: str
    compass_innovation: float = 0.0
    possible_bias: bool = False

@dataclass(frozen=True)
class LocalizationResult:
    pose: PoseEstimate
    diagnostics: LocalizationDiagnostics
    frame_transform: Optional[FrameTransform] = None
    frame_event: Optional[str] = None
```

`frame_event` 限定为 `LOCAL_STARTED`、`WORLD_COMMITTED`、
`FRAME_QUARANTINED`、`RESET` 或 `None`。本机协调器消费完整
结果；建图和返航消费 `FrameTransform`；日志消费诊断；
通信层仅选择性发送压缩后的 `world` 位姿摘要，不发送
内部 `P`、GPS 原始值、local 轨迹或恢复候选缓存。

实际输出示例：

```python
LocalizationResult(
    pose=PoseEstimate(
        position=(132.4, -87.1), heading=1.28,
        position_variance=120.0, heading_variance=0.012,
        source="ODOM_COMPASS_FUSED", frame_id="world",
        step=932, valid=True,
    ),
    diagnostics=LocalizationDiagnostics(
        mode="DEAD_RECKONING", accepted_gps=False,
        accepted_compass=True, accepted_odometry=True,
        gps_innovation=0.0, rejected_reason="gps_disabled",
    ),
)
```

上例数值仅示意，实际以滤波结果和协方差地板为准。
可广播摘要至少包括版本、`drone_id`、`step`、`frame_id`、
`position`、`heading`、两项方差和 `source`；步龄/TTL 由
接收侧检查。

14. 通信和多机坐标边界

同为 `world` 的坐标可以在不确定度和时间戳门控后融合；
同为字符串 `local:d6:e0` 也只属于编号 6 的该回合该段，
其他飞机不能冒充相同局部原点。不同 frame 没有显式可靠
变换时，不得计算队友-炸弹距离，也不得合并地图/炸弹点。

队友世界位姿不能单独校正本机位姿：缺少本机到队友的
独立相对距离/方位测量，只有队友坐标并没有几何约束。
若后续开发相对观测，应另立协同定位设计并处理相关误差。
frame transform 默认不广播，因为 local frame 仅本机有
意义；在需要共享本机局部图的高级版本中，须连同完整
变换置信度和 epoch 一起发，接收侧可选择拒绝。

通信半径只有 250 px，通信中断不应影响本机 EKF 预测、
重定位候选或本机 Lidar 避障。低置信度点由本机编码层
拦截，不能寄希望于队友再做清洗。

15. 单步调用时序

仿真先采集广播，再调用控制器；因此 `define_message_for_all()`
只读上一控制步完成的快照，消息 `step` 必须写那个旧步数。
控制器本步伪代码：

```python
def control(self) -> CommandsDict:
    step = int(self.elapsed_timestep)
    frame = sample_once(self, step)  # 只在此处测 GPS/指南针/odom
    result = self._localizer.update(frame)
    self._pose = result.pose
    if result.frame_event is not None:
        self._coordinator.apply_frame_event(result)
    if result.diagnostics.mode == "RELOCALIZING":
        self._coordinator.freeze_world_writes_and_paths()
    self._coordinator.update_map_explore_tasks_return(self._pose)
    return self._coordinator.choose_safe_command(self._pose)

def define_message_for_all(self) -> dict:
    return self._message_codec.pack_cached(self._pose)
```

`sample_once()` 对每个 API 调一次，先取得禁用标志，后复制
GPS/指南针/里程计；不调用 `measured_velocity()` 去制造
额外“独立观测”。单步顺序严格为：校验输入、里程计预测、
指南针门控校正、GPS 门控校正、角度归一化、方差保护、
发布不可变结果、处理 frame 事件、下游消费。

16. 与现有代码的迁移步骤

1. 先冻结 `contracts.py` 中 `PoseEstimate` 的八个字段、
   `frame_id` 规则及阈值语义；运行旧蓝队控制器回归基线。
2. `MyDroneBlueBasic.__init__()` 创建每机独立 manager；
   `control()` 首句按 `elapsed_timestep` 采样并更新一次，
   将本步结果缓存为 `self._pose`。
3. 把 `_remember_home_position()`、`_choose_remembered_bomb()`、
   `_control_to_remembered_bomb()`、`_record_breadcrumb()`、
   `_next_return_waypoint()`、`_control_carry()`、
   `_disposal_target()` 中的直接读取替换成缓存 pose，
   所有记忆增加 frame/step/方差元数据。
4. 在子类覆盖父类 `_semantic_to_world()`、`_move_toward()`、
   `_within_radius()`，保留原签名以兼容父类调用，但内部
   统一读取缓存 pose；同时给协调器新增显式接收 pose
   的纯函数版本。`_find_closest_semantic()` 通过动态分派
   使用覆盖后的转换；父类 `_control_carry()` 和
   `_control_carry_push()` 也会动态分派到覆盖后的导航方法。
5. 覆盖后的导航方法必须校验目标所属 frame；原父类
   `_DISPOSAL_APPROACH` 等硬编码回退不能当作可靠地图点。
   当 pose 无效时，不能执行父类向目标前进的默认分支，
   应转局部安全动作。确认 `_disposal_target()` 已由子类
   给出“已观测且同 frame”的目标；否则不要调用父类
   世界坐标投放流程。
6. 接入地图、A*、探索、竞价、返航后，只传本步同一
   `PoseEstimate`；接收 frame 事件时统一清理旧缓存。
7. 配置开关 `use_fused_localization` 对比旧直读行为与
   新位姿行为；稳定后删掉散落的测量调用，但保留开关
   一段时间用于回归。`my_drone_eval.py` 入口无需改。

`_point_from_sensor()` 可继续作为输入解析器，但不能成为
其他模块绕过 `PoseEstimate` 的“位置来源”。

17. 参数和默认值

`LocalizationConfig` 建议字段及初值如下，均允许在训练
场景实验后调整：

```python
@dataclass(frozen=True)
class LocalizationConfig:
    gps_sigma_px: float = 5.0
    compass_sigma_rad: float = 0.0698132
    ar_rho: float = 0.98
    odom_dist_sigma_px: float = 0.2
    odom_alpha_sigma_rad: float = 0.1396263
    odom_theta_sigma_rad: float = 0.0174533
    correction_period_steps: int = 10
    q_xy_px2_per_step: float = 0.25
    q_heading_rad2_per_step: float = 0.00007615
    initial_position_variance_px2: float = 100.0
    initial_heading_variance_rad2: float = 0.0048739
    max_odom_distance_px: float = 40.0
    gps_bounds_margin_px: float = 30.0
    gps_nis_limit: float = 11.83
    compass_nis_limit: float = 9.0
    relocalization_samples: int = 3
    relocalization_baseline_px: float = 35.0
    max_world_correction_px_per_step: float = 8.0
    max_heading_correction_rad_per_step: float = 0.0872665
    max_position_variance_px2: float = 1000000.0
    max_heading_variance_rad2: float = 3.2898681
    v_map_px2: float = 400.0
    v_plan_px2: float = 225.0
    v_track_px2: float = 225.0
    v_safe_px2: float = 1600.0
```

`max_heading_variance_rad2` 是约 104 度标准差对应的上限，
只是数值夹限；达到之前就应因 `valid=False` 退出世界
导航。若合成试验发现卡方拒绝率过高，优先检查噪声模型、
时序和坐标系，而不是无限放大门槛。

18. 数值稳定性和性能

状态维数固定为 3，每步矩阵乘法和 2x2/1x1 更新都是
`O(1)` 时间、`O(1)` 内存；每机一个实例，不需要大型
滤波框架。NumPy 可复用固定尺寸临时矩阵，但首版应先
追求正确性，避免原地覆盖造成 `P` 别名错误。

每次校正使用 Joseph 形式更新协方差；之后执行
`P=(P+P.T)/2`，特征值低于 `1e-9` 时抬高至地板，
高于配置上限时进入降级而非继续信任。求逆用
`np.linalg.solve()`，若 `S` 非正定或非有限，拒绝该观测。
角度的预测、innovation、校正后输出都调用同一
`normalize_angle()`。输入任何 NaN/inf 不得进入状态；
一旦内部状态非有限，回退最近安全快照并输出 `LOST`。

长时间无 GPS 时可把协方差限幅，但不能因限幅而把
`valid` 重新置真；必须另存 `lost_since_step` 和
`uncertainty_saturated`。每次新回合或新的无人机对象
构建时调用 `reset()`，清空状态、候选点、诊断计数和
局部 epoch；绝不跨回合复用上回合世界坐标。

19. 失败模式和退化策略

- GPS 与指南针同时消失：`NoGpsZone` 的正常情况。
  里程计继续积分，方差增长；超过阈值后停止世界建图
  和竞价，必要时开启 local 子图和局部返航段。
- 仅指南针消失：用里程计 `theta` 推朝向，角方差增长；
  仍有 GPS 时可修正位置，朝向不足则不投影语义世界点。
- 仅里程计消失：不假设控制命令一定生效；GPS/指南针
  可继续更新，期间加时间缺口方差；两者再失效则停止。
- 全部暂时消失：输出上一可信位置的预测/失效摘要，
  不写地图、不沿旧路径加速；Lidar 若仍可用则只低速
  避障，KillZone 中 Lidar 也可能禁用，应安全停机。
- 有限值大跳点：按 NIS 和几何连续性拒绝，不更新
  世界记忆；连续拒绝进入恢复候选，不直接重置 EKF。
- GPS 恢复但漂移很大：候选验证后转 frame，暂停
  相关模块；转换不可靠的数据隔离或舍弃，宁可重新建图。
- 传感器连续返回重复旧值：检查 step、里程计运动与
  GPS/指南针停滞是否矛盾，报告诊断并提高测量方差。
- 地图边界附近测量略超界：30 px 余量内允许观测，
  禁止把估计点硬裁到边界墙上。

20. 测试与验收

纯单元测试必须覆盖：三传感器正常、GPS 丢失后直线与
转弯、指南针丢失而 `theta` 可用、里程计丢失而 GPS
可用、全部失效、冷启动无 GPS、冷启动仅 GPS、
GPS 恢复与 frame 变换、GPS 跳点拒绝、指南针跨
`pi/-pi`、NaN/inf/错误 shape、无 GPS 方差单调增加、
恢复后方差下降、同 step 重复 update 返回同一对象或
等价快照且不重复积分、跳步不重复应用单步 odom。

合成轨迹包括直线、圆周、方形闭环、静止、GPS 中断
100/500/1000 步、带 `rho=0.98` 的 GPS/指南针偏置、
带三项里程计噪声。每类固定随机种子至少 100 条轨迹；
用外部轨迹真值计算位置 RMSE、角度 RMSE、断 GPS
末端漂移、95% 方差覆盖率、正常观测误拒率、重定位
收敛步数和每步滤波耗时。真值只在测试器外部使用，
不得流入控制器或 `SensorFrame`。

集成测试：正常区域 Map01/Map02 与当前蓝队基线对比，
不得系统性降低成功率；包含 `NO_GPS_ZONE` 的计划不崩溃；
GPS 中断仍能记录同 frame 的局部面包屑；恢复后无
地图伪墙和危险目标跳变。固定炸弹布局每配置重复至少
20 轮，报告均值与分位数，不只挑最高分。建议首版
可执行验收目标：控制步定位耗时中位数 <0.2 ms、
99 分位 <1 ms；正常观测误拒率 <5%；恢复后 50 步
内完成变换或明确保持降级；地图伪墙数不高于直读
基线。漂移和成功率目标须以实际场景标定，不虚构数字。

验证命令示意：

```text
python -m pytest tests/solutions/blue_team/test_localization.py -q
python -m swarm_rescue.submission_check --help
```

提交检查的确切 CLI 参数应在实施时先读 `--help`，不要把
上面的示意当作已运行通过的验证结果。图形仿真仍需要
有效 X11 或 Xvfb；纯定位单元测试不需要它。

21. 分阶段实施清单

阶段 1：冻结契约和阈值，建立纯单元测试数据生成器；
实现 `SensorFrame` 采样、验证、`PoseEstimate` 和
`LocalizationConfig`。先保证同一步只更新一次。

阶段 2：实现 3x3 预测、独立指南针/GPS 校正、Joseph
协方差和异常门控；通过直线、转弯、静止与缺失传感器
测试。此阶段先保留单一 `world` frame。

阶段 3：实现 local frame 初始化、无 GPS 方差增长、
恢复候选与 `FrameTransform`；补 frame 事件的原子消费
接口。先允许保守丢弃不可靠的 local 地图，再做重采样。

阶段 4：修改 `MyDroneBlueBasic` 单次采样与父类方法覆盖，
接入其炸弹、回收区、面包屑记忆；用旧/新开关做同种子
回归。不要一次重写抓取和避障动作。

阶段 5：与第 01 至 04 模块联调阈值、消息与 frame 事件；
跑 20 轮固定布局、无 GPS 区以及提交检查。以得分、
误差覆盖率、伪墙和安全性共同决定是否默认启用。

22. 尚需确认的问题

1. 五模块团队需冻结 `PoseEstimate.valid` 的统一阈值，
   尤其地图 04 方案的 `400 px²` 与竞价的 `225 px²`。
2. local 子图恢复时是保守重采样还是直接丢弃？首版建议
   高变换不确定度时丢弃，避免地图伪墙。
3. 当前比赛评测配置中 `NO_GPS_ZONE` 出现频率、大小和
   持续步数是多少？源码支持该区，但不能据此断言每轮
   都会出现；阈值需通过实际计划调校。
4. 地图/任务/返航模块如何一次性确认 frame 事件已处理？
   协调器应定义提交屏障，不能某模块已切 world、另一
   模块仍读取 local 旧路径。
5. 是否需要增广 GPS/指南针 AR 偏置状态？若 3 状态方案
   在长距离 GPS 区域系统性偏移，第二版可评估 5/6 状态
   偏置滤波，但不能把同一测量当作独立证据。
6. `measured_velocity()` 注释称 px/s，而实现直接使用
   单步里程计距离；在未核实时间换算前不要用它参与
   运动模型或速度门控。
