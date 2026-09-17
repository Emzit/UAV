# 队伍开发指南

> 红队（Place）和蓝队（Rescue）的统一 API 参考与代码示例。

---

## 0. 编码规则 — 务必遵守

### 修改范围限制

**你的所有代码只能写在 `src/swarm_rescue/solutions/` 目录中。**

| 允许 | 禁止 |
|------|------|
| 在 `solutions/my_drone_eval.py` 中编写无人机控制逻辑 | 修改 `simulation/` 下的模拟器核心代码 |
| 在 `solutions/` 下添加额外的 `.py` 辅助模块 | 修改 `maps/` 下的地图文件 |
| 通过 `solutions/team_info.yml` 填写队伍信息 | 修改 `config/` 下的评测配置文件 |
| 通过 `solutions/req.txt` 声明额外依赖 | 修改 `launcher.py` 或任何 `eval_cli/` 代码 |

### 禁止动态修改框架代码

**即使在 `solutions/` 内部，也不允许通过运行时动态修改模拟器框架的行为。**

禁止但不限于以下手段：
- 通过 `setattr()` / `__dict__` 修改框架类的私有属性或方法
- 通过 `sys.modules` 替换框架模块
- 通过 `gc.get_objects()` 遍历并修改框架对象
- 通过 `ctypes` / `inspect` 篡改模拟器内存
- 任何试图绕过 `competition_mode` 真值 API 拦截的手段
- 任何试图绕过传感器噪声 / 干扰区的手段
- 任何行数过多的做法

**AI 审查**：提交代码将由 AI 进行代码审查。一旦发现上述修改框架代码的证据，该次提交**直接计为 0 分**。

### 依赖声明

如需使用第三方库（如 `numpy`, `scipy` 等），请在项目根目录的 `req.txt` 中逐行声明。评测环境会根据 `req.txt` 自动安装依赖。

### 提交流程

1. 仅在 `solutions/` 下编写代码
2. 运行 `./check_submission.sh` 验证无误
3. 运行 `./create_zip_submission.sh` 打包
4. 再次验证 zip：`./check_submission.sh teamNNN_evalstep.zip`

详见 [`03-提交检查与打包.md`](03-提交检查与打包.md)。

---

## 1. 红蓝队任务对比

| | 红队（Place） | 蓝队（Rescue） |
|------|-------------|-------------|
| **目标** | 探索地图 → 放置炸弹 → 提交探索地图 | 搜索炸弹 → 抓取 → 运送至 DisposalCenter |
| **炸弹来源** | 地图初始 **0 枚**，无人机携带库存（`initial_bombs_per_drone`） | 地图预置炸弹（由红队布设或 JSON 文件加载） |
| **独占能力** | `place_bomb`（放置炸弹）、`submit_exploration_map()`（提交地图） | —— |
| **回合结束** | 步数上限 / 时间上限 / 调用 `submit_exploration_map()` | 步数上限 / 时间上限 |

---

## 2. 快速开始 — 无人机类结构

你需要创建一个继承自 `DroneAbstract` 的子类，并**必须实现**两个抽象方法：

- `define_message_for_all(self)` — 定义广播消息
- `control(self) -> CommandsDict` — 定义每步控制指令

```python
import numpy as np
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.drone.controller import CommandsDict


class MyDroneRescue(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 在此初始化你的内部状态

    def define_message_for_all(self):
        """向范围内其他无人机广播的消息"""
        return self.measured_gps_position()

    def control(self) -> CommandsDict:
        """每仿真步调用一次，返回控制指令字典"""
        cmd = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        # === 你的控制逻辑在此 ===
        return cmd
```

**关键说明**：
- `__init__` 接收 `identifier`（int）、`misc_data` 等参数，必须透传给 `super().__init__()`。
- 蓝队的 CommandsDict 只有 **4 个键**；红队多一个 `"place_bomb"` 键。
- 所有传感器数据通过 `self.xxx_values()` 方法获取，受噪声和干扰区影响。

---

## 3. 传感器 API 完整参考

所有传感器返回的数值都可能受到噪声影响（GPS 含 AR(1) 噪声，Lidar 含高斯噪声等），且可能在特殊区域（禁能区）内被禁用。

### 3.1 GPS 传感器

获取无人机当前位置的测量值（像素坐标）。

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `measured_gps_position()` | `np.ndarray \| None` | `[x, y]` 像素坐标，GPS 被禁用时返回 `None` |
| `gps_values()` | `np.ndarray \| None` | 原始 GPS 值 |
| `gps_is_disabled()` | `bool` | GPS 是否被禁用 |

```python
pos = self.measured_gps_position()
if pos is not None:
    x, y = pos[0], pos[1]
```

### 3.2 罗盘传感器

获取无人机当前的朝向角度。

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `measured_compass_angle()` | `float \| None` | 弧度值，范围 `[-pi, pi]`，被禁用时返回 `None` |
| `compass_values()` | `float \| None` | 原始罗盘值 |
| `compass_is_disabled()` | `bool` | 罗盘是否被禁用 |

```python
heading = self.measured_compass_angle()
if heading is not None:
    pass  # heading 范围 [-pi, pi]
```

### 3.3 里程计传感器

获取无人机当前的运动状态。

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `measured_velocity()` | `np.ndarray \| None` | `[vx, vy]` 像素/秒，被禁用时返回 `None` |
| `measured_angular_velocity()` | `float \| None` | 角速度，弧度/秒，被禁用时返回 `None` |
| `odometer_values()` | `np.ndarray \| None` | `[dist_travel, alpha, theta]` 原始里程计三元组 |
| `odometer_is_disabled()` | `bool` | 里程计是否被禁用 |

**里程计三元组说明**：

| 分量 | 含义 |
|------|------|
| `dist_travel` | 上一步到这一步的位移距离（像素） |
| `alpha` | 当前位置相对于上一步无人机朝向的相对角度（弧度） |
| `theta` | 上一步到这一步的朝向变化量（弧度） |

```python
odom = self.odometer_values()
if odom is not None:
    dist_travel = odom[0]   # 单步位移
    alpha = odom[1]          # 相对角度
    theta = odom[2]          # 朝向变化

vel = self.measured_velocity()
if vel is not None:
    vx, vy = vel[0], vel[1]

ang_vel = self.measured_angular_velocity()
```

### 3.4 Lidar 传感器

激光雷达，用于测距和避障。

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `lidar_values()` | `np.ndarray \| None` | 181 个距离值，被禁用时返回 `None` |
| `lidar_rays_angles()` | `np.ndarray` | 181 个射线角度（弧度），从 `-pi` 到 `pi` 均匀分布 |
| `lidar_is_disabled()` | `bool` | Lidar 是否被禁用 |

**规格**：分辨率 181 条射线，视场角 360°，最大测距 300 像素。

```python
distances = self.lidar_values()
angles = self.lidar_rays_angles()

if distances is not None:
    # 前方 ±20° 范围内的最小距离
    front_mask = (angles >= -math.pi / 9) & (angles <= math.pi / 9)
    front_min = np.min(distances[front_mask])
```

### 3.5 语义传感器

识别周围实体的类型和状态。

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `semantic_values()` | `list[Data] \| None` | 35 条检测结果列表，被禁用时返回 `None` |
| `semantic_is_disabled()` | `bool` | 语义传感器是否被禁用 |

**规格**：分辨率 35 条射线，视场角 360°，最大检测距离 200 像素。**墙壁（WALL）会被自动过滤，不会出现在返回列表中。**

**Data 命名元组结构**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `distance` | `float` | 到检测实体的距离（像素） |
| `angle` | `float` | 射线角度（弧度，相对于无人机朝向） |
| `entity_type` | `TypeEntity` | 实体类型枚举 |
| `grasped` | `bool` | 是否已被其他无人机抓取（仅对 BOMB 有效） |

**TypeEntity 枚举值**：

| 枚举值 | 对应实体 |
|--------|----------|
| `TypeEntity.BOMB` | 炸弹 |
| `TypeEntity.DISPOSAL_CENTER` | 处置中心 |
| `TypeEntity.DRONE` | 其他无人机 |
| `TypeEntity.OTHER` | 其他实体 |

```python
semantic = self.semantic_values()
if semantic is not None:
    for data in semantic:
        if data.entity_type.name == "BOMB" and not data.grasped:
            # 发现未被抓取的炸弹
            dist = data.distance
            angle = data.angle
        elif data.entity_type.name == "DRONE":
            # 发现其他无人机
            pass
        elif data.entity_type.name == "DISPOSAL_CENTER":
            # 发现处置中心
            pass
```

### 3.6 通信器

无人机间通信由 `communicator` 对象管理。

| 属性/方法 | 类型 | 说明 |
|-----------|------|------|
| `communicator.received_messages` | `list[tuple[Communicator, Any]]` | 当前步收到的消息列表，每条为 `(发送者通信器, 消息)` |
| `communicator_is_disabled()` | `bool` | 通信是否被禁用 |

**通信规格**：通信半径 **250 像素**。每步自动清除上一轮消息。消息内容为任意 Python 对象。

```python
# 发送：在 define_message_for_all() 中返回消息
def define_message_for_all(self):
    return {
        "id": self.identifier,
        "pos": self.measured_gps_position(),
        "has_bomb": len(self.grasped_bombs()) > 0,
    }

# 接收：在 control() 中读取
def control(self) -> CommandsDict:
    for sender_comm, msg in self.communicator.received_messages:
        drone_id = msg["id"]
        pos = msg["pos"]
```

---

## 4. 执行器（CommandsDict）参考

`control()` 方法必须返回一个 `CommandsDict` 字典，支持以下键：

| 键 | 类型 | 范围 | 说明 | 队伍 |
|----|------|------|------|------|
| `"forward"` | `float` | `[-1.0, 1.0]` | 前向推力，正值前进 | 通用 |
| `"lateral"` | `float` | `[-1.0, 1.0]` | 横向推力，正值右移 | 通用 |
| `"rotation"` | `float` | `[-1.0, 1.0]` | 角速度，正值逆时针 | 通用 |
| `"grasper"` | `int` | `{0, 1}` | 1=尝试抓取，0=释放 | 通用 |
| `"place_bomb"` | `int` | `{0, 1}` | 上升沿触发：0→1 放置炸弹 | 红队 |

### 抓取机制（通用）

- 设置 `"grasper": 1` 启用抓取器，与炸弹重叠即自动抓取（最多 1 枚）
- 设置 `"grasper": 0` 释放当前持有的炸弹
- `grasped_bombs()` 返回当前持有的炸弹列表

### `place_bomb` 上升沿机制（红队）

`place_bomb` 采用**上升沿**检测，只有 0→1 的转换才会触发一次放置：

```python
def control(self) -> CommandsDict:
    cmd = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0, "place_bomb": 0}

    if self._steps % 60 == 0 and self.carried_bombs_count() > 0:
        cmd["place_bomb"] = 1  # 触发上升沿放置

    self._steps += 1
    return cmd
```

放置失败的条件（静默失败，不报错）：库存用完、无人机正在抓持炸弹、抓取器处于可抓取状态、与已有炸弹间距 < 24 像素。

---

## 5. 禁止在 `control()` 中使用的 API

在比赛模式（`--competition`）或批量评估期间，以下 API 在 `control()` 方法内部调用会**抛出异常**：

| 方法 | 替代方案 |
|------|----------|
| `true_position()` | `measured_gps_position()` |
| `true_angle()` | `measured_compass_angle()` |
| `true_velocity()` | `measured_velocity()` |
| `true_angular_velocity()` | `measured_angular_velocity()` |

以下属性访问**任何时候都会抛出异常**（已被彻底禁用）：

| 禁止访问 | 替代方案 |
|----------|---------|
| `self.position` | `measured_gps_position()` |
| `self.angle` | `measured_compass_angle()` |
| `self.velocity` | `measured_velocity()` |
| `self.angular_velocity` | `measured_angular_velocity()` |

**注意**：即使在本地调试（非 competition 模式）下，也不应在 `control()` 中使用真实值 API。好的策略应仅依赖传感器数据。

---

## 6. 其他可用属性与方法

| 属性/方法 | 类型 | 说明 |
|-----------|------|------|
| `identifier` | `int` | 无人机唯一标识符 |
| `size_area` | `tuple \| None` | 地图尺寸 `(width, height)`，像素 |
| `drone_health` | `int` | 当前生命值（初始 **50**，碰墙或碰无人机减 1，归零被移除） |
| `elapsed_timestep` | `int` | 从回合开始到当前的仿真步数 |
| `elapsed_walltime` | `float` | 从回合开始到当前的墙钟时间（秒） |
| `is_inside_return_area` | `bool` | 当前是否处于返回区域内 |
| `grasped_bombs()` | `list` | 当前抓持的炸弹对象列表（最多 1 个） |
| `carried_bombs_count()` | `int` | 剩余可放置炸弹数（红队），蓝队始终返回 0 |
| `submit_exploration_map(grid)` | `None` | 红队提交探索地图并立即结束当前回合 |

---

## 7. 红队（Place）专用功能

### 7.1 `submit_exploration_map(grid)` 详解

调用后提交探索地图并**立即结束当前回合**。

```python
def submit_exploration_map(self, grid: np.ndarray) -> None:
```

- `grid`：二维 `np.ndarray`，形状 `(H, W)`，`H = self.size_area[1]`，`W = self.size_area[0]`
  - **1** / 非零 = 障碍物，**0** = 自由空间
  - 推荐 `np.uint8` 类型
  - **坐标约定**：`grid[row, col]` 为图像坐标，原点在左上角。世界坐标 `(x, y)` 的换算为 `col = round(x + W/2)`，`row = round(-y + H/2)`（注意 y 轴翻转）
  - 分辨率为 **1 格 = 1 世界像素**，评分时逐像素比对，不做缩放。粗分辨率栅格上采样提交会让边缘像素越过免罚区而被扣分

```python
import numpy as np

def _build_and_submit_map(self) -> None:
    if self.size_area is None:
        return
    w, h = self.size_area
    grid = np.zeros((h, w), dtype=np.uint8)

    # 基于 Lidar 和定位信息标记障碍物
    lidar = self.lidar_values()
    gps = self.measured_gps_position()
    heading = self.measured_compass_angle()
    if lidar is not None and gps is not None and heading is not None:
        angles = self.lidar_rays_angles()
        for dist, angle in zip(lidar, angles):
            if dist < 300:  # 检测到障碍物
                world_angle = heading + angle
                x = gps[0] + dist * math.cos(world_angle)
                y = gps[1] + dist * math.sin(world_angle)
                col = round(x + w / 2)
                row = round(-y + h / 2)  # y 轴翻转
                if 0 <= col < w and 0 <= row < h:
                    grid[row, col] = 1  # 标记为障碍物

    self.submit_exploration_map(grid)
```

### 7.2 探索地图如何计分

计分方式是「**命中得分，误画扣分**」。加分只看真实障碍被你覆盖了多少，扣分只看你画的障碍里有多少离真墙太远：

```
地图分 = clamp(加分 - 扣分, 0, 1)
探索分 = 100 × 地图分        # 时间分单独计算（详见下方“红队总分”说明）
```

> **真值只含真实墙体**：计分对比的真值来自地图的真实墙体实体（线段墙、矩形墙、箱子、边框墙）。
> **处置中心、返回区、干扰区不算墙**——把它们画进地图不会加分，还会因远离真墙而按误检扣分。

红队总分由四部分组成（默认权重，可在 `place_scoring` 调整）：

```
红队总分 = 50% × 探索分 + 20% × 炸弹难度分 + 20% × 炸墙分 + 10% × 时间分
```

其中**时间分按推演步数**计算：用时 ≤ 一半时限即满分，达到最长时限为 0 分，中间线性递减（红队回合在提交探索地图时结束，`elapsed_timestep` 即整轮用时）。

**加分（命中率）** 把真实障碍按厚度分为两个区域，分别统计被覆盖的比例：

| 区域 | 权重 | 说明 |
|------|-----|------|
| 墙体表面（距自由空间 ≤ 4 px） | 1.0 | 地图分只看这部分。无人机真正能观测到的部分。引擎默认墙厚 6 px，因此一整面标准墙都算在这里 |
| 深层内部（更深处） | 0.0 | 不计分。大片实心区块，靠整片填充就能猜中 |

只要该像素上你也画了障碍就算命中，**不要求精确对齐**。

**扣分** 遍历你画的每个障碍像素，按它到最近真墙的距离收费：

| 距离 | 收费 |
|------|-----|
| ≤ 10 px | **0**（免罚区） |
| 11 px 起 | 按 `0.1 0.1 0.2 0.2 0.3 0.3 ...` 递增 |
| ≥ 30 px | 1.0（触顶） |

扣分总和除以**墙体表面像素数**，再乘系数 0.2。分母用墙面而非全部障碍，是因为误检本质上是「墙的位置判断错了」，深层大块不提供这类信息。这个分母是固定的，多画多少就扣多少，无法靠画满来稀释。

实践建议：

1. **填充障碍物**：墙体按实际厚度（默认 6 px）整体填充，而不是只画 1 px 的线。只描轮廓在示例地图上只能拿 0.27 的地图分。
2. **不要整片涂黑**：加分侧确实会因此满分，但扣分侧会把它彻底抵消——实测全黑提交扣分达 5.47，地图分为 0。同理，随机噪声也是 0 分。
3. **10 px 以内的误差是免费的**：GPS（σ=5 px）与罗盘（σ=4°）噪声会使激光点投影产生约 10~22 px 的误差，免罚区正是按这个量级设定的。轻微的墙厚误差或 1~2 px 偏移不会扣分，不必为此过度优化。
4. **但超出免罚区代价很快上升**：与其把墙画粗以求「覆盖到」，不如控制定位误差。同一帧内所有射线共享同一偏差，多帧累积、优先使用近距离射线、或做扫描匹配校正都有帮助。实测在真实地图上，把提交整体加粗**不会**提分。
5. **提交 1:1 分辨率**：内部用粗栅格建图可以，但提交前应重建全分辨率地图（例如将激光点拟合成线段后按墙厚绘制）。直接上采样粗栅格会让边缘像素越过免罚区而被收费。
6. **时间分只占 10%，且按推演步数计**：用时 ≤ 一半时限即满分，达到最长时限为 0，线性递减。但**不要为赶时间牺牲其他分**——提前提交只拿到时间分，探索/炸弹/炸墙分可能很低；全队放完炸弹、完成建图后再提交是更稳的选择。
7. **不要画处置中心/返回区/干扰区**：真值只统计真实墙体，这些地块不算墙。它们位于真实墙体之外，画上去即落入扣分区（超过 10 px 免罚区就开始收费）。

评测会为每轮生成一张三面板对比图 `*_exploration_diff.png`（真值[真实墙体栅格] | 你的提交 | 计分视图）。计分视图用颜色标出每个像素的贡献：绿色为得分（越亮代表该区域权重越高）、蓝灰为计分区域内的漏检、灰色为当前权重下不计分的真值（`w_interior = 0` 时的深层内部，命中与否都不影响得分）、琥珀为免罚区内的预测、红色深浅表示扣分多少。图例里直接写出了得分算式，可以据此定位失分位置。

另外每轮还会生成一张炸墙可视化图 `*_wall_destruction.png`（单面板：地图墙体 + 炸弹位置与爆炸覆盖范围，被炸毁的墙体单独着色）：绿色/蓝色为被炸毁的内部墙/边界墙，灰色为完好墙体，蓝色区为爆炸覆盖范围，图例写出加权炸毁比例与炸墙分。可以用来直观核对炸弹放在哪些位置才覆盖了你想要的墙体。

### 7.3 Leader 汇总后提交

```python
def _leader_try_submit(self) -> None:
    """仅 0 号机在全队库存耗尽后提交"""
    if int(self.identifier or 0) != 0:
        return

    # 通过通信汇总所有无人机的库存
    total_inventory = self.carried_bombs_count()
    for _, msg in self.communicator.received_messages:
        if isinstance(msg, dict):
            total_inventory += msg.get("inv", 0)

    if total_inventory == 0:
        self._build_and_submit_map()
```

---

## 8. 蓝队（Rescue）专用功能

### 8.1 评分构成（满分 100）

蓝队总分三项加权（与红队分项独立，权重不可调）：

- **炸弹处置 60%**：已处置炸弹数 / 炸弹总数 × 100。
- **返航健康 20%**：返回区（或无返航区时全场）内无人机平均健康度（初始健康满分）。
- **时间 20%**：与红队时间分同一公式——按推演步数计，**用时 ≤ 一半时限满分，达到最长时限为 0 分**，中间线性递减，无完成门槛。回合在**全部炸弹处置完成时提前结束**（未拆完则跑到步数/时间上限，时间分通常为 0；回合崩溃时间分为 0）。

> 例：地图上限 2000 步，1500 步拆完全部炸弹 → `时间分 = (2000−1500)/(2000−1000) × 100 = 50`。想拿满分时间分需在**一半时限内**拆完全部炸弹。

### 8.2 寻找 DisposalCenter

蓝队需要将炸弹运送至 DisposalCenter 拆除。可通过语义传感器发现：

```python
semantic = self.semantic_values()
if semantic is not None:
    for hit in semantic:
        if hit.entity_type.name == "DISPOSAL_CENTER":
            world_angle = compass + hit.angle
            dp_x = gps[0] + hit.distance * np.cos(world_angle)
            dp_y = gps[1] + hit.distance * np.sin(world_angle)
            self._disposal_pos = (dp_x, dp_y)
            break
```

### 8.3 多机协作建议

- **角色分工**：利用 `self.identifier` 分配不同角色（搜索、搬运、协调）
- **信息共享**：通过 `define_message_for_all()` 广播 DisposalCenter 位置和已发现炸弹坐标
- **生命值管理**：`is_inside_return_area` 为 `True` 时进入回血区，生命值逐渐恢复
- **通信范围**：仅 250 像素，大地图需编队飞行或约定汇合点

---

## 9. 完整示例：红队探索与放置

```python
import math
import numpy as np

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.drone.controller import CommandsDict


class MyDronePlace(DroneAbstract):
    """红队无人机：探索 + 放置炸弹 + 提交地图"""

    _LEADER_ID = 0
    _PLACE_INTERVAL = 60
    _STRAIGHT_STEPS = 90
    _TURN_STEPS = 28
    _FORWARD_SPEED = 0.7
    _TURN_SPEED = 0.35
    _BLOCK_DIST = 35.0
    _PATH_CONE = math.pi / 9

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._steps = 0
        self._explore_tick = 0
        self._exploration_submitted = False
        self._peer_inventory = {}

    def define_message_for_all(self):
        return {"id": self.identifier, "inv": self.carried_bombs_count()}

    def control(self) -> CommandsDict:
        self._steps += 1
        cmd = self._empty_command()

        self._roam(cmd)

        if self._steps % self._PLACE_INTERVAL == 0:
            self._try_place(cmd)

        if int(self.identifier or 0) == self._LEADER_ID and not self._exploration_submitted:
            self._try_submit()

        return cmd

    def _roam(self, cmd):
        lidar = self.lidar_values()
        angles = self.lidar_rays_angles()
        if lidar is not None:
            front_mask = (angles >= -self._PATH_CONE) & (angles <= self._PATH_CONE)
            front_min = np.min(lidar[front_mask]) if np.any(front_mask) else float("inf")
            if front_min < self._BLOCK_DIST:
                left_mask = (angles > 0.1) & (angles < 0.8)
                right_mask = (angles < -0.1) & (angles > -0.8)
                left_min = np.min(lidar[left_mask]) if np.any(left_mask) else 0
                right_min = np.min(lidar[right_mask]) if np.any(right_mask) else 0
                cmd["rotation"] = self._TURN_SPEED if left_min >= right_min else -self._TURN_SPEED
                return

        cycle = self._STRAIGHT_STEPS + self._TURN_STEPS
        if self._explore_tick % cycle < self._STRAIGHT_STEPS:
            cmd["forward"] = self._FORWARD_SPEED
        else:
            spin = self._TURN_SPEED if (self.identifier or 0) % 2 == 0 else -self._TURN_SPEED
            cmd["rotation"] = spin
        self._explore_tick += 1

    def _try_place(self, cmd):
        if self.carried_bombs_count() > 0:
            cmd["place_bomb"] = 1

    def _try_submit(self):
        for _, msg in self.communicator.received_messages:
            if isinstance(msg, dict):
                self._peer_inventory[msg["id"]] = msg.get("inv", 0)
        self._peer_inventory[self.identifier] = self.carried_bombs_count()
        if all(v == 0 for v in self._peer_inventory.values()):
            if self.size_area is not None:
                w, h = self.size_area
                grid = np.zeros((h, w), dtype=np.uint8)
                self.submit_exploration_map(grid)
                self._exploration_submitted = True

    @staticmethod
    def _empty_command() -> CommandsDict:
        return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0, "place_bomb": 0}
```

---

## 10. 完整示例：蓝队探索 + 抓取 + 运送

```python
import numpy as np

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.drone.controller import CommandsDict


class MyDroneRescue(DroneAbstract):
    """蓝队无人机：搜索炸弹 → 抓取 → 运送至 DisposalCenter"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state = "search"
        self._target_pos = None
        self._disposal_pos = None
        self._has_bomb = False

    def define_message_for_all(self):
        return {
            "id": self.identifier,
            "pos": self.measured_gps_position(),
            "disposal": self._disposal_pos,
        }

    def control(self) -> CommandsDict:
        cmd = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        gps = self.measured_gps_position()
        if gps is None:
            return cmd
        compass = self.measured_compass_angle()
        if compass is None:
            return cmd

        lidar = self.lidar_values()
        self._has_bomb = len(self.grasped_bombs()) > 0

        for _, msg in self.communicator.received_messages:
            dp = msg.get("disposal")
            if dp is not None and self._disposal_pos is None:
                self._disposal_pos = dp

        if self._state == "search":
            self._search_phase(cmd, gps, compass, lidar)
        elif self._state == "approach_bomb":
            self._approach_phase(cmd, gps, compass, lidar)
        elif self._state == "return_to_disposal":
            self._return_phase(cmd, gps, compass, lidar)

        return cmd

    def _search_phase(self, cmd, gps, compass, lidar):
        semantic = self.semantic_values()
        if semantic is not None:
            for hit in semantic:
                if hit.entity_type.name == "BOMB" and not hit.grasped:
                    self._target_pos = self._polar_to_world(
                        hit.distance, hit.angle, gps, compass
                    )
                    self._state = "approach_bomb"
                    return

        if lidar is not None:
            front = np.min(lidar[80:101])
            if front > 80:
                cmd["forward"] = 0.5
            else:
                left = np.mean(lidar[110:140])
                right = np.mean(lidar[40:70])
                cmd["rotation"] = 0.6 if left > right else -0.6
        else:
            cmd["forward"] = 0.4

    def _approach_phase(self, cmd, gps, compass, lidar):
        if self._has_bomb:
            self._state = "return_to_disposal"
            return
        if self._target_pos is None:
            self._state = "search"
            return
        self._navigate_to(cmd, gps, compass, self._target_pos)
        cmd["grasper"] = 1

    def _return_phase(self, cmd, gps, compass, lidar):
        if not self._has_bomb:
            self._state = "search"
            return
        if self._disposal_pos is None:
            semantic = self.semantic_values()
            if semantic is not None:
                for hit in semantic:
                    if hit.entity_type.name == "DISPOSAL_CENTER":
                        self._disposal_pos = self._polar_to_world(
                            hit.distance, hit.angle, gps, compass
                        )
                        break
        if self._disposal_pos is not None:
            self._navigate_to(cmd, gps, compass, self._disposal_pos)
            cmd["grasper"] = 1
        else:
            cmd["forward"] = 0.5
            cmd["grasper"] = 1

    def _polar_to_world(self, dist, angle, gps, compass):
        world_angle = compass + angle
        return np.array([
            gps[0] + dist * np.cos(world_angle),
            gps[1] + dist * np.sin(world_angle),
        ])

    def _navigate_to(self, cmd, gps, compass, target):
        delta = np.array(target) - gps
        distance = np.linalg.norm(delta)
        if distance < 8:
            return
        target_angle = np.arctan2(delta[1], delta[0])
        angle_diff = target_angle - compass
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        if abs(angle_diff) > 0.25:
            cmd["rotation"] = np.clip(angle_diff * 0.5, -0.5, 0.5)
        else:
            cmd["forward"] = min(0.5, distance * 0.01)
```

---

## 11. 测试方法

### 本地测试

```bash
# 红队
.venv/bin/python -m swarm_rescue.launcher -c config/competition_place_eval_plan.yml

# 蓝队
.venv/bin/python -m swarm_rescue.launcher -c config/competition_rescue_eval_plan.yml
```

### 无头模式（服务器）

```bash
xvfb-run -s "-screen 0 1920x1080x24" \
    .venv/bin/python -m swarm_rescue.launcher --headless \
    -c config/competition_rescue_eval_plan.yml
```

### 竞赛模式（模拟评测环境）

```bash
.venv/bin/python -m swarm_rescue.launcher --competition \
    -c config/competition_rescue_eval_plan.yml
```

建议提交前务必运行一次 `--competition` 模式，确保 `control()` 中没有调用被禁止的真值 API。

---

## 12. 炸弹放置工具（自建测试用）

> 本节面向需要在本地测试蓝队（Rescue）算法的选手。你可以使用炸弹编辑器为任意地图交互式添加炸弹，导出为 JSON 后用于测试。

### 12.1 启动炸弹编辑器

```bash
.venv/bin/python -m swarm_rescue.tools.bomb_editor Map01
```

支持的地图：`Map01`、`Map04`、`Map05`。

### 12.2 操作说明

| 操作 | 功能 |
|------|------|
| **左键点击** | 在点击位置添加一枚炸弹 |
| **右键点击** | 移除最近的炸弹 |
| **J 键** | 导出当前所有炸弹为 JSON 文件 |
| **I 键** | 从 JSON 文件导入炸弹 |
| **K 键** | 清除所有炸弹 |
| **Enter 键** | 锁定编辑并开始模拟 |
| **Q 键** | 退出 |

默认导出文件名为 `bombs_<地图名>.json`，也可通过 `-o` 指定输出路径：

```bash
.venv/bin/python -m swarm_rescue.tools.bomb_editor Map04 -o my_bombs.json
```

### 12.3 使用导出的炸弹测试

**方法一：手动编辑模式加载**

```bash
.venv/bin/python -m swarm_rescue.launcher \
    --manual-bomb-edit \
    --manual-bombs-in bombs_Map01.json \
    -c config/competition_rescue_eval_plan.yml
```

启动后按 **Enter** 即可用预设炸弹开始蓝队救援模拟。

**方法二：YAML 配置加载（批量评估用）**

在 YAML 配置中设置 `bombs_file` 字段：

```yaml
evaluation_plan:
  - map_name: Map01
    bombs_file: bombs_Map01.json
    nb_rounds: 1
    number_drones: 10
    zones_config: []
```

然后运行：

```bash
.venv/bin/python -m swarm_rescue.launcher --bombs-file bombs_Map01.json -c config/custom_eval_plan.yml
```

### 12.4 JSON 格式参考

```json
{
  "map_name": "Map01",
  "zones_config": [],
  "bombs": [
    {"x": -220, "y": 80, "theta": 0},
    {"x": 0, "y": 0, "theta": 0}
  ]
}
```
