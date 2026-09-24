MobaXterm 可视化与无头模式运行指南
(VSCODE没有配置X11转发，可视化不了，想在vscode中无头模式运行的话需要装Xvfb，似乎需要sudo权限)
==================================
（基于蓝队（排炸弹））
1. 运行命令(环境装好并且mobaxterm配置好了)

1.1 MobaXterm 可视化

确认 MobaXterm 的 X Server 已启动，且 `echo $DISPLAY` 输出非空，然后运行：

```bash
conda activate swarm
cd /data-ssd/guoyu/uav/baseline/swarm-eod

python -m swarm_rescue.launcher \
  --competition \
  --stop_at_first_crash \
  -c config/competition_rescue_eval_plan.yml
```

该方式会在 Windows 桌面显示仿真窗口。

1.2 MobaXterm 无头运行（不需要Xvfb）

```bash
conda activate swarm
cd /data-ssd/guoyu/uav/baseline/swarm-eod

python -m swarm_rescue.launcher \
  --competition \
  --headless \
  --stop_at_first_crash \
  -c config/competition_rescue_eval_plan.yml
```

程序不显示窗口，但仍会完成物理仿真、控制和评分。

1.3 VS Code Remote SSH 无头运行

VS Code SSH 终端通常没有 `DISPLAY`，需要服务器安装 Xvfb。

Ubuntu 或 Debian 安装命令：

```bash
sudo apt update
sudo apt install -y xvfb libgl1-mesa-dri mesa-utils
```

安装只需执行一次。以后运行不需要 sudo：

```bash
conda activate swarm
cd /data-ssd/guoyu/uav/baseline/swarm-eod

LIBGL_ALWAYS_SOFTWARE=1 \
xvfb-run -a -s "-screen 0 1250x800x24" \
python -m swarm_rescue.launcher \
  --competition \
  --headless \
  --stop_at_first_crash \
  -c config/competition_rescue_eval_plan.yml
```

Xvfb 只提供内存中的虚拟显示器，不会在本地弹出窗口。

2. 如何选择运行方式

```text
观察无人机运动：MobaXterm 可视化
测试和比较得分：无头模式
VS Code 没有 DISPLAY：Xvfb + 无头模式
服务器没有 Xvfb：在 MobaXterm 终端运行
```

推荐使用 VS Code 编辑代码，使用 MobaXterm 运行。二者访问服务器上的同一套
文件，不需要复制代码。

3. 配置 MobaXterm

打开 MobaXterm 后：

- 确认顶部工具栏中的 X Server 已启动；
- 点击 `Session`，选择 `SSH`；
- 填写服务器地址、端口和用户名；
- 在 `Advanced SSH settings` 中启用 `X11-Forwarding`；
- 建立连接后执行 `echo $DISPLAY`。

正常结果类似：

```text
localhost:10.0
localhost:15.0
```

具体数字不重要，但输出不能为空。如果输出为空，请重新检查 X Server 和
`X11-Forwarding`，然后断开并重新建立 SSH 会话。

4. 为什么 headless 仍需要显示环境

本项目使用 Arcade/Pyglet。即使添加 `--headless`，程序仍会创建不可见窗口：

```python
arcade.Window(width=1, height=1, visible=False)
```

`visible=False` 只表示不展示窗口。Pyglet 仍需要显示服务创建 OpenGL 上下文。

```text
MobaXterm 使用 X11 提供显示环境
VS Code 使用 Xvfb 提供虚拟显示环境
```

如果二者都没有，通常会报错：

```text
pyglet.canvas.xlib.NoSuchDisplayException:
Cannot connect to "None"
```

5. 当前测试配置

评测计划是 `config/competition_rescue_eval_plan.yml`，核心配置如下：

```yaml
stat_saving_enabled: false
state_recording_enabled: true
video_capture_enabled: false

evaluation_plan:
  - map_name: Map01
    team_mode: rescue
    nb_rounds: 2
    config_weight: 1
    number_drones: 10
    zones_config: []
    bombs_file: config/demo_bombs_map01.json
```

它表示：

- 使用 `Map01` 和蓝方 `rescue` 模式；
- 创建 10 架无人机，连续运行 2 个独立回合；
- 不启用特殊区域；
- 加载两枚测试炸弹；
- 保存状态记录，不录制视频。

配置已经指定 `bombs_file`，运行时不需要再次传入炸弹文件。

6. 命令参数

- `python -m swarm_rescue.launcher`：启动仿真入口；
- `--competition`：禁止控制器读取位置和角度等仿真真值；
- `--headless`：不显示窗口，但不跳过仿真和评分；
- `--stop_at_first_crash`：控制器异常时立即停止并显示 traceback；
- `-c`：指定评测计划文件。

7. 运行过程与评分

当前配置的 `nb_rounds: 2` 表示程序会依次执行两个独立回合。可视化模式下，
第一个窗口自动关闭后会出现第二个窗口，这不是程序重复启动。

不要手动关闭仿真窗口，否则程序可能提前结束并输出无效分数。

每回合结束后会输出类似：

```text
rescued nb: 2/2
health return score: 97.6%
time score: 100.0%
elapse timestep: 300/2000 steps
time to dispose all: 300 steps
round score: 99.5%
```

- `rescued nb`：已处理炸弹数和总炸弹数；
- `health return score`：无人机返航健康得分；
- `time score`：任务完成速度得分；
- `elapse timestep`：已运行步数和最大步数；
- `round score`：本回合蓝方总分。

正常结束通常满足以下条件之一：

- `rescued nb: 2/2`，所有炸弹均已处理；
- `elapse timestep: 2000/2000`，达到最大步数；
- `walltime elapsed: 90s/90s`，达到 Map01 墙钟时间上限。

如果炸弹未处理完、步数和时间均未达到上限，并且健康统计异常为 0，通常是
窗口被外部关闭。这次分数不适合用来评价算法。

简要结论：

```text
MobaXterm X11 = 能看到窗口，适合行为调试
--headless     = 不显示窗口，适合测试得分
Xvfb           = 为无 DISPLAY 的终端提供虚拟显示器
```

（我只新增了swarm-eod/src/swarm_rescue/solutions/my_drone_blue_basic.py，以及部分测试配置，其他都是官方的baseline代码）