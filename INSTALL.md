
# Table of Content
- [Table of Content](#table-of-content)
- [Introduction](#introduction)
- [Installation on macOS](#installation-on-macos)
- [Installation on Ubuntu (recommended)](#installation-on-ubuntu-recommended)
  - [**Warning** for Ubuntu users on VirtualBox](#warning-for-ubuntu-users-on-virtualbox)
  - [Git and Arcade library dependencies](#git-and-arcade-library-dependencies)
  - [*Python* installation](#python-installation)
  - [Installing this *swarm-rescue* repository](#installing-this-swarm-rescue-repository)
- [Installation on Windows 10/11 with WSL2](#installation-on-windows-1011-with-wsl2)
  - [WSL2: OpenGL and GPU selection](#wsl2-opengl-and-gpu-selection)
- [Installation on Windows 10/11 with GitBash](#installation-on-windows-1011-with-gitbash)
  - [**Warning** for Windows users](#warning-for-windows-users)
  - [*Python* installation](#python-installation-1)
  - [*Git* installation](#git-installation)
  - [Configure *Git Bash*](#configure-git-bash)
  - [Install this *swarm-rescue* repository](#install-this-swarm-rescue-repository)
- [Troubleshooting](#troubleshooting)
  - [Deactivation of OpenGL shaders](#deactivation-of-opengl-shaders)
  - [Check software versions](#check-software-versions)
  - [Find OpenGL version on Ubuntu and WSL2](#find-opengl-version-on-ubuntu-and-wsl2)
  - [Update Mesa library](#update-mesa-library)
- [Python IDE](#python-ide)
- [Contact](#contact)

# Introduction

This installation procedure has been successfully tested on **Ubuntu** and **Windows 11 with Git Bash**. Ubuntu is the recommended platform for optimal performance and stability.

# Installation on macOS

Installation on **macOS systems** is challenging and is **not officially supported**. *Swarm-Rescue* requires a recent version of *OpenGL* to function properly. However, recent versions of macOS no longer use *OpenGL* and instead rely on Apple's *Metal* graphics framework.

In any case, the code has had to be adapted for macOS in the past, resulting in significant performance degradation compared to Ubuntu systems. If you must use macOS, refer to [Deactivation of OpenGL shaders](#deactivation-of-opengl-shaders)

# Installation on Ubuntu (recommended)

This installation procedure has been tested on Ubuntu 20.04, 22.04 and 24.04.

## **Warning** for Ubuntu users on VirtualBox

**VirtualBox users have reported compatibility issues**.

*Swarm-Rescue* uses *OpenGL* shaders to accelerate computations. The lidar and semantic sensor calculations are performed directly on the GPU via these shaders for optimal performance.

VirtualBox has known issues with graphics drivers and OpenGL handling, which can cause problems with *Swarm-Rescue*.

You have several solutions:
- **Recommended**: Install Ubuntu natively (not in VirtualBox)
- **Alternative**: Follow the workaround described in [Deactivation of OpenGL shaders](#deactivation-of-opengl-shaders)

## Git and Arcade library dependencies

First, you will obviously have to install the Git tools.

The *Arcade* library (used for 2D graphics) requires *libjpeg-dev* and *zlib1g-dev* for proper functionality.

```bash
sudo apt update
sudo apt install git git-gui gitk libjpeg-dev zlib1g-dev
```

## *Python* installation

*Swarm-Rescue* requires **Python 3.8 or higher**. Python 3.8 is the default version on Ubuntu 20.04.

```bash
sudo apt update
sudo apt install python3 python3-venv python3-dev python3-pip 
```

Verify your Python version:
```bash
python3 --version
```

## Installing this *swarm-rescue* repository

**Step 1: Prepare your workspace**
Navigate to your desired working directory (e.g., *~/code/*):

```bash
cd
mkdir code
cd code
```

**Step 2: Clone the repository**
Download the code from [*Swarm-Rescue*](https://github.com/emmanuel-battesti/swarm-rescue):

```bash
git clone https://github.com/emmanuel-battesti/swarm-rescue.git
```

This creates the *swarm-rescue* directory with all the source code.

**Step 3: Create a virtual environment**
Set up an isolated Python environment for the project:

```bash
cd swarm-rescue
python3 -m venv .venv
```
It creates a *.venv* directory where all dependencies are installed.

**Step 4: Activate the virtual environment**
Activate the environment (required each time you work with the project):

```bash
source .venv/bin/activate
```

To deactivate the virtual environment when finished: `deactivate`

**Step 5: Install dependencies**
With the virtual environment activated, install all required packages:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install --editable .
```

**Step 6: Test the installation**
Verify everything works by running the launcher:

```bash
python3 ./src/swarm_rescue/launcher.py
```

**Simulator elements:** The codebase uses *Bomb* and *DisposalCenter* (not `WoundedPerson` / `RescueCenter`). See the “API naming” table in `README.md` if you port older team code.

# Installation on Windows 10/11 with WSL2

It's still **very experimental and untested**, but it seems possible to install *Swarm-Rescue* on WSL2 on Windows.
For this to work, certain points need to be checked:
- Use WSL 2 and not WSL 1,
- Have Windows 11 or Windows 10 with the latest updates,
- Update with the latest graphics card drivers, and reboot,
- Use Ubuntu 24.04 for WSL (Ubuntu 22.04 or earlier will not work).

Then simply follow the Ubuntu installation instructions: [Installation on Ubuntu (recommended)](#installation-on-ubuntu-recommended)

> [!NOTE]
> To be more precise, there's no need to install a driver for the graphics card under WSL. WSL uses the Windows graphics driver through a virtual interface. OpenGL support is provided by the Mesa library. Ubuntu 22.04's Mesa version in WSL does not support the required OpenGL 4.4 features, which is why Ubuntu 24.04 is mandatory.

## WSL2: OpenGL and GPU selection

On WSL2, GUI OpenGL goes through Mesa's **d3d12** Gallium driver and the Windows display stack—not a separate Linux `nvidia-driver` package. **`nvidia-smi` only confirms CUDA**; it does not show whether OpenGL uses your discrete GPU.

Ray sensors use GLSL `#version 440` (see `src/swarm_rescue/simulation/ray_sensors/shaders/id_compute.glsl`), which requires **GLSL 4.40+** and therefore an **OpenGL 4.4+** context. The integrated-GPU path on some systems only reports OpenGL 4.1 / GLSL 4.10, which is **not sufficient** even though the version number looks "close".

### Enable hardware OpenGL (avoid llvmpipe)

Without configuration, `glxinfo` may report `llvmpipe` (CPU software rendering). Force the D3D12 backend:

```bash
sudo apt install mesa-utils   # provides glxinfo
export GALLIUM_DRIVER=d3d12
glxinfo | grep "OpenGL renderer"
```

You should see `D3D12 (...)` instead of `llvmpipe`.

### Select NVIDIA (or another discrete GPU)

On systems with multiple GPUs, Mesa often picks the first enumerated adapter (often Intel integrated graphics). To use an NVIDIA GPU, set these in `~/.bashrc` or `~/.zshrc`:

```bash
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=nvidia
```

`MESA_D3D12_DEFAULT_ADAPTER_NAME` is a case-insensitive **substring** of the GPU name from Windows Device Manager (e.g. `nvidia`, `NVIDIA`, `4090`). See [GPU selection in WSLg](https://github.com/microsoft/wslg/wiki/GPU-selection-in-WSLg).

Reload your shell, then verify:

```bash
glxinfo | grep -E "OpenGL renderer|OpenGL version|shading language"
```

Example of a working discrete-GPU setup:

```text
OpenGL renderer string: D3D12 (NVIDIA GeForce RTX 4090)
OpenGL version string: 4.6 (Compatibility Profile) Mesa ...
OpenGL shading language version string: 4.60
```

Then verify from the project virtual environment:

```bash
python src/swarm_rescue/tools/opengl_info.py
```

### Windows-side settings (if OpenGL still uses Intel)

If environment variables are not enough, in Windows **Settings → System → Display → Graphics** (or the NVIDIA control panel), assign **High performance** / NVIDIA to:

- `C:\Windows\System32\wsl.exe`
- `C:\Windows\System32\wslhost.exe`

Then from PowerShell: `wsl --shutdown`, and reopen Ubuntu.

### What not to do on WSL2

- Do **not** install `nvidia-driver-*` inside WSL for GUI OpenGL (Windows drivers are exposed under `/usr/lib/wsl/lib`).
- Do **not** use `__GLX_VENDOR_LIBRARY_NAME=nvidia` in WSL; it often falls back to `llvmpipe`.

Upgrading Mesa via the [kisak PPA](#update-mesa-library) is optional on WSL2 when Ubuntu 24.04 already ships a recent Mesa; fixing **GPU selection** (`GALLIUM_DRIVER` + `MESA_D3D12_DEFAULT_ADAPTER_NAME`) is usually what matters.

# Installation on Windows 10/11 with GitBash

This installation procedure has been tested on Windows 11 (october 2025)

## **Warning** for Windows users

**Windows users have reported various issues**, ranging from unexpected behavior to complete program failure.

**Common Problems:**
- Lidar sensors detecting through walls
- Semantic sensors continuing to detect bombs even when grasped (grasped bombs should be hidden from the carrier’s semantic rays; if not, check drivers/OpenGL)
- General simulation instabilities

**Root Cause:**

*Swarm-Rescue* uses OpenGL shaders for GPU-accelerated computations (lidar and semantic sensor calculations). These issues appear to stem from graphics driver compatibility or Windows' OpenGL implementation.
The problem doesn't seem to occur on Ubuntu. On some "more powerful" Windows machines (desktop), it also works correctly.

**Environment Requirements:**
- Windows 11 or Windows 10 with latest updates
- Up-to-date graphics card drivers (restart after installation)
- Optimal performance settings configured

If problems persist, you have several solutions:
- Change to Ubuntu operating system,
- Change your machine, more powerful desktop machines tend to have fewer issues
- Use the workaround in [Deactivation of OpenGL shaders](#deactivation-of-opengl-shaders)

## *Python* installation

- Open the following link in your web browser: https://www.python.org/downloads/windows/
- The program will **not** work with a Python version greater than or equal to 3.12. (3.13.7 aug. 14 2025)
- Don't choose the latest version of Python, but choose version 3.11.9. Currently (10/2023), it is "*Python 3.11.9 - April 2, 2024*".
- For modern machines, you must select the *Windows installer (64-bit)*.
- Once the installer is downloaded, run the Python installer.
- **Important**: you need to check the "**Add python.exe to path**" check box to include the interpreter in the execution path.

## *Git* installation

Git is essential for source code management and tracking changes in the *swarm-rescue* project.

**Installation Steps:**
1. Download the [latest version of Git](https://git-scm.com/download/win)
2. Select "Git for Windows/x64 Setup" in "Standalone Installer"
3. Install with default configuration settings
4. Select "Launch Git Bash" and click "Finish"

The *Git Bash* terminal will be your primary interface for project work.

## Configure *Git Bash*

- Run the *Git Bash* terminal.
- **Warning**, by default you may **not** be in your home directory. So to get there, just type `cd`.
- If everything works, the command `python --version` should show the installed Python version, for example: `Python 3.11.9`.

## Install this *swarm-rescue* repository

- To install this git repository, go to the directory you want to work in (for example: *~/code/*).
- With *Git Bash*, you have to use those Linux commands, for example:
```bash
cd
mkdir code
cd code
```

Clone the code from [*Swarm-Rescue*](https://github.com/emmanuel-battesti/swarm-rescue):

```bash
git clone https://github.com/emmanuel-battesti/swarm-rescue.git
```

This creates the *swarm-rescue* directory with all the source code.

- Create your virtual environment. This command will create a *.venv* directory where all dependencies will be installed:

```bash
cd swarm-rescue
python -m venv .venv
```

- To use this newly created virtual environment, as each time you need it, use the command:

```bash
source .venv/Scripts/activate
```

To deactivate this virtual environment when finished, just type: `deactivate`

- With this virtual environment activated, we can install all the dependencies with the command:

```bash
python -m pip install --upgrade pip
python -m pip install --editable .
```

Test the installation:

```bash
python ./src/swarm_rescue/launcher.py
```
# Troubleshooting

## Deactivation of OpenGL shaders

Shaders are GPU-accelerated programs written in GLSL that significantly improve performance. However, compatibility issues may require disabling them.

To disable shaders:
1. Open file *src/swarm_rescue/simulation/gui_map/closed_playground.py*
2. In `ClosedPlayground.__init__`, change `use_shaders` from *True* to *False*

**Note**: This reduces performance, especially with multiple drones.

## Check software versions

*Swarm-Rescue* requires **GLSL 4.40+** (shader `#version 440`), which in practice means **OpenGL 4.4+**. There is no separate OpenGL API version "4.40"; that number refers to the shading language version.

Use this script to verify your system:
```bash
python src/swarm_rescue/tools/opengl_info.py
```

On WSL2, also check the renderer and GLSL version with `glxinfo` after setting [WSL2: OpenGL and GPU selection](#wsl2-opengl-and-gpu-selection)—a high OpenGL number on `llvmpipe` or on an integrated GPU with GLSL 4.10 is not enough.

## Find OpenGL version on Ubuntu and WSL2

Check your OpenGL version using *glxinfo*:

Install *glxinfo*:
```bash
sudo apt update
sudo apt install mesa-utils
```

Use *glxinfo*:
```bash
glxinfo | grep -E "OpenGL renderer|OpenGL version|shading language"
```

On WSL2, run the same command **after** exporting `GALLIUM_DRIVER=d3d12` and, if you have a discrete GPU, `MESA_D3D12_DEFAULT_ADAPTER_NAME=nvidia` (see above).

## Update Mesa library

Under Linux or WSL2, to update the Mesa library to the lastest (but not-official) version:
```bash
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt update
sudo apt upgrade
```

# Python IDE

While optional, using an IDE improves the Python development experience.

**Recommended**: [*PyCharm Community*](https://www.jetbrains.com/pycharm/) (free)
- Configure the interpreter path to your virtual environment (.venv) for proper integration 

# Contact

For questions about installation or code:

**Email**: emmanuel . battesti at ensta . fr  
**Discord**: Available on the project's Discord server

