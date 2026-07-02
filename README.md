![Project banner](https://raw.githubusercontent.com/mujocolab/mjlab/main/docs/source/_static/mjlab-banner.jpg)

# mjlab

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjlab/ci.yml?branch=main)](https://github.com/mujocolab/mjlab/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation](https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg)](https://mujocolab.github.io/mjlab/)
[![License](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)
[![Nightly Benchmarks](https://img.shields.io/badge/Nightly-Benchmarks-blue)](https://mujocolab.github.io/mjlab/nightly/)
[![PyPI](https://img.shields.io/pypi/v/mjlab)](https://pypi.org/project/mjlab/)
[![PyPI downloads](https://img.shields.io/pypi/dm/mjlab?color=blue)](https://pypistats.org/packages/mjlab)

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), a GPU-accelerated version of [MuJoCo](https://github.com/google-deepmind/mujoco).
The framework provides composable building blocks for environment design,
with minimal dependencies and direct access to native MuJoCo data structures.



## Moz training

当前创建的环境配置在mjlab/src/mjlab/tasks/manipulation/config 目录下，配置有
1. Mjlab-Moz1-Lh-Reach           |  给定目标位姿的到达任务
2. Mjlab-Moz1-Lh-Reach-Smooth    |  配置1加入了关节平滑的奖励奖励，解决到位后的抖动问题
3. Mjlab-Moz1-Lh-Reach-Delta     |  持续输出微小增量的训练


**训练脚本举例**
```bash
uv run python -m mjlab.scripts.train Mjlab-Moz1-Lh-Reach --enable-nan-guard True
```
如果接着停止的训练继续，添加参数 --agent.resume True


**推理脚本举例**

```bash
# 1. 运行 Reach Smooth 模型：
uv run play Mjlab-Moz1-Lh-Reach-Smooth --interactive-commands True --checkpoint-file pretrained_models/moz1_lh_reach_smooth_best.pt

# 2. 运行 Reach Delta 模型：
uv run play Mjlab-Moz1-Lh-Reach-Delta --interactive-commands True --checkpoint-file pretrained_models/moz1_lh_reach_delta_best.pt

# 3. 运行基础 Reach 模型：
uv run play Mjlab-Moz1-Lh-Reach --interactive-commands True --checkpoint-file pretrained_models/moz1_lh_reach_best.pt

# --interactive-commands： 是否开启目标位姿拖拽
```

### 交互式拖拽与实时评估说明

在使用 `--interactive-commands True` 启动推理脚本后，可以直接在仿真界面中进行操作和精度评估：

**1. 拖拽目标的操作方法：**
- 在 MuJoCo 仿真窗口中，按住键盘的 **`Ctrl` 键**（Mac 为 `Cmd` 键）。
- 鼠标悬停在绿色的目标球（Target）上，**双击鼠标左键**，选中该目标。选中后会显示出控制坐标轴。
- **平移目标**：按住 `Ctrl` 键，按住鼠标 **右键** 并拖动，即可平移目标球。
- **旋转目标**：按住 `Ctrl` 键，按住鼠标 **左键** 并拖动，即可在原地旋转目标球的姿态。
- 机械臂会实时、平滑地跟随拖拽的目标位姿。

**2. 精度和响应速度评估说明：**
当你拖拽目标时，脚本会在后台自动监听的操作，并在启动程序的终端 (Terminal) 中实时打印两项核心评估数据：
- **响应速度 (Response Settling Time)**：当拖拽目标并**松手**的那一刻，系统会启动内置秒表。当机械臂末端与目标球的绝对直线距离首次进入 **2cm** 阈值内时(可自定义)，秒表停止并在终端打印耗时（例如：`[Eval] Response Settling Time (<2cm): 0.35s`）。这反映了机械臂应对突发指令的动态敏捷性。
- **稳态精度 (Steady-State Precision)**：当目标球保持静止超过 **1秒钟** 后，系统认为机械臂已进入稳态。随后系统每隔 0.5 秒会计算一次当前的绝对欧氏距离误差，并打印在终端（例如：`[Eval] Steady-State Precision: 0.8 mm`）。注意，该精度计算的是严苛的三维空间绝对直线距离（L2 Norm），这意味着实际的 X、Y、Z 单轴误差只会更小。



---

## Getting Started

mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Try it now:**

Run the demo (no installation needed):

```bash
uvx --from mjlab --refresh demo
```

Or try in [Google Colab](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb) (no local setup required).

**Install from source:**

```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv run demo
```

For alternative installation methods (PyPI, Docker), see the [Installation Guide](https://mujocolab.github.io/mjlab/main/source/installation.html).

## Training Examples

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**Multi-GPU Training:** Scale to multiple GPUs using `--gpu-ids`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 2. Motion Imitation

Train a humanoid to mimic reference motions. See the [motion imitation guide](https://mujocolab.github.io/mjlab/main/source/training/motion_imitation.html) for preprocessing setup.

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name your-org/motions/motion-name --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 3. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions
```

When running motion-tracking tasks, add `--registry-name your-org/motions/motion-name` to the command.


## Documentation

Full documentation is available at **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**.

## Development

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format and lint
make docs          # Build docs locally
```

For development setup: `uvx pre-commit install`

## Citation

mjlab is used in published research and open-source robotics projects. See the [Research](https://mujocolab.github.io/mjlab/main/source/research.html) page for publications and projects, or share your own in [Show and Tell](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell).

If you use mjlab in your research, please consider citing:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.
