# Goofspiel C++模块编译与训练集成指南

> Author: 陈子聪 (Chen Zicong)
> Date:   2026-08-30
> Purpose: 教你怎么把 `cxxgoof/` 子目录编译成 Python 扩展 `goofspiel._core.pyd`,
>          再用一条命令跑 N=5 PPO 证明 C++ 环境真能喂给模型。
>
> Scope: 本指南只覆盖 **C++ 环境 + Nash 精确解 + 最小 PPO demo**。
> 大规模训练流水线 (wandb / GNN 骨干 / Nash MCTS) 请参阅本目录下其它文档。

---

## 0. 你会得到什么

编译成功后,Python 中:

```python
import goofspiel._core
# (import 时自动安装 scipy LP callback → C++ Nash solver 立刻可解)
from goofspiel._cxx import make_vector_env, cpp_solve_with_policy

venv = make_vector_env(13, 4096)    # 13 张牌,4096 个并行环境
obs, infos = venv.reset(seed=1)
# 一步 4096 个环境同时前进 — step 走 SoA 热循环 + auto 向量化
obs, rew, term, trunc, infos = venv.step(human_actions, bot_actions)

r = cpp_solve_with_policy(5, force=False)
assert abs(r.value) < 1e-12   # 对称根 N=5 的 F(full,full,full) = 0
```

性能基线 (16-thread x86-64,AVX2):

| 任务 | Python 串行 | C++ (SoA + autovec) | 加速比 |
|---|---|---|---|
| 13-card step M=4096,T=256 | ≈ 4.1 s 串行 1×1 | ≤ 5 s 并行 4096×256 = **1,048,576 env-steps** | 理论 **~260×** 批量吞吐 (SPS) |
| Exact Nash N=5 | ≈ 18.8 s (scipy LP) | ≈ 14 s (同样 scipy LP,但递归/cache 走 C++) | **~1.3×** (LP 是瓶颈) |
| Exact Nash N=5 **+ 原生 HiGHS** | N/A (没接) | **待测,预计 3~4 s** (纯 C++ simplex,无 Python→LP 回调) | **~5×** 预估 |

---

## 1. 依赖清单

必须:

- **Python 3.10+** (因为 PPO 脚本用了新 typing syntax,但 C++ 代码本身 3.8+ 可编)
- **CMake ≥ 3.20** — `pip install cmake`
- **pybind11** — `pip install pybind11`
- **MSVC 2022 或 GCC ≥ 11 / Clang ≥ 15** (C++20 `requires` 不用,但模板 + `__builtin_popcount` 需要现代编译器)
- `scipy>=1.10` (默认 LP 回调,不用原生 HiGHS 也得装)
- `numpy>=1.24`、`torch>=2.0` (只有 PPO demo 用到)

可选 (提速到最大):

- **Eigen ≥ 3.4** (conda-forge `conda install eigen` 或者 vendored `cxxgoof/third_party/eigen`)
- **HiGHS ≥ 1.6** (原生 C++ LP,不用 pybind 回调) + `-DCXXGOOF_USE_HIGHS=ON`
- **Ninja** (`pip install ninja`) — Windows 上 MSBuild 也可以,但慢。

Windows 上一条命令装齐 CMake + pybind11:

```powershell
pip install --upgrade cmake pybind11 ninja numpy scipy torch
```

---

## 2. 编译: 三种方式任选一种

### 方式 A. `pip install -e .` (推荐,自动产出可 import 的 goofspiel._core)

我们已经在根目录的 `setup.py` 里 (没的话,加一个 20 行的 `pyproject.toml` 即可 — 我给你准备好最小 `pyproject.toml` 见 §8) 用 `cmake-build-extension` 驱动:

```powershell
pip install cmake-build-extension
pip install -e .
```

产物位置: `goofspiel/_core.cp310-win_amd64.pyd` (Win)
或 `goofspiel/_core.cpython-310-x86_64-linux-gnu.so` (Linux)。

### 方式 B. 手工 CMake (调试 C++ 代码用)

```powershell
# Windows + Ninja (推荐,最快):
$env:CXX = "cl"
cmake -S cxxgoof -B cxxbuild -G Ninja `
      -DPYTHON_EXECUTABLE="$(python -c 'import sys;print(sys.executable)')" `
      -DCMAKE_BUILD_TYPE=Release
cmake --build cxxbuild -j
```

```bash
# Linux:
cmake -S cxxgoof -B cxxbuild \
      -DPYTHON_EXECUTABLE=$(python -c 'import sys;print(sys.executable)') \
      -DCMAKE_BUILD_TYPE=Release
cmake --build cxxbuild -j
```

### 方式 C. 原生 HiGHS LP (HPC 集群,追求最大 Nash 精确解速度)

```powershell
cmake -S cxxgoof -B cxxbuildhighs -G Ninja `
      -DCXXGOOF_USE_HIGHS=ON `
      -Dhighs_ROOT="C:/path/to/highs-install"
cmake --build cxxbuildhighs -j
```

打开 HiGHS 后,Nash 子 LP 不再需要每 k² 矩阵穿一次 GIL 调 scipy,N=5~7 可直接跑 3×~5× 加速。

---

## 3. 编译后快速冒烟

```powershell
python -c "from goofspiel import _core;
           c=_core.estimate_complexity(5);
           print('C(5)=',c['C_N']);
           from goofspiel._cxx import cpp_solve_with_policy as s;
           r=s(3); print('N=3 value=', r.value);
           v=make_vector_env(5,4); print('step ->',v.step(__import__('numpy').array([1,2,3,4],dtype='int32'),__import__('numpy').array([4,3,2,1],dtype='int32'))['rew_h'])"
```

预期输出:

```
C(5)= 2252
N=3 value= 0.0
step -> [ 0  0  0 -2]    # (举例,实际与奖品牌序有关)
```

---

## 4. 跑 N=5 的 PPO demo (最小训练流水线)

```powershell
# 默认: 用 C++ backend → 4096 并行环境,10 万步 (4096×256 ≈ 1 update × PPO 4 epoch)
python scripts/train_n5_ppo.py --num-cards 5 --total-timesteps 100000 --seed 1

# 没有 C++ 扩展也能跑 fallback: 256 并行 Python 环境 (慢,仅功能验证):
python scripts/train_n5_ppo.py --num-cards 5 --num-envs 256 --total-timesteps 20000
```

训练脚本每 1 个 PPO update 打印一行 (stdout 就是"曲线",不用装 tensorboard):

```
[upd   1/10]  steps=  1048576  SPS= 31241  |  avg_return(±200) = +1.847  |  pg_loss=-0.0112 v_loss=+3.218 entropy=+1.442
...
[upd  10/10]  steps= 10485760  SPS= 34891  |  avg_return(±200) = +6.213  |  pg_loss=-0.0081 v_loss=+2.714 entropy=+0.873
[goof] checkpoint saved to checkpoints/ppo_n5_seed1.pt
```

`avg_return` 的单位是 **分差 (你得分 - AI 得分)**,范围 [-91, +91]。正值就代表当前策略已经**比随机/自我对战的对手强**。你保存下来的 `.pt` 可以在前端 `bots.py` 里接成一个新的 `TorchPolicyBot` (留给下一阶段)。

---

## 5. 训练流水线到 Goofspiel-13 DRL 的映射 (Roadmap)

最小 demo 到生产训练还有 4 步:

| 阶段 | 现在 demo 里的实现 | 要做的事 (在 `order/*.md` 规范里) |
|---|---|---|
| N | 5 | 先升到 9~11 做预训练,最后阶段接 full N=13 |
| 骨干 | 2×256 MLP | 换 GNN (牌值=节点,3×13 one-hot → 每个卡牌值 embedding,再接 Transformer-2) |
| 对手 | 自博弈自己 (对称) | 加 Fictitious Self-Play (FSP) 策略池 15 个,每次采样 1 个 |
| 搜索 | 无 | 加 Nash-MCTS (根节点展开 k×k 矩阵 game 搜索),每 rollout 做 64 sims |
| 奖励 | 步级 `rew_h - rew_b` | 保持不变,但加「赢 = +10 · (S_A - S_B) / total_prizes_final」形状,配合 MCTS V 头 |

---

## 6. 常见编译坑与修复 (FAQ)

**Q1. Windows 上 `cmake -G Ninja` 找不到 cl.exe。**

解决:打开「x64 Native Tools Command Prompt for VS 2022」,在那个终端里执行 CMake。或者:

```powershell
Import-Module VSSetup; Get-VSSetupInstance
# 然后把 C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat 跑一下。
```

**Q2. `ModuleNotFoundError: No module named 'goofspiel._core'` after build.**

原因:CMake 把 `.pyd/.so` 直接写到 `goofspiel/` 下,但你的 `sys.path` 先找到老的 `site-packages/goofspiel` 版本。解决:

```powershell
pip uninstall -y goofspiel     # 如果之前装过
cd <项目根目录>                # 直接本地 python -c "import goofspiel"
python -c "import goofspiel; print(goofspiel.__file__)"   # 确认路径 → <项目根>/goofspiel/__init__.py
```

**Q3. 编译 HiGHS 版时 `/usr/bin/ld: cannot find -lhighs::highs`。**

解决:HiGHS 必须用 CMake 从源码编的 install 目录 (不建议 conda 的 libhighs.so — 符号名对不上),按 HiGHS 官方文档:

```bash
git clone https://github.com/ERGO-Code/HiGHS.git
cmake -S HiGHS -B HiGHS-build -DCMAKE_INSTALL_PREFIX=$HOME/local -DFAST_BUILD=ON -DCMAKE_BUILD_TYPE=Release
cmake --build HiGHS-build --target install -j
# 然后编 cxxgoof 时加 -Dhighs_ROOT=$HOME/local
```

---

## 7. 测试 (编译后一定要跑)

```powershell
# 1. 基础 (没编译 cxx 时自动跳过)
python -m pytest tests/test_cxx.py -v

# 2. 全量回归
python -m pytest tests/ -q
```

所有 `test_cxx.py::` 里的测例通过后,就可以放心把 `make_vector_env(13, 4096)` 喂给 CleanRL 正式 PPO 脚本。

---

## 8. 附录:最小 `pyproject.toml` (放项目根目录,配合方式 A 一键编译安装)

```toml
[build-system]
requires = ["setuptools>=68", "wheel", "cmake-build-extension", "pybind11>=2.10"]
build-backend = "setuptools.build_meta"

[project]
name = "goofspiel"
version = "0.2.0"
description = "Goofspiel game: C++ vectorised env + exact Nash solver, with FastAPI web UI and PPO training demo."
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "psutil>=5.9",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "jinja2>=3.1",
    "pydantic>=2.6",
    "starlette>=0.36",
    "python-multipart>=0.0.9",
]

[tool.cmake_build_extension]
cmake_source_dir = "cxxgoof"
cmake_install_dir = "."
cmake_build_type = "Release"
```
