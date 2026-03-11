# GPU Global Routing + cuPDLPx 详细实施计划（可直接执行）

## 0. 目标与原则

### 0.1 总目标
以 `cuPDLPx` 为核心求解后端，构建一条“理论可解释 + 工程可运行”的新 GPU global routing 路线；`InstantGR` 仅作为 baseline 与算法参考，不作为融合对象：

1. 先有可运行基线（InstantGR 原流程不破坏）。
2. 再把路由问题逐步 LP 化（从小规模子问题开始）。
3. 最后形成可回退、可对比、可扩展的新路由原型框架。

### 0.2 借鉴现有 md 的核心立场
参考 `gpu_global_routing_cupdlpx_analysis.md`：

1. 不走“先枚举大量 pattern 再做大 ILP”的重路径。
2. 把 global routing 看成可优化、可分解的 solver-like 模块。
3. 先构建清晰变量/约束/目标，再做 GPU 化与工程化。

### 0.3 执行原则
1. 每阶段都有可运行产物和评估指标。
2. 每阶段都可独立回滚（Git checkpoint）。
3. 小步快跑：先 correctness，再性能。
4. 独立工程边界：不改 `InstantGR` 源码，不链接/调用 `InstantGR` 代码，仅使用其结果做 baseline 对比。

---

## 1. 代码现状映射（作为计划基座）

### 1.1 InstantGR（当前可用主链）
用途定位：`baseline + 算法借鉴`（RSMT、batch、detour、拥塞视图），不作为主线集成目标，不复用其代码实现。

- 入口：`InstantGR/src/main.cpp`
- 数据读取：`InstantGR/src/database.hpp`
- GPU 数据与 RSMT/batch：`InstantGR/src/database_cuda.hpp`
- 成本/需求/拥塞与提交：`InstantGR/src/graph.hpp`
- 初始路由：`InstantGR/src/Lshape_route.hpp`
- detour 路由：`InstantGR/src/Lshape_route_detour.hpp`

### 1.2 cuPDLPx（当前可用主链）
- C API：`cuPDLPx/include/cupdlpx.h`
- LP 数据结构：`cuPDLPx/include/cupdlpx_types.h`
- 接口封装：`cuPDLPx/src/cupdlpx.c`
- 主求解：`cuPDLPx/src/solver.cu`
- 预处理/缩放：`cuPDLPx/src/presolve.c`, `cuPDLPx/src/preconditioner.cu`

### 1.3 关键差距（必须补齐）
1. 缺少“routing LP 问题构造 -> cuPDLPx 求解 -> 离散恢复”的完整闭环实现。
2. 还没有统一的 benchmark I/O 和指标对比框架。
3. 缺少从 baseline 到新原型的公平对比与回归机制。

### 1.4 cuPDLPx 复用策略（强约束）
本项目默认“优先复用 cuPDLPx”，避免自研重复求解器内核：

1. 直接复用（不重写）：
  - `cuPDLPx/include/cupdlpx.h` C API（`create_lp_problem`, `solve_lp_problem`）
  - `cuPDLPx` 的 presolve / scaling / restart / feasibility polishing 能力
  - 终止准则与状态码体系（OPTIMAL/INFEASIBLE/TIME_LIMIT 等）
2. 借鉴思想（不直接拷贝实现）：
  - restarted Halpern PDHG 的“评估频率 + 自适应重启”节奏
  - bound/objective rescaling 对数值稳定性的处理思路
3. 必须新增（适配层）：
  - routing -> LP 的问题构造器（变量索引、CSR 装配）
  - LP 解 -> 路由结构回写器（rounding + 修复）

---

## 2. 分阶段实施计划（详细）

## Phase 0: 环境与依赖锁定（0.5-1 天）

### 0.1 新建独立 conda 环境
- 环境文件：`envs/gagr_dev.yml`
- 一键脚本：`scripts/setup_conda_env.sh`
- 环境名建议：`gagr-dev`

### 0.2 工具链检查
1. `python --version`（建议 3.11）
2. `cmake --version`（>= 3.20）
3. `nvcc --version`（建议 CUDA >= 12.4）
4. `nvidia-smi`（确认 GPU 可用）

### 0.3 验收标准
1. 能激活环境并导入核心 Python 包。
2. 能编译 InstantGR（至少 tiny case）。
3. 能调用 `cupdlpx`（Python 或 C API 任一条链路打通）。

---

## Phase A: 基线固化与实验脚手架（1-2 天）

### A1. 固化 InstantGR baseline
- 任务：固定编译参数、输入输出路径、日志格式。
- 交付：
  - `scripts/run_instantgr_baseline.sh`
  - `results/baseline/<case>/`（输出与统计）
- 验收：同一输入重复运行结果一致（允许浮点微扰但指标一致）。

### A2. 建立统一评估脚本
- 任务：统一提取 `wirelength/via/overflow/runtime`。
- 交付：
  - `scripts/eval_routing_result.py`
  - `docs/metrics_definition.md`
- 验收：可对 InstantGR 原生输出自动打分并生成 CSV。

### A3. 小规模 case 集
- 任务：准备 tiny/small/medium 三档测试。
- 交付：`benchmarks/local_suite_manifest.yaml`
- 验收：`scripts/run_suite.sh` 一条命令执行全套。

---

## Phase B: LP 映射最小闭环（2-4 天）

### B0. 先做 cuPDLPx 最小直连 PoC（必须先于 B1）
- 任务：在不改 cuPDLPx 核心代码前提下，做一条最小路由 LP 的端到端调用。
- 交付：
  - `router_lp/examples/min_route_lp_with_cupdlpx.cpp`
  - `scripts/run_min_lp_poc.sh`
- 验收：
  1. 能调用 `create_lp_problem + solve_lp_problem`。
  2. 输出求解状态、obj、residual、runtime。
  3. 失败时有清晰 fallback 日志。

### B1. 定义 LP 中间表示（IR）
- 任务：在本仓库新增独立模块，作为“新路由原型”的核心，不依赖改动 InstantGR 主逻辑。
- 新增目录建议：
  - `router_lp/include/`
  - `router_lp/src/`
- 关键结构：
  - `RoutingLPProblem`（变量索引、约束索引、稀疏矩阵构造器）
  - `RoutingLPSolution`（x/dual/状态）

### B2. 首版变量与约束（先做最小可解）
- 变量：边使用量 `f_{n,e}`（先连续放松，后续再加强）。
- 约束：
  1. flow conservation（源汇/中间点守恒）
  2. edge capacity（按 2D 或单层简化）
  3. 变量边界（`0 <= f <= 1` 或适配连续流）
- 目标：`wire cost + lambda * overflow_slack`

### B3. 导出到 cuPDLPx 输入
- 任务：把 IR 转成 `cupdlpx` C API 需要的 CSR + bounds。
- 对接文件建议：
  - `router_lp/src/to_cupdlpx.cpp`
  - `router_lp/src/solve_with_cupdlpx.cpp`
- 验收：小案例可调用 `solve_lp_problem` 返回可解释状态。

### B3.1 cuPDLPx 参数映射规范（新增）
- 任务：给 routing 场景定义默认参数配置，形成可复现实验基线。
- 建议初始参数（可调）：
  1. `presolve = true`
  2. `termination_evaluation_frequency = 200`
  3. `eps_opt = 1e-4`, `eps_feas = 1e-4`
  4. `time_limit = 30~120s`（按子图规模）
  5. `feasibility_polishing = true`（仅在后处理阶段）
- 交付：
  - `router_lp/config/cupdlpx_routing_default.yaml`
  - `router_lp/src/cupdlpx_param_loader.cpp`

### B4. LP 结果回写路由（最小版本）
- 任务：将连续解投影/抽取为可行路由骨架（先允许启发式后处理，可借鉴 InstantGR 的 detour 思想）。
- 交付：
  - `router_lp/src/solution_rounding.cpp`
- 验收：生成可被现有 evaluator 消费的输出。

---

## Phase C: 新路由原型闭环（3-5 天）

### C1. 新原型入口与运行模式
新增独立入口（建议：`router_lp/main.cpp`）：
1. `--mode pure_lp`（LP + rounding + repair）
2. `--mode lagrangian_path`（Pathfinding 风格 shortest-path + multiplier）
3. `--mode hybrid`（pricing/path pool + cuPDLPx master LP）

### C2. 局部子问题策略
- 先只对 overflow hotspot 子图做 LP（避免全图过大）。
- hotspot 生成可借鉴 InstantGR 的拥塞视图思路，但数据结构在新原型内独立实现。

### C3. 回退机制
- 若 LP 超时/失败：记录失败并切换到 `lagrangian_path` 或 `repair-only`，保证流程可完成。

---

## Phase D: 工程稳健性与性能优化（持续）

### D1. 正确性与鲁棒性
- 单测：IR 构造、索引一致性、约束维度检查。
- 回归：3 档 case 的指标不退化阈值。

### D2. 性能优化优先级
1. LP 构造阶段（CPU 端稀疏拼装）。
2. 子图裁剪策略（减少变量/约束规模）。
3. 新原型内部的数据搬运与 kernel 启动开销。

### D3. 参数扫描
- `lambda`、子图大小、迭代次数、time limit。
- 输出 Pareto 曲线：`overflow vs runtime`, `wirelength vs runtime`。

---

## 3. Git 管理策略（随时可执行）

## 3.1 分支策略
- 主分支：`main`（只接收可运行结果）
- 功能分支统一前缀：`codex/`

建议分支序列：
1. `codex/phase-a-baseline-harness`
2. `codex/phase-b0-cupdlpx-direct-poc`
3. `codex/phase-b-lp-ir-minimal`
4. `codex/phase-c-router-lp-prototype`
5. `codex/phase-d-tuning-and-regression`

## 3.2 提交粒度
每个 commit 只做一类变化：
1. `feat:` 新能力
2. `refactor:` 重构不改行为
3. `test:` 测试与基线
4. `docs:` 文档/实验记录

## 3.3 提交模板（建议）
```
<type>(<scope>): <summary>

What:
- ...

Why:
- ...

Validation:
- command: ...
- result: ...
```

## 3.4 强制检查清单（commit 前）
1. 能编译。
2. 至少一个小 case 跑通。
3. 关键指标有记录（CSV 或日志）。
4. 有回退开关（避免主流程断裂）。

---

## 4. 第一周可执行任务单（具体到动作）

### Day 1
1. 建 `phase-a` 分支。
2. 加 baseline 运行脚本与评估脚本。
3. 产出第一版 baseline 指标表。

### Day 2
1. 先做 `cuPDLPx` 最小直连 PoC（B0）。
2. 搭 `router_lp` 目录与数据结构。
3. 实现最小 CSR 构造与维度校验。

### Day 3
1. 把 tiny routing 子问题接入 `cuPDLPx` C API。
2. 输出 LP 状态、目标值、残差、终止原因。

### Day 4
1. 做最小 rounding 到路由输出。
2. 接入 evaluator，对比 baseline。

### Day 5
1. 新增 `router_lp` 入口的 `--mode` 开关与 fallback。
2. 做一次完整小规模回归并归档结果。

---

## 5. 风险与应对

1. LP 规模爆炸：
- 应对：先 hotspot 子图 + net 分批 + 约束裁剪。

2. 连续解不可直接布线：
- 应对：两段式（LP 给引导 + 自研/借鉴 detour repair 修复）。

3. 运行时回退不足：
- 应对：`pure_lp` 失败时自动退化到 `lagrangian_path` 或 `repair-only`。

4. 结果不可比较：
- 应对：统一评估脚本 + 固定 case manifest。

---

## 6. 立即开始的 Git 命令（建议）

```bash
cd /Users/haoning/project/eda-3d-arch-research/GPU_Accelerate_Global_Routing

# 1) 新分支
git checkout -b codex/phase-a-baseline-harness

# 2) 阶段提交
git add scripts/ docs/ IMPLEMENTATION_PLAN_GPU_ROUTING_CUPDLPX.md
git commit -m "docs(plan): add phased integration plan for InstantGR + cuPDLPx"

# 3) 持续小步提交
# feat/test/docs 分开提交，避免大杂烩 commit
```
