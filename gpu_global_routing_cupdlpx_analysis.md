# 面向 GPU Global Routing 的优化建模路线分析

## 研究对象

本文综合分析三份材料：

1. **Pathfinding Model and Lagrangian-Based Global Routing**（DAC 2023）
2. **RUPlace: Optimizing Routability via Unified Placement and Routing Formulation**（DAC 2025）
3. **cuPDLPx**（MIT Lu Lab 的 GPU 一阶 LP 求解器）

目标不是讨论 placement-routing 协同优化本身，而是围绕一个更具体的问题：

> **如果目标是把 global routing 做成一个有明确理论支撑、且真正适合 GPU 的优化求解框架，那么“Pathfinding/Lagrangian 模型 + cuPDLPx/PDLP 这条线”是否成立？与现有 GPU global routing 相比，新意在哪里，边界在哪里，真正的技术难点又在哪里？**

---

## 1. 三份材料的核心信息

### 1.1 DAC 2023: Pathfinding Model and Lagrangian-Based Global Routing

这篇论文把 global routing 在 2D 初始布线阶段写成一个 **ILP-based pathfinding model**。在其建模中，先把 3D routing area 压缩到 2D，再用 FLUTE 将多 pin net 分解为 two-pin nets，然后在有向网格图上定义二值变量 `x_{n,ê}`，表示 two-pin net `n` 是否使用 directed edge `ê`。约束包括：

- edge capacity 约束：每条无向边对应的双向有向边使用总和不超过容量；
- source / sink / intermediate node 的流守恒约束；
- 目标是最小化总 routing cost（若 cost 取 1，则对应最小化线长）。 fileciteturn1file0L84-L123

论文没有直接用商业 ILP 求解器求解，而是把容量约束拉格朗日松弛到目标函数里，引入 multiplier `μ_e`，从而把原问题分解成**每个 two-pin net 的最短路问题**；随后用 **gradient ascent** 更新 multipliers，并用一个 **direction-aware weighted A\*** 算法求各 net 的路径。之后再配合三阶段 rip-up and reroute 进一步优化 overflow、wirelength 和 via 数。 fileciteturn1file0L124-L153 fileciteturn1file0L154-L214

实验上，这篇工作在 ISPD08/18 benchmark 上对 SPRoute 和 CUGR 给出不错结果：

- 与 CUGR 相比，平均 via 数降低约 5.1%，平均 runtime 有 4.89× speedup；
- 与 SPRoute 2.0 相比，平均 wirelength 降低约 1.7%，via 数相当。 fileciteturn1file0L215-L246

**对本文主题最重要的启示**是：

1. 它把 global routing 直接表述为一个优化问题，而不是纯启发式搜索；
2. 它没有依赖事先枚举候选 routing patterns；
3. 它通过拉格朗日松弛把“大 ILP”转成“很多 shortest path + multiplier update”。

这条线与 GPU 加速天然相关，因为“并行 shortest path + 并行 edge cost update”看上去很像 GPU-friendly 工作负载。

---

### 1.2 DAC 2025: RUPlace

RUPlace 的重点不是 standalone global routing，而是 **placement 和 routing 的统一建模**。它给出了一个 routing ILP：变量 `f_{n,e}` 表示 net `n` 是否使用 routing edge `e`，约束包括 edge capacity 和 `Af = h(x)` 形式的 flow conservation，其中 `h(x)` 由 placement 决定。接着它把 congestion 定义为 capacity violation，并把 routing cost 写成 `wire cost + μ * congestion penalty`。 fileciteturn1file1L53-L88

再往上，RUPlace 把 placement 的 density 约束和 routing 的 flow/capacity 约束统一进一个 **UCP（Unified Concurrent Placement and Routing）** 问题中，其中 `Af = h(x)` 是 placement 与 routing 的唯一耦合约束。 fileciteturn1file1L88-L99

求解层面，RUPlace 用 ADMM 交替优化 placement 与 routing：

- 给定 placement，解 routing 子问题；
- 给定 routing 解，做一个近似 placement update；
- 用 Wasserstein regularization 保证 pin supply/sink distribution 的变化不过于剧烈。 fileciteturn1file1L99-L142

RUPlace 还引入了 clustering-based convex global inflation 和局部 cell area adjustment。值得注意的是，它在实验设置中明确写到：实现基于 DREAMPlace，并用 **HeLEM-GR 作为 global router，从而形成 full GPU acceleration flow**。 fileciteturn1file1L164-L186

**对本文主题的意义**主要有两点：

1. 它提供了一个非常清楚的 routing ILP / congestion penalty / flow conservation 表达方式，可以作为后续把 routing 放松为 LP 的出发点；
2. 它说明了当前前沿 GPU-aware physical design 论文，已经倾向于把 global routing 视作一个可嵌入更大优化框架的“solver-like”子模块，而不是纯启发式黑盒。

不过，RUPlace 的目标是协同优化 placement + routing，不是专门研究 GPU global routing kernel 本身。因此它更像是“建模参考”和“系统定位参考”，而不是直接给出一个可替换 cuPDLPx 的 global routing solver。

---

### 1.3 cuPDLPx

cuPDLPx 的官方 README 将其定义为：

> **A GPU-accelerated first-order LP solver**，基于 **restarted Halpern PDHG**，面向 GPU 架构专门优化。 citeturn772186search0turn772186search2

它支持的标准形式是：

- `min c^T x`
- `l_c <= A x <= u_c`
- `l_v <= x <= u_v` citeturn772186search2

Python 接口也明确说明，它面向的是 **large-scale linear programming (LP)**，并接受 NumPy / SciPy 稀疏数据结构。 citeturn772186search3

仓库与相关论文介绍强调了几个关键点：

- 核心不是 branch-and-bound，而是一阶 primal-dual 方法；
- 主计算模式是稀疏线性代数与向量更新；
- 设计目标就是让计算模式尽量贴合 GPU 的 streaming memory 和大规模并行结构。 citeturn772186search0turn772186search1turn772186search4

因此，cuPDLPx **不是一个 ILP/MILP 通用整数求解器**，而是一个 **LP solver**。这点对 routing 非常关键：

- 如果你坚持保留 `x ∈ {0,1}` 的离散约束，cuPDLPx 不直接适用；
- 如果你接受 LP relaxation / path-based LP / edge-based LP relaxation，再配合 rounding / repair / column generation，那么它就有潜力成为 global routing 的一个新型 GPU backend。

---

## 2. 三者之间的结构关系

从最粗略的角度看，这三份材料可以排成一条谱系：

- **Pathfinding DAC 2023**：routing-only，ILP + Lagrangian + shortest path；
- **RUPlace DAC 2025**：placement + routing unified formulation，routing 是嵌入式 ILP 子问题；
- **cuPDLPx**：不特定于 EDA，但给出一个适合 GPU 的大型 LP 求解基础设施。

如果只考虑 global routing 本身，那么最值得抓住的桥梁是：

1. Pathfinding paper 已经给出了 routing 的 **离散优化 formulation**；
2. RUPlace 则给出了更清楚的 **edge-based ILP + congestion penalty** 表达；
3. cuPDLPx 提供了一个 **GPU-friendly LP solving engine**。

因此，一个自然的问题是：

> 能否把 Pathfinding / RUPlace 里 routing 的 ILP 做成某种 LP relaxation，然后把“逐 net shortest path + heuristic congestion update”的结构，替换为“GPU 上统一的 primal-dual LP solver + fractional-to-discrete recovery”？

这条路线理论上可行，但它不是直接照搬现有方法，而是一次 **solver paradigm shift**。

---

## 3. Pathfinding 模型为什么不等于 cuPDLPx 模型

### 3.1 Pathfinding 模型的求解本质

Pathfinding paper 虽然起点是 ILP，但真正的算法不是求大 ILP 本身，而是：

- 把容量约束对偶化；
- 对每个 net 解 shortest path；
- 用 gradient ascent 更新 multiplier；
- 最后再用 multi-stage rip-up & reroute 修正不可行或次优结果。 fileciteturn1file0L124-L214

所以它本质上更像：

> **Lagrangian decomposition + parallelizable shortest path oracle**

而不是：

> **Solve one monolithic optimization problem end-to-end.**

### 3.2 cuPDLPx 的求解本质

cuPDLPx 假设问题已经是 LP，并把大量计算统一为：

- `A x`
- `A^T y`
- box / interval projection
- primal/dual vector update

这种方法的好处是 **算子统一、内存访问更规律、GPU 更擅长**；缺点是 **它不直接给离散 path**。

所以 Pathfinding 与 cuPDLPx 的差别不只是“一个是 CPU，一个是 GPU”，而是：

- Pathfinding 的“原子操作”是 **shortest path per net**；
- cuPDLPx 的“原子操作”是 **SpMV / vector ops**。

前者偏 combinatorial optimization oracle，后者偏 continuous optimization solver。

---

## 4. 如果目标是 GPU global routing，真正的建模分叉点在哪里

要把 global routing 做成 GPU-friendly，有三条路线。

### 路线 A：继续走 Pathfinding/Lagrangian，重点做 shortest path GPU 化

这是最直观的路线。你保留 DAC 2023 的整体框架：

- 继续使用拉格朗日松弛；
- GPU 上做 batched shortest path / A* / wave propagation；
- GPU 上更新 multipliers 和 edge costs；
- 之后用 rip-up & reroute 或 repair 完成离散可行解。

**优点**：

- 保留离散路径语义；
- 与现有 router 生态最接近；
- 比较容易与 DAWA*、HeLEM-GR、GAMER、InstantGR 等现有 GPU routing 思路对接。

**缺点**：

- 仍然高度依赖 irregular graph traversal；
- congestion update 常常落在 `edge_usage[e]++` 这类 atomic-heavy 操作上；
- memory bandwidth 和 contention 仍然是核心瓶颈。

### 路线 B：走 edge-based LP relaxation，用 cuPDLPx 直接解 relaxed flow

这条路线对应我们前面推导过的形式：

- 变量是每个 net 在每条 edge 上的流变量 `f_{n,e}`；
- 约束包括 flow conservation、capacity 和 box constraints；
- 可进一步加入 overflow slack 变量；
- 用 cuPDLPx 解 relaxed LP，再做 rounding / repair。

**优点**：

- 理论上最接近 RUPlace 的 routing ILP；
- 求解结构完全 solver-centric；
- GPU workload 从 graph search 转为 sparse linear algebra。

**缺点**：

- 变量规模会达到 `|N| * |E|`，往往不可承受；
- 即便 LP 可解，fractional flow 到离散路径的恢复也不简单；
- 对大规模 benchmark，可能不具备工程可落地性。

### 路线 C：走 path-based LP / column generation，再用 cuPDLPx 做 master LP

这条路线是最值得认真考虑的折中方案：

- 不对所有 `(net, edge)` 建变量；
- 对每个 net 只保留少量 candidate paths；
- master problem 是 LP，可交给 cuPDLPx；
- pricing subproblem 仍是 shortest path，可以单独 GPU 化；
- 需要时通过 column generation 扩充 path 集。

**优点**：

- 变量规模从 `|N||E|` 降到 `Σ_n K_n`；
- cuPDLPx 真正能用起来；
- shortest path oracle 仍保留，便于继承 Pathfinding 模型的优点；
- 很适合写成“solver-centric + bandwidth-aware”的论文定位。

**缺点**：

- 要设计 path pool、pricing、rounding 和 repair；
- 不是拿现成 router 或现成 LP solver 就能直接跑通；
- 论文工作量会落在 formulation + decomposition + system design 三方面。

从研究价值和可实现性平衡看，**路线 C 最值得深入**。

---

## 5. 为什么“直接把 Pathfinding ILP 扔给 cuPDLPx”不成立

这是最容易被误解的一点。

### 5.1 因为 cuPDLPx 解的是 LP，不是 IP

Pathfinding paper 的变量 `x_{n,ê}` 是二值变量。RUPlace 的 `f_{n,e}` 也是二值变量。 fileciteturn1file0L106-L123 fileciteturn1file1L63-L88

而 cuPDLPx 面向的是 LP 形式 `l_c <= Ax <= u_c`、`l_v <= x <= u_v`。它没有 branch-and-bound 或 cutting-plane 框架来强制整数性。 citeturn772186search2turn772186search3

### 5.2 因为 edge-based full relaxation 规模太大

对大型 global routing graph：

- nets 可能在 10^5 量级；
- edges 可能在 10^6 量级；
- `|N||E|` 变量直接爆炸。

这不是 cuPDLPx 特有的问题，而是 edge-based multicommodity flow formulation 的结构问题。

### 5.3 因为 relaxed LP 的输出不是最终 legal route

即便 LP relaxation 解出来，结果可能是：

- 一个 net 同时走了多条分数路径；
- 不一定对应单条 simple path；
- 容量 slack 虽然给出 lower bound / tradeoff，但不等于最好的最终离散解。

所以必须补上：

- path extraction；
- randomized / greedy rounding；
- repair / reroute；
- layer assignment / via-aware refinement。

这意味着 cuPDLPx 在这个问题里更适合作为 **continuous master solver**，而不是 full router。

---

## 6. 为什么说 cuPDLPx 路线是 solver-centric

“solver-centric” 不是一句包装词，而是一个实质性区别。

### 6.1 与传统 GPU global routing 的区别

很多 GPU global routing 论文的基本结构是：

- 先有一个 router 流程；
- 再把 maze routing / pattern routing / A* / edge update 搬到 GPU；
- GPU 的角色是 **kernel accelerator**。

而如果你以 cuPDLPx 为中心，GPU 的角色就会变成：

- 统一执行 primal-dual iteration；
- 统一执行稀疏矩阵-向量乘；
- 统一处理约束残差和 dual updates；
- 整个 routing flow 是“在解一个松弛优化问题”。

这时 GPU 不是“帮某个 router kernel 变快”，而是“整个 routing backend 的 solver”。

### 6.2 RUPlace 提供了一个很强的论证参照

RUPlace 把 routing 明确建成 ILP，再作为统一 placement-routing formulation 的一部分处理。它表明：

- routing 不是只能做 heuristic search；
- routing 完全可以被看成一个数学规划子问题；
- solver-level 设计是合理的。 fileciteturn1file1L53-L142

虽然 RUPlace 本身不是用 cuPDLPx，也不是做 pure global routing，但它帮助论证了：

> **把 routing 看成一个 optimization subproblem，而非单纯 pathfinding procedure，是合理且前沿的。**

这正是 “solver-centric” 的理论基础。

---

## 7. 为什么说 cuPDLPx 路线是 bandwidth-aware

### 7.1 现有 GPU routing 的真实瓶颈往往不在 shortest path 本身

如果沿着 DAC 2023 的思路做 GPU 化，你很快会遇到两个痛点：

1. **congestion accumulation**：很多 nets 的路径都要对 edge usage 做更新；
2. **edge cost / multiplier update**：需要频繁扫描大量 edge 状态。

这两步常常比 shortest path kernel 本身更吃带宽、更容易产生 atomic contention。

### 7.2 Pathfinding 框架中的更新更偏“irregular graph workload”

以 Lagrangian/pathfinding 路线为例，典型操作是：

- 每个 net 走出一条 path；
- path 上每条 edge 计数加 1；
- 基于 overflow 更新 multiplier；
- 重新跑 pathfinding。

这类访问模式通常：

- 不规则；
- 跨 net 路径重叠导致 atomic 冲突；
- 难以做到高效 memory coalescing；
- 在大设计上更容易 memory-bound。

### 7.3 cuPDLPx 的主算子更接近 GPU 友好的 streaming workload

cuPDLPx 的核心工作负载是：

- 稀疏矩阵-向量乘 `Ax`、`A^T y`；
- 向量更新；
- 盒约束投影。 citeturn772186search0turn772186search2turn772186search3

这些操作虽然也受内存带宽限制，但它们：

- 访问模式更规则；
- 更接近标准 GPU 稀疏线性代数；
- 更容易用 CSR/CSC + fused kernels + streaming loads 优化；
- 没有大规模随机 atomic edge increments 这一类最坏情况。

因此，如果 routing formulation 能被改写成适合 cuPDLPx 的 LP/master problem，那么你就有充分理由说：

> **我们的设计不是只在“算路核”上做 GPU 化，而是围绕 bandwidth bottleneck 重塑了整个求解过程。**

这就是 “bandwidth-aware” 的真正含义。

---

## 8. 你真正的创新点应该放在哪里

如果未来你要写论文或开题，创新点不能简单写成：

- “我们把 shortest path 放到 GPU 上”；
- “我们把 Lagrangian routing 用 GPU 做了”；
- “我们用 cuPDLPx 解 LP 了”。

这些都不够强，也容易与现有 GPU routing 或 generic LP solver 应用撞车。

更合理的创新定位应是以下四点。

### 8.1 从 router-kernel acceleration 转向 optimization-solver backend

定位语句可以是：

> Prior GPU global routing methods primarily accelerate discrete routing kernels, whereas our method treats GPU as the backend of a relaxed optimization solver for routing.

对应中文就是：

- 现有方法多把 GPU 当作 routing kernel 的加速器；
- 你的方法把 GPU 当作 routing relaxed problem 的求解引擎。

### 8.2 从 irregular path updates 转向 structured sparse linear operators

你需要强调：

- 传统 pathfinding/Lagrangian 在 GPU 上真正麻烦的是 path accumulation 和 edge update；
- 你的 formulation 把大部分计算转成 `Ax` / `A^T y` / vector updates；
- 这不是“换一个 solver 名字”，而是 workload 结构发生了变化。

### 8.3 relaxed master + discrete recovery 的双层框架

真正可发表的路线不是“LP 替代 router”，而是：

- relaxed master problem 负责给出全局 congestion-aware guidance；
- discrete path extraction / pricing / repair 负责恢复 legal routing；
- 二者协同，而不是谁单独替代谁。

### 8.4 formulation–algorithm–system co-design

你最有希望和现有 GPU routing 拉开差距的地方，是把创新同时放在：

- formulation：path-based LP / edge-based relaxation / soft capacity；
- algorithm：PDLP / column generation / rounding / repair；
- system：CSR data layout / fused dual update / low-contention path recovery。

如果三层都做，创新性就足够明确。

---

## 9. 推荐的技术路线：不是“全 LP 替代”，而是“LP master + path pricing + repair”

综合三份材料，我认为最有前途的方案如下。

### 9.1 Step 1: 从 DAC 2023 借 pathfinding 结构

沿用它的 two-pin decomposition 和 directed grid graph 思路：

- multi-pin net 先用 FLUTE 或 Steiner decomposition 变成 two-pin commodities；
- directed graph 上路径语义保持清晰；
- shortest path 仍然可以作为 pricing oracle 或 repair oracle。 fileciteturn1file0L84-L123

### 9.2 Step 2: 不直接保留 edge-based full IP，而改成 path-based LP master

对每个 net 只维护少量 candidate paths：

- 变量从 `x_{n,ê}` 改成 `z_{n,p}`；
- 约束包括：每个 net 选一路径、edge capacity、必要时 overflow slack；
- 这部分可以直接映射到 cuPDLPx 的 `l_c <= A x <= u_c` 形式。 citeturn772186search2turn772186search3

### 9.3 Step 3: 用 cuPDLPx 解 LP master

这样做的目的不是直接得到最终 legal route，而是得到：

- congestion-aware path weights；
- dual signals；
- 对当前 path set 的高质量 relaxed allocation。

### 9.4 Step 4: 用 pricing / shortest path 产生新路径

这一步可以借鉴 DAC 2023 的 DAWA* 思路，但角色变了：

- 不再是 full router 的唯一核心；
- 而是 column generation 里的 path generation oracle；
- 或 rounding 失败后的 repair oracle。

### 9.5 Step 5: 用 discrete recovery 生成最终 routing

可以包括：

- greedy selection based on fractional weights；
- randomized rounding；
- local repair；
- rip-up & reroute。

这样，你既继承了 Pathfinding paper 对路径语义和离散可行性的优势，又发挥了 cuPDLPx 在 GPU 上的 solver 优势。

---

## 10. 与 RUPlace 的关系：该借什么，不该借什么

RUPlace 对你的研究有帮助，但不是主线。

### 应该借的

1. **routing ILP 表达方式**：`f_{n,e}`、capacity、`Af = h(x)` 这些表达方式很利于你写 formal problem statement。 fileciteturn1file1L53-L99
2. **congestion penalty 思路**：把 overflow 作为显式 penalty/slack，而不是纯硬约束。 fileciteturn1file1L63-L88
3. **solver-level framing**：routing 是 optimization subproblem，不是黑盒。

### 不应该借的

1. **placement-routing unified 目标本身**：这会把问题拉大，偏离你的目标；
2. **ADMM + Wasserstein 正则**：这更适合 placement-coupled setting，不适合作为 standalone global routing 的主框架；
3. **cell inflation / area adjustment**：这是 placement refinement 技术，不是你想做的 routing GPU solver 的核心。

一句话概括：

> RUPlace 更适合作为“怎么把 routing 写成数学规划”的参考，而不是“怎么把 routing 在 GPU 上高效求解”的直接模板。

---

## 11. 现阶段最现实的研究主张

如果要把这个方向概括成一句研究主张，我建议写成：

> **We do not aim to merely GPU-accelerate a routing kernel. Instead, we seek to redesign global routing as a bandwidth-aware, solver-centric optimization pipeline, where cuPDLPx serves as the LP master solver and pathfinding is retained only as a pricing/recovery oracle.**

中文可表述为：

> 我们的目标不是把某个 routing kernel 搬到 GPU，而是把 global routing 重构为一个面向带宽瓶颈、以求解器为中心的优化流水线：由 cuPDLPx 负责 LP 主问题求解，由 pathfinding 负责列生成与离散恢复。

这一定义可以非常清楚地把你和现有 GPU global routing 拉开。

---

## 12. 结论：可行，但前提是“变问题”，而不是“硬套工具”

综合三份材料，可以得到一个很明确的结论：

1. **Pathfinding DAC 2023** 说明，global routing 可以从启发式 search 升级为离散优化问题，并通过 Lagrangian decomposition 变成一系列 shortest path 子问题。 fileciteturn1file0L124-L214
2. **RUPlace DAC 2025** 说明，routing ILP 可以被正式嵌入更大的优化框架中，routing 完全可以被视为 solver-managed subproblem，而不是黑盒。 fileciteturn1file1L53-L142
3. **cuPDLPx** 提供了一个真正适合 GPU 的 LP 求解基础设施，但它只能直接处理 LP，而不是原始整数 routing 问题。 citeturn772186search0turn772186search2turn772186search3

因此：

- **不能**把原始 Pathfinding / RUPlace 的 routing IP 直接扔给 cuPDLPx；
- **可以**把它们重写成 path-based 或局部化的 LP/master problem，再用 cuPDLPx 作为 GPU 求解器；
- 最值得做的不是“套一个 LP solver”，而是提出一个 **solver-centric + bandwidth-aware + discrete-recovery-aware** 的 global routing 新框架。

如果要进一步推进，这个方向下一步最关键的技术问题有三个：

1. **path-based LP master 的具体矩阵构造与稀疏布局**；
2. **fractional solution 到 legal discrete routing 的恢复策略**；
3. **path pricing / column generation 与 GPU shortest path oracle 的协同设计**。

这三点解决后，这条路线才会从“有想法”变成“可发表、可实现、可对比”的完整方案。


可以。下面我把 **global routing 的 ILP → LP relaxation → 适配 PDLP / cuPDLPx 的形式**完整推导一遍。
我会尽量写成你后面可以直接实现的数据结构形式。
其中 ILP 基本出发点和 flow conservation 写法，与 RUPlace 文中给出的 routing formulation 一致：用 (f_{n,e}) 表示 net (n) 是否使用 routing edge (e)，并用 (Af=h(x)) 表达流守恒。

---

# 1. 先定义 routing graph

设 global routing graph 为

[
G=(V,E)
]

其中：

* (V)：gcell 节点集合
* (E)：gcell 之间的 routing edge 集合

对每条 edge (e\in E)，有：

* 容量 (u_e)
* 单位代价 (c_e)

---

# 2. 多商品流 ILP 形式

设 net 集合为 (\mathcal N)。
先从最简单的 **2-pin net** 讲起。对每个 net (n\in\mathcal N)，它有：

* source (s_n)
* sink (t_n)

定义二值变量：

[
f_{n,e}\in{0,1},\quad \forall n\in\mathcal N, e\in E
]

表示：

[
f_{n,e}=1 \iff \text{net } n \text{ 使用 edge } e
]

---

## 2.1 目标函数

最基本目标是总 routing cost：

[
\min \sum_{n\in\mathcal N}\sum_{e\in E} c_e f_{n,e}
]

如果你想允许 overflow penalty，可以稍后再加松弛变量。

---

## 2.2 容量约束

每条 edge 上所有 net 的使用量不能超过容量：

[
\sum_{n\in\mathcal N} f_{n,e}\le u_e,\quad \forall e\in E
]

---

## 2.3 流守恒约束

为了写成线性形式，先给 routing graph 任意定向。
对每条有向 edge (e=(i,j))，定义节点-边关联矩阵 (B\in\mathbb R^{|V|\times |E|})：

[
B_{v,e}=
\begin{cases}
+1,& \text{if } e \text{ leaves } v\
-1,& \text{if } e \text{ enters } v\
0,& \text{otherwise}
\end{cases}
]

对每个 net (n)，定义其 demand 向量 (b_n\in\mathbb R^{|V|})：

[
(b_n)_v=
\begin{cases}
+1,& v=s_n\
-1,& v=t_n\
0,& \text{otherwise}
\end{cases}
]

则 net (n) 的 flow conservation 为：

[
B f_n = b_n
]

其中 (f_n\in\mathbb R^{|E|}) 是 net (n) 在所有 edge 上的变量向量。

于是整体 ILP 为：

[
\min_{{f_n}} \sum_{n}\sum_e c_e f_{n,e}
]

s.t.

[
Bf_n=b_n,\quad \forall n
]

[
\sum_n f_{n,e}\le u_e,\quad \forall e
]

[
f_{n,e}\in{0,1}
]

这就是标准 **edge-based multi-commodity flow ILP**。
它和论文里的写法本质一致，只是把 (Af=h(x)) 展开成了每个 net 的 incidence form。

---

# 3. 写成一个大矩阵形式

为了适配 LP / PDLP，需要把所有变量拼成一个大向量。

定义总变量向量：

[
f=
\begin{bmatrix}
f_1\
f_2\
\vdots\
f_{|\mathcal N|}
\end{bmatrix}
\in \mathbb R^{|\mathcal N||E|}
]

也就是把所有 net 的 edge variables 串起来。

---

## 3.1 目标向量

定义

[
c^{(all)}=
\begin{bmatrix}
c\
c\
\vdots\
c
\end{bmatrix}
\in \mathbb R^{|\mathcal N||E|}
]

其中每个 (c\in\mathbb R^{|E|}) 是 edge cost 向量。

则目标写成：

[
\min (c^{(all)})^T f
]

---

## 3.2 流守恒矩阵

对每个 net 都有一个 (Bf_n=b_n)。
把它们拼成块对角矩阵：

[
A_{flow}=
\begin{bmatrix}
B & 0 & \cdots & 0\
0 & B & \cdots & 0\
\vdots & \vdots & \ddots & \vdots\
0 & 0 & \cdots & B
\end{bmatrix}
\in \mathbb R^{|\mathcal N||V| \times |\mathcal N||E|}
]

右端项：

[
b_{flow}=
\begin{bmatrix}
b_1\
b_2\
\vdots\
b_{|\mathcal N|}
\end{bmatrix}
]

于是：

[
A_{flow} f = b_{flow}
]

---

## 3.3 容量矩阵

容量约束需要对每条 edge，把所有 net 在该 edge 上的变量加起来。

定义 (A_{cap}\in\mathbb R^{|E|\times |\mathcal N||E|})，其第 (e) 行在所有对应 (f_{n,e}) 的位置填 1，其余为 0。

更形式化地说：

[
(A_{cap}f)*e = \sum_n f*{n,e}
]

于是容量约束变成：

[
A_{cap}f \le u
]

其中 (u\in\mathbb R^{|E|}) 是所有 edge capacity 组成的向量。

---

# 4. 原始 ILP 的大矩阵形式

因此原始问题可写成：

[
\min_f (c^{(all)})^T f
]

s.t.

[
A_{flow}f=b_{flow}
]

[
A_{cap}f\le u
]

[
f\in{0,1}^{|\mathcal N||E|}
]

---

# 5. LP relaxation

为了适配 cuPDLPx，先把整数约束放松成 box constraint：

[
0\le f_{n,e}\le 1
]

于是得到 LP：

[
\min_f (c^{(all)})^T f
]

s.t.

[
A_{flow}f=b_{flow}
]

[
A_{cap}f\le u
]

[
0\le f\le 1
]

这就是最直接的 **LP relaxation**。

---

# 6. 变成 cuPDLPx 标准形式

cuPDLPx 支持的标准形式本质是：

[
\min c^T x
]

s.t.

[
l_c \le Ax \le u_c
]

[
l_v \le x \le u_v
]

所以我们只需把 flow equality 和 capacity inequality 合并。

---

## 6.1 合并约束矩阵

定义总约束矩阵：

[
A=
\begin{bmatrix}
A_{flow}\
A_{cap}
\end{bmatrix}
]

变量就是：

[
x=f
]

---

## 6.2 约束上下界

对于 flow conservation：

[
A_{flow}f=b_{flow}
]

在 cuPDLPx 里等价于：

[
l_{flow}=u_{flow}=b_{flow}
]

对于 capacity：

[
A_{cap}f\le u
]

可写成：

[
-\infty \le A_{cap}f \le u
]

因此：

[
l_c=
\begin{bmatrix}
b_{flow}\
-\infty
\end{bmatrix},\qquad
u_c=
\begin{bmatrix}
b_{flow}\
u
\end{bmatrix}
]

更完整地：

[
l_c=
\begin{bmatrix}
b_{flow}\
-\infty\cdot \mathbf 1_{|E|}
\end{bmatrix},
\quad
u_c=
\begin{bmatrix}
b_{flow}\
u
\end{bmatrix}
]

变量上下界：

[
l_v=0,\qquad u_v=1
]

于是就完全进入 cuPDLPx 形式了。

---

# 7. 如果你要允许 overflow，而不是 hard capacity

很多 global router 不会强制容量完全满足，而是允许 overflow，并在目标里惩罚。
这比 hard feasibility 更符合真实 global routing。

引入 overflow 变量 (s_e\ge 0)，表示 edge (e) 的超容量量：

[
\sum_n f_{n,e} \le u_e + s_e
]

并在目标里加罚项：

[
\mu \sum_e s_e
]

---

## 7.1 新变量

定义总变量：

[
x=
\begin{bmatrix}
f\
s
\end{bmatrix}
]

其中：

* (f\in\mathbb R^{|\mathcal N||E|})
* (s\in\mathbb R^{|E|})

目标向量变成：

[
c'=
\begin{bmatrix}
c^{(all)}\
\mu \mathbf 1_{|E|}
\end{bmatrix}
]

---

## 7.2 约束

### 流守恒

overflow 不参与 flow conservation：

[
[A_{flow}\ \ 0]x=b_{flow}
]

### 容量松弛

[
A_{cap}f - s \le u
]

即：

[
[A_{cap}\ \ -I]x \le u
]

### 变量范围

[
0\le f\le 1,\qquad s\ge 0
]

如果不想给 (s) 上界，可设 (u_s=+\infty)。

---

## 7.3 cuPDLPx 形式

总矩阵：

[
A=
\begin{bmatrix}
A_{flow} & 0\
A_{cap} & -I
\end{bmatrix}
]

约束界：

[
l_c=
\begin{bmatrix}
b_{flow}\
-\infty
\end{bmatrix},
\qquad
u_c=
\begin{bmatrix}
b_{flow}\
u
\end{bmatrix}
]

变量界：

[
l_v=
\begin{bmatrix}
0\
0
\end{bmatrix},
\qquad
u_v=
\begin{bmatrix}
1\
+\infty
\end{bmatrix}
]

目标：

[
\min (c')^T x
]

这个版本通常比 hard-capacity 版本更实用。

---

# 8. 若一个 net 有多个 pins，怎么处理

真实 net 通常不是 2-pin，而是 multi-pin。这里有两条路。

## 路线 A：先做 net decomposition

把 multi-pin net 拆成若干 2-pin connection，比如：

* MST / FLUTE tree edges
* star decomposition
* Steiner tree decomposition

然后每个 2-pin 子连接当成一个 commodity。

这是最容易直接落进上面的 LP 形式。

## 路线 B：用多端流模型

对每个 multi-pin net 选一个 root，向其他 terminal 发 flow。
这会让 commodity 数增大，但形式一样，本质还是多个 2-pin demand。

工程上一般先走路线 A。

---

# 9. 这个 LP 为什么是“solver-centric”

因为经过 relaxation 后，问题不再是“逐条 net 跑 maze routing”，而是统一变成：

[
\min c^T x
\quad
\text{s.t. linear constraints}
]

求解过程由 primal-dual / PDHG 驱动。
核心计算是：

* (Ax)
* (A^Ty)
* projection to box constraints

所以 GPU 处理的是一个统一 solver loop，而不是一堆离散 routing kernels。

---

# 10. 这个 LP 为什么更“bandwidth-aware”

因为它把原先很不规则的操作：

* 对每条路径上的 edge 做原子加
* 随机更新 congestion map
* 不规则 path traceback

转成更规则的线性代数：

* sparse matrix-vector multiply
* vector updates
* box projection

在 GPU 上，这通常比大量原子 edge updates 更适合高带宽 streaming 访问。

---

# 11. 但这里有一个非常重要的现实问题

上面这个 **edge-based LP** 虽然形式漂亮，但变量规模可能极大：

[
|\mathcal N||E|
]

如果：

* nets = (10^5)
* edges = (10^6)

那变量数会到 (10^{11})，直接不可行。

所以真正落地时，一般要做 **降维/分解**。

---

# 12. 两个更可实现的方向

## 方向 A：局部窗口 LP

只在 hotspot window 内建模：

* 只取窗口内 edges
* 只取经过该窗口的 nets

这样变量量级能降很多。

## 方向 B：path-based LP

不对每个 edge 建变量，而是对每个 net 的若干 candidate paths 建变量。

---

# 13. path-based LP 更适合实现

对每个 net (n)，先生成一小组 candidate paths：

[
\mathcal P_n = {p_{n,1},p_{n,2},\dots,p_{n,K_n}}
]

定义变量：

[
z_{n,k}\in[0,1]
]

表示 net (n) 选第 (k) 条候选路径的程度。

---

## 13.1 路径选择约束

每个 net 必须选一条路径：

[
\sum_{k=1}^{K_n} z_{n,k}=1,\quad \forall n
]

## 13.2 容量约束

若 path (p_{n,k}) 经过 edge (e)，定义常数

[
\delta_{e,n,k}=
\begin{cases}
1,& e\in p_{n,k}\
0,& \text{otherwise}
\end{cases}
]

则容量约束：

[
\sum_n \sum_k \delta_{e,n,k} z_{n,k} \le u_e
]

## 13.3 目标

若路径代价为 (C_{n,k})，则：

[
\min \sum_n \sum_k C_{n,k} z_{n,k}
]

这是一个更小的 LP：

* 变量数：(\sum_n K_n)
* 通常远小于 (|\mathcal N||E|)

它也完全能写成 cuPDLPx 标准形式。

---

# 14. 对你来说最推荐的路线

如果你真想用 cuPDLPx 做 GPU global routing，我建议顺序是：

### 第一阶段

先实现 **path-based LP relaxation**

因为：

* 变量规模可控
* 容易接 cuPDLPx
* 容易从 fractional solution 恢复到离散路径

### 第二阶段

再考虑 column generation / pricing：

* 初始给每个 net 少量 candidate paths
* 解 LP
* 根据 dual cost 重新找更优 path
* 加入候选集合继续解

这就和 pathfinding model 很自然结合起来了。

---

# 15. 最后给你一个可直接实现的 cuPDLPx 映射

以 **path-based LP + overflow slack** 为例。

变量：

[
x = [z; s]
]

其中：

* (z_{n,k}\in[0,1])
* (s_e\ge 0)

目标：

[
\min \sum_{n,k} C_{n,k} z_{n,k} + \mu \sum_e s_e
]

约束：

### 每个 net 选一路径

[
\sum_k z_{n,k}=1
]

### 容量带 slack

[
\sum_{n,k}\delta_{e,n,k} z_{n,k} - s_e \le u_e
]

变量界：

[
0\le z_{n,k}\le 1,\qquad s_e\ge 0
]

这时：

* (A) 是“net-path incidence + edge-path incidence”拼出来的稀疏矩阵
* 特别适合 PDLP / cuPDLPx

---

# 16. 一句话总结

如果严格从 ILP 推到 PDLP，可得到：

### edge-based 形式

[
\min c^T f
\quad
\text{s.t. }A_{flow}f=b,\ A_{cap}f\le u,\ 0\le f\le 1
]

这是最标准的推导。

但工程上更推荐：

### path-based 形式

[
\min C^T z + \mu 1^Ts
\quad
\text{s.t. path selection, capacity constraints, }0\le z\le 1, s\ge 0
]

这个更适合 cuPDLPx 落地。

---

如果你愿意，我下一步可以继续给你写两样东西中的一个：

1. **把 path-based LP 写成明确的 CSR 矩阵构造方法**
2. **讲怎么从 cuPDLPx 的 fractional solution 恢复成离散 global routing path**


https://github.com/MIT-Lu-Lab/cuPDLPx