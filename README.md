# TopBlown-CausalDiag: 顶吹炉合成数据的因果发现

本项目面向你的 `topblown_synth_8000.csv`（18个过程变量 + fault_id 多类别标签），提供：

## 1) 因果发现（输出因果图 + TPR/FDR/SHD/F1）
对比方法：
- **PC**（线性高斯/偏相关 Fisher-Z CI test）
- **GES**（提供两种：
  - 默认：BIC 分数的 greedy search（HC-BIC 变体，代码纯 Python）
  - 可选：若你安装 `causal-learn`，可切换到其 GES 实现）
- **NOTEARS**（线性 NOTEARS，L1 稀疏 + DAG 约束）
- **RL-BIC**（DQN 搜索 DAG，reward=BIC 改善 + DAG 罚项；提供 quick/full 两种运行模式）

输出：
- `outputs/causal/<method>/graph.png` 因果图
- `outputs/causal/<method>/adjacency.csv` 邻接矩阵
- `outputs/causal/<method>/metrics.json` TPR/FDR/SHD/F1

> 说明：TPR/FDR/SHD/F1 需要 **真值因果图**。本项目提供了 `priors/ground_truth_edges.csv` 作为默认真值（来自工艺先验/合成生成假设）。
> 如果你的合成数据生成时采用了不同 DAG，请把该文件替换为你的真实边集。

---

## 快速开始

### 0) 安装依赖
```bash
pip install -r requirements.txt
```

### 1) 一键跑全流程（EDA -> 因果发现四方法 -> GNN四模型）
```bash
python experiment/run_all.py --csv data/topblown_synth_8000.csv --outdir outputs --device auto
```

### 2) 只跑因果发现
```bash
python experiment/run_causal_discovery.py --csv data/topblown_synth_8000.csv --outdir outputs --use_normal_only
```

---

## 目录结构
```
./data/                      # 数据
./priors/                    # 真值图/先验
./experiment/                   # 可执行脚本
./src/topblown_causal_diag/  # 核心库
./outputs/                   # 运行输出
```


## Ours (你的方法) 已集成

本项目额外集成了你自己的方法，用于对比实验：

1) **因果发现 Ours**：Prior-injected RL-BIC（在 RL-BIC reward 里加入专家先验 shaping）
   - 专家先验边文件：`priors/expert_prior_edges.csv`（你可用自己的先验替换）
   - 可选禁用边：`priors/forbidden_edges.csv`（不存在则忽略）
