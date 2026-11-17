# PerfectDou 独立推理引擎（脱 .so / 脱 torch）

把 [PerfectDou](https://github.com/Netease-Games-AI-Lab-Guangzhou/PerfectDou) 的**观测编码**与**动作编码**用纯 Python 逐位复现，
让模型脱离两个 Linux/cp37 的 `.so`、脱离 torch / rlcard，只依赖 **numpy + onnxruntime** 即可推理。

一句话：**同样的局面，本仓库算出的 32738 维输入向量与官方 `.so` 几乎逐位一致（实测整局动作一致率 99.67%）。**

---

## 为什么需要它

官方 PerfectDou 的特征编码由两个二进制提供：

| 文件 | 问题 |
|---|---|
| `perfectdou/env/encode.cpython-37m-x86_64-linux-gnu.so` | cp37 + Linux ABI，Windows / 高版本 Python 直接 import 失败 |
| `libCalculateLeftHands.so`（仓库根） | 相对路径 dlopen，必须 `cd` 到仓库根；内部是 C++ 的最小步数搜索 |

模型本身是标准 ONNX：输入 `[batch, 32738]`，输出 `action_logit [batch, 621]`。
只要把 32738 维输入按同样的规则算出来，推理就完全不需要那两个 `.so`。

32738 的构成：

```
32738 = 4688（局面特征 x_no_action） + 150 × 187（合法的动作特征 legal_actions_arr）
```

本仓库把这两段都复现了。

---

## 验证结果

对拍方式：同一局、同一步，`.so` 与本仓库各算一次输入向量，用**同一份** onnx 推理，比较 `argmax(logit[:len(legal)])` 是否指向同一个动作。

| 项目 | 规模 | 结果 |
|---|---|---|
| 局面特征 `x_no_action`（4688 维） | 1600 个样本（三视角） | 1590 个 **bit-exact**，其余 10 个仅差 4686 一位 |
| 动作特征（5459 个含牌动作行 × 187 位） | 1600 个样本 | 5436 行 187 位全同 = **99.58%**；位级错误 0.0023% |
| 合法动作枚举（集合 + 顺序） | 600 个样本 | **600 / 600 完全一致** |
| **整链动作一致率** | **1000 局 / 61803 步** | **61600 步相同 = 99.67%** |

1000 局用固定种子（`--seed 2025`）可复现。203 个不一致步的逐位诊断显示差异全部落在动作段第 2/3/4 列与 4686（炸弹位），局面特征其余位零差异。

---

## 目录结构

```
pd_policy.py          运行时策略：act(position, hand, base, seq) -> 出哪几张牌

tools/
  obs_encoder.py      局面特征：4688 维（26 帧 × 180 + 8 标量），含 minSteps 纯 Python 实现
  action_encoder.py   动作特征：150 行 × 187 列，含 32738 拼接
  movegen_py.py       合法动作枚举，与官方 move_generator 同集合、同顺序
  action_space_py.py  621 动作空间：specific_map(27472) → action_space(621) 两级映射
  chain_check_v2.py   整链对拍执行器（v2，推荐）：直接复用本仓库的 obs_encoder / movegen_py，
                      只需 Linux + py3.7 + 官方 .so 环境
  chain_check.py      整链对拍执行器（v1，早期版本）：依赖 /tmp 下另放的临时编码文件，仅作留档
  probe_obs.py        单维探针：改一个字段，看哪些位变了（解题过程留档）
  dump_golden.py      金标准导出：随机对局 → npz（用于回归）
  onnx_selftest.py    端到端自检：真实局面 → 32738 → onnx → 621 → 动作

docs/
  obs_structure_v2.md   局面特征 4688 维的完整结构（帧布局、块/行语义、标量含义）
  obs_structure_v1.md   旧版，已被 v2 覆盖，保留为演进记录
  action_structure.md   动作特征 187 列语义、621 分段表、已知偏差
```

需要自行准备、**不入库**的 `assets/`：

```
assets/
  action_space.json      621 个动作模板 ↔ id
  specific_map.json      27472 个具体牌面 ↔ 模板名
  card_type.json         牌型识别
  action_values.json     另一套索引口径（主链未使用）
  landlord.onnx          地主模型
  landlord_up.onnx       上家模型
  landlord_down.onnx     下家模型
```

四个 json 与三个 onnx 均取自上游 PerfectDou 仓库。

---

## 快速使用

### 只做编码（任意平台，numpy 即可）

```python
import sys
sys.path.insert(0, 'tools')

from obs_encoder import encode_x_no_action
from action_encoder import ActionEncoder
from movegen_py import legal_moves_with_pass

# 局面：只需要这五个字段，其余全部自推
meta = {
    'position': 'landlord',                 # landlord / landlord_up / landlord_down
    'hand':     [...],                      # 自己的手牌（牌值约定见下）
    'other':    [],                         # 别人手牌；盲牌置空
    'base':     [...],                      # 三张底牌
    'seq':      [[...], [], [...]],         # 出牌历史，按发生顺序，pass 用空列表
}

obs   = encode_x_no_action(meta)            # 4688
enc   = ActionEncoder('assets')             # 目录里放那四个 json
last_move = ...                             # 场上待压的牌；连续 pass 之后为空 []
legal     = legal_moves_with_pass(meta['hand'], last_move)
flat  = enc.flat(meta['hand'], legal, other=[])   # 150 × 187 展平 → 28050
x32738 = list(obs) + list(flat)             # 长度 32738，可直接喂 onnx
```

### 一步到位：给手牌直接出牌

`pd_policy.py` 把「合法动作枚举 → 32738 编码 → onnx 推理 → argmax」封成一个方法：

```python
from pd_policy import PerfectDouPolicy

policy = PerfectDouPolicy()                       # 默认读同目录 assets/

out = policy.act(
    'landlord',
    hand=[3,3,3,4,5,6,7,8,8,9,10,11,12,13,14,14,17,17,20,30],
    base=[17,20,30],
    seq=[],                                       # 出牌历史；pass 记 []
    last_move=[],                                 # 自由出牌；要压牌时给那手牌
)
out['cards']      # [10, 11, 12, 13, 14]
out['id']         # 386
out['top'][:3]    # 前 3 名候选及分值（分值 = logit，越大越偏好，不是胜率）
```

本机实测（Python 3.11 / CPU）：

```
地主 20 张自由出牌 -> [10,11,12,13,14]  id=386  legal=74   首次 250 ms
农民 17 张压 33     -> [14,14]          id=26   legal=4     34 ms
热路径复算          -> 5.4 ms
```

首次慢是模型加载 + minSteps 冷算；同一局面重复决策会命中缓存，稳态 5~15 ms。

### 跑自测（无需 .so）

```bash
python tools/action_encoder.py     # 动作 id 映射与 quirk
python tools/onnx_selftest.py      # 端到端：编码 → 拼 32738 → onnx → 621 → 动作
```

本机基线（Python 3.11 / onnxruntime 1.32，CPU）：

```
[1] landlord 手牌 20 张，legal 74 个
      编码 4688 + 28050 = 32738  ✓   60.4 ms
      推理 -> 621 ✓   197.1 ms（首次含模型加载；热身后续约 10 ms）
      选中 legal[45] = [10,11,12,13,14]   动作 id = 386
[2] landlord_up 手牌 17 张，压 33，legal 4 个
      选中 legal[0] = [14,14]   动作 id = 26
```

编码里的 `minSteps` 是主要开销（每个合法动作算一次），加一层缓存还能再压。

### 跑整链对拍（需要官方仓库与原版 .so）

```bash
cd /path/to/PerfectDou             # 必须：.so 按相对路径 dlopen
python tools/chain_check_v2.py --repo /path/to/PerfectDou --games 1000 --seed 2025
```

`chain_check_v2.py` 是当前使用的版本，直接复用本仓库的 `obs_encoder.py` / `movegen_py.py`，
不依赖 `/tmp` 下的临时文件。`chain_check.py` 是早期版本，依赖 `/tmp/obs_py.py` 与
`/tmp/minsteps.py` 这两个另放的临时编码文件，只作留档，不要用它复现这里的数字。

脚本最终打印两行，格式为：

```
步数 <总步数>，动作一致 <相同步数> (<百分比>%)
每局: [...]
```

---

## 牌值约定

```
3..9 -> 3..9      10 -> 10      11 J      12 Q      13 K      14 A
2 -> 17          小王 -> 20     大王 -> 30
```

字符串表 `'3456789TJQKA2BR'`，拼 key 时必须按 rank 升序（`3 < ... < A < 2 < B < R`）。

---

## 关键实现要点

局面特征 `4688 = 26 帧 × 180 + 8 标量`，每帧 180 = 3 块 × 60，块 = 4 行 × 15（列序 `3..A,2,B,R`）：

- 块 A 行 k = 张数 `>= k`；块 B 行 k = 张数 `== k`
- 块 C：**行 8 = 王炸标记，行 9 顺子(>=5)，行 10 连对(>=3)，行 11 飞机(>=2)**
- 链扫描把 `2` 接在 A 后参与**长度**计算，但 `2` 自己那列不点亮
- 标量：`4682 = minSteps(手牌)/10`、`4683~4685` 按角色分母、`4686 = (炸弹数+1)/16`、`4687 = 场上无待压牌`
- 剩余张数全部从动作序列自推，**不读** `num_cards_left_dict`

动作特征 187 列：

| 列 | 含义 |
|---|---|
| 0 | 621 动作 id（两级查表；查不到时整行全零，复刻官方 quirk） |
| 1 | 是否炸弹 / 王炸 |
| 2 | 1 表示别人压不住这手（盲牌时恒 1） |
| 5 | `minSteps(手牌 − 本动作)/10`（pass 行用 `minSteps(手牌)/10`） |
| 6 | 恒 1.0 |
| 7 ~ 186 | `card_frame(本动作的牌)` |

150 行只填前 `len(legal)` 行，其余保持 0。

`minSteps` 是从 C++ 移植的最小步数搜索（张数 DP + 枚举链），移植时必须做三处修正：
**去掉原实现的剪枝**（其下界是"忽略链"的上界，会漏最优解）、**允许单王当翅膀**、**fast-path 仅在无链且无王时启用**。

---

## 已知偏差（不打算修）

| 偏差 | 影响面 | 原因 |
|---|---|---|
| 动作段第 3/4 列恒 0 | 11 / 5459 行 | 依赖上家与下家**各自的牌分配**，真机盲牌拿不到 —— 结构性缺失 |
| 第 2 列长顺子例外 | 10 行 | 官方在该情形给 0 而非 1 |
| 4686 王炸位 | 10 / 1600 样本 | 同时有王炸与普通炸弹时官方多 1，只影响 1/16 刻度 |

这三类把一致率锁在 **99.67%**，理论封顶约 99.7%。100% 不可达，因为第一类缺的是信息而不是算法。

---

## 不包含什么

- **不含原版 `.so` 与上游仓库源码**。`chain_check.py` 对拍时才需要它们。
- **不含界面与对局引擎**（发牌、叫地主、加分、出牌轮转、结算）。本仓库只到"局面 → 32738 输入向量"这一层，推理与交互请接在自己那一层。

---

## 环境

- Python ≥ 3.8（实测 3.11）
- `numpy`：编码与拼接
- `onnxruntime`：推理（编码本身不需要）
- 对拍额外需要：Linux + Python 3.7 + 上游仓库 + 两个 `.so`

---

## 许可与致谢

- 特征定义、动作空间、模型权重与训练算法归上游作者（PerfectDou，Netease Games AI Lab Guangzhou）所有。
- 本仓库只包含独立复现的**编码实现**，不含上游代码副本；使用与分发请遵守上游许可。
- 上游许可为 **Apache License 2.0**，全文见仓库根目录 `LICENSE`。若要连 `assets/` 下的
  模型权重与四个 json 一起分发，请一并向使用者提供该许可并注明来源为上游 PerfectDou 仓库。
- `minSteps` 的算法思路参考自 [esrrhs/PerfectDouCPP](https://github.com/esrrhs/PerfectDouCPP) 的 `oracle.cpp`（论文附录 E.1 Algorithm 1）。
- 动作空间的构造顺序参考其对 `action_space.cpp` 的整理（论文附录 E.2 Algorithm 2）。

论文：*PerfectDou: Dominating DouDizhu with Perfect Information Distillation*
