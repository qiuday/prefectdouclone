# -*- coding: utf-8 -*-
"""动作段编码 —— 复刻官方 .so 的 legal_actions_arr（150 行 × 187 维）

这是 32738 维输入的后 28050 维（150×187），与 obs 段的 4688 维拼成整输入。

列布局（与金标准逐位对拍确认，见 docs/action_structure.md）：

  列  0     621 动作 id。需要**两级映射**：
              具体牌排序拼串 -> specific_map.json -> 抽象串 -> action_space.json -> id
              specific_map 查不到时（如四带二带双王 TTTT+大小王），官方 .so 给**整行全零**
  列  1     炸弹 / 王炸标记（type 4 / type 5）
  列  2     1 = 外面压不住这手牌（other_hand_cards 里没有能压的牌）
  列  3/4   官方有值但语义未解（约占 0.2% 的行，依赖上家/下家各自的牌分配）；
              程序里恒填 0，只影响这些边缘行
  列  5     min_steps(出掉这手后的剩牌) / 10 ；pass 行 = min_steps(当前手牌) / 10
  列  6     恒 1.0
  列  7..186 card_frame(动作牌) 180 维（与 obs 帧同一函数）

行：只填前 len(legal) 行，其余全 0；pass 行 = 空牌动作（id=620）。

实测（1600 样本 5459 行）：187 位完全一致 5436 行 = 99.58%，
位级错误 23 位 / 1,020,833 位 = 0.0023%（全部落在列 2/3/4/5）。
"""
import json
import os

try:
    import numpy as np
except ImportError:                                   # 允许无 numpy 时退化为 list
    np = None

try:
    from obs_encoder import card_frame, min_steps
    from movegen_py import get_move_type, legal_moves
except ImportError:
    from .obs_encoder import card_frame, min_steps
    from .movegen_py import get_move_type, legal_moves

ROWS = 150          # 动作段行数
COLS = 187          # 每行维度
DIM = ROWS * COLS   # 28050

RANKS = list(range(3, 15)) + [17, 20, 30]
CHAR = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T',
        11: 'J', 12: 'Q', 13: 'K', 14: 'A', 17: '2', 20: 'B', 30: 'R'}


class ActionEncoder(object):
    """asset_dir 需含 action_space.json（621 项）与 specific_map.json（27472 项）。"""

    def __init__(self, asset_dir):
        with open(os.path.join(asset_dir, 'action_space.json'), 'r', encoding='utf-8') as f:
            self.space = json.load(f)
        with open(os.path.join(asset_dir, 'specific_map.json'), 'r', encoding='utf-8') as f:
            self.spec = json.load(f)

    def action_id(self, cards):
        """具体牌 -> 621 空间 id；查不到返回 None（对应官方整行全零）。"""
        key = ''.join(CHAR[c] for c in sorted(int(c) for c in cards))
        abs_list = self.spec.get(key)
        if not abs_list:
            return None
        return self.space.get(abs_list[0])

    def rows(self, hand, legal, other=None):
        """hand: 自己当前手牌；legal: 官方顺序的合法动作（含 [] 表示 pass）；
        other: 外面已知的牌（真机盲牌时传 None）。返回 (150,187) float32。
        """
        hand = [int(c) for c in hand]
        arr = np.zeros((ROWS, COLS), dtype=np.float32) if np is not None else \
            [[0.0] * COLS for _ in range(ROWS)]
        outside = [int(c) for c in (other or [])]

        for i, act in enumerate(legal[:ROWS]):
            a = sorted(int(c) for c in act)
            if not a:                                             # pass
                rid = 620.0
                h5 = round(min_steps(hand) / 10.0, 6)
                vals = (rid, 0.0, 0.0, h5, 1.0, [0] * 180)
                if np is not None:
                    arr[i, 0] = rid
                    arr[i, 5] = h5
                    arr[i, 6] = 1.0
                else:
                    arr[i][0] = rid
                    arr[i][5] = h5
                    arr[i][6] = 1.0
                continue

            aid = self.action_id(a)
            if aid is None:                                       # 官方查不到 -> 整行留零
                continue

            bomb = 1.0 if get_move_type(a)['type'] in (4, 5) else 0.0
            can_beat = 1.0 if not legal_moves(outside, a) else 0.0   # 外面压不住
            rest = list(hand)
            for c in a:
                if c in rest:
                    rest.remove(c)
            h5 = round(min_steps(rest) / 10.0, 6)
            frame = card_frame(a)
            if np is not None:
                arr[i, 0] = float(aid)
                arr[i, 1] = bomb
                arr[i, 2] = can_beat
                arr[i, 5] = h5
                arr[i, 6] = 1.0
                arr[i, 7:] = frame
            else:
                arr[i][0] = float(aid)
                arr[i][1] = bomb
                arr[i][2] = can_beat
                arr[i][5] = h5
                arr[i][6] = 1.0
                arr[i][7:] = frame
        return arr

    def flat(self, hand, legal, other=None):
        """直接返回拼进 32738 用的 28050 维一维数组。"""
        arr = self.rows(hand, legal, other)
        return arr.reshape(-1) if np is not None else [v for row in arr for v in row]


def build_x(obs4688, action_dim):
    """4688 维 obs + 28050 维动作段 -> 32738 维模型输入。"""
    if np is None:
        raise RuntimeError('build_x 需要 numpy')
    obs = np.asarray(obs4688, dtype=np.float32).reshape(-1)
    act = np.asarray(action_dim, dtype=np.float32).reshape(-1)
    if obs.size != 4688 or act.size != DIM:
        raise ValueError('维度不符: obs=%d (应 4688), action=%d (应 %d)' % (obs.size, act.size, DIM))
    return np.concatenate([obs, act])
