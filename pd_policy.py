# -*- coding: utf-8 -*-
"""PerfectDou 策略封装：给手牌和局面，直接返回"出哪几张牌"。

把三件事包成一个函数：合法动作枚举 -> 32738 编码 -> onnx 推理 + argmax。
调用方不需要知道 4688 / 28050 / 621 这些数字。

    from pd_policy import PerfectDouPolicy

    policy = PerfectDouPolicy()                     # 默认读同目录的 assets/
    out = policy.act('landlord', hand, base=base, seq=seq, last_move=[])
    out['cards']                                    # 建议出的牌，[] 表示 pass
    out['id']                                       # 621 抽象动作 id
    out['top']                                      # 前 5 名候选及分值，便于展示"第二选择"

约定（与 tools/ 的编码器一致）：
  手牌用整数牌值：3..14、17(2)、20(小王)、30(大王)
  seq 是出牌历史，按发生顺序，pass 记为空列表 []
  last_move 是"轮到我之前那手要压的牌"；连续 pass 后的自由出牌请显式传 []

不依赖 torch / rlcard / 任何 .so，只需要 numpy + onnxruntime。
"""
import os
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, 'tools')
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from action_encoder import ActionEncoder                       # noqa: E402
from movegen_py import legal_moves, legal_moves_with_pass      # noqa: E402
from obs_encoder import encode_x_no_action                     # noqa: E402

DEFAULT_ASSETS = os.path.join(HERE, 'assets')
POSITIONS = ('landlord', 'landlord_up', 'landlord_down')
PASS_ID = 620


class PerfectDouPolicy(object):
    """一个进程里只建一个实例；内部对 onnx session 做懒加载并复用。"""

    def __init__(self, asset_dir=None, intra_op_threads=1):
        self.asset_dir = asset_dir or DEFAULT_ASSETS
        if not os.path.isdir(self.asset_dir):
            raise ValueError('assets 目录不存在：%s' % self.asset_dir)
        self.enc = ActionEncoder(self.asset_dir)
        self._sess = {}
        self._lock = threading.Lock()
        self._threads = intra_op_threads

    # ------------------------------------------------------------------ 内部
    def _session(self, position):
        if position not in self._sess:
            import onnxruntime as ort
            opt = ort.SessionOptions()
            opt.intra_op_num_threads = self._threads
            opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            path = os.path.join(self.asset_dir, position + '.onnx')
            if not os.path.exists(path):
                raise ValueError('缺模型：%s' % path)
            self._sess[position] = ort.InferenceSession(path, opt)
        return self._sess[position]

    @staticmethod
    def last_from_seq(seq):
        """seq 里最后一个非空动作；全 pass 或空序列时返回 []。"""
        for move in reversed(seq or []):
            if move:
                return list(move)
        return []

    # ------------------------------------------------------------------ 对外
    def act(self, position, hand, base=None, seq=None, last_move=None,
            other=None, top_k=5, played=None, last_nonpass=None, bomb=None):
        """返回建议出牌与候选列表。

        position      landlord / landlord_up / landlord_down
        hand          自己手牌（整数牌值）
        base          三张底牌（看不到就给 []）
        seq           出牌历史
        last_move     场上待压的牌；None 时从 seq 末尾推
        other         别人手牌；盲牌保持 None/[]（推荐，与对拍口径一致）
        top_k         额外返回前 top_k 名候选
        played        可选：{landlord: [...], landlord_down: [...], landlord_up: [...]}
                      三家各自已出的牌。给定时不再靠 seq 轮转反推归属（读屏拿不到
                      pass 时刻，seq 缺 pass 会让 pid 整段错位）。
        last_nonpass  可选：最近一手实际出牌；不给就取 last_move
        bomb          可选：已出炸弹数（含王炸）；不给就按 seq 推

        返回 dict：
            cards      建议出的牌，[] = pass
            id         621 抽象动作 id
            idx        在 legal 列表里的下标
            n_legal    本次合法动作数
            score      该动作的 logit（越大越偏好，不是胜率）
            top        [{cards, id, idx, score}, ...] 按分值降序
        """
        if position not in POSITIONS:
            position = 'landlord'

        hand = [int(c) for c in (hand or [])]
        if not hand:
            raise ValueError('hand 为空')
        base = [int(c) for c in (base or [])]
        seq = [[int(c) for c in m] for m in (seq or [])]
        other = [int(c) for c in (other or [])]

        if last_move is None:
            last_move = self.last_from_seq(seq)
        else:
            last_move = [int(c) for c in last_move]

        legal = (legal_moves_with_pass(hand, last_move) if last_move
                 else legal_moves(hand, last_move))
        legal = [list(m) for m in legal][:150]
        if not legal:
            raise ValueError('legal 为空：手牌与待压牌不自洽？')

        meta = {'position': position, 'hand': hand, 'other': other,
                'base': base, 'seq': seq}
        if played:
            meta['played'] = {k: [int(c) for c in (v or [])] for k, v in played.items()}
            if last_nonpass:
                meta['last_nonpass'] = [int(c) for c in last_nonpass]
            if bomb is not None:
                meta['bomb'] = int(bomb)
        obs = np.asarray(encode_x_no_action(meta), dtype=np.float32).ravel()
        flat = np.asarray(self.enc.flat(hand, legal, other=other),
                          dtype=np.float32).ravel()
        x = np.concatenate([obs, flat]).reshape(1, obs.size + flat.size)

        with self._lock:
            sess = self._session(position)
            logit = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]

        n = len(legal)
        # 模型输出的是 621 个"固定抽象动作"的分值，不是按本次 legal 的顺序排的。
        # 原写法 scores = logit[:n] 按顺序取前 n 个，等于把动作 id 0..n-1 的分
        # 硬套给 legal[0..n-1]，整体错位：模型把某个动作 mask 成 -inf（不可行）时，
        # 错位后会把这个 -inf 摊给一堆本来可行的动作，表现为"候选全 -3.4e38"，
        # argmax 退化成按索引取第一个 —— 实测让 PerfectDou 的胜率掉到 3%。
        # 正确做法是按每个动作自己的抽象 id 去取分。
        ids = []
        for mv in legal:
            aid = PASS_ID if not mv else self.enc.action_id(mv)
            ids.append(PASS_ID if aid is None else int(aid))
        scores = np.asarray([logit[i] if 0 <= i < logit.size else -np.inf
                             for i in ids], dtype=np.float64)
        # 模型对绝大多数抽象动作输出 float32 的 -FLT_MAX（硬 mask），个别位置还会
        # 溢出成 +inf。+inf 不是"极推荐"，是数值异常，直接当不可选处理；否则它会
        # 压过所有有限值，导致 PD 反复去出四带二这类动作。
        scores[~np.isfinite(scores)] = -np.inf
        # 同一抽象 id 可能对应多种具体牌（如三带一/三带二共用一个 id），这时分值相同，
        # 保持 legal 的既有顺序即可，不额外打散。
        order = np.argsort(-scores)
        idx = int(order[0])
        if not np.isfinite(scores[idx]):
            # 全部被 mask（罕见）：退而出 pass；连 pass 都没有才取第一个候选。
            for i, mv in enumerate(legal):
                if not mv:
                    idx = i
                    break
            else:
                idx = 0
        play = legal[idx]
        aid = ids[idx]

        top = []
        for i in order[:max(1, int(top_k))]:
            i = int(i)
            mv = legal[i]
            top.append({
                'cards': list(mv),
                'id': ids[i],
                'idx': i,
                'score': float(scores[i]),
            })

        return {'cards': list(play), 'id': aid, 'idx': idx, 'n_legal': n,
                'score': float(scores[idx]), 'top': top}

    def act_cards(self, position, hand, **kw):
        """只要建议出的牌。"""
        return self.act(position, hand, **kw)['cards']


# ====================================================================== 自测
def _main():
    import time

    policy = PerfectDouPolicy()

    hand_landlord = [3, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10,
                     11, 12, 13, 14, 14, 17, 17, 20, 30]
    t0 = time.perf_counter()
    out = policy.act('landlord', hand_landlord, base=[17, 20, 30], seq=[], last_move=[])
    t1 = time.perf_counter()
    print('地主 20 张自由出牌 -> %s  id=%s  legal=%d  %.1f ms'
          % (out['cards'], out['id'], out['n_legal'], (t1 - t0) * 1000))
    print('  前 3 名：', [(t['cards'], round(t['score'], 3)) for t in out['top'][:3]])

    hand_farmer = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 17, 17, 20, 30]
    t0 = time.perf_counter()
    out2 = policy.act('landlord_up', hand_farmer, base=[3, 3, 3], seq=[[3, 3]],
                      last_move=[3, 3])
    t1 = time.perf_counter()
    print('农民 17 张压 33 -> %s  id=%s  legal=%d  %.1f ms'
          % (out2['cards'], out2['id'], out2['n_legal'], (t1 - t0) * 1000))

    # 第二次调用走热路径（session 已就绪）
    t0 = time.perf_counter()
    policy.act('landlord', hand_landlord, base=[17, 20, 30], seq=[], last_move=[])
    t1 = time.perf_counter()
    print('热路径复算：%.1f ms' % ((t1 - t0) * 1000))


if __name__ == '__main__':
    _main()
