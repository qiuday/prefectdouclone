# -*- coding: utf-8 -*-
"""端到端自检：真实局面 -> 4688 + 28050 -> 32738 -> onnx -> 621 -> argmax 动作。

不依赖 .so、不依赖 torch，只需要 numpy + onnxruntime 和 assets/ 目录：

    python tools/onnx_selftest.py

两个用例覆盖两种视角：
  1) 地主 20 张、自由出牌（legal 不含 pass）
  2) 上家农民 17 张、压一手对子（legal 含 pass 与炸弹）
任一断言失败即以非零码退出，可直接当发版闸门。
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, 'assets')
sys.path.insert(0, HERE)

from action_encoder import ActionEncoder                      # noqa: E402
from movegen_py import legal_moves, legal_moves_with_pass      # noqa: E402
from obs_encoder import encode_x_no_action                     # noqa: E402

REQUIRED = ['action_space.json', 'specific_map.json', 'card_type.json',
            'action_values.json', 'landlord.onnx', 'landlord_up.onnx',
            'landlord_down.onnx']

OBS_DIM = 4688
ACT_DIM = 150 * 187
TOTAL_DIM = OBS_DIM + ACT_DIM


def check_assets():
    missing = [n for n in REQUIRED if not os.path.exists(os.path.join(ASSETS, n))]
    if missing:
        raise SystemExit('assets 缺少文件：%s\n应放在：%s' % (', '.join(missing), ASSETS))
    print('assets 齐全（%d 个文件）：%s' % (len(REQUIRED), ASSETS))


def run_case(tag, position, hand, base, seq, last_move, use_pass):
    enc = ActionEncoder(ASSETS)

    t0 = time.perf_counter()
    legal = (legal_moves_with_pass(hand, last_move) if use_pass
             else legal_moves(hand, last_move))
    meta = {'position': position, 'hand': hand, 'other': [], 'base': base, 'seq': seq}
    obs = np.asarray(encode_x_no_action(meta), dtype=np.float32).ravel()
    flat = np.asarray(enc.flat(hand, legal, other=[]), dtype=np.float32).ravel()
    t_enc = (time.perf_counter() - t0) * 1000

    assert len(legal) <= 150, 'legal 数 %d 超过 150 行上限' % len(legal)
    assert obs.size == OBS_DIM, 'obs 维度 %d != %d' % (obs.size, OBS_DIM)
    assert flat.size == ACT_DIM, '动作段维度 %d != %d' % (flat.size, ACT_DIM)

    x = np.concatenate([obs, flat]).reshape(1, TOTAL_DIM)
    assert x.shape == (1, TOTAL_DIM), '拼接维度错：%s' % (x.shape,)

    t1 = time.perf_counter()
    sess = onnx_session(position)
    logit = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    t_inf = (time.perf_counter() - t1) * 1000

    assert logit.shape == (1, 621), '输出形状 %s != (1, 621)' % (logit.shape,)
    row = logit[0]
    idx = int(np.argmax(row[:len(legal)]))
    assert 0 <= idx < len(legal), 'argmax %d 越界（legal %d）' % (idx, len(legal))

    play = legal[idx]
    aid = enc.action_id(play) if play else 620
    print('[%s] %s 手牌 %d 张，legal %d 个（含 pass=%s）'
          % (tag, position, len(hand), len(legal), use_pass))
    print('      编码 %d + %d = %d  ✓   %.1f ms' % (obs.size, flat.size, x.size, t_enc))
    print('      推理 -> 621 ✓   %.1f ms' % t_inf)
    print('      选中 legal[%d] = %s   动作 id = %s' % (idx, play, aid))
    return {'idx': idx, 'play': play, 'aid': aid, 'n_legal': len(legal)}


_SESS = {}


def onnx_session(position):
    if position not in _SESS:
        import onnxruntime as ort
        path = os.path.join(ASSETS, position + '.onnx')
        _SESS[position] = ort.InferenceSession(path)
    return _SESS[position]


def main():
    check_assets()

    # 用例 1：地主 20 张（含底牌），开局自由出牌
    hand_landlord = [3, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10,
                     11, 12, 13, 14, 14, 17, 17, 20, 30]
    base = [17, 20, 30]
    r1 = run_case('1', 'landlord', hand_landlord, base, [], [], use_pass=False)

    # 用例 2：上家农民 17 张，要压一手对子 33
    hand_farmer = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 17, 17, 20, 30]
    seq = [[3, 3]]
    r2 = run_case('2', 'landlord_up', hand_farmer, [3, 3, 3], seq, [3, 3], use_pass=True)

    assert r1['aid'] is not None, '地主用例的动作 id 落空（specific_map 查不到）'
    assert r2['aid'] is not None, '农民用例的动作 id 落空（specific_map 查不到）'

    print('全部通过：2 / 2')


if __name__ == '__main__':
    main()
