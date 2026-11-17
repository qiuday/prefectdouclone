# -*- coding: utf-8 -*-
"""整链对拍：官方 .so 编码 + onnx   vs   纯 Python 编码 + onnx，逐手比动作。

**只能在 Linux + python3.7 + 官方仓库（含两个 .so）上跑**，Windows 独立程序用不到它。
它是"验收 1000 局动作一致率"的执行器，不是运行时代码。

跑法（必须在官方仓库根，encode.so 按相对路径 dlopen）：

    cd /tmp/PerfectDou
    /tmp/py37/bin/python tools/chain_check.py --repo /tmp/PerfectDou --games 50 --seed 11

依赖：onnxruntime、numpy、官方 perfectdou.env；以及同目录的 movegen_py.py、
另放的纯 Python obs 编码（vm 上是 /tmp/obs_py.py，含 encode_x_no_action / card_frame）
与 /tmp/minsteps.py。

口径：每一步把 infoset.other_hand_cards 置空（盲牌 = 真机视角），
两边各推一次 onnx，取 argmax(logit[:len(legal)]) 为动作，用 gold 的动作推进对局，
统计两个动作是否相同。
"""
import argparse
import copy
import json
import sys

import numpy as np
import onnxruntime as ort


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/tmp/PerfectDou', help='官方仓库根（含 .so 与四个 json）')
    ap.add_argument('--games', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tools', default='/tmp', help='纯 Python obs 编码所在目录（obs_py.py / minsteps.py）')
    return ap.parse_args()


A = parse_args()
sys.path.insert(0, A.repo)
sys.path.insert(0, A.tools)

from perfectdou.env.env import Env                                    # noqa: E402
from perfectdou.env.encode import encode_obs_landlord, encode_obs_peasant  # noqa: E402
from obs_py import encode_x_no_action, card_frame                     # noqa: E402
import minsteps                                                       # noqa: E402
from movegen_py import get_move_type, legal_moves                     # noqa: E402

minsteps.init_f()

CHAR = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T',
        11: 'J', 12: 'Q', 13: 'K', 14: 'A', 17: '2', 20: 'B', 30: 'R'}
with open(A.repo + '/action_space.json', 'r', encoding='utf-8') as f:
    AS = json.load(f)
with open(A.repo + '/specific_map.json', 'r', encoding='utf-8') as f:
    SM = json.load(f)

ENCODE = {'landlord': encode_obs_landlord, 'landlord_up': encode_obs_peasant,
          'landlord_down': encode_obs_peasant}
MODEL = {p: '%s/perfectdou/model/perfectdou/%s.onnx' % (A.repo, p)
         for p in ('landlord', 'landlord_up', 'landlord_down')}
SESS = {}


def sess_of(pos):
    if pos not in SESS:
        so = ort.InferenceSession(MODEL[pos])
        SESS[pos] = (so, so.get_inputs()[0].name)
    return SESS[pos]


def act_id(acts):
    key = ''.join(CHAR[c] for c in sorted(acts))
    ab = SM.get(key)
    return AS.get(ab[0]) if ab else None


def my_action_rows(hand, legal, other=None):
    """(150,187)；与 tools/action_encoder.py 同一套规则（此处不依赖 obs_encoder 包）。"""
    outside = [int(c) for c in (other or [])]
    arr = np.zeros((150, 187), dtype=np.float32)
    for i, a in enumerate(legal[:150]):
        a = list(a) if a else []
        if not a:                                                  # pass
            arr[i][0] = 620.0
            arr[i][5] = round(minsteps.min_steps(hand) / 10.0, 6)
            arr[i][6] = 1.0
            continue
        aid = act_id(a)
        if aid is None:                                            # 官方查不到 -> 整行全零
            continue
        arr[i][0] = float(aid)
        arr[i][1] = 1.0 if get_move_type(sorted(a))['type'] in (4, 5) else 0.0
        arr[i][2] = 0.0 if legal_moves(outside, sorted(a)) else 1.0
        rest = list(hand)
        for c in a:
            if c in rest:
                rest.remove(c)
        arr[i][5] = round(minsteps.min_steps(rest) / 10.0, 6)
        arr[i][6] = 1.0
        arr[i][7:] = np.array(card_frame(a), dtype=np.float32)
    return arr


def my_x(info):
    meta = {'position': info.player_position,
            'hand': [int(c) for c in info.player_hand_cards],
            'other': [],                                           # 盲牌：真机看不到别人的牌
            'base': [int(c) for c in info.three_landlord_cards],
            'seq': [[int(c) for c in m] for m in info.card_play_action_seq]}
    x = np.array(encode_x_no_action(meta), dtype=np.float32)
    rows = my_action_rows(meta['hand'], info.legal_actions, other=None)
    return np.concatenate([x, rows.ravel()])[None, :]


def gold_x(info):
    g = ENCODE[info.player_position](info)
    return np.concatenate([np.asarray(g['x_no_action'], dtype=np.float32).ravel(),
                           np.asarray(g['legal_actions_arr'], dtype=np.float32).ravel()])[None, :]


def main():
    np.random.seed(A.seed)
    env = Env('wp')
    tot = same = 0
    per_game = []
    for _ in range(A.games):
        env.reset()
        gs = gc = 0
        while True:
            pos = env._acting_player_position
            info = copy.deepcopy(env.infoset)
            info.other_hand_cards = []                             # 两边同等信息
            legal = info.legal_actions
            so, name = sess_of(pos)
            lg = so.run(None, {name: gold_x(info)})[0][0]
            lm = so.run(None, {name: my_x(info)})[0][0]
            k = len(legal)
            ig, im = int(np.argmax(lg[:k])), int(np.argmax(lm[:k]))
            tot += 1
            gs += 1
            if ig == im:
                same += 1
                gc += 1
            env.step(legal[ig])                                    # 用 gold 的动作推进
            if env._game_over:
                break
        per_game.append((gs, gc))
    print('步数 %d，动作一致 %d (%.2f%%)' % (tot, same, 100.0 * same / max(1, tot)))
    print('每局:', per_game)


if __name__ == '__main__':
    main()
