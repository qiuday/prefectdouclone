"""导出金标准（.so 版本的 obs），供纯 Python 复写版逐位对拍。

必须在 PerfectDou 仓库根目录运行（.so 以相对路径加载 libCalculateLeftHands.so，
并且要能读到 card_type.json）：

    cd /path/to/PerfectDou
    python /path/to/dump_golden.py --games 50 --max-per-game 6 --out golden.npz

产出 golden.npz：
    x    (N, 4688)       金标准状态向量
    act  (N, 150, 187)   金标准动作表
    meta (N,)            JSON 字符串，含 position/hand/legal/base/seq，可重建局面

两个必须注意的点（已在代码里处理）：
1. env.game_infoset 是缓存对象，step() 用的就是它；encode 前必须重新赋值同步，
   否则会拿过期 legal 出牌，报 ValueError: list.remove(x): x not in list。
2. encode 会改动传入的 infoset，必须 copy.deepcopy，否则污染对局状态。
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.getcwd())   # 必须在 PerfectDou 仓库根目录运行

import numpy as np
from perfectdou.env.game import GameEnv
from perfectdou.env.encode import encode_obs_landlord, encode_obs_peasant
from perfectdou.evaluation.random_agent import RandomAgent

DECK = [c for c in range(3, 15) for _ in range(4)] + [17] * 4 + [20, 30]
ENCODE = {"landlord": encode_obs_landlord,
          "landlord_up": encode_obs_peasant,
          "landlord_down": encode_obs_peasant}


def gen_deal(rng):
    d = DECK.copy()
    rng.shuffle(d)
    return {
        "landlord": sorted(d[:20]),
        "landlord_up": sorted(d[20:37]),
        "landlord_down": sorted(d[37:54]),
        "three_landlord_cards": sorted(d[17:20]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--max-per-game", type=int, default=6)
    ap.add_argument("--out", default="golden.npz")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    players = {"landlord": RandomAgent(), "landlord_up": RandomAgent(),
               "landlord_down": RandomAgent()}
    env = GameEnv(players)

    xs, acts, metas = [], [], []
    for _ in range(args.games):
        env.card_play_init(gen_deal(rng))
        n = 0
        while not env.game_over and n < args.max_per_game:
            env.game_infoset = env.get_infoset()      # 与当前局面同步
            pos = env.acting_player_position
            info = copy.deepcopy(env.game_infoset)    # 避免 encode 污染对局状态
            obs = ENCODE[pos](info)

            xs.append(np.asarray(obs["x_no_action"], dtype=np.float32).flatten())
            acts.append(np.asarray(obs["legal_actions_arr"], dtype=np.float32))
            metas.append(json.dumps({
                "position": pos,
                "hand": [int(c) for c in info.player_hand_cards],
                "legal": [[int(c) for c in a] for a in info.legal_actions],
                "base": [int(c) for c in info.three_landlord_cards],
                "seq": [[int(c) for c in a] for a in info.card_play_action_seq],
                "actions_len": len(obs["actions"]),
                "actions_head": [list(a) for a in obs["actions"][:5]],
            }, ensure_ascii=False))
            n += 1
            env.step()
        env.reset()

    np.savez_compressed(args.out, x=np.stack(xs), act=np.stack(acts),
                        meta=np.array(metas, dtype=object))
    print("saved %d samples -> %s" % (len(xs), args.out))
    print("x shape=%s act shape=%s" % (np.stack(xs).shape, np.stack(acts).shape))


if __name__ == "__main__":
    main()
