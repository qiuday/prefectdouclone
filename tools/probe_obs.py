"""obs 结构差分探针：定位 4688 维里每一段被哪个局面字段驱动。

必须在 PerfectDou 仓库根目录运行（.so 以相对路径加载 libCalculateLeftHands.so，
并且要能读到 card_type.json）：

    cd /path/to/PerfectDou
    python /path/to/probe_obs.py

只做只读的 encode 调用，不跑对局、不改仓库文件。
"""
import numpy as np
from perfectdou.env.game import InfoSet
from perfectdou.env.encode import encode_obs_landlord

ORDER = ["landlord", "landlord_down", "landlord_up"]
START = {"landlord": 20, "landlord_up": 17, "landlord_down": 17}
BLOCK = 60      # 一个 4x15 矩阵
FRAME = 180     # 一帧 = 3 个矩阵

HAND = [3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30]
LEGALS = [[3], [3, 3], [3, 3, 3], [4], [5], [6], [7], [8], [9], [10],
          [11], [12], [13], [14], [17], [20], [30], []]


def build(hand=None, base=None, seq=None, lm=None, nleft=None, played=None, bomb=0):
    hand = list(HAND if hand is None else hand)
    seq = [list(s) for s in (seq or [])]
    info = InfoSet("landlord")
    info.player_hand_cards = hand
    info.legal_actions = [list(a) for a in LEGALS]
    info.last_move = list(lm or [])
    info.last_two_moves = []
    info.three_landlord_cards = list(base if base is not None else [5, 12, 17])
    info.card_play_action_seq = seq
    info.other_hand_cards = []
    info.all_handcards = {p: (hand if p == "landlord" else []) for p in ORDER}
    info.last_pid = "landlord"
    info.bomb_num = bomb

    if played is None:
        played = {p: [] for p in ORDER}
        for idx, mv in enumerate(seq):
            played[ORDER[idx % 3]].extend(mv)
    info.played_cards = played
    info.num_cards_left_dict = nleft or {p: START[p] - len(played[p]) for p in ORDER}
    lmd = {p: [] for p in ORDER}
    for idx, mv in enumerate(seq):
        if mv:
            lmd[ORDER[idx % 3]] = mv
    info.last_move_dict = lmd
    return info


def vec(**kw):
    obs = encode_obs_landlord(build(**kw))
    return np.asarray(obs["x_no_action"], dtype=np.float32).flatten()


def blocks(changed):
    """把变化维度按帧/块压缩成可读字符串。"""
    if not len(changed):
        return "无变化"
    out = []
    for f in sorted(set(i // FRAME for i in changed)):
        sub = [i for i in changed if i // FRAME == f]
        items = []
        for b in sorted(set(i // BLOCK for i in sub)):
            offs = sorted(i - b * BLOCK for i in sub if i // BLOCK == b)
            items.append("块%d+%s" % (b, offs[:8]))
        out.append("帧%d[%s]" % (f, " ".join(items)))
    return " ".join(out)


def main():
    base = vec()
    nz = np.nonzero(base)[0]
    print("基线非零 %d 维，按帧分布：" % len(nz))
    for f in sorted(set(i // FRAME for i in nz)):
        sub = [i - f * FRAME for i in nz if i // FRAME == f]
        print("  帧%-3d 帧内偏移 %s" % (f, sub))
    print()

    cases = [
        ("换一张手牌 3->4", dict(hand=[4 if c == 3 else c for c in HAND])),
        ("手牌 3 换成 2", dict(hand=[17 if c == 3 else c for c in HAND])),
        ("删一张手牌", dict(hand=HAND[1:])),
        ("底牌 5->6", dict(base=[6, 12, 17])),
        ("底牌 12->13", dict(base=[5, 13, 17])),
        ("自己剩余 20->14", dict(nleft={"landlord": 14, "landlord_up": 17, "landlord_down": 17})),
        ("上家剩余 17->14", dict(nleft={"landlord": 20, "landlord_up": 14, "landlord_down": 17})),
        ("下家剩余 17->14", dict(nleft={"landlord": 20, "landlord_up": 17, "landlord_down": 14})),
        ("自己已出 333", dict(played={"landlord": [3, 3, 3], "landlord_up": [], "landlord_down": []})),
        ("上家已出 K", dict(played={"landlord": [], "landlord_up": [13], "landlord_down": []})),
        ("下家已出 X", dict(played={"landlord": [], "landlord_up": [], "landlord_down": [20]})),
        ("历史 1 手(我 3334)", dict(seq=[[3, 3, 3, 4]])),
        ("历史 1 手(上家 5)", dict(seq=[[3, 3, 3, 4], [], [5]])),
        ("last_move=5", dict(lm=[5])),
        ("bomb_num=1", dict(bomb=1)),
    ]

    for name, kw in cases:
        diff = np.nonzero(vec(**kw) != base)[0]
        print("%-20s %3d 维 | %s" % (name, len(diff), blocks(diff)))


if __name__ == "__main__":
    main()
