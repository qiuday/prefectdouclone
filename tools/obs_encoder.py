# -*- coding: utf-8 -*-
"""PerfectDou 观测编码器 + minSteps —— 纯 Python，不依赖 torch / rlcard / 任何 .so

与官方 encode_obs_landlord / encode_obs_peasant 的 .so 金标准对拍：
1600 个样本（三批、三种视角）中 1590 个样本的 4688 维完全 bit-exact，
其余 10 个只差 4686（炸弹数）1 位，见文件末尾"已知偏差"。

输入只需要 (position, hand, other, base, seq)，其余全部自推：
    vec = encode_x_no_action({"position": "landlord", "hand": [...], "other": [],
                               "base": [...], "seq": [[...], [], ...]})

帧布局（26 帧 × 180 + 8 标量；每帧 180 = 3 块 × 60，块 = 4 行 × 15，列序 3..A,2,B,R）
    f2   自己手牌
    f3 ~ f17   最近 15 手动作：f17 = 最新一手，往前递减，f3 = 第 15 手前（不足 15 手时前部留空）
    f18  对手手牌（两家合并；盲牌时整帧全 0）
    f19  自己已出    f20  地主局=下家 / 农民局=队友    f21  地主局=上家 / 农民局=地主
    f22  底牌
    f23  landlord_down 取 last_two[1]，其余位置取 last_two[0]
    f24  landlord_down 取 last_two[0]，其余位置取 last_two[1]
    f25  最近一手非 pass 动作（连续 pass 后不清空）
    f0 / f1 恒 0

块 A 行 k = 张数 >= k；块 B 行 k = 张数 == k；块 C：
    行 8  = 王炸标记（同时有 20 和 30 → 整行 15 位全 1）
    行 9  = 顺子区间（>=1 张的连续段，长度 >=5）
    行 10 = 连对区间（>=2 张的连续段，长度 >=3）
    行 11 = 飞机区间（>=3 张的连续段，长度 >=2）
    ⚠ 链扫描把 2 接在 A 后面参与长度计算，但 2 的列不输出。

8 个标量
    4680 / 4681 = 0
    4682 = minSteps(自己手牌) / 10            （值域 0.3 ~ 1.1）
    4683 = 地主局：上家剩余/17；农民局：队友剩余/17
    4684 = 地主局：下家剩余/17；农民局：地主剩余/20
    4685 = 地主局：自己剩余/20；农民局：自己剩余/17
    4686 = (bomb_num + 1) / 16
    4687 = 1.0 若场上没有待压的牌（自己可自由出牌），否则 0.0

已知偏差
    含"王炸 + 普通炸弹"的少数对局，官方 4686 会比我算的多 1（1600 样本里 10 个，
    集中在同一局）。其余 4687 维全部一致。
"""
RANK15 = list(range(3, 15)) + [17, 20, 30]      # 3..A,2,小王,大王
IDX = {v: i for i, v in enumerate(RANK15)}
ORDER = ["landlord", "landlord_down", "landlord_up"]
CHAIN_MIN = ((1, 5, 9), (2, 3, 10), (3, 2, 11))  # (最少张数, 最短链长, 输出行)
START = {"landlord": 20, "landlord_up": 17, "landlord_down": 17}
VAL = RANK15


# ===========================================================================
# 一、牌 -> 180 维帧
# ===========================================================================
def cnt_of(cards):
    c = {}
    for v in cards:
        v = int(v)
        c[v] = c.get(v, 0) + 1
    return c


def chain_runs(cnt, mincnt, minlen):
    """满足条件的连续牌值区间（含 2，索引 0..12）。"""
    runs, cur = [], []
    for i in range(13):
        if cnt.get(RANK15[i], 0) >= mincnt:
            cur.append(i)
        else:
            if len(cur) >= minlen:
                runs.append(list(cur))
            cur = []
    if len(cur) >= minlen:
        runs.append(list(cur))
    return runs


def card_frame(cards):
    """一组牌 -> 180 维（12 行 × 15 列）。"""
    cnt = cnt_of(cards)
    out = [0] * 180
    for k in range(1, 5):
        b = (k - 1) * 15                       # 块A 第 k 行（>= k）
        for i, v in enumerate(RANK15):
            if cnt.get(v, 0) >= k:
                out[b + i] = 1
        b = (3 + k) * 15                       # 块B 第 k 行（== k）
        for i, v in enumerate(RANK15):
            if cnt.get(v, 0) == k:
                out[b + i] = 1
    if cnt.get(20, 0) and cnt.get(30, 0):      # 王炸
        out[120:135] = [1] * 15
    for mincnt, minlen, row in CHAIN_MIN:
        for run in chain_runs(cnt, mincnt, minlen):
            for i in run:
                if i < 12:                     # 2 参与长度但不输出
                    out[row * 15 + i] = 1
    return out


# ===========================================================================
# 二、动作序列 -> 局面
# ===========================================================================
def replay(seq):
    """出牌顺序固定 landlord -> landlord_down -> landlord_up，pass 也轮转。"""
    pid = "landlord"
    played = {p: [] for p in ORDER}
    last_move = []        # 场上待压的牌，两家连续 pass 后清空
    last_nonpass = []     # 最近一手实际出牌，不清空
    last_two = []
    passes = 0
    bomb = 0
    for a in seq:
        a = [int(c) for c in a]
        last_two = ([a] + last_two)[:2]
        if a:
            last_move = a
            last_nonpass = a
            passes = 0
            played[pid].extend(a)
            if len(a) == 2 and 20 in a and 30 in a:
                bomb += 1
            elif len(a) >= 4 and len(set(a)) == 1:
                bomb += 1
        else:
            passes += 1
            if passes >= 2:
                last_move = []
        pid = ORDER[(ORDER.index(pid) + 1) % 3]
    return played, last_move, last_nonpass, last_two, bomb


def encode_x_no_action(meta):
    pos = meta["position"]
    seq = meta.get("seq") or []
    if meta.get("played"):
        # 调用方已经掌握"三家各自出了哪些牌"（手工读屏能按座位直接确定归属），
        # 就不必靠 seq 的严格轮转去反推。原因：replay 假设 landlord->down->up
        # 一手不漏地轮转，而读屏轮询会漏掉 pass 时刻，seq 里少几个 [] 就会让
        # pid 整段错位，三家的已出牌全串到别人头上 —— 编码出来的局面完全失真。
        played = {p: [int(c) for c in (meta["played"].get(p) or [])] for p in ORDER}
        last_move = [int(c) for c in (meta.get("last_move") or [])]
        last_nonpass = [int(c) for c in (meta.get("last_nonpass") or last_move)]
        last_two = [[int(c) for c in m] for m in (meta.get("last_two") or [])][:2]
        bomb = int(meta.get("bomb") or 0)
    else:
        played, last_move, last_nonpass, last_two, bomb = replay(seq)

    frames = [[0] * 180 for _ in range(26)]
    frames[2] = card_frame(meta["hand"])
    if meta.get("other"):
        frames[18] = card_frame(meta["other"])

    # f19/f20/f21 三家已出：地主局按 自己/下家/上家，农民局按 自己/队友/地主
    if pos == "landlord":
        mate = None
        frames[19] = card_frame(played["landlord"])
        frames[20] = card_frame(played["landlord_down"])
        frames[21] = card_frame(played["landlord_up"])
    else:
        mate = "landlord_down" if pos == "landlord_up" else "landlord_up"
        frames[19] = card_frame(played[pos])
        frames[20] = card_frame(played[mate])
        frames[21] = card_frame(played["landlord"])

    frames[22] = card_frame(meta.get("base") or [])

    # f3..f17 = 最近 15 手，f17 是最新一手
    for k in range(15):
        idx = len(seq) - 15 + k
        if idx >= 0:
            frames[3 + k] = card_frame(seq[idx])

    # f23/f24：landlord_down 与另外两个位置相反
    if pos == "landlord_down":
        if len(last_two) > 1:
            frames[23] = card_frame(last_two[1])
        if len(last_two) > 0:
            frames[24] = card_frame(last_two[0])
    else:
        if len(last_two) > 0:
            frames[23] = card_frame(last_two[0])
        if len(last_two) > 1:
            frames[24] = card_frame(last_two[1])
    frames[25] = card_frame(last_nonpass)

    flat = []
    for f in frames:
        flat.extend(f)

    left = {p: START[p] - len(played[p]) for p in ORDER}
    if pos == "landlord":
        s4683 = left["landlord_up"] / 17.0
        s4684 = left["landlord_down"] / 17.0
        s4685 = left["landlord"] / 20.0
    else:
        s4683 = left[mate] / 17.0
        s4684 = left["landlord"] / 20.0
        s4685 = left[pos] / 17.0

    flat.extend([
        0.0, 0.0,                              # 4680, 4681
        min_steps(meta["hand"]) / 10.0,        # 4682
        s4683, s4684, s4685,                   # 4683, 4684, 4685
        (bomb + 1) / 16.0,                     # 4686
        1.0 if not last_move else 0.0,         # 4687
    ])
    return flat


# ===========================================================================
# 三、minSteps：这手牌最少几手打完
#    移植 esrrhs/PerfectDouCPP 的 oracle.cpp，并修掉它对"王可当翅膀"的高估
# ===========================================================================
_F = {}
_F_READY = False


def _fget(a, b, c, d):
    if a < 0 or b < 0 or c < 0 or d < 0:
        return 100
    if a >= 22 or b >= 15 or c >= 9 or d >= 6:
        return a + b + c + d
    v = _F.get((a, b, c, d))
    return a + b + c + d if v is None else v


def _enum_loose(a, b, c, d, need):
    """从单/对/三/炸里抽 need 张散牌，抽剩的组自动降级。"""
    for s in range(0, min(need, a) + 1):
        r1 = need - s
        if r1 == 0:
            yield (a - s, b, c, d)
            continue
        for bp in range(0, min(b, r1) + 1):
            for pu in range(bp, min(2 * bp, r1) + 1):
                r2 = r1 - pu
                pl = 2 * bp - pu                 # 拆对后剩的单张数
                if r2 == 0:
                    yield (a - s + pl, b - bp, c, d)
                    continue
                for bt in range(0, min(c, r2) + 1):
                    for tu in range(bt, min(3 * bt, r2) + 1):
                        r3 = r2 - tu
                        for t2 in range(0, bt + 1):
                            for t1 in range(0, bt - t2 + 1):
                                t3 = bt - t1 - t2
                                if t1 + 2 * t2 + 3 * t3 != tu:
                                    continue
                                if r3 == 0:
                                    yield (a - s + pl + t2, b - bp + t1, c - bt, d)
                                    continue
                                lo = (r3 + 3) // 4       # 每个炸弹最多供 4 张
                                for bd in range(lo, min(d, r3) + 1):
                                    for u3 in range(0, bd + 1):
                                        for u2 in range(0, bd - u3 + 1):
                                            for u1 in range(0, bd - u2 - u3 + 1):
                                                u4 = bd - u1 - u2 - u3
                                                if u1 + 2 * u2 + 3 * u3 + 4 * u4 != r3:
                                                    continue
                                                yield (a - s + pl + t2 + u3,
                                                       b - bp + t1 + u2,
                                                       c - bt + u1, d - bd)


def _init_f():
    """F(a,b,c,d)：不含链、不含王时，a 单 / b 对 / c 三张 / d 炸弹 最少几手。"""
    global _F_READY
    if _F_READY:
        return
    _F.clear()
    _F[(0, 0, 0, 0)] = 0
    for total in range(1, 21):
        for a in range(22):
            for b in range(15):
                for c in range(9):
                    for d in range(6):
                        if a + 2 * b + 3 * c + 4 * d != total:
                            continue
                        best = a + b + c + d
                        if a >= 1:
                            best = min(best, 1 + _fget(a - 1, b, c, d))
                        if b >= 1:
                            best = min(best, 1 + _fget(a, b - 1, c, d))
                        if c >= 1:
                            best = min(best, 1 + _fget(a, b, c - 1, d))
                        if d >= 1:
                            best = min(best, 1 + _fget(a, b, c, d - 1))
                        if c >= 1:                       # 三带一
                            for t in _enum_loose(a, b, c - 1, d, 1):
                                best = min(best, 1 + _fget(*t))
                            if b >= 1:                   # 三带一对
                                best = min(best, 1 + _fget(a, b - 1, c - 1, d))
                            if c >= 2:                   # 两个三张连着出
                                best = min(best, 1 + _fget(a + 1, b, c - 2, d))
                            if d >= 1:
                                best = min(best, 1 + _fget(a, b, c - 1, d - 1))
                        if d >= 1:                       # 四带二
                            for t in _enum_loose(a, b, c, d - 1, 2):
                                best = min(best, 1 + _fget(*t))
                            if b >= 2:                   # 四带二对
                                best = min(best, 1 + _fget(a, b - 2, c, d - 1))
                            if b >= 1 and c >= 1:
                                best = min(best, 1 + _fget(a + 1, b - 1, c - 1, d - 1))
                            if c >= 2:
                                best = min(best, 1 + _fget(a + 2, b, c - 2, d - 1))
                        _F[(a, b, c, d)] = best
    _F_READY = True


def _groups(cnt):
    a = b = c = d = 0
    for r in range(13):
        t = cnt[r]
        if t == 1: a += 1
        elif t == 2: b += 1
        elif t == 3: c += 1
        elif t == 4: d += 1
    return a, b, c, d


def _lower_bound(cnt):
    _init_f()
    a, b, c, d = _groups(cnt)
    jokers = (1 if cnt[13] else 0) + (1 if cnt[14] else 0)
    if cnt[13] and cnt[14]:
        jokers = 1
    return _fget(a, b, c, d) + jokers


def _has_chain(cnt):
    m1 = m2 = m3 = c1 = c2 = c3 = 0
    for r in range(13):
        t = cnt[r]
        c1 = c1 + 1 if t >= 1 else 0; m1 = max(m1, c1)
        c2 = c2 + 1 if t >= 2 else 0; m2 = max(m2, c2)
        c3 = c3 + 1 if t >= 3 else 0; m3 = max(m3, c3)
    return m1 >= 5 or m2 >= 3 or m3 >= 2


def _enum_wing(cnt, rank, need, pick, out):
    if need == 0:
        out.append(tuple(pick))
        return
    if rank == 15:
        return
    for take in range(0, min(cnt[rank], need) + 1):
        pick[rank] = take
        _enum_wing(cnt, rank + 1, need - take, pick, out)
    pick[rank] = 0


_MEMO = {}


def _solve(cnt):
    key = tuple(cnt)
    if not any(key):
        return 0
    if key in _MEMO:
        return _MEMO[key]
    # 无链且无王时，纯张数 DP 就是精确值（王能当翅膀，不能走这条快路）
    if not _has_chain(key) and not key[13] and not key[14]:
        ans = _lower_bound(key)
        _MEMO[key] = ans
        return ans

    best = 100
    if key[13] and key[14]:                 # 王炸当一手
        rem = list(key); rem[13] = rem[14] = 0
        best = min(best, 1 + _solve(rem))
    elif key[13] or key[14]:
        rem = list(key); rem[13] = rem[14] = 0
        best = min(best, 1 + _solve(rem))

    a, b, c, d = _groups(key)
    jg = (1 if key[13] else 0) + (1 if key[14] else 0)
    rocket = 1 if (key[13] and key[14]) else 0
    if rocket:
        jg = 1
    best = min(best, _fget(a, b, c, d) + jg)
    loose = 0 if rocket else jg           # 单王可以当单张/翅膀用掉
    if loose:
        best = min(best, rocket + _fget(a + loose, b, c, d))

    if best > 1:
        def enum_chains(repeat, minlen, trios):
            nonlocal best
            i = 0
            while i < 13:
                if key[i] < repeat:
                    i += 1
                    continue
                j = i
                while j + 1 < 13 and key[j + 1] >= repeat and VAL[j + 1] - VAL[j] == 1:
                    j += 1
                runlen = j - i + 1
                for ln in range(minlen, runlen + 1):
                    for s in range(i, j - ln + 2):
                        rem = list(key)
                        for x in range(ln):
                            rem[s + x] -= repeat
                        best = min(best, 1 + _solve(rem))
                        if trios:
                            if 4 * ln <= 20:                   # 飞机带单
                                out = []
                                _enum_wing(rem, 0, ln, [0] * 15, out)
                                for w in out:
                                    after = [rem[r] - w[r] for r in range(15)]
                                    best = min(best, 1 + _solve(after))
                            if 5 * ln <= 20:                   # 飞机带对
                                pr = [r for r in range(13) if rem[r] >= 2]
                                if len(pr) >= ln:
                                    import itertools
                                    for sel in itertools.combinations(pr, ln):
                                        after = list(rem)
                                        for r in sel:
                                            after[r] -= 2
                                        best = min(best, 1 + _solve(after))
                i = j + 1
        enum_chains(1, 5, False)     # 顺子
        enum_chains(2, 3, False)     # 连对
        enum_chains(3, 2, True)      # 飞机
    _MEMO[key] = best
    return best


def min_steps(cards):
    """这手牌最少几手出完。"""
    cnt = [0] * 15
    for v in cards:
        v = int(v)
        if v == 17: cnt[12] += 1
        elif v == 20: cnt[13] += 1
        elif v == 30: cnt[14] += 1
        else: cnt[v - 3] += 1
    return _solve(cnt)


if __name__ == "__main__":
    demo = {
        "position": "landlord",
        "hand": [3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30],
        "other": [], "base": [5, 12, 17], "seq": [],
    }
    v = encode_x_no_action(demo)
    print("维度:", len(v), "非零:", sum(1 for t in v if t))
    print("minSteps(3..A+2+王炸) =", min_steps(demo["hand"]))
    h2 = [3, 4, 5, 6, 6, 6, 6, 7, 9, 9, 9, 9, 10, 11, 12, 12, 12, 17, 17, 30]
    print("minSteps(含炸弹、王当翅膀) =", min_steps(h2), "(官方值 4)")
