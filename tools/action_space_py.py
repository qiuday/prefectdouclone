# -*- coding: utf-8 -*-
"""PerfectDou 621 抽象动作空间 —— 纯 Python 实现（不依赖 torch / rlcard / .so）

数据来源是官方仓库里的两张表（不是算法反推）：
    action_space.json   621 个动作模板名，value 就是动作 id（0..620）
    specific_map.json  27472 个具体牌面 -> 模板名，用于把任意具体出牌归到模板

只要这两张表在手，动作 id 的求法与 .so 内部一致，无需任何二进制。
配套的 card_type.json（27471 条，具体牌面 -> [牌型名, 牌型内序号]）可用于另做校验。

用法：
    from action_space_py import ActionSpace
    asp = ActionSpace(asset_dir=r"...\\pd_docker\\PerfectDou")
    asp.abstract_id([3, 3, 3, 4])          # -> 41
    asp.abstract_id([])                    # -> 620 (pass)
    asp.decode(41, legal_actions, hand)    # -> [3, 3, 3, 4]

排布顺序（与 build 顺序严格一致，已用 JSON 校验）：
    0    .. 14    solo            15
    15   .. 27    pair            13
    28   .. 40    trio            13
    41   .. 222   trio + solo     182 = 13 * 14
    223  .. 378   trio + pair     156 = 13 * 12
    379  .. 414   straight        36
    415  .. 466   pair chain      52
    467  .. 511   plane           45
    512  .. 549   plane + solo    38
    550  .. 579   plane + pair    30
    580  .. 592   four + two      13
    593  .. 605   four + 2 pairs  13
    606  .. 618   bomb            13
    619           rocket
    620           pass
"""
import json
import os

RANK_ORDER = list(range(3, 15)) + [17, 20, 30]   # 3..A,2,小王,大王

CHAR_OF = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
           10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A',
           17: '2', 20: 'B', 30: 'R'}
IDX_OF = {v: i for i, v in enumerate(RANK_ORDER)}

SEGMENT = [
    ('solo', 0, 15), ('pair', 15, 13), ('trio', 28, 13),
    ('trio_solo', 41, 182), ('trio_pair', 223, 156),
    ('straight', 379, 36), ('pair_chain', 415, 52),
    ('plane', 467, 45), ('plane_solo', 512, 38), ('plane_pair', 550, 30),
    ('four_two', 580, 13), ('four_two_pair', 593, 13),
    ('bomb', 606, 13), ('rocket', 619, 1), ('pass', 620, 1),
]


def key_of(cards):
    """具体牌面（DouZero 数值序列）-> 升序牌面字符串，如 [4,3,3,3] -> '3334'。"""
    cnt = {}
    for c in cards:
        cnt[int(c)] = cnt.get(int(c), 0) + 1
    return ''.join(CHAR_OF[v] * cnt.get(v, 0) for v in RANK_ORDER)


def segment_of(action_id):
    for name, base, size in SEGMENT:
        if base <= action_id < base + size:
            return name, action_id - base
    return None, -1


def chain_offset(base, length, min_len):
    """链式动作在段内的起点偏移：长度为 L 的链从 sum(13-k) 开始。"""
    return base + sum(13 - L for L in range(min_len, length))


class ActionSpace(object):
    def __init__(self, asset_dir, replicate_so_quirks=False):
        self.asset_dir = asset_dir
        self.space = self._load('action_space.json')
        self.specific = self._load('specific_map.json')
        self.name_to_id = self.space
        self.id_to_name = {v: k for k, v in self.space.items()}
        # .so 对 8/10 张的带翅膀动作大量不填（见 docs/obs_structure_v2.md）。
        # 打开这个开关后，本模块会同样返回 None，交由上层留空行。
        self.replicate_so_quirks = replicate_so_quirks

    def _load(self, name):
        path = os.path.join(self.asset_dir, name)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ---------- 具体 -> 抽象 ----------
    def abstract_id(self, cards):
        cards = [int(c) for c in cards]
        if not cards:
            return 620
        if self.replicate_so_quirks and len(cards) in (8, 10):
            kind = self._guess_kind(cards)
            if kind in ('plane_solo', 'plane_pair', 'four_two_pair'):
                return None
        k = key_of(cards)
        tpl = self.specific.get(k)
        if tpl:
            for name in tpl:
                if name in self.space:
                    return self.space[name]
        return self._rule_id(cards)

    def _guess_kind(self, cards):
        cnt = {}
        for c in cards:
            cnt[int(c)] = cnt.get(int(c), 0) + 1
        quad = [v for v, n in cnt.items() if n == 4]
        if len(cards) == 8 and quad and len([v for v, n in cnt.items() if n == 2]) == 2:
            return 'four_two_pair'
        tri = sorted(v for v, n in cnt.items() if n >= 3 and v <= 14)
        if len(cards) == 8 and tri:
            return 'plane_solo'
        if len(cards) == 10 and tri:
            return 'plane_pair'
        return None

    def _rule_id(self, cards):
        """兜底：按牌型规则算 id（用于表里查不到的边角情况）。"""
        cnt = {}
        for c in cards:
            cnt[int(c)] = cnt.get(int(c), 0) + 1
        n = len(cards)
        ranks = sorted(cnt, key=lambda v: IDX_OF[v])
        idxs = [IDX_OF[v] for v in ranks]
        # 单/对/三/炸/火箭
        if n == 2 and 20 in cnt and 30 in cnt:
            return 619
        if n == 1:
            return 0 + idxs[0]
        if n == 2 and len(cnt) == 1:
            return 15 + idxs[0] if idxs[0] < 13 else None
        if n == 3 and len(cnt) == 1:
            return 28 + idxs[0] if idxs[0] < 13 else None
        if n == 4 and len(cnt) == 1:
            return 606 + idxs[0] if idxs[0] < 13 else None
        # 三带一
        tri = [v for v, c in cnt.items() if c == 3]
        if n == 4 and len(tri) == 1:
            t = tri[0]
            if t <= 14:
                kick = [v for v in cnt if v != t][0]
                pos = sum(1 for v in RANK_ORDER if v != t and IDX_OF[v] < IDX_OF[kick])
                return 41 + IDX_OF[t] * 14 + pos
        # 三带二
        if n == 5 and len(tri) == 1:
            t = tri[0]
            pairs = [v for v, c in cnt.items() if c == 2]
            if t <= 14 and len(pairs) == 1 and pairs[0] < 17:
                p = pairs[0]
                pos = sum(1 for v in RANK_ORDER[:13] if v != t and IDX_OF[v] < IDX_OF[p])
                return 223 + IDX_OF[t] * 12 + pos
        # 四带二 / 四带两对
        quad = [v for v, c in cnt.items() if c == 4]
        if quad:
            q = quad[0]
            if q < 13:
                if n == 6:
                    return 580 + q
                if n == 8 and len([v for v, c in cnt.items() if c == 2]) == 2:
                    return 593 + q
        # 链式：顺子 / 连对 / 飞机（含翅膀）
        run = self._longest_run(cnt, need=1)
        if run and (3 * len(run) == n or 4 * len(run) == n or 5 * len(run) == n):
            start, ln = run
            return chain_offset(467, ln, 2) + start
        run2 = self._longest_run(cnt, need=2)
        if run2 and 2 * len(run2) == n:
            start, ln = run2
            return chain_offset(415, ln, 3) + start
        run1 = self._longest_run(cnt, need=1)
        if run1 and len(run1) == n:
            start, ln = run1
            return chain_offset(379, ln, 5) + start
        return None

    @staticmethod
    def _longest_run(cnt, need):
        best = None
        cur = []
        for v in range(3, 15):
            if cnt.get(v, 0) >= need:
                cur.append(v)
                if len(cur) >= 2 and (best is None or len(cur) > len(best[1])):
                    best = (IDX_OF[cur[0]], list(cur))
            else:
                cur = []
        if not best:
            return None
        start_idx, chained = best
        return start_idx, len(chained)

    # ---------- 抽象 -> 具体 ----------
    def decode(self, action_id, legal_actions, hand=None):
        """把抽象动作还原成一手具体出牌。

        legal_actions: 当前合法动作（每项是 DouZero 数值列表）
        hand:          当前手牌（用于 kicker 选择，可为 None）
        """
        if action_id is None:
            return None
        if action_id == 620:
            return []
        name = self.id_to_name.get(action_id)
        if name is None:
            return None
        main, wings = name.split('*')[0], name.count('*')
        main_cards = []
        i = 0
        while i < len(main):
            ch = main[i]
            run = 1
            while i + run < len(main) and main[i + run] == ch:
                run += 1
            value = {c: v for v, c in CHAR_OF.items()}[ch]
            main_cards += [value] * run
            i += run
        cands = []
        for move in legal_actions:
            move = [int(c) for c in move]
            if len(move) != len(main_cards) + wings:
                continue
            k = key_of(move)
            if not k.startswith(key_of(main_cards)):
                # 主牌必须整体包含在出牌里（顺序无关，比计数）
                pass
            cnt = {}
            for c in move:
                cnt[c] = cnt.get(c, 0) + 1
            need = {}
            for c in main_cards:
                need[c] = need.get(c, 0) + 1
            if all(cnt.get(v, 0) >= n for v, n in need.items()):
                cands.append(move)
        if not cands:
            return main_cards
        if not wings:
            for m in cands:
                if len(m) == len(main_cards):
                    return m
            return cands[0]
        # 带翅膀：选最"便宜"的一组（牌值和小者优先），与官方 Algorithm 2 近似
        def cost(move):
            extra = list(move)
            for c in main_cards:
                extra.remove(c)
            return (sum(extra), len(extra))
        cands.sort(key=cost)
        return cands[0]


if __name__ == '__main__':
    import sys
    asset = sys.argv[1] if len(sys.argv) > 1 else '.'
    asp = ActionSpace(asset)
    print('动作数:', len(asp.space))
    for cards in ([3], [3, 3], [3, 3, 3], [3, 3, 3, 4], [3, 5, 5, 5],
                  [3, 4, 5, 6, 7], [3, 3, 4, 4, 5, 5], [3, 3, 3, 4, 4, 4],
                  [10, 10, 10, 10, 3, 4], [20, 30], []):
        print('%-24s -> %s' % (cards, asp.abstract_id(cards)))
