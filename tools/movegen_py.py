# -*- coding: utf-8 -*-
"""PerfectDou 官方合法动作枚举 —— 纯 Python 复刻

逐行照抄官方 perfectdou/env/move_generator.py + move_selector.py + game.py 的
get_legal_card_play_actions，保持**完全相同的生成顺序**（模型的动作段按这个顺序排）。
实测：与 .so 金标准 600/600 样本的合法动作集合与顺序完全一致。

用法：
    from movegen_py import legal_moves, legal_moves_with_pass, get_move_type
    acts = legal_moves(hand=[3, 3, 5, ...], last_move=[...])   # 不含 pass
    acts = legal_moves_with_pass(hand, last_move)              # last_move 非空时末尾追加 []
"""
import collections
import itertools

MIN_SINGLE_CARDS = 5
MIN_PAIRS = 3
MIN_TRIPLES = 2

TYPE_0_PASS = 0
TYPE_1_SINGLE = 1
TYPE_2_PAIR = 2
TYPE_3_TRIPLE = 3
TYPE_4_BOMB = 4
TYPE_5_KING_BOMB = 5
TYPE_6_3_1 = 6
TYPE_7_3_2 = 7
TYPE_8_SERIAL_SINGLE = 8
TYPE_9_SERIAL_PAIR = 9
TYPE_10_SERIAL_TRIPLE = 10
TYPE_11_SERIAL_3_1 = 11
TYPE_12_SERIAL_3_2 = 12
TYPE_13_4_2 = 13
TYPE_14_4_22 = 14
TYPE_15_WRONG = 15


def select(cards, num):
    return [list(i) for i in itertools.combinations(cards, num)]


def is_continuous_seq(move):
    for i in range(len(move) - 1):
        if move[i + 1] - move[i] != 1:
            return False
    return True


def get_move_type(move):
    move_size = len(move)
    move_dict = collections.Counter(move)
    if move_size == 0:
        return {'type': TYPE_0_PASS}
    if move_size == 1:
        return {'type': TYPE_1_SINGLE, 'rank': move[0]}
    if move_size == 2:
        if move[0] == move[1]:
            return {'type': TYPE_2_PAIR, 'rank': move[0]}
        elif move == [20, 30]:
            return {'type': TYPE_5_KING_BOMB}
        return {'type': TYPE_15_WRONG}
    if move_size == 3:
        if len(move_dict) == 1:
            return {'type': TYPE_3_TRIPLE, 'rank': move[0]}
        return {'type': TYPE_15_WRONG}
    if move_size == 4:
        if len(move_dict) == 1:
            return {'type': TYPE_4_BOMB, 'rank': move[0]}
        elif len(move_dict) == 2:
            if move[0] == move[1] == move[2] or move[1] == move[2] == move[3]:
                return {'type': TYPE_6_3_1, 'rank': move[1]}
            return {'type': TYPE_15_WRONG}
        return {'type': TYPE_15_WRONG}
    if is_continuous_seq(move):
        return {'type': TYPE_8_SERIAL_SINGLE, 'rank': move[0], 'len': len(move)}
    if move_size == 5:
        if len(move_dict) == 2:
            return {'type': TYPE_7_3_2, 'rank': move[2]}
        return {'type': TYPE_15_WRONG}

    count_dict = collections.defaultdict(int)
    for c, n in move_dict.items():
        count_dict[n] += 1

    if move_size == 6:
        if (len(move_dict) == 2 or len(move_dict) == 3) and count_dict.get(4) == 1 and \
                (count_dict.get(2) == 1 or count_dict.get(1) == 2):
            return {'type': TYPE_13_4_2, 'rank': move[2]}
    if move_size == 8 and (((len(move_dict) == 3 or len(move_dict) == 2) and
            (count_dict.get(4) == 1 and count_dict.get(2) == 2)) or count_dict.get(4) == 2):
        return {'type': TYPE_14_4_22, 'rank': max([c for c, n in move_dict.items() if n == 4])}

    mdkeys = sorted(move_dict.keys())
    if len(move_dict) == count_dict.get(2) and is_continuous_seq(mdkeys):
        return {'type': TYPE_9_SERIAL_PAIR, 'rank': mdkeys[0], 'len': len(mdkeys)}
    if len(move_dict) == count_dict.get(3) and is_continuous_seq(mdkeys):
        return {'type': TYPE_10_SERIAL_TRIPLE, 'rank': mdkeys[0], 'len': len(mdkeys)}

    if count_dict.get(3, 0) >= MIN_TRIPLES:
        serial_3, single, pair = list(), list(), list()
        for k, v in move_dict.items():
            if v == 3:
                serial_3.append(k)
            elif v == 1:
                single.append(k)
            elif v == 2:
                pair.append(k)
            else:
                return {'type': TYPE_15_WRONG}
        serial_3.sort()
        if is_continuous_seq(serial_3):
            if len(serial_3) == len(single) + len(pair) * 2:
                return {'type': TYPE_11_SERIAL_3_1, 'rank': serial_3[0], 'len': len(serial_3)}
            if len(serial_3) == len(pair) and len(move_dict) == len(serial_3) * 2:
                return {'type': TYPE_12_SERIAL_3_2, 'rank': serial_3[0], 'len': len(serial_3)}
        if len(serial_3) == 4:
            if is_continuous_seq(serial_3[1:]):
                return {'type': TYPE_11_SERIAL_3_1, 'rank': serial_3[1], 'len': len(serial_3) - 1}
            if is_continuous_seq(serial_3[:-1]):
                return {'type': TYPE_11_SERIAL_3_1, 'rank': serial_3[0], 'len': len(serial_3) - 1}
    return {'type': TYPE_15_WRONG}


class MovesGener(object):
    """与官方 MovesGener 同名同序：初始化时按 单/对/三/炸/王炸 预生成。"""

    def __init__(self, cards_list):
        self.cards_list = list(cards_list)
        self.cards_dict = collections.defaultdict(int)
        for i in self.cards_list:
            self.cards_dict[i] += 1

        self.single_card_moves = []
        self.gen_type_1_single()
        self.pair_moves = []
        self.gen_type_2_pair()
        self.triple_cards_moves = []
        self.gen_type_3_triple()
        self.bomb_moves = []
        self.gen_type_4_bomb()
        self.final_bomb_moves = []
        self.gen_type_5_king_bomb()

    def _gen_serial_moves(self, cards, min_serial, repeat=1, repeat_num=0):
        if repeat_num < min_serial:
            repeat_num = 0
        single_cards = sorted(list(set(cards)))
        seq_records = list()
        moves = list()
        start = i = 0
        longest = 1
        while i < len(single_cards):
            if i + 1 < len(single_cards) and single_cards[i + 1] - single_cards[i] == 1:
                longest += 1
                i += 1
            else:
                seq_records.append((start, longest))
                i += 1
                start = i
                longest = 1
        for seq in seq_records:
            if seq[1] < min_serial:
                continue
            start, longest = seq[0], seq[1]
            longest_list = single_cards[start: start + longest]
            if repeat_num == 0:
                steps = min_serial
                while steps <= longest:
                    index = 0
                    while steps + index <= longest:
                        target_moves = sorted(longest_list[index: index + steps] * repeat)
                        moves.append(target_moves)
                        index += 1
                    steps += 1
            else:
                if longest < repeat_num:
                    continue
                index = 0
                while index + repeat_num <= longest:
                    target_moves = sorted(longest_list[index: index + repeat_num] * repeat)
                    moves.append(target_moves)
                    index += 1
        return moves

    def gen_type_1_single(self):
        self.single_card_moves = []
        for i in set(self.cards_list):
            self.single_card_moves.append([i])
        return self.single_card_moves

    def gen_type_2_pair(self):
        self.pair_moves = []
        for k, v in self.cards_dict.items():
            if v >= 2:
                self.pair_moves.append([k, k])
        return self.pair_moves

    def gen_type_3_triple(self):
        self.triple_cards_moves = []
        for k, v in self.cards_dict.items():
            if v >= 3:
                self.triple_cards_moves.append([k, k, k])
        return self.triple_cards_moves

    def gen_type_4_bomb(self):
        self.bomb_moves = []
        for k, v in self.cards_dict.items():
            if v == 4:
                self.bomb_moves.append([k, k, k, k])
        return self.bomb_moves

    def gen_type_5_king_bomb(self):
        self.final_bomb_moves = []
        if 20 in self.cards_list and 30 in self.cards_list:
            self.final_bomb_moves.append([20, 30])
        return self.final_bomb_moves

    def gen_type_6_3_1(self):
        result = []
        for t in self.single_card_moves:
            for i in self.triple_cards_moves:
                if t[0] != i[0]:
                    result.append(t + i)
        return result

    def gen_type_7_3_2(self):
        result = list()
        for t in self.pair_moves:
            for i in self.triple_cards_moves:
                if t[0] != i[0]:
                    result.append(t + i)
        return result

    def gen_type_8_serial_single(self, repeat_num=0):
        return self._gen_serial_moves(self.cards_list, MIN_SINGLE_CARDS, repeat=1, repeat_num=repeat_num)

    def gen_type_9_serial_pair(self, repeat_num=0):
        single_pairs = list()
        for k, v in self.cards_dict.items():
            if v >= 2:
                single_pairs.append(k)
        return self._gen_serial_moves(single_pairs, MIN_PAIRS, repeat=2, repeat_num=repeat_num)

    def gen_type_10_serial_triple(self, repeat_num=0):
        single_triples = list()
        for k, v in self.cards_dict.items():
            if v >= 3:
                single_triples.append(k)
        return self._gen_serial_moves(single_triples, MIN_TRIPLES, repeat=3, repeat_num=repeat_num)

    def gen_type_11_serial_3_1(self, repeat_num=0):
        serial_3_moves = self.gen_type_10_serial_triple(repeat_num=repeat_num)
        serial_3_1_moves = list()
        for s3 in serial_3_moves:
            s3_set = set(s3)
            new_cards = [i for i in self.cards_list if i not in s3_set]
            subcards = select(new_cards, len(s3_set))
            for i in subcards:
                serial_3_1_moves.append(s3 + i)
        return list(k for k, _ in itertools.groupby(serial_3_1_moves))

    def gen_type_12_serial_3_2(self, repeat_num=0):
        serial_3_moves = self.gen_type_10_serial_triple(repeat_num=repeat_num)
        serial_3_2_moves = list()
        pair_set = sorted([k for k, v in self.cards_dict.items() if v >= 2])
        for s3 in serial_3_moves:
            s3_set = set(s3)
            pair_candidates = [i for i in pair_set if i not in s3_set]
            subcards = select(pair_candidates, len(s3_set))
            for i in subcards:
                serial_3_2_moves.append(sorted(s3 + i * 2))
        return serial_3_2_moves

    def gen_type_13_4_2(self):
        four_cards = list()
        for k, v in self.cards_dict.items():
            if v == 4:
                four_cards.append(k)
        result = list()
        for fc in four_cards:
            cards_list = [k for k in self.cards_list if k != fc]
            subcards = select(cards_list, 2)
            for i in subcards:
                result.append([fc] * 4 + i)
        return list(k for k, _ in itertools.groupby(result))

    def gen_type_14_4_22(self):
        four_cards = list()
        for k, v in self.cards_dict.items():
            if v == 4:
                four_cards.append(k)
        result = list()
        for fc in four_cards:
            cards_list = [k for k, v in self.cards_dict.items() if k != fc and v >= 2]
            subcards = select(cards_list, 2)
            for i in subcards:
                result.append([fc] * 4 + [i[0], i[0], i[1], i[1]])
        return result

    def gen_moves(self):
        moves = []
        moves.extend(self.gen_type_1_single())
        moves.extend(self.gen_type_2_pair())
        moves.extend(self.gen_type_3_triple())
        moves.extend(self.gen_type_4_bomb())
        moves.extend(self.gen_type_5_king_bomb())
        moves.extend(self.gen_type_6_3_1())
        moves.extend(self.gen_type_7_3_2())
        moves.extend(self.gen_type_8_serial_single())
        moves.extend(self.gen_type_9_serial_pair())
        moves.extend(self.gen_type_10_serial_triple())
        moves.extend(self.gen_type_11_serial_3_1())
        moves.extend(self.gen_type_12_serial_3_2())
        moves.extend(self.gen_type_13_4_2())
        moves.extend(self.gen_type_14_4_22())
        return moves


def _common_handle(moves, rival_move):
    return [move for move in moves if move[0] > rival_move[0]]


def _filter_by_rank(moves, rival_move, idx):
    rival_move.sort()
    rival_rank = rival_move[idx]
    out = []
    for move in moves:
        move.sort()
        if move[idx] > rival_rank:
            out.append(move)
    return out


def _filter_by_max_trio(moves, rival_move):
    rival = collections.Counter(rival_move)
    rival_rank = max([k for k, v in rival.items() if v == 3])
    out = []
    for move in moves:
        mymove = collections.Counter(move)
        my_rank = max([k for k, v in mymove.items() if v == 3])
        if my_rank > rival_rank:
            out.append(move)
    return out


def _filter_4_22(moves, rival_move):
    rival = collections.Counter(rival_move)
    rival_rank = 0
    for k, v in rival.items():
        if v == 4:
            rival_rank = k
    out = []
    for move in moves:
        mymove = collections.Counter(move)
        my_rank = 0
        for k, v in mymove.items():
            if v == 4:
                my_rank = k
        if my_rank > rival_rank:
            out.append(move)
    return out


def legal_moves(hand, last_move):
    """不包含 pass。last_move 为空 = 自由出牌，返回全部牌型。"""
    mg = MovesGener(list(hand))
    rival_move = list(last_move) if last_move else []
    rival = get_move_type(rival_move)
    rtype = rival['type']
    rlen = rival.get('len', 1)

    if rtype == TYPE_0_PASS:
        moves = mg.gen_moves()
    elif rtype == TYPE_1_SINGLE:
        moves = _common_handle(mg.gen_type_1_single(), rival_move)
    elif rtype == TYPE_2_PAIR:
        moves = _common_handle(mg.gen_type_2_pair(), rival_move)
    elif rtype == TYPE_3_TRIPLE:
        moves = _common_handle(mg.gen_type_3_triple(), rival_move)
    elif rtype == TYPE_4_BOMB:
        moves = _common_handle(mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb(), rival_move)
    elif rtype == TYPE_5_KING_BOMB:
        moves = []
    elif rtype == TYPE_6_3_1:
        moves = _filter_by_rank(mg.gen_type_6_3_1(), rival_move, 1)
    elif rtype == TYPE_7_3_2:
        moves = _filter_by_rank(mg.gen_type_7_3_2(), rival_move, 2)
    elif rtype == TYPE_8_SERIAL_SINGLE:
        moves = _common_handle(mg.gen_type_8_serial_single(repeat_num=rlen), rival_move)
    elif rtype == TYPE_9_SERIAL_PAIR:
        moves = _common_handle(mg.gen_type_9_serial_pair(repeat_num=rlen), rival_move)
    elif rtype == TYPE_10_SERIAL_TRIPLE:
        moves = _common_handle(mg.gen_type_10_serial_triple(repeat_num=rlen), rival_move)
    elif rtype == TYPE_11_SERIAL_3_1:
        moves = _filter_by_max_trio(mg.gen_type_11_serial_3_1(repeat_num=rlen), rival_move)
    elif rtype == TYPE_12_SERIAL_3_2:
        moves = _filter_by_max_trio(mg.gen_type_12_serial_3_2(repeat_num=rlen), rival_move)
    elif rtype == TYPE_13_4_2:
        moves = _filter_by_rank(mg.gen_type_13_4_2(), rival_move, 2)
    elif rtype == TYPE_14_4_22:
        moves = _filter_4_22(mg.gen_type_14_4_22(), rival_move)
    else:
        moves = []

    if rtype not in (TYPE_0_PASS, TYPE_4_BOMB, TYPE_5_KING_BOMB):
        moves = moves + mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()

    for m in moves:
        m.sort()
    return moves


def legal_moves_with_pass(hand, last_move):
    moves = legal_moves(hand, last_move)
    if last_move:
        moves = moves + [[]]
    return moves
