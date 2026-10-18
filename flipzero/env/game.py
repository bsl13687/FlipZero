import random
from enum import IntEnum


class PieceType(IntEnum):
    EMPTY = 0
    UNFLIPPED = 1
    RED_KING = 2
    RED_MANDARIN = 3
    RED_ELEPHANT = 4
    RED_KNIGHT = 5
    RED_ROOK = 6
    RED_CANNON = 7
    RED_SOLDIER = 8
    BLACK_KING = 9
    BLACK_MANDARIN = 10
    BLACK_ELEPHANT = 11
    BLACK_KNIGHT = 12
    BLACK_ROOK = 13
    BLACK_CANNON = 14
    BLACK_SOLDIER = 15


class ActionType(IntEnum):
    FLIP = 0
    MOVE = 1
    ATTACK = 2


RED_PIECES = {PieceType.RED_KING, PieceType.RED_MANDARIN, PieceType.RED_ELEPHANT,
              PieceType.RED_KNIGHT, PieceType.RED_ROOK, PieceType.RED_CANNON, PieceType.RED_SOLDIER}

BLACK_PIECES = {PieceType.BLACK_KING, PieceType.BLACK_MANDARIN, PieceType.BLACK_ELEPHANT,
                PieceType.BLACK_KNIGHT, PieceType.BLACK_ROOK, PieceType.BLACK_CANNON, PieceType.BLACK_SOLDIER}

PIECE_SCORES = {
    PieceType.RED_KING: 30, PieceType.RED_MANDARIN: 10, PieceType.RED_ELEPHANT: 5,
    PieceType.RED_KNIGHT: 5, PieceType.RED_ROOK: 5, PieceType.RED_CANNON: 5, PieceType.RED_SOLDIER: 1,
    PieceType.BLACK_KING: 30, PieceType.BLACK_MANDARIN: 10, PieceType.BLACK_ELEPHANT: 5,
    PieceType.BLACK_KNIGHT: 5, PieceType.BLACK_ROOK: 5, PieceType.BLACK_CANNON: 5, PieceType.BLACK_SOLDIER: 1,
}

PIECE_RANK = {
    PieceType.RED_KING: 7, PieceType.RED_MANDARIN: 6, PieceType.RED_ELEPHANT: 5,
    PieceType.RED_KNIGHT: 4, PieceType.RED_ROOK: 3, PieceType.RED_CANNON: 2, PieceType.RED_SOLDIER: 1,
    PieceType.BLACK_KING: 7, PieceType.BLACK_MANDARIN: 6, PieceType.BLACK_ELEPHANT: 5,
    PieceType.BLACK_KNIGHT: 4, PieceType.BLACK_ROOK: 3, PieceType.BLACK_CANNON: 2, PieceType.BLACK_SOLDIER: 1,
}

ALL_PIECE_TYPES = [
    PieceType.RED_KING,
    PieceType.RED_MANDARIN, PieceType.RED_MANDARIN,
    PieceType.RED_ELEPHANT, PieceType.RED_ELEPHANT,
    PieceType.RED_KNIGHT, PieceType.RED_KNIGHT,
    PieceType.RED_ROOK, PieceType.RED_ROOK,
    PieceType.RED_CANNON, PieceType.RED_CANNON,
    PieceType.RED_SOLDIER, PieceType.RED_SOLDIER, PieceType.RED_SOLDIER, PieceType.RED_SOLDIER, PieceType.RED_SOLDIER,
    PieceType.BLACK_KING,
    PieceType.BLACK_MANDARIN, PieceType.BLACK_MANDARIN,
    PieceType.BLACK_ELEPHANT, PieceType.BLACK_ELEPHANT,
    PieceType.BLACK_KNIGHT, PieceType.BLACK_KNIGHT,
    PieceType.BLACK_ROOK, PieceType.BLACK_ROOK,
    PieceType.BLACK_CANNON, PieceType.BLACK_CANNON,
    PieceType.BLACK_SOLDIER, PieceType.BLACK_SOLDIER, PieceType.BLACK_SOLDIER, PieceType.BLACK_SOLDIER, PieceType.BLACK_SOLDIER,
]

BOARD_HEIGHT = 8
BOARD_WIDTH = 4
INITIAL_SCORE = 60
MAX_DRAW_STEP = 20

FLIPPED_POSITIONS = [(1, 1), (3, 0), (4, 3), (6, 2)]


class Action:
    __slots__ = ('action_type', 'from_row', 'from_col', 'to_row', 'to_col')

    def __init__(self, action_type, from_row=-1, from_col=-1, to_row=-1, to_col=-1):
        self.action_type = action_type
        self.from_row = from_row
        self.from_col = from_col
        self.to_row = to_row
        self.to_col = to_col

    def __eq__(self, other):
        return (self.action_type == other.action_type and self.from_row == other.from_row
                and self.from_col == other.from_col and self.to_row == other.to_row
                and self.to_col == other.to_col)

    def __hash__(self):
        return hash((self.action_type, self.from_row, self.from_col, self.to_row, self.to_col))

    def __repr__(self):
        if self.action_type == ActionType.FLIP:
            return f"FLIP({self.to_row},{self.to_col})"
        elif self.action_type == ActionType.MOVE:
            return f"MOVE({self.from_row},{self.from_col})->({self.to_row},{self.to_col})"
        else:
            return f"ATTACK({self.from_row},{self.from_col})->({self.to_row},{self.to_col})"


def is_red(piece_type):
    return piece_type in RED_PIECES


def is_black(piece_type):
    return piece_type in BLACK_PIECES


def same_side(p1, p2):
    return (is_red(p1) and is_red(p2)) or (is_black(p1) and is_black(p2))


def can_capture(attacker, defender):
    if same_side(attacker, defender):
        return False
    if attacker in (PieceType.RED_CANNON, PieceType.BLACK_CANNON):
        return True
    attacker_rank = PIECE_RANK[attacker]
    defender_rank = PIECE_RANK[defender]
    if attacker_rank == 1 and defender_rank == 7:
        return True
    if attacker_rank == 7 and defender_rank == 1:
        return False
    return attacker_rank >= defender_rank


def _is_valid_cannon_attack(board, fr, fc, tr, tc):
    """炮攻击校验：from 与 to 必须同行或同列，且中间恰好有一个非空棋子作为炮架，目标本身非空。"""
    if fr != tr and fc != tc:
        return False
    if fr == tr and fc == tc:
        return False
    if board[tr][tc] == PieceType.EMPTY:
        return False
    dr = 0 if fr == tr else (1 if tr > fr else -1)
    dc = 0 if fc == tc else (1 if tc > fc else -1)
    r, c = fr + dr, fc + dc
    pivots = 0
    while (r, c) != (tr, tc):
        if board[r][c] != PieceType.EMPTY:
            pivots += 1
            if pivots > 1:
                return False
        r += dr
        c += dc
    return pivots == 1


class GameEnv:
    def __init__(self):
        self.board = None
        self.true_board = None
        self.current_player = 0
        self.red_score = INITIAL_SCORE
        self.black_score = INITIAL_SCORE
        self.peace_steps = 0
        self.unflipped_positions = set()
        self.done = False
        self.winner = -1
        self.reset()

    def reset(self):
        pieces = ALL_PIECE_TYPES[:]
        random.shuffle(pieces)
        self.board = [[PieceType.EMPTY for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.true_board = [[PieceType.EMPTY for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.unflipped_positions = set()
        self.current_player = 0
        self.red_score = INITIAL_SCORE
        self.black_score = INITIAL_SCORE
        self.peace_steps = 0
        self.done = False
        self.winner = -1

        idx = 0
        flipped_set = set(FLIPPED_POSITIONS)
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                self.true_board[r][c] = pieces[idx]
                if (r, c) in flipped_set:
                    self.board[r][c] = pieces[idx]
                else:
                    self.board[r][c] = PieceType.UNFLIPPED
                    self.unflipped_positions.add((r, c))
                idx += 1

        return self._get_obs()

    def _get_obs(self):
        return {
            'board': self.board,
            'true_board': self.true_board,
            'current_player': self.current_player,
            'red_score': self.red_score,
            'black_score': self.black_score,
            'peace_steps': self.peace_steps,
            'done': self.done,
            'winner': self.winner,
            'unflipped_count': len(self.unflipped_positions),
        }

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True

        # 防御性校验：炮不能自由移动到空格，也不能相邻攻击
        if action.action_type in (ActionType.MOVE, ActionType.ATTACK):
            mover = self.board[action.from_row][action.from_col]
            if mover in (PieceType.RED_CANNON, PieceType.BLACK_CANNON):
                if action.action_type == ActionType.MOVE:
                    raise ValueError("Cannon cannot perform MOVE action; it can only ATTACK with a screen piece")
                # 攻击：必须不是相邻，且 from-to 之间恰好有一个非空棋子作为炮架
                if not _is_valid_cannon_attack(self.board, action.from_row, action.from_col,
                                               action.to_row, action.to_col):
                    raise ValueError("Invalid cannon attack: requires exactly one screen piece between from and to")

        captured = False
        if action.action_type == ActionType.FLIP:
            #翻开棋子后重置和棋步数
            r, c = action.to_row, action.to_col
            self.board[r][c] = self.true_board[r][c]
            self.unflipped_positions.discard((r, c))
            captured = True
        elif action.action_type == ActionType.MOVE:
            piece = self.board[action.from_row][action.from_col]
            self.board[action.to_row][action.to_col] = piece
            self.board[action.from_row][action.from_col] = PieceType.EMPTY
            self.true_board[action.to_row][action.to_col] = self.true_board[action.from_row][action.from_col]
            self.true_board[action.from_row][action.from_col] = PieceType.EMPTY
        elif action.action_type == ActionType.ATTACK:
            target_board = self.board[action.to_row][action.to_col]
            attacker = self.board[action.from_row][action.from_col]
            if target_board == PieceType.UNFLIPPED:
                target_true = self.true_board[action.to_row][action.to_col]
                self.unflipped_positions.discard((action.to_row, action.to_col))
            else:
                target_true = target_board
            score_loss = PIECE_SCORES.get(target_true, 0)
            if is_red(target_true):
                self.red_score -= score_loss
            else:
                self.black_score -= score_loss
            self.board[action.to_row][action.to_col] = attacker
            self.board[action.from_row][action.from_col] = PieceType.EMPTY
            self.true_board[action.to_row][action.to_col] = self.true_board[action.from_row][action.from_col]
            self.true_board[action.from_row][action.from_col] = PieceType.EMPTY
            captured = True

        if captured:
            self.peace_steps = 0
        else:
            self.peace_steps += 1

        self.current_player = 1 - self.current_player
        self._check_winner()
        reward = self._get_reward()
        return self._get_obs(), reward, self.done

    def _check_winner(self):
        if self.red_score <= 0:
            self.done = True
            self.winner = 1
            return
        if self.black_score <= 0:
            self.done = True
            self.winner = 0
            return
        if self.peace_steps > MAX_DRAW_STEP:
            self.done = True
            self.winner = -2
            return
        if not self._has_legal_actions(self.current_player):
            self.done = True
            self.winner = 1 - self.current_player
            return

    def _has_legal_actions(self, player):
        from flipzero.env.move_generator import MoveGenerator
        return len(MoveGenerator.get_legal_actions(self.board, player, self.unflipped_positions)) > 0

    def _get_reward(self):
        if not self.done:
            return 0
        if self.winner == -2:
            return 0
        if self.winner == 0:
            return 1
        return -1

    def get_legal_actions(self):
        from flipzero.env.move_generator import MoveGenerator
        return MoveGenerator.get_legal_actions(self.board, self.current_player, self.unflipped_positions)

    def deepcopy(self):
        new_env = GameEnv.__new__(GameEnv)
        new_env.board = [row[:] for row in self.board]
        new_env.true_board = [row[:] for row in self.true_board]
        new_env.current_player = self.current_player
        new_env.red_score = self.red_score
        new_env.black_score = self.black_score
        new_env.peace_steps = self.peace_steps
        new_env.unflipped_positions = set(self.unflipped_positions)
        new_env.done = self.done
        new_env.winner = self.winner
        return new_env
