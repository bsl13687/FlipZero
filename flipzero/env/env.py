import numpy as np

from flipzero.env.game import (
    PieceType, ActionType, Action, GameEnv, ALL_PIECE_TYPES,
    BOARD_HEIGHT, BOARD_WIDTH, INITIAL_SCORE, MAX_DRAW_STEP,
    is_red, is_black,
)
from flipzero.env.move_generator import MoveGenerator

BOARD_SIZE = BOARD_HEIGHT * BOARD_WIDTH

RED_PIECE_ORDER = [
    PieceType.RED_KING, PieceType.RED_MANDARIN, PieceType.RED_ELEPHANT,
    PieceType.RED_KNIGHT, PieceType.RED_ROOK, PieceType.RED_CANNON, PieceType.RED_SOLDIER,
]
BLACK_PIECE_ORDER = [
    PieceType.BLACK_KING, PieceType.BLACK_MANDARIN, PieceType.BLACK_ELEPHANT,
    PieceType.BLACK_KNIGHT, PieceType.BLACK_ROOK, PieceType.BLACK_CANNON, PieceType.BLACK_SOLDIER,
]
# 保持向后兼容的旧名字（红方视角）
OWN_PIECE_ORDER = RED_PIECE_ORDER
OPP_PIECE_ORDER = BLACK_PIECE_ORDER

# 各棋子类型的初始数量（用于信念特征归一化）
INITIAL_COUNTS = {t: ALL_PIECE_TYPES.count(t) for t in RED_PIECE_ORDER + BLACK_PIECE_ORDER}

NUM_BOARD_CHANNELS = 16
BOARD_FEATURE_DIM = NUM_BOARD_CHANNELS * BOARD_SIZE
SCALAR_FEATURE_DIM = 4
# 信念特征：剩余未翻开棋子中，红/黑各 7 种类型的归一化计数。
REMAINING_COUNT_DIM = 14
X_NO_ACTION_DIM = BOARD_FEATURE_DIM + SCALAR_FEATURE_DIM + REMAINING_COUNT_DIM
ACTION_DIM = 3 + BOARD_SIZE + BOARD_SIZE
Z_DIM = ACTION_DIM
HISTORY_LENGTH = 10


def _pos_index(r, c):
    return r * BOARD_WIDTH + c


def board2array(game_env, player, perfect_info=False):
    board = game_env.board
    true_board = game_env.true_board
    features = np.zeros(BOARD_FEATURE_DIM, dtype=np.float32)
    # 按当前玩家选择己方/对方棋子通道顺序，保证两个视角的编码对称
    if player == 0:
        own_order, opp_order = RED_PIECE_ORDER, BLACK_PIECE_ORDER
    else:
        own_order, opp_order = BLACK_PIECE_ORDER, RED_PIECE_ORDER

    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            pos = _pos_index(r, c)
            piece = board[r][c]
            true_piece = true_board[r][c]

            if piece == PieceType.EMPTY:
                features[15 * BOARD_SIZE + pos] = 1.0
                continue

            if piece == PieceType.UNFLIPPED:
                features[14 * BOARD_SIZE + pos] = 1.0
                if perfect_info:
                    src = true_piece
                    is_own = (player == 0 and is_red(src)) or (player == 1 and is_black(src))
                    if is_own:
                        ch = own_order.index(src) if src in own_order else -1
                        if ch >= 0:
                            features[ch * BOARD_SIZE + pos] = 1.0
                    else:
                        ch = opp_order.index(src) if src in opp_order else -1
                        if ch >= 0:
                            features[(7 + ch) * BOARD_SIZE + pos] = 1.0
                continue

            is_own = (player == 0 and is_red(piece)) or (player == 1 and is_black(piece))
            if is_own:
                ch = own_order.index(piece) if piece in own_order else -1
                if ch >= 0:
                    features[ch * BOARD_SIZE + pos] = 1.0
            else:
                ch = opp_order.index(piece) if piece in opp_order else -1
                if ch >= 0:
                    features[(7 + ch) * BOARD_SIZE + pos] = 1.0

    return features


def scalar_features(game_env, player):
    own_score = game_env.red_score if player == 0 else game_env.black_score
    opp_score = game_env.black_score if player == 0 else game_env.red_score
    return np.array([
        own_score / INITIAL_SCORE,
        opp_score / INITIAL_SCORE,
        game_env.peace_steps / MAX_DRAW_STEP,
        len(game_env.unflipped_positions) / 28.0,
    ], dtype=np.float32)


def belief_features(game_env):
    """剩余未翻开棋子的类型计数（信念特征，14 维）。

    暗棋中玩家能精确得知"还有哪些类型的棋子尚未翻开"：总数固定，已翻开的可见，
    被吃掉的在吃子时会亮明身份。因此未翻开位置上棋子的多集对玩家是可观测的，
    不确定的只是"哪个类型藏在哪个格子"。这里遍历 unflipped_positions 取 true_board
    得到的正是这个诚实的多集（只泄露各类型的剩余数量，不泄露每个格子的具体身份）。
    """
    counts = {t: 0 for t in RED_PIECE_ORDER + BLACK_PIECE_ORDER}
    for r, c in game_env.unflipped_positions:
        t = game_env.true_board[r][c]
        if t in counts:
            counts[t] += 1
    feat = np.zeros(REMAINING_COUNT_DIM, dtype=np.float32)
    for i, t in enumerate(RED_PIECE_ORDER):
        feat[i] = counts[t] / INITIAL_COUNTS[t]
    for i, t in enumerate(BLACK_PIECE_ORDER):
        feat[7 + i] = counts[t] / INITIAL_COUNTS[t]
    return feat


def action2array(action):
    features = np.zeros(ACTION_DIM, dtype=np.float32)
    features[action.action_type] = 1.0
    if action.from_row >= 0 and action.from_col >= 0:
        features[3 + _pos_index(action.from_row, action.from_col)] = 1.0
    features[3 + BOARD_SIZE + _pos_index(action.to_row, action.to_col)] = 1.0
    return features


class InfoSet:
    def __init__(self, player, game_env, action_history, perfect_info=False):
        self.player = player
        self.legal_actions = game_env.get_legal_actions()
        self.x_no_action = np.concatenate([
            board2array(game_env, player, perfect_info),
            scalar_features(game_env, player),
            belief_features(game_env),
        ])
        # 预 stack 成 [N, ACTION_DIM]，避免每次推理都重做 np.stack。
        # 没有合法动作时给一个空数组（保持二维形状以便下游索引/shape 假设一致）。
        if self.legal_actions:
            self.action_encodings = np.stack(
                [action2array(a) for a in self.legal_actions], axis=0
            )
        else:
            self.action_encodings = np.zeros((0, ACTION_DIM), dtype=np.float32)
        self.z = np.zeros((HISTORY_LENGTH, Z_DIM), dtype=np.float32)
        history = action_history[-HISTORY_LENGTH:]
        for i, a in enumerate(history):
            self.z[HISTORY_LENGTH - len(history) + i] = action2array(a)


class Env:
    def __init__(self, perfect_info=False):
        self.game = GameEnv()
        self.perfect_info = perfect_info
        self.action_history = []

    def reset(self):
        self.game.reset()
        self.action_history = []
        return self._get_infoset()

    def step(self, action):
        self.action_history.append(action)
        self.game.step(action)
        return self._get_infoset()

    def _get_infoset(self):
        return InfoSet(
            player=self.game.current_player,
            game_env=self.game,
            action_history=self.action_history,
            perfect_info=self.perfect_info,
        )

    def get_legal_actions(self):
        return self.game.get_legal_actions()

    @property
    def done(self):
        return self.game.done

    @property
    def winner(self):
        return self.game.winner

    @property
    def current_player(self):
        return self.game.current_player
