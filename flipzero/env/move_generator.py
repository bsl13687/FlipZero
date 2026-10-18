from flipzero.env.game import (
    PieceType, ActionType, Action, BOARD_HEIGHT, BOARD_WIDTH,
    is_red, is_black, same_side, can_capture,
)

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class MoveGenerator:
    @staticmethod
    def get_legal_actions(board, current_player, unflipped_positions):
        actions = []
        has_unflipped = len(unflipped_positions) > 0
        _add_flip_actions(board, unflipped_positions, actions)
        _add_move_and_attack_actions(board, current_player, has_unflipped, actions)
        return actions


def _add_flip_actions(board, unflipped_positions, actions):
    for r, c in unflipped_positions:
        if board[r][c] == PieceType.UNFLIPPED:
            actions.append(Action(ActionType.FLIP, to_row=r, to_col=c))


def _add_move_and_attack_actions(board, current_player, has_unflipped, actions):
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            p = board[r][c]
            if p == PieceType.EMPTY or p == PieceType.UNFLIPPED:
                continue
            is_player_piece = (current_player == 0 and is_red(p)) or (current_player == 1 and is_black(p))
            if not is_player_piece:
                continue
            is_cannon = p in (PieceType.RED_CANNON, PieceType.BLACK_CANNON)
            if is_cannon:
                # 炮不能自由移动到空格，也不能相邻攻击；只能通过隔一个棋子的方式攻击对方
                _add_cannon_attack_actions(board, r, c, p, actions)
            else:
                _add_adjacent_actions(board, r, c, p, actions)


def _add_adjacent_actions(board, r, c, p, actions):
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < BOARD_HEIGHT and 0 <= nc < BOARD_WIDTH):
            continue
        target = board[nr][nc]
        if target == PieceType.EMPTY:
            actions.append(Action(ActionType.MOVE, r, c, nr, nc))
        elif target != PieceType.UNFLIPPED and not same_side(p, target):
            if can_capture(p, target):
                actions.append(Action(ActionType.ATTACK, r, c, nr, nc))


def _add_cannon_attack_actions(board, r, c, p, actions):
    for dr, dc in DIRECTIONS:
        pivot = False
        nr, nc = r + dr, c + dc
        while 0 <= nr < BOARD_HEIGHT and 0 <= nc < BOARD_WIDTH:
            if board[nr][nc] != PieceType.EMPTY:
                if not pivot:
                    pivot = True
                else:
                    target = board[nr][nc]
                    if target == PieceType.UNFLIPPED or not same_side(p, target):
                        actions.append(Action(ActionType.ATTACK, r, c, nr, nc))
                    break
            nr += dr
            nc += dc
