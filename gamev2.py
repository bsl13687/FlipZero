"""gamev2.py —— 用蒙特卡洛搜索增强的 FlipZero 对局 / 分析程序。

思路
----
`game.py` 里的 AI 是"单步贪心"：对当前局面的每个合法动作各算一次价值，取最大。
`gamev2.py` 换成**模拟对局 + 胜率估计**：

    对每个候选动作 a，反复模拟整局对局若干次（双方都由价值网络走棋），
    统计在 a 之后「我方最终获胜」的次数，得到 a 的胜率估计；
    搜索预算按 UCB1 自适应地更多地投给有希望的动作；最后选胜率最高的动作。

三个关键设计
------------
1. **不完美信息（PIMC）**：暗棋中未翻开棋子的身份对双方都不可见。模拟时不能直接
   用 `true_board`（那是作弊）。每次模拟前先从当前信息集**采样一个一致的真实世界**：
   把当前仍未翻开的棋子多重集随机打乱到这些位置上。已翻开的棋盘保持不变。
   由于模型的特征只依赖公开信息（已翻开棋子 + 剩余类型计数），策略对同一多重集的
   不同排列是**不敏感**的——变的只是翻棋时揭出的具体棋子，这正是我们要平均掉的部分。
2. **截断 + 价值网络**：完整模拟一局约 58 步（实测），1 万次模拟要 8 分钟/步。
   用 `--leaf-steps K` 可以在走 K 步后直接用价值网络的估值代替继续对局，
   把 1 万次压到约 2 分钟/步。
3. **UCB1 分配预算**：不是对每个动作平摊，而是优先把模拟次数投给有希望的动作，
   这是"蒙特卡洛搜索"相对暴力枚举的核心收益。

用法
----
    # 分析一个局面：打印每个候选动作的模拟胜率表（1 万次模拟，截断到 12 步）
    python gamev2.py --mode analyze --rollouts 10000 --leaf-steps 12 --verbose

    # 竞技场：搜索 AI 对阵原始贪心模型，下 10 局（交替先后手）
    python gamev2.py --mode arena --opponent model --games 10 --rollouts 128 --leaf-steps 10

    # pygame 图形界面：双方都可选 人类 / 贪心模型 / 搜索AI
    python gamev2.py --mode gui

    # 人机对战（命令行界面）
    python gamev2.py --mode human --rollouts 512 --leaf-steps 12

    # 最忠实但最慢：1 万次完整模拟（不截断），约 8 分钟/步
    python gamev2.py --mode analyze --rollouts 10000 --leaf-steps 0

分阶段预算
----------
默认 `--schedule auto`：前 `--early-plies`（10）步每步只跑 `--early-rollouts`（500）次模拟，
之后在 `--ramp-plies`（10）步内线性增长到 `--rollouts`（上限），此后保持上限。
理由：开局可选动作多、彼此差异小，少采样即可；随着局面收窄、单步权重变大，再加大预算。
用 `--schedule fixed` 可让每步都用固定 `--rollouts`。

注意
----
搜索的代价与「模拟次数 × 每步网络前向」成正比。启动时会打印配置，
`--verbose` 会显示实时进度与预计剩余时间。

`--mode gui` 复用 `game.py` 的绘制与菜单风格，但搜索是在**搜索循环内部**用
进度回调来刷新画面的（不是后台线程）。原因：CPython 的 GIL 会让线程版搜索
被主线程的重绘严重饿死，实测慢几十倍；回调式则只损失极小比例的时间。
"""
import argparse
import glob
import os
import random
import sys
import time

import numpy as np
import torch

from flipzero.env.game import (
    PieceType, ActionType, GameEnv, BOARD_HEIGHT, BOARD_WIDTH,
    is_red, is_black,
)
from flipzero.env.env import InfoSet
from flipzero.dmc.models import FlipZeroModel

# pygame 只在 --mode gui 时才真正导入，保证 CLI 模式不依赖图形库。
pygame = None


# --------------------------------------------------------------------------- #
# 棋盘渲染用的单字母缩写：红方大写、黑方小写
# --------------------------------------------------------------------------- #
_PIECE_LETTER = {
    PieceType.RED_KING: 'K', PieceType.RED_MANDARIN: 'M', PieceType.RED_ELEPHANT: 'E',
    PieceType.RED_KNIGHT: 'N', PieceType.RED_ROOK: 'R', PieceType.RED_CANNON: 'C',
    PieceType.RED_SOLDIER: 'S',
    PieceType.BLACK_KING: 'k', PieceType.BLACK_MANDARIN: 'm', PieceType.BLACK_ELEPHANT: 'e',
    PieceType.BLACK_KNIGHT: 'n', PieceType.BLACK_ROOK: 'r', PieceType.BLACK_CANNON: 'c',
    PieceType.BLACK_SOLDIER: 's',
}

LEGEND = '图例: 大写=红(K将 M士 E象 N马 R车 C炮 S兵) 小写=黑  ?=未翻开  .=空'


def render_board(game):
    """把棋盘渲染成等宽文本。"""
    lines = ['    ' + ''.join(f' {c}  ' for c in range(BOARD_WIDTH))]
    for r in range(BOARD_HEIGHT):
        cells = []
        for c in range(BOARD_WIDTH):
            p = game.board[r][c]
            if p == PieceType.EMPTY:
                ch = '.'
            elif p == PieceType.UNFLIPPED:
                ch = '?'
            else:
                ch = _PIECE_LETTER[p]
            cells.append(f' {ch} ')
        lines.append(f' {r}  ' + ''.join(cells))
    return '\n'.join(lines)


def side_name(player):
    return '红方' if player == 0 else '黑方'


# --------------------------------------------------------------------------- #
# 模型加载与价值评估
# --------------------------------------------------------------------------- #
def find_latest_checkpoint(dir_path='log/flipzero'):
    """返回 `log/flipzero/` 下步数最大的 checkpoint 路径。"""
    files = glob.glob(os.path.join(dir_path, 'model_step_*.tar'))
    if not files:
        return None

    def step_of(path):
        try:
            return int(os.path.basename(path).replace('model_step_', '').replace('.tar', ''))
        except ValueError:
            return -1

    return max(files, key=step_of)


def load_net(model_path, device):
    """加载 checkpoint 到 `FlipZeroModel`，兼容 {'model_state_dict': ...} 与裸 state_dict。"""
    state = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    net = FlipZeroModel()
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


class Evaluator:
    """价值网络封装：对当前信息集的所有合法动作求值（一次批前向）。"""

    def __init__(self, net, device):
        self.net = net
        self.device = device
        self.calls = 0

    def values(self, infoset):
        n = len(infoset.legal_actions)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        x = torch.from_numpy(infoset.x_no_action).to(self.device).unsqueeze(0).expand(n, -1).contiguous()
        z = torch.from_numpy(infoset.z).to(self.device).unsqueeze(0).expand(n, -1, -1).contiguous()
        a = torch.from_numpy(infoset.action_encodings).to(self.device)
        with torch.no_grad():
            v = self.net(x, a, z, return_value=True)['value']
        self.calls += 1
        return v.cpu().numpy()

    def best_action(self, infoset):
        """贪心取价值最大的动作（`game.py` 的行为，作为搜索的对照基线）。"""
        v = self.values(infoset)
        return infoset.legal_actions[int(np.argmax(v))]


# --------------------------------------------------------------------------- #
# PIMC：从一个信息集采样一致的真实世界
# --------------------------------------------------------------------------- #
def sample_world(base_game, rng, peek=False):
    """采样一个与当前公开观测一致的真实世界。

    只打乱**未翻开格子背后的棋子身份**；`board`（公开可见部分）、分数、
    和棋计数与当前行动方全部保持不变。`peek=True` 时直接返回真实棋盘
    （等价于作弊 / 完美信息），仅用于诊断对比。
    """
    world = base_game.deepcopy()
    if peek:
        return world
    poss = sorted(world.unflipped_positions)
    if not poss:
        return world
    pieces = [world.true_board[r][c] for r, c in poss]
    rng.shuffle(pieces)
    for (r, c), p in zip(poss, pieces):
        world.true_board[r][c] = p
    return world


# --------------------------------------------------------------------------- #
# 搜索结果
# --------------------------------------------------------------------------- #
class SearchResult:
    """一次搜索的结果：每个候选动作的统计量、估计胜率、被选动作。

    要点：胜/和/负只在**模拟真正跑到终局**时统计；截断模拟贡献的是连续估值，
    计入 `total_value`。因此：
      - `score`（平均估值）在两种模式下都可用，是默认排序与选择依据；
      - `win_rate`（终局纯胜率）只在 `--leaf-steps 0` 时才有完整意义。
    """

    def __init__(self, player, actions, visits, total_value, wins, draws, losses,
                 static_values, elapsed, rollouts_done, select='score'):
        self.player = player
        self.actions = actions
        self.visits = visits
        self.total_value = total_value
        self.wins = wins
        self.draws = draws
        self.losses = losses
        self.static_values = static_values
        self.elapsed = elapsed
        self.rollouts_done = rollouts_done
        self.select = select

    @property
    def mean_value(self):
        """平均估值 = 每次模拟结果的均值（终局时等价于 胜+0.5和 / 次数）。"""
        return np.where(self.visits > 0, self.total_value / np.maximum(self.visits, 1), 0.0)

    @property
    def terminal_count(self):
        return self.wins + self.draws + self.losses

    @property
    def win_rate(self):
        """终局纯胜率 = 胜 / 终局次数；无终局模拟时为 0。"""
        tc = self.terminal_count
        return np.where(tc > 0, self.wins / np.maximum(tc, 1), 0.0)

    @property
    def score(self):
        """默认排序指标：平均估值。"""
        return self.mean_value

    @property
    def chosen_index(self):
        """按 `select` 准则挑出的下标。"""
        if len(self.actions) == 1:
            return 0
        if self.rollouts_done == 0:
            return int(np.argmax(self.static_values))
        if self.select == 'winrate' and np.any(self.terminal_count > 0):
            metric = self.win_rate
        elif self.select == 'visits':
            metric = self.visits.astype(float)
        else:
            metric = self.mean_value
        return int(np.argmax(metric))

    def table(self):
        """把每个候选动作的模拟结果排成文本表（按平均估值降序）。"""
        order = np.argsort(-self.mean_value)
        rows = [f'{"#":>2}  {"动作":<32} {"模拟":>6} {"均值":>7} '
                f'{"胜":>4} {"和":>4} {"负":>4} {"终局胜率":>9} {"静态值":>8}']
        rows.append('-' * 92)
        for rank, i in enumerate(order, 1):
            tc = int(self.terminal_count[i])
            wr = f'{self.wins[i] / tc:.3f}' if tc > 0 else '   -'
            rows.append(
                f'{rank:>2}  {str(self.actions[i]):<32} {self.visits[i]:>6d} '
                f'{self.mean_value[i]:>+7.3f} {self.wins[i]:>4.0f} {self.draws[i]:>4.0f} '
                f'{self.losses[i]:>4.0f} {wr:>9} {self.static_values[i]:>+8.3f}'
            )
        return '\n'.join(rows)


# --------------------------------------------------------------------------- #
# 蒙特卡洛搜索
# --------------------------------------------------------------------------- #
class MonteCarloSearch:
    """基于模拟对局的蒙特卡洛动作选择。"""

    def __init__(self, evaluator, rollouts=1000, leaf_steps=0, c_ucb=0.7,
                 init_visits=3, select='score', seed=0, peek=False,
                 time_budget=0.0, verbose=False, label='搜索AI',
                 early_rollouts=500, early_plies=10, ramp_plies=10, schedule='auto',
                 progress_cb=None, progress_every=0):
        self.ev = evaluator
        self.rollouts = int(rollouts)
        self.leaf_steps = int(leaf_steps)
        self.c_ucb = float(c_ucb)
        self.init_visits = int(init_visits)
        self.select = select
        self.rng = random.Random(seed)
        self.peek = peek
        self.time_budget = float(time_budget)
        self.verbose = verbose
        self.label = label
        # 分阶段预算：前 early_plies 步用 early_rollouts，之后在 ramp_plies 步内
        # 线性增长到 rollouts（上限），再之后保持 rollouts。
        self.schedule = schedule
        self.early_rollouts = int(early_rollouts)
        self.early_plies = int(early_plies)
        self.ramp_plies = max(1, int(ramp_plies))
        # 进度回调：GUI 用它周期性泵事件并重绘。回调返回 False 表示中止搜索。
        self.progress_cb = progress_cb
        self.progress_every = int(progress_every)

    def budget_for_ply(self, ply):
        """第 `ply` 步（1 起算）应使用的模拟次数。"""
        if self.schedule == 'fixed' or self.early_rollouts <= 0 \
                or self.early_rollouts >= self.rollouts:
            return self.rollouts
        if ply <= self.early_plies:
            return self.early_rollouts
        t = min(1.0, (ply - self.early_plies) / self.ramp_plies)
        return int(round(self.early_rollouts + (self.rollouts - self.early_rollouts) * t))

    # ---------------- 单次模拟 ---------------- #
    def _rollout(self, world, history, root_player, first_action):
        """从 `first_action` 出发模拟。

        返回 (value, terminal)：value 是根玩家视角的结果（终局为 ±1/0，截断为连续估值），
        terminal 表示这次模拟是否真的跑到了终局。
        """
        world.step(first_action)
        hist = history + [first_action]
        depth = 0
        while not world.done:
            if self.leaf_steps and depth >= self.leaf_steps:
                return self._leaf_value(world, hist, root_player), False
            infoset = InfoSet(world.current_player, world, hist, perfect_info=False)
            if not infoset.legal_actions:
                break
            action = self.ev.best_action(infoset)
            world.step(action)
            hist.append(action)
            depth += 1
        return self._terminal_value(world, root_player), True

    def _leaf_value(self, world, hist, root_player):
        """截断点估值：价值网络对"当前行动方的所有动作"取最大，再折算到根玩家视角。"""
        infoset = InfoSet(world.current_player, world, hist, perfect_info=False)
        if not infoset.legal_actions:
            return self._terminal_value(world, root_player)
        best = float(np.max(self.ev.values(infoset)))
        return best if world.current_player == root_player else -best

    @staticmethod
    def _terminal_value(world, root_player):
        w = world.winner
        if w < 0:            # -2 和棋，或 -1 未结束
            return 0.0
        return 1.0 if w == root_player else -1.0

    # ---------------- 搜索主循环 ---------------- #
    def search(self, game, history, player=None):
        """在 `game` 当前局面上搜索。返回 `SearchResult`（含被选动作与全部统计）。"""
        if player is None:
            player = game.current_player

        root_info = InfoSet(player, game, history, perfect_info=False)
        actions = root_info.legal_actions
        n = len(actions)

        if n == 0:
            z = np.zeros(0)
            return SearchResult(player, [], np.zeros(0, dtype=np.int64), z, z, z, z,
                                z, 0.0, 0, self.select)
        static_values = self.ev.values(root_info)
        budget = self.budget_for_ply(len(history) + 1)
        if n == 1 or budget <= 0:
            z = np.zeros(n)
            return SearchResult(player, actions, np.zeros(n, dtype=np.int64), z, z, z, z,
                                static_values, 0.0, 0, self.select)

        visits = np.zeros(n, dtype=np.int64)
        total_value = np.zeros(n)   # 连续结果累加（终局 ±1/0，截断为估值）
        wins = np.zeros(n)          # 仅统计真正跑到终局的模拟
        draws = np.zeros(n)
        losses = np.zeros(n)

        t0 = time.time()
        done = 0
        progress_step = max(1, budget // 10)
        min_visits = max(1, self.init_visits)

        while done < budget:
            if self.time_budget and (time.time() - t0) >= self.time_budget:
                break

            # --- 选一个动作：先保证每个动作有最低采样，再按 UCB1 --- #
            under = np.where(visits < min_visits)[0]
            if len(under) > 0:
                idx = int(under[done % len(under)])
            else:
                mean = self._objective(total_value, wins, draws, visits)
                bonus = self.c_ucb * np.sqrt(np.log(done + 1) / np.maximum(visits, 1))
                idx = int(np.argmax(mean + bonus))

            # --- 采样一个世界并模拟 --- #
            world = sample_world(game, self.rng, peek=self.peek)
            value, terminal = self._rollout(world, list(history), player, actions[idx])

            visits[idx] += 1
            total_value[idx] += value
            if terminal:
                if value > 0.5:
                    wins[idx] += 1
                elif value < -0.5:
                    losses[idx] += 1
                else:
                    draws[idx] += 1

            done += 1
            if self.verbose and done % progress_step == 0:
                el = time.time() - t0
                eta = el / done * (budget - done)
                print(f'  [{self.label}] {done}/{budget} 次模拟  '
                      f'已用 {el:.1f}s  预计剩余 {eta:.1f}s', flush=True)
            if self.progress_cb is not None and self.progress_every > 0 \
                    and done % self.progress_every == 0:
                if self.progress_cb(done, budget) is False:
                    break

        return SearchResult(player, actions, visits, total_value, wins, draws, losses,
                            static_values, time.time() - t0, done, self.select)

    def _objective(self, total_value, wins, draws, visits):
        """UCB 的即时值。默认用平均估值；`--select winrate` 时用终局胜率。"""
        if self.select == 'winrate' and np.any((wins + draws + losses) > 0):
            tc = wins + draws + losses
            return np.where(tc > 0, wins / np.maximum(tc, 1), -np.inf)
        return np.where(visits > 0, total_value / np.maximum(visits, 1), 0.0)


# --------------------------------------------------------------------------- #
# 智能体封装（统一签名：act(game, history) -> Action）
# --------------------------------------------------------------------------- #
class ModelAgent:
    """原始贪心模型（`game.py` 的行为），作为基线。"""

    def __init__(self, evaluator):
        self.ev = evaluator
        self.name = '贪心模型'

    def act(self, game, history):
        infoset = InfoSet(game.current_player, game, history, perfect_info=False)
        if not infoset.legal_actions:
            return None
        return self.ev.best_action(infoset)


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.name = '随机'

    def act(self, game, history):
        infoset = InfoSet(game.current_player, game, history, perfect_info=False)
        if not infoset.legal_actions:
            return None
        return self.rng.choice(infoset.legal_actions)


class SearchAgent:
    """蒙特卡洛搜索智能体：每步搜索一次，按准则选动作。"""

    def __init__(self, searcher, name='搜索AI'):
        self.searcher = searcher
        self.name = name
        self.last_result = None
        self.total_rollouts = 0
        self.total_time = 0.0

    def act(self, game, history):
        result = self.searcher.search(game, history)
        self.last_result = result
        self.total_rollouts += result.rollouts_done
        self.total_time += result.elapsed
        if not result.actions:
            return None
        return result.actions[result.chosen_index]


# --------------------------------------------------------------------------- #
# 对局循环
# --------------------------------------------------------------------------- #
def play_game(agent_red, agent_black, seed=None):
    """下一整局。返回 (winner, steps)。winner: 0 红胜 / 1 黑胜 / -2 和棋。"""
    if seed is not None:
        random.seed(seed)
    game = GameEnv()
    history = []
    while not game.done:
        agent = agent_red if game.current_player == 0 else agent_black
        action = agent.act(game, list(history))
        if action is None:
            break
        game.step(action)
        history.append(action)
    return game.winner, len(history)


# --------------------------------------------------------------------------- #
# 各模式
# --------------------------------------------------------------------------- #
def _new_searcher(args, evaluator, seed, label='搜索AI', verbose=None):
    return MonteCarloSearch(
        evaluator,
        rollouts=args.rollouts,
        leaf_steps=args.leaf_steps,
        c_ucb=args.c_ucb,
        select=args.select,
        seed=seed,
        peek=args.peek,
        time_budget=args.time_budget,
        verbose=args.verbose if verbose is None else verbose,
        label=label,
        early_rollouts=args.early_rollouts,
        early_plies=args.early_plies,
        ramp_plies=args.ramp_plies,
        schedule=args.schedule,
    )


def mode_analyze(args, evaluator):
    """走到指定相位后分析该局面：打印每个候选动作的模拟胜率表。"""
    print(f'== 局面分析 (seed={args.seed}, 走到第 {args.ply} 步) ==')
    # 用贪心模型走到指定相位，得到一个有代表性的中局局面
    random.seed(args.seed)
    game = GameEnv()
    history = []
    while not game.done and len(history) < args.ply:
        infoset = InfoSet(game.current_player, game, history, perfect_info=False)
        if not infoset.legal_actions:
            break
        action = evaluator.best_action(infoset)
        game.step(action)
        history.append(action)

    print(render_board(game))
    print(LEGEND)
    print(f'当前行动方: {side_name(game.current_player)}   '
          f'分数 红 {game.red_score} : 黑 {game.black_score}   '
          f'和棋计数 {game.peace_steps}   未翻开 {len(game.unflipped_positions)}')
    print()

    searcher = _new_searcher(args, evaluator, args.seed)
    result = searcher.search(game, history)
    if not result.actions:
        print('该局面无合法动作。')
        return

    print(f'== 每个候选动作的模拟结果（{len(result.actions)} 个候选，'
          f'{result.rollouts_done} 次模拟，耗时 {result.elapsed:.1f}s）==')
    print(result.table())
    print()

    chosen = result.actions[result.chosen_index]
    best = result.chosen_index
    static_idx = int(np.argmax(result.static_values))
    static_best = result.actions[static_idx]
    print('== 结论 ==')
    print(f'  搜索选择    : {chosen}   平均估值 {result.mean_value[best]:+.3f}')
    if result.terminal_count[best] > 0:
        print(f'                 终局统计  胜 {result.wins[best]:.0f} / 和 {result.draws[best]:.0f} / '
              f'负 {result.losses[best]:.0f}   终局胜率 {result.win_rate[best]:.3f}')
    else:
        print(f'                 截断模式（{result.visits[best]} 次模拟均未跑到终局，'
              f'用价值网络估值代替结果）')
    print(f'  贪心模型选择: {static_best}   （静态值 {result.static_values[static_idx]:+.3f}）')
    print(f'  两者{"一致" if chosen == static_best else "不一致"}。')


# --------------------------------------------------------------------------- #
# 竞技场（支持多进程并行；每局任务可独立复现）
# --------------------------------------------------------------------------- #
_ARENA = {}


def _arena_cfg(args, evaluator, model_path=None, device=None):
    """把一次竞技场评测的全部参数收集成可 pickle 的字典。"""
    if model_path is None:
        model_path = getattr(args, 'model', None) or find_latest_checkpoint()
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return dict(
        model_path=model_path, device=str(device),
        rollouts=args.rollouts, leaf_steps=args.leaf_steps, c_ucb=args.c_ucb,
        select=args.select, seed=args.seed, peek=args.peek, opponent=args.opponent,
        schedule=args.schedule, early_rollouts=args.early_rollouts,
        early_plies=args.early_plies, ramp_plies=args.ramp_plies,
        time_budget=args.time_budget,
    )


def _arena_init(cfg):
    """worker 初始化：每进程只加载一次模型，并限制 torch 线程数避免超订。"""
    torch.set_num_threads(1)
    dev = torch.device(cfg['device'])
    _ARENA['ev'] = Evaluator(load_net(cfg['model_path'], dev), dev)
    _ARENA['cfg'] = cfg


def _arena_task(g):
    """下第 g 局，返回 (搜索AI执哪方, winner, 步数, 模拟次数)。可在任意进程里执行。"""
    cfg = _ARENA['cfg']
    ev = _ARENA['ev']

    def make(seed):
        return MonteCarloSearch(
            ev, rollouts=cfg['rollouts'], leaf_steps=cfg['leaf_steps'], c_ucb=cfg['c_ucb'],
            select=cfg['select'], seed=seed, peek=cfg['peek'],
            time_budget=cfg['time_budget'], verbose=False,
            early_rollouts=cfg['early_rollouts'], early_plies=cfg['early_plies'],
            ramp_plies=cfg['ramp_plies'], schedule=cfg['schedule'])

    search_red = (g % 2 == 0)
    search_agent = SearchAgent(make(cfg['seed'] + g))
    if cfg['opponent'] == 'model':
        opponent = ModelAgent(ev)
    elif cfg['opponent'] == 'random':
        opponent = RandomAgent(seed=cfg['seed'] + g)
    else:
        opponent = SearchAgent(make(cfg['seed'] + 1000 + g), name='搜索AI(对手)')

    if search_red:
        winner, steps = play_game(search_agent, opponent, seed=cfg['seed'] + g)
        search_player = 0
    else:
        winner, steps = play_game(opponent, search_agent, seed=cfg['seed'] + g)
        search_player = 1
    return search_player, winner, steps, search_agent.total_rollouts


def mode_arena(args, evaluator, model_path=None, device=None):
    """竞技场：搜索 AI 对阵指定对手，交替先后手，统计胜负。"""
    cfg = _arena_cfg(args, evaluator, model_path, device)
    t0 = time.time()
    workers = max(1, min(int(getattr(args, 'workers', 1)), args.games))

    if workers == 1:
        _arena_init(cfg)
        results = []
        for g in range(args.games):
            results.append(_arena_task(g))
            sp, winner, steps, rolls = results[-1]
            tag = '和' if winner < 0 else ('胜' if winner == sp else '负')
            print(f'  第 {g + 1:>3}/{args.games} 局  搜索AI执{"红" if sp == 0 else "黑"} -> {tag}   '
                  f'{steps} 步  {rolls} 次模拟  累计 {time.time() - t0:.0f}s', flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        results = []
        print(f'  并行评测：{workers} 进程（每进程 1 个 torch 线程）', flush=True)
        with ctx.Pool(workers, initializer=_arena_init, initargs=(cfg,)) as pool:
            for i, r in enumerate(pool.imap_unordered(_arena_task, range(args.games), chunksize=1), 1):
                results.append(r)
                if i % max(1, args.games // 20) == 0 or i == args.games:
                    print(f'  进度 {i}/{args.games}  累计 {time.time() - t0:.0f}s', flush=True)

    wins = draws = losses = 0
    total_steps = 0
    total_rollouts = 0
    for sp, winner, steps, rolls in results:
        total_steps += steps
        total_rollouts += rolls
        if winner < 0:
            draws += 1
        elif winner == sp:
            wins += 1
        else:
            losses += 1

    total = wins + draws + losses
    print()
    print(f'== 竞技场结果：搜索 AI vs {args.opponent}（{total} 局，交替先后手）==')
    print(f'  搜索配置: 每步 {args.rollouts} 次模拟  截断 {args.leaf_steps or "不截断"}  '
          f'预算 {args.schedule}')
    print(f'  胜 {wins}   和 {draws}   负 {losses}')
    if total:
        n = total
        se = (wins / n * (1 - wins / n) / n) ** 0.5
        print(f'  胜率 {wins / n:.4f} (±{1.96 * se:.4f}, 95% CI)   '
              f'期望得分 {(wins + 0.5 * draws) / n:.4f}   '
              f'排除和棋后胜率 {wins / max(1, wins + losses):.4f}')
        print(f'  平均局长 {total_steps / n:.1f} 步   总模拟 {total_rollouts:,} 次   '
              f'总耗时 {time.time() - t0:.0f}s')


def mode_human(args, evaluator):
    """命令行人机对战。输入动作编号走棋，b 重看棋盘，q 退出。"""
    print('== 人机对战 ==  输入动作编号走棋，b 看棋盘，q 退出')
    game = GameEnv()
    history = []
    human = 0  # 人类执红
    agent = SearchAgent(_new_searcher(args, evaluator, args.seed, verbose=False))

    while not game.done:
        print()
        print(render_board(game))
        print(LEGEND)
        print(f'{side_name(game.current_player)} 行动   分数 红 {game.red_score} : 黑 {game.black_score}   '
              f'和棋计数 {game.peace_steps}   未翻开 {len(game.unflipped_positions)}')

        if game.current_player == human:
            infoset = InfoSet(game.current_player, game, history, perfect_info=False)
            actions = infoset.legal_actions
            if not actions:
                break
            for i, a in enumerate(actions, 1):
                print(f'  [{i:>2}] {a}')
            sys.stdout.write('> ')
            sys.stdout.flush()
            raw = sys.stdin.readline().strip()
            if raw.lower() == 'q':
                print('已退出。')
                return
            if raw.lower() == 'b' or raw == '':
                continue
            try:
                idx = int(raw)
            except ValueError:
                print('请输入动作编号。')
                continue
            if not (1 <= idx <= len(actions)):
                print(f'编号需在 1..{len(actions)}。')
                continue
            action = actions[idx - 1]
        else:
            print(f'  {agent.name} 思考中（{side_name(game.current_player)}）...')
            action = agent.act(game, list(history))
            result = agent.last_result
            if result is not None and result.actions:
                best = result.chosen_index
                print(f'  {agent.name} 选择 {action}   平均估值 {result.mean_value[best]:+.3f}   '
                      f'{result.rollouts_done} 次模拟   {result.elapsed:.1f}s')
                if args.verbose:
                    print(result.table())
            if action is None:
                break

        game.step(action)
        history.append(action)

    print()
    print(render_board(game))
    if game.winner < 0:
        print('结果：和棋')
    else:
        print(f'结果：{side_name(game.winner)} 胜')


# --------------------------------------------------------------------------- #
# pygame 界面（复用 game.py 的绘制，搜索放到后台线程）
# --------------------------------------------------------------------------- #
class _GuiGreedyAgent:
    """单步贪心：直接调用价值网络，作为「原始 game.py AI」的对照。"""

    kind = 'greedy'

    def __init__(self, evaluator, name='贪心模型'):
        self.ev = evaluator
        self.name = name

    def act(self, game, history):
        infoset = InfoSet(game.current_player, game, history, perfect_info=False)
        if not infoset.legal_actions:
            return None
        return self.ev.best_action(infoset)


_SIDE_OPTIONS = [('人类', 'human'), ('贪心模型', 'greedy'), ('搜索AI', 'search')]
_BUDGET_OPTIONS = [('快 200', 200), ('中 500', 500), ('慢 2000', 2000)]


def _select_sides(screen, clock, assets, gui, args):
    """启动菜单：选择红/黑双方的角色与搜索预算。返回 (red, black, budget) 或 None。"""
    font_title = gui.load_font(34, bold=True)
    font_label = gui.load_font(24, bold=True)
    font_btn = gui.load_font(20)
    font_small = gui.load_font(18)

    cx = gui.WINDOW_WIDTH // 2
    btn_w, btn_h, gap = 132, 44, 10

    def make_row(cy, n):
        total_w = n * btn_w + (n - 1) * gap
        x0 = cx - total_w // 2
        return [pygame.Rect(x0 + i * (btn_w + gap), cy, btn_w, btn_h) for i in range(n)]

    red_rects = make_row(180, len(_SIDE_OPTIONS))
    black_rects = make_row(310, len(_SIDE_OPTIONS))
    budget_rects = make_row(440, len(_BUDGET_OPTIONS))
    start_btn = pygame.Rect(cx - 90, 540, 180, 56)

    red_sel = black_sel = 0
    budget_sel = 1

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if start_btn.collidepoint(mx, my):
                    return _SIDE_OPTIONS[red_sel][1], _SIDE_OPTIONS[black_sel][1], _BUDGET_OPTIONS[budget_sel][1]
                for i, rect in enumerate(red_rects):
                    if rect.collidepoint(mx, my):
                        red_sel = i
                for i, rect in enumerate(black_rects):
                    if rect.collidepoint(mx, my):
                        black_sel = i
                for i, rect in enumerate(budget_rects):
                    if rect.collidepoint(mx, my):
                        budget_sel = i

        screen.fill(gui.COLOR_BG)
        if assets is not None and assets.get('background') is not None:
            overlay = pygame.Surface((gui.WINDOW_WIDTH, gui.WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 110))
            screen.blit(assets['background'], (0, 0))
            screen.blit(overlay, (0, 0))

        title = font_title.render('翻翻棋 FlipZero — 蒙特卡洛搜索版', True, gui.COLOR_TEXT)
        screen.blit(title, title.get_rect(center=(cx, 80)))
        sub = font_small.render('AI 走棋时会模拟对局并估计每个动作的胜率；搜索在后台线程进行', True, gui.COLOR_TEXT)
        screen.blit(sub, sub.get_rect(center=(cx, 120)))

        lab = font_label.render('红方（先手）', True, gui.COLOR_RED)
        screen.blit(lab, lab.get_rect(center=(cx, 152)))
        for i, rect in enumerate(red_rects):
            gui._draw_choice(screen, rect, _SIDE_OPTIONS[i][0], i == red_sel, font_btn, True)

        lab = font_label.render('黑方（后手）', True, gui.COLOR_BLACK)
        screen.blit(lab, lab.get_rect(center=(cx, 282)))
        for i, rect in enumerate(black_rects):
            gui._draw_choice(screen, rect, _SIDE_OPTIONS[i][0], i == black_sel, font_btn, True)

        lab = font_label.render(f'搜索预算（每步模拟次数上限）', True, gui.COLOR_TEXT)
        screen.blit(lab, lab.get_rect(center=(cx, 412)))
        for i, rect in enumerate(budget_rects):
            gui._draw_choice(screen, rect, _BUDGET_OPTIONS[i][0], i == budget_sel, font_btn, True)

        hint = f'阶梯预算: 前 {args.early_plies} 步 {args.early_rollouts} 次 → {args.ramp_plies} 步内升到上限'
        hs = font_small.render(hint, True, gui.COLOR_TEXT)
        screen.blit(hs, hs.get_rect(center=(cx, 500)))

        mx, my = pygame.mouse.get_pos()
        hover = start_btn.collidepoint(mx, my)
        pygame.draw.rect(screen, gui.COLOR_BTN_ACTIVE if hover else gui.COLOR_BTN, start_btn, border_radius=8)
        pygame.draw.rect(screen, gui.COLOR_BOARD_LINE, start_btn, 2, border_radius=8)
        s = font_label.render('开始游戏', True, gui.COLOR_TEXT)
        screen.blit(s, s.get_rect(center=start_btn.center))
        pygame.display.flip()
        clock.tick(gui.FPS)


class _Match:
    """一局对战的界面状态与交互（移植自 game.py 的 _play_one_match，改为后台搜索）。"""

    def __init__(self, screen, clock, fonts, assets, gui, evaluator, red_kind, black_kind, budget, args):
        self.screen = screen
        self.clock = clock
        self.fonts = fonts
        self.assets = assets
        self.gui = gui
        self.args = args
        self.ev = evaluator
        self.env = GameEnv()
        self.history = []
        self.selected = None
        self.target_map = {}
        self.last_move_origin = None
        self.move_animation = None
        self.next_action_at = 0.0
        self.search_progress = None  # (done, total) 搜索进行中的进度
        self.search_player = None
        self.last_search = None      # (player, SearchResult)
        self.budget = budget
        self.quit = False

        self.red_kind = red_kind
        self.black_kind = black_kind
        self._greedy = _GuiGreedyAgent(evaluator)
        self._searcher = MonteCarloSearch(
            evaluator,
            rollouts=budget,
            leaf_steps=args.leaf_steps,
            c_ucb=args.c_ucb,
            select=args.select,
            seed=args.seed,
            peek=args.peek,
            verbose=False,
            early_rollouts=min(args.early_rollouts, budget),
            early_plies=args.early_plies,
            ramp_plies=args.ramp_plies,
            schedule=args.schedule,
        )

    # ---------------- 小工具 ---------------- #
    def kind_of(self, player):
        return self.red_kind if player == 0 else self.black_kind

    def is_human_turn(self):
        return not self.env.done and self.kind_of(self.env.current_player) == 'human'

    def reset_selection(self):
        self.selected = None
        self.target_map = {}

    def build_targets_for(self, r, c):
        legals = self.env.get_legal_actions()
        piece = self.env.board[r][c]
        is_cannon = piece in (PieceType.RED_CANNON, PieceType.BLACK_CANNON)
        m = {}
        for a in legals:
            if a.from_row != r or a.from_col != c:
                continue
            # 炮只能隔子 ATTACK，不能 MOVE（MoveGenerator 理论上已保证，这里再过滤一次）
            if is_cannon and a.action_type != ActionType.ATTACK:
                continue
            m[(a.to_row, a.to_col)] = a
        return m

    def find_flip_action(self, r, c):
        for a in self.env.get_legal_actions():
            if a.action_type == ActionType.FLIP and a.to_row == r and a.to_col == c:
                return a
        return None

    def own_piece_at(self, player, r, c):
        p = self.env.board[r][c]
        if p in (PieceType.EMPTY, PieceType.UNFLIPPED):
            return False
        return is_red(p) if player == 0 else is_black(p)

    def apply_action(self, action):
        moving_piece = None
        if action.action_type in (ActionType.MOVE, ActionType.ATTACK):
            self.last_move_origin = (action.from_row, action.from_col)
            moving_piece = self.env.board[action.from_row][action.from_col]
        else:
            self.last_move_origin = None
        self.history.append(action)
        self.env.step(action)
        if moving_piece is not None:
            self.move_animation = {
                'piece': moving_piece,
                'from': (action.from_row, action.from_col),
                'to': (action.to_row, action.to_col),
                'start_time': time.time(),
            }
        else:
            self.move_animation = None
        self.reset_selection()
        self.next_action_at = time.time() + 0.3

    # ---------------- 事件 ---------------- #
    def handle_events(self):
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
                return 'quit'
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit = True
                    return 'quit'
                if event.key == pygame.K_r and self.env.done:
                    return 'menu'
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.env.done:
                    return 'menu'
                if not self.is_human_turn() or self.move_animation:
                    continue
                pos = self.gui.pos_from_mouse(*event.pos)
                if pos is None:
                    continue
                r, c = pos
                cell = self.env.board[r][c]
                player = self.env.current_player
                if self.selected is not None:
                    if (r, c) in self.target_map:
                        self.apply_action(self.target_map[(r, c)])
                        continue
                    if (r, c) == self.selected:
                        self.reset_selection()
                        continue
                    if self.own_piece_at(player, r, c):
                        self.selected = (r, c)
                        self.target_map = self.build_targets_for(r, c)
                        continue
                    if cell == PieceType.UNFLIPPED:
                        a = self.find_flip_action(r, c)
                        if a is not None:
                            self.apply_action(a)
                        continue
                    self.reset_selection()
                else:
                    if cell == PieceType.UNFLIPPED:
                        a = self.find_flip_action(r, c)
                        if a is not None:
                            self.apply_action(a)
                    elif self.own_piece_at(player, r, c):
                        self.selected = (r, c)
                        self.target_map = self.build_targets_for(r, c)
        return None

    # ---------------- AI ---------------- #
    def _on_progress(self, done, total):
        """搜索进度回调：泵事件 + 重绘。返回 False 表示中止搜索。

        用回调而不是后台线程，是因为 CPython 的 GIL 会让线程版搜索被主线程
        的重绘严重饿死（实测慢几十倍）。回调在搜索循环内部执行，搜索绝大部分
        时间是全速跑的，只是每隔一小段刷新一次画面。
        """
        self.search_progress = (done, total)
        if self.handle_events() is not None:
            self.quit = True
            return False
        self.draw()
        return True

    def step_ai(self):
        if self.env.done or self.move_animation or self.is_human_turn():
            return
        if time.time() < self.next_action_at:
            return

        if self.kind_of(self.env.current_player) != 'search':
            action = self._greedy.act(self.env, self.history)
            if action is not None:
                self.apply_action(action)
            return

        player = self.env.current_player
        self.search_player = player
        self.search_progress = (0, 0)
        budget_guess = max(1, self.budget)
        self._searcher.progress_cb = self._on_progress
        self._searcher.progress_every = max(1, budget_guess // 40)
        try:
            result = self._searcher.search(self.env, list(self.history), player=player)
        finally:
            self._searcher.progress_cb = None
            self.search_progress = None
        if self.quit:
            return

        if result is None or not result.actions:
            # 搜索异常：退化为贪心，保证对局能继续
            info = InfoSet(self.env.current_player, self.env, self.history, perfect_info=False)
            action = self.ev.best_action(info) if info.legal_actions else None
        else:
            self.last_search = (player, result)
            action = result.actions[result.chosen_index]
        if action is not None:
            self.apply_action(action)

    # ---------------- 绘制 ---------------- #
    def status_lines(self):
        import numpy as np
        env = self.env
        gui = self.gui
        cur = env.current_player
        cur_text = '红方' if cur == 0 else '黑方'
        cur_color = gui.COLOR_RED if cur == 0 else gui.COLOR_BLACK
        role = {0: self.red_kind, 1: self.black_kind}
        role_cn = {'human': '人类', 'greedy': '贪心AI', 'search': '搜索AI'}
        status = [
            (f'红方({role_cn[role[0]]}) 分数: {env.red_score}', gui.COLOR_RED),
            (f'黑方({role_cn[role[1]]}) 分数: {env.black_score}', gui.COLOR_BLACK),
            (f'和棋计数: {env.peace_steps}/20', gui.COLOR_TEXT),
            (f'未翻棋子: {len(env.unflipped_positions)}', gui.COLOR_TEXT),
            ('', gui.COLOR_TEXT),
        ]
        if env.done:
            if env.winner == 0:
                status.append(('红方胜！点击返回菜单', gui.COLOR_RED))
            elif env.winner == 1:
                status.append(('黑方胜！点击返回菜单', gui.COLOR_BLACK))
            else:
                status.append(('和棋！点击返回菜单', gui.COLOR_TEXT))
        else:
            status.append((f'当前: {cur_text}', cur_color))

        if self.search_progress is not None:
            done, total = self.search_progress
            if total:
                status.append((f'搜索中 {done}/{total} ({done / total:.0%})', (200, 120, 0)))
            else:
                status.append(('搜索中...', (200, 120, 0)))
        elif self.last_search is not None:
            player, res = self.last_search
            if res is not None and res.actions:
                idx = res.chosen_index
                who = '红' if player == 0 else '黑'
                status.append(('', gui.COLOR_TEXT))
                status.append((f'{who}方搜索 {res.rollouts_done} 次 / {len(res.actions)} 候选', gui.COLOR_TEXT))
                status.append((f'选中估计 {res.mean_value[idx]:+.3f}', gui.COLOR_TEXT))
                order = np.argsort(-res.mean_value)[:3]
                for k, i in enumerate(order, 1):
                    txt = str(res.actions[i])
                    if len(txt) > 15:
                        txt = txt[:15] + '..'
                    status.append((f' {k} {res.mean_value[i]:+.3f} {txt}', (100, 100, 100)))
        return status

    def draw(self):
        target_set = set(self.target_map.keys())
        self.gui.draw_board(self.screen, self.env, self.selected, target_set,
                            self.last_move_origin, self.fonts, self.status_lines(),
                            self.assets, self.move_animation)
        self.gui.pygame.display.flip()

    def run(self):
        while True:
            if self.move_animation and time.time() - self.move_animation['start_time'] >= self.gui.MOVE_ANIMATION_DURATION:
                self.move_animation = None
            nav = self.handle_events()
            if nav is not None:
                return nav
            self.step_ai()
            self.draw()
            self.clock.tick(self.gui.FPS)


def gui_main(args, evaluator, model_path):
    global pygame
    try:
        import pygame
        import game as gui
    except Exception as exc:
        print(f'无法加载 pygame 界面: {exc}', file=sys.stderr)
        print('请先安装 pygame：pip install pygame', file=sys.stderr)
        return 1

    pygame.init()
    pygame.display.set_caption('翻翻棋 FlipZero — 蒙特卡洛搜索版')
    screen = pygame.display.set_mode((gui.WINDOW_WIDTH, gui.WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    assets = gui.load_assets()
    fonts = (gui.load_font(40, bold=True), gui.load_font(22, bold=True), gui.load_font(18))

    while True:
        choice = _select_sides(screen, clock, assets, gui, args)
        if choice is None:
            break
        red_kind, black_kind, budget = choice
        match = _Match(screen, clock, fonts, assets, gui, evaluator,
                       red_kind, black_kind, budget, args)
        if match.run() == 'quit':
            break
    pygame.quit()
    return 0


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description='FlipZero 蒙特卡洛搜索对局程序')
    p.add_argument('--mode', choices=['analyze', 'arena', 'human', 'gui'], default='analyze',
                   help='analyze=分析局面; arena=搜索AI对阵对手; human=命令行对战; gui=pygame 界面')
    p.add_argument('--model', type=str, default=None,
                   help='checkpoint 路径，默认取 log/flipzero/ 下步数最大的')
    p.add_argument('--device', type=str, default=None, help='cpu / cuda，默认自动选择')

    # 搜索预算
    p.add_argument('--rollouts', type=int, default=10000,
                   help='模拟次数上限（阶梯模式下为后期使用的上限，默认 10000）')
    p.add_argument('--early-rollouts', type=int, default=500,
                   help='阶梯模式下前期每步使用的模拟次数（默认 500）')
    p.add_argument('--early-plies', type=int, default=10,
                   help='阶梯模式下前多少步使用 early-rollouts（默认 10）')
    p.add_argument('--ramp-plies', type=int, default=10,
                   help='阶梯模式从 early-rollouts 线性升到 rollouts 所用的步数')
    p.add_argument('--schedule', choices=['auto', 'fixed'], default='auto',
                   help='auto=前 N 步少跑、之后递增; fixed=每步都用 rollouts')
    p.add_argument('--leaf-steps', type=int, default=12,
                   help='模拟 K 步后用价值网络估值截断；0 表示模拟到终局（很慢，约 8 分钟/步）')
    p.add_argument('--time-budget', type=float, default=0.0,
                   help='单步搜索的墙钟预算（秒），0 表示不限制')
    p.add_argument('--c-ucb', type=float, default=0.7, help='UCB1 探索系数')
    p.add_argument('--select', choices=['score', 'winrate', 'visits'], default='score',
                   help='最终选择准则：score=胜+0.5和(默认); winrate=纯胜率; visits=访问最多')
    p.add_argument('--peek', action='store_true',
                   help='诊断用：模拟时使用真实棋盘（作弊/完美信息），仅用于对比')

    # 模式参数
    p.add_argument('--games', type=int, default=10, help='arena 模式的对局数')
    p.add_argument('--workers', type=int, default=1,
                   help='arena 模式的并行进程数（>1 时多进程并行，每进程限 1 个 torch 线程）')
    p.add_argument('--opponent', choices=['model', 'random', 'search'], default='model',
                   help='arena 模式的对手')
    p.add_argument('--ply', type=int, default=12, help='analyze 模式走到第几步后分析')
    p.add_argument('--seed', type=int, default=0, help='随机种子')
    p.add_argument('--verbose', action='store_true', help='打印搜索进度与动作表')
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model_path = args.model or find_latest_checkpoint()
    if not model_path or not os.path.exists(model_path):
        print('找不到 checkpoint，请用 --model 指定。', file=sys.stderr)
        return 1

    evaluator = Evaluator(load_net(model_path, device), device)
    if args.schedule == 'auto' and 0 < args.early_rollouts < args.rollouts:
        sched = (f'阶梯 前 {args.early_plies} 步 {args.early_rollouts} 次 '
                 f'→ {args.ramp_plies} 步内升至 {args.rollouts} 次')
    else:
        sched = f'固定 {args.rollouts} 次'
    print(f'模型: {os.path.basename(model_path)}   设备: {device}   '
          f'模拟预算: {sched}   截断步数: {args.leaf_steps or "不截断(全程)"}   '
          f'选择准则: {args.select}')
    if args.leaf_steps == 0 and args.rollouts >= 2000:
        print('提示: 完整模拟 + 高预算会非常慢（1 万次约 8 分钟/步）。'
              '可改 --leaf-steps 12 或降低 --rollouts；交互对战建议 --rollouts 500 左右。')

    t0 = time.time()
    if args.mode == 'gui':
        return gui_main(args, evaluator, model_path)
    elif args.mode == 'analyze':
        mode_analyze(args, evaluator)
    elif args.mode == 'arena':
        mode_arena(args, evaluator, model_path, device)
    else:
        mode_human(args, evaluator)
    print(f'\n总耗时 {time.time() - t0:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
