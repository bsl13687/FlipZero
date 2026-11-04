"""单机版翻翻棋（暗棋）GUI。

使用 pygame 实现的图形界面，支持选择红方/黑方分别由人类或 AI 操作。
AI 调用本地训练好的模型权重（log/flipzero/ 下最新的 .tar）进行推理。

运行：
    python game.py
"""
import os
import sys
import glob
import time
import pickle
import threading
import multiprocessing as mp

import pygame
import numpy as np
import torch

from flipzero.env.game import (
    PieceType, ActionType, Action, GameEnv,
    BOARD_HEIGHT, BOARD_WIDTH, INITIAL_SCORE,
    is_red, is_black, ALL_PIECE_TYPES,
)
from flipzero.env.env import InfoSet
from flipzero.dmc.models import FlipZeroModel


# ---------------- 配置 ----------------
CELL_SIZE = 80
BOARD_MARGIN = 30
SIDEBAR_WIDTH = 260
# 墓地区域：左侧放红方阵亡，右侧放黑方阵亡
CEMETERY_WIDTH = 2 * CELL_SIZE + 20  # 每行最多 2 枚棋子
CEMETERY_GAP = 12
BOARD_LEFT = BOARD_MARGIN + CEMETERY_WIDTH + CEMETERY_GAP
WINDOW_WIDTH = BOARD_LEFT + BOARD_WIDTH * CELL_SIZE + CEMETERY_GAP + CEMETERY_WIDTH + BOARD_MARGIN + SIDEBAR_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT * CELL_SIZE + BOARD_MARGIN * 2
FPS = 30
MOVE_ANIMATION_DURATION = 0.18

# 颜色
COLOR_BG = (245, 230, 200)
COLOR_BOARD_LINE = (90, 60, 30)
COLOR_CELL = (220, 195, 150)
COLOR_CELL_HL = (255, 230, 130)
COLOR_CELL_TARGET = (140, 220, 140)
COLOR_UNFLIPPED = (110, 80, 50)
COLOR_RED = (200, 30, 30)
COLOR_BLACK = (20, 20, 20)
COLOR_TEXT = (40, 40, 40)
COLOR_BTN = (180, 180, 180)
COLOR_BTN_HL = (210, 210, 210)
COLOR_BTN_ACTIVE = (120, 200, 120)

# 棋子中文名（兜底显示，没图片素材时使用）
PIECE_NAMES = {
    PieceType.RED_KING: "将",
    PieceType.RED_MANDARIN: "士",
    PieceType.RED_ELEPHANT: "象",
    PieceType.RED_KNIGHT: "马",
    PieceType.RED_ROOK: "车",
    PieceType.RED_CANNON: "炮",
    PieceType.RED_SOLDIER: "兵",
    PieceType.BLACK_KING: "将",
    PieceType.BLACK_MANDARIN: "士",
    PieceType.BLACK_ELEPHANT: "象",
    PieceType.BLACK_KNIGHT: "马",
    PieceType.BLACK_ROOK: "车",
    PieceType.BLACK_CANNON: "炮",
    PieceType.BLACK_SOLDIER: "兵",
}

# 图片素材
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "image")
PIECE_IMAGE_FILES = {
    PieceType.RED_KING: "RedKing.png",
    PieceType.RED_MANDARIN: "RedMandarins.png",
    PieceType.RED_ELEPHANT: "RedElephant.png",
    PieceType.RED_KNIGHT: "RedKnight.png",
    PieceType.RED_ROOK: "RedRook.png",
    PieceType.RED_CANNON: "RedCannons.png",
    PieceType.RED_SOLDIER: "RedSoldier.png",
    PieceType.BLACK_KING: "BlackKing.png",
    PieceType.BLACK_MANDARIN: "BlackMandarins.png",
    PieceType.BLACK_ELEPHANT: "BlackElephant.png",
    PieceType.BLACK_KNIGHT: "BlackKnight.png",
    PieceType.BLACK_ROOK: "BlackRook.png",
    PieceType.BLACK_CANNON: "BlackCannons.png",
    PieceType.BLACK_SOLDIER: "BlackSoldier.png",
}
BACK_IMAGE_FILE = "Item.png"
BACKGROUND_IMAGE_FILE = "Background.jpg"
TAP_IMAGE_FILE = "tap.png"

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")

# 可选 AI 模型槽位：显示名 -> 模型目录（各自取目录下最新的 model_step_*.tar）
MODEL_SLOTS = [
    ("latest", os.path.join(LOG_DIR, "flipzero")),
    ("rank1", os.path.join(LOG_DIR, "rank", "1")),
    ("rank2", os.path.join(LOG_DIR, "rank", "2")),
    ("rank3", os.path.join(LOG_DIR, "rank", "3")),
]

# ---------------- MCTS 配置 ----------------
# 多进程 rollout 模拟：从根局面按神经网络预测权重采样若干次，
# 统计每个根动作的 win/visit，最后选胜率最高的动作。
MCTS_NUM_WORKERS = 6          # 并行模拟进程数（按需调整）
MCTS_THINK_TIME = 10.0        # 每次走棋思考时间（秒）
MCTS_ROLLOUT_TEMPERATURE = 1.0  # rollout 中 softmax 温度，越大越随机
MCTS_ROOT_TEMPERATURE = 1.0     # 根动作采样温度
MCTS_ROOT_BOOST = 4.0           # 模型推荐动作的权重放大系数（1.0=不放大；2.0=最看好动作权重×2后再归一化，例如0.6/0.4→0.75/0.25）
MCTS_FORCE_MODEL_STEPS = 1     # 对局前 N 步强制用模型 argmax，不跑 MCTS（开局局面信息少，模拟意义不大）
MCTS_TOP_K = 3                  # 每个局面只保留神经网络预测权重最高的前 K 个合法动作（MCTS 搜索剪枝，0=不剪枝）
MCTS_CACHE_MAX = 200000        # 单 worker 局面缓存上限，超过则清空
MCTS_MAX_ROLLOUT_STEPS = 400  # 单局 rollout 最大步数，防止异常死循环


def _load_scaled(filename, size):
    path = os.path.join(ASSETS_DIR, filename)
    if not os.path.isfile(path):
        return None
    try:
        img = pygame.image.load(path).convert_alpha()
    except Exception:
        try:
            img = pygame.image.load(path)
        except Exception:
            return None
    return pygame.transform.smoothscale(img, size)


def load_assets():
    """加载棋子/背面/背景图片，返回 dict。失败时返回的图片为 None，绘制端会回退到几何+文字。"""
    piece_size = (CELL_SIZE - 8, CELL_SIZE - 8)
    pieces = {}
    for pt, fname in PIECE_IMAGE_FILES.items():
        pieces[pt] = _load_scaled(fname, piece_size)
    back = _load_scaled(BACK_IMAGE_FILE, piece_size)
    background = _load_scaled(BACKGROUND_IMAGE_FILE, (WINDOW_WIDTH, WINDOW_HEIGHT))
    tap = _load_scaled(TAP_IMAGE_FILE, (CELL_SIZE, CELL_SIZE))
    return {"pieces": pieces, "back": back, "background": background, "tap": tap}


def find_latest_checkpoint(dir_path):
    """返回 dir_path 目录下最新的 model_step_*.tar 路径，没找到返回 None。"""
    if not os.path.isdir(dir_path):
        return None
    files = glob.glob(os.path.join(dir_path, "model_step_*.tar"))
    if not files:
        return None

    def _step(p):
        try:
            return int(os.path.basename(p).replace("model_step_", "").replace(".tar", ""))
        except ValueError:
            return -1

    files.sort(key=_step)
    return files[-1]


def checkpoint_step(path):
    """从 model_step_*.tar 路径中解析训练步数，失败返回 None。"""
    try:
        return int(os.path.basename(path).replace("model_step_", "").replace(".tar", ""))
    except (ValueError, AttributeError):
        return None


def discover_models():
    """发现可用的 AI 模型：返回 [(显示名, 模型路径), ...]，缺失目录/文件的槽位会被跳过。"""
    models = []
    for name, dir_path in MODEL_SLOTS:
        path = find_latest_checkpoint(dir_path)
        if path is not None:
            models.append((name, path))
    return models


# ---------------- AI 封装 ----------------
class GuiDeepAgent:
    """加载本地 checkpoint，对当前 InfoSet 选择最优动作。

    兼容两种保存格式：
    - dict 含 'model_state_dict'（训练时保存）
    - 直接 state_dict

    act() 内部跑多进程 MCTS rollout：从根局面按神经网络预测权重采样
    若干局，统计每个根动作的 win/visit/draw/fail，最后选 win/visit 最大的动作。

    worker 进程在 __init__ 时常驻启动（避免每次走棋都 spawn，spawn 期间
    主线程 GIL 被占用会导致 pygame 窗口冻结）；act() 只通过 Queue 发任务/收结果。
    """

    def __init__(self, model_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = FlipZeroModel()
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        self.model.eval()

        # 启动常驻 MCTS worker 进程池
        self._ctx = mp.get_context("spawn")
        self._task_queues = []
        self._result_queue = self._ctx.Queue()
        self._workers = []
        for i in range(MCTS_NUM_WORKERS):
            q = self._ctx.Queue()
            p = self._ctx.Process(
                target=_mcts_worker_loop,
                args=(model_path, q, self._result_queue, 1000 + i),
                daemon=True,
            )
            p.start()
            self._task_queues.append(q)
            self._workers.append(p)

    def close(self):
        """通知所有 worker 退出并等待回收。"""
        for q in self._task_queues:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self._workers:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        self._workers = []
        self._task_queues = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _predict_values(self, infoset):
        """对 infoset.legal_actions 中每个动作返回模型 value（np.ndarray, shape=[N]）。"""
        legal_actions = infoset.legal_actions
        n = len(legal_actions)
        x_no_action = torch.from_numpy(infoset.x_no_action).to(self.device).unsqueeze(0).expand(n, -1).contiguous()
        z = torch.from_numpy(infoset.z).to(self.device).unsqueeze(0).expand(n, -1, -1).contiguous()
        actions = torch.from_numpy(infoset.action_encodings).to(self.device)
        with torch.no_grad():
            values = self.model(x_no_action, actions, z, return_value=True)["value"]
        return values.cpu().numpy()

    def act(self, infoset, env=None, action_history=None):
        legal_actions = infoset.legal_actions
        if not legal_actions:
            return None
        if len(legal_actions) == 1:
            return legal_actions[0]

        root_values = self._predict_values(infoset)

        # 一步绝杀检查：某个动作执行后对方分数直接归零 → 胜率 100%，直接走
        # 注意：炮打暗棋（ATTACK 目标是 UNFLIPPED）结果不确定，可能翻出对方将也可能翻出己方将，
        #       这种情况跳过绝杀判断，交给 MCTS 模拟
        if env is not None:
            root_player = infoset.player
            side = "红" if root_player == 0 else "黑"
            for a in legal_actions:
                if a.action_type == ActionType.ATTACK and env.board[a.to_row][a.to_col] == PieceType.UNFLIPPED:
                    continue  # 炮/车打暗棋，结果未知，交给 MCTS
                sim = env.deepcopy()
                sim.step(a)
                if sim.done and sim.winner == root_player:
                    print(f"[绝杀] {side}方 {a} 直接获胜，胜率 100%，跳过 MCTS")
                    return a

        # 未启用多进程 MCTS，或没有 env/history 可用于 rollout：退化为 argmax
        if (MCTS_NUM_WORKERS <= 0 or MCTS_THINK_TIME <= 0
                or env is None or action_history is None
                or not self._workers):
            best_idx = int(np.argmax(root_values))
            return legal_actions[best_idx]

        # 对局前 N 步强制用模型 argmax（开局信息少，MCTS 模拟意义不大且耗时）
        step_no = len(action_history)
        if step_no < MCTS_FORCE_MODEL_STEPS:
            best_idx = int(np.argmax(root_values))
            side = "红" if infoset.player == 0 else "黑"
            print(f"[步骤 {step_no}] {side}方 前 {MCTS_FORCE_MODEL_STEPS} 步强制模型推荐: "
                  f"{legal_actions[best_idx]}  (nn={root_values[best_idx]:+.3f})")
            return legal_actions[best_idx]

        root_player = infoset.player
        root_env_state = pickle.dumps(env)
        root_history = list(action_history)

        task = dict(
            root_env_state=root_env_state,
            root_history=root_history,
            root_player=root_player,
            think_time=MCTS_THINK_TIME,
            temperature=MCTS_ROLLOUT_TEMPERATURE,
            root_temperature=MCTS_ROOT_TEMPERATURE,
            root_boost=MCTS_ROOT_BOOST,
            top_k=MCTS_TOP_K,
            cache_max=MCTS_CACHE_MAX,
            max_rollout_steps=MCTS_MAX_ROLLOUT_STEPS,
        )

        # 给每个常驻 worker 发任务
        for q in self._task_queues:
            q.put(task)

        # 聚合统计：key 用动作元组，避免跨进程对象 identity 问题
        agg = {}
        for a in legal_actions:
            agg[_action_key(a)] = {"visit": 0, "win": 0, "fail": 0, "draw": 0}
        total_hits = 0
        total_lookups = 0

        for _ in range(len(self._workers)):
            try:
                result = self._result_queue.get(timeout=MCTS_THINK_TIME + 60.0)
            except Exception:
                continue
            # worker 返回 (stats_dict, cache_hits, cache_lookups)
            if isinstance(result, tuple) and len(result) == 3:
                stats, hits, lookups = result
            else:
                stats, hits, lookups = result, 0, 0
            if not isinstance(stats, dict):
                continue
            total_hits += int(hits)
            total_lookups += int(lookups)
            for key, s in stats.items():
                if key in agg:
                    agg[key]["visit"] += int(s.get("visit", 0))
                    agg[key]["win"] += int(s.get("win", 0))
                    agg[key]["fail"] += int(s.get("fail", 0))
                    agg[key]["draw"] += int(s.get("draw", 0))

        # ---- 打印 MCTS 统计 ----
        side = "红" if root_player == 0 else "黑"
        opp_side = "黑" if root_player == 0 else "红"
        root_probs = _softmax(root_values, MCTS_ROOT_TEMPERATURE)
        model_best_idx = int(np.argmax(root_values))
        model_best = legal_actions[model_best_idx]
        print(f"\n=== MCTS [{side}方] {len(legal_actions)} 个合法动作"
              f"{f'，剪枝 top-{MCTS_TOP_K}' if MCTS_TOP_K > 0 else ''}, "
              f"{MCTS_NUM_WORKERS} workers x {MCTS_THINK_TIME:.0f}s ===")
        total_visit = 0
        total_win = 0
        total_fail = 0
        total_draw = 0
        for i, a in enumerate(legal_actions):
            key = _action_key(a)
            s = agg[key]
            total_visit += s["visit"]
            total_win += s["win"]
            total_fail += s["fail"]
            total_draw += s["draw"]
            wr = (s["win"] / s["visit"] * 100) if s["visit"] > 0 else 0.0
            dr = (s["draw"] / s["visit"] * 100) if s["visit"] > 0 else 0.0
            fr = (s["fail"] / s["visit"] * 100) if s["visit"] > 0 else 0.0
            nn_v = root_values[i]
            nn_p = root_probs[i] * 100
            tag = ""
            if a is model_best:
                tag = "  <-模型推荐"
            print(f"  {str(a):40s} nn={nn_v:+.3f}({nn_p:5.1f}%)  "
                  f"visit={s['visit']:4d}  "
                  f"win={s['win']:4d}({wr:5.1f}%)  "
                  f"fail={s['fail']:4d}({fr:5.1f}%)  "
                  f"draw={s['draw']:4d}({dr:5.1f}%){tag}")
        hit_rate = (total_hits / total_lookups * 100) if total_lookups > 0 else 0.0
        print(f"  总模拟局数: {total_visit}  |  "
              f"缓存命中: {total_hits}/{total_lookups} ({hit_rate:.1f}%)")

        # 选 win/visit 最大；visit=0 的动作跳过
        best_action = None
        best_score = -1.0
        for a in legal_actions:
            key = _action_key(a)
            v = agg[key]["visit"]
            if v <= 0:
                continue
            score = agg[key]["win"] / v
            if score > best_score:
                best_score = score
                best_action = a

        # 决策标记 + 双方胜率
        win_rate = (total_win / total_visit * 100) if total_visit > 0 else 0.0
        fail_rate = (total_fail / total_visit * 100) if total_visit > 0 else 0.0
        draw_rate = (total_draw / total_visit * 100) if total_visit > 0 else 0.0
        print(f"  胜率: {side}方 {win_rate:.1f}%  |  {opp_side}方 {fail_rate:.1f}%  |  "
              f"和棋 {draw_rate:.1f}%")

        if best_action is None:
            print(f"  >> 无有效模拟，退化为模型推荐: {model_best}")
            return model_best

        if best_action is model_best:
            print(f"  >> 最终选择: {best_action}  (模型与模拟一致，推荐)")
        else:
            print(f"  >> 最终选择: {best_action}  (模拟推荐，模型原本推荐 {model_best})")
        return best_action


def get_agent(model_path, cache):
    """按模型路径缓存并返回 GuiDeepAgent；加载失败返回 None（视作无 AI）。"""
    if model_path is None:
        return None
    if model_path not in cache:
        try:
            cache[model_path] = GuiDeepAgent(model_path)
        except Exception:
            cache[model_path] = None
    return cache[model_path]


def _action_key(a):
    """把 Action 转成可跨进程比较的 tuple。"""
    return (int(a.action_type), int(a.from_row), int(a.from_col),
            int(a.to_row), int(a.to_col))


def _softmax(x, temperature=1.0):
    """数值稳定的 softmax。x: np.ndarray -> np.ndarray，和为 1。"""
    x = np.asarray(x, dtype=np.float64) / max(float(temperature), 1e-6)
    x = x - x.max()
    e = np.exp(x)
    s = e.sum()
    if s <= 0 or not np.isfinite(s):
        return np.ones_like(x) / len(x)
    return e / s


def _mcts_worker_loop(model_path, task_queue, result_queue, seed):
    """常驻 MCTS worker 子进程。

    - 启动时加载一次模型到 CPU；
    - 循环从 task_queue 取任务（dict），跑 think_time 秒 rollout，
      把统计结果 put 到 result_queue；
    - 收到 None 任务时退出。

    任务 dict 字段：root_env_state / root_history / root_player /
    think_time / temperature / root_temperature / cache_max / max_rollout_steps。
    """
    # 子进程内重新 import，避免依赖 spawn 继承的模块状态
    import numpy as _np
    import torch as _torch
    from flipzero.env.env import InfoSet as _InfoSet
    from flipzero.dmc.models import FlipZeroModel as _FlipZeroModel

    _np.random.seed(seed)
    _torch.manual_seed(seed)

    device = _torch.device("cpu")
    model = _FlipZeroModel()
    ckpt = _torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    cache = {}
    cache_hits = [0]
    cache_lookups = [0]

    def _cache_key(env, hist):
        board_t = tuple(tuple(int(p) for p in row) for row in env.board)
        unflipped_t = frozenset((int(r), int(c)) for (r, c) in env.unflipped_positions)
        hist_t = tuple(_action_key(a) for a in hist[-10:])
        return (board_t, int(env.current_player), int(env.red_score),
                int(env.black_score), int(env.peace_steps), unflipped_t, hist_t)

    def _top_k_actions(legals, vals, top_k):
        """MCTS 剪枝：只保留神经网络预测权重最高的前 top_k 个动作。top_k<=0 或不够多时不裁剪。"""
        if top_k is None or top_k <= 0 or len(legals) <= top_k:
            return legals, vals
        order = _np.argsort(vals)[::-1][:top_k]
        return [legals[i] for i in order], vals[order]

    def _policy_values(env, hist, cache_max, top_k):
        legal = env.get_legal_actions()
        if not legal:
            return legal, _np.array([])
        key = _cache_key(env, hist)
        cache_lookups[0] += 1
        cached = cache.get(key)
        if cached is not None and len(cached) == len(legal):
            cache_hits[0] += 1
            return _top_k_actions(legal, cached, top_k)
        n = len(legal)
        # InfoSet 内部按 player 交换 own/opp 通道 -> 自动完成视角转换
        infoset = _InfoSet(player=env.current_player, game_env=env,
                           action_history=hist, perfect_info=False)
        x = _torch.from_numpy(infoset.x_no_action).unsqueeze(0).expand(n, -1).contiguous()
        z = _torch.from_numpy(infoset.z).unsqueeze(0).expand(n, -1, -1).contiguous()
        a = _torch.from_numpy(infoset.action_encodings)
        with _torch.no_grad():
            v = model(x, a, z, return_value=True)["value"].numpy()
        if len(cache) >= cache_max:
            cache.clear()
        cache[key] = v
        return _top_k_actions(legal, v, top_k)

    while True:
        task = task_queue.get()
        if task is None:
            return

        try:
            root_env = pickle.loads(task["root_env_state"])
            history = list(task["root_history"])
            root_player = task["root_player"]
            think_time = task["think_time"]
            temperature = task["temperature"]
            root_temperature = task["root_temperature"]
            root_boost = task.get("root_boost", 1.0)
            top_k = task.get("top_k", 0)
            cache_max = task["cache_max"]
            max_rollout_steps = task["max_rollout_steps"]

            # 每次任务重置缓存命中计数（cache dict 本身跨任务保留）
            cache_hits[0] = 0
            cache_lookups[0] = 0

            root_legal, root_vals = _policy_values(root_env, history, cache_max, top_k)
            if not root_legal:
                result_queue.put(({}, 0, 0))
                continue
            root_probs = _softmax(root_vals, root_temperature)
            # 对模型最看好的动作权重放大 root_boost 倍后重新归一化
            # 例如 0.6/0.4, boost=2.0 -> 1.2/0.4 -> 0.75/0.25
            if root_boost and root_boost != 1.0 and len(root_probs) > 1:
                best_i = int(_np.argmax(root_vals))
                root_probs = root_probs.copy()
                root_probs[best_i] *= root_boost
                root_probs /= root_probs.sum()

            stats = {}
            for a in root_legal:
                stats[_action_key(a)] = {"visit": 0, "win": 0, "fail": 0, "draw": 0}

            deadline = time.time() + think_time
            while time.time() < deadline:
                if len(root_legal) == 1:
                    a0 = root_legal[0]
                else:
                    a0 = root_legal[int(_np.random.choice(len(root_legal), p=root_probs))]

                sim_env = root_env.deepcopy()
                sim_hist = list(history)
                sim_hist.append(a0)
                sim_env.step(a0)

                steps = 0
                while not sim_env.done and steps < max_rollout_steps:
                    legals, vals = _policy_values(sim_env, sim_hist, cache_max, top_k)
                    if not legals:
                        break
                    if len(legals) == 1:
                        a_next = legals[0]
                    else:
                        probs = _softmax(vals, temperature)
                        a_next = legals[int(_np.random.choice(len(legals), p=probs))]
                    sim_hist.append(a_next)
                    sim_env.step(a_next)
                    steps += 1

                key0 = _action_key(a0)
                if key0 not in stats:
                    stats[key0] = {"visit": 0, "win": 0, "fail": 0, "draw": 0}
                stats[key0]["visit"] += 1
                if sim_env.winner == root_player:
                    stats[key0]["win"] += 1
                elif sim_env.winner == -2:
                    stats[key0]["draw"] += 1
                else:
                    stats[key0]["fail"] += 1

            result_queue.put((stats, cache_hits[0], cache_lookups[0]))
        except Exception:
            import traceback
            traceback.print_exc()
            try:
                result_queue.put(({}, 0, 0))
            except Exception:
                pass


# ---------------- 字体辅助 ----------------
_CN_FONT_PATH = None


def _find_chinese_font_path():
    """在 Windows 系统字体目录中查找一个包含中文字形的字体文件路径。"""
    global _CN_FONT_PATH
    if _CN_FONT_PATH is not None:
        return _CN_FONT_PATH

    # 优先文件名（按优先级）
    preferred_files = [
        "msyh.ttc", "msyh.ttf", "msyhbd.ttc", "msyhbd.ttf",
        "msyhl.ttc", "msyhl.ttf",
        "simhei.ttf", "simsun.ttc", "simsun.ttf",
        "Deng.ttf", "Dengb.ttf", "Dengl.ttf",
        "STZHONGS.TTF", "STKAITI.TTF", "STSONG.TTF",
        "mingliu.ttc", "mingliub.ttc",
    ]
    font_dirs = []
    win_dir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or r"C:\Windows"
    font_dirs.append(os.path.join(win_dir, "Fonts"))
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        font_dirs.append(os.path.join(local_appdata, "Microsoft", "Windows", "Fonts"))

    for d in font_dirs:
        if not os.path.isdir(d):
            continue
        for fname in preferred_files:
            p = os.path.join(d, fname)
            if os.path.isfile(p):
                _CN_FONT_PATH = p
                return p

    # 退而求其次：用 pygame.font.match_font 按字体名查
    for name in ["microsoftyahei", "msyh", "simhei", "simsun", "dengxian", "kaiti", "fangsong"]:
        try:
            p = pygame.font.match_font(name)
            if p and os.path.isfile(p):
                _CN_FONT_PATH = p
                return p
        except Exception:
            continue

    return None


def load_font(size, bold=False):
    """加载支持中文的字体；优先使用系统中文字体文件路径，避免 pygame.font.SysFont 在 Win+Py3.9 上的 None bug 与字形缺失问题。"""
    path = _find_chinese_font_path()
    if path:
        try:
            font = pygame.font.Font(path, size)
            if bold:
                font.set_bold(True)
            return font
        except Exception:
            pass

    # 最终兜底（不含中文字形，可能显示为方块；仅保证不崩溃）
    try:
        font = pygame.font.Font(None, size)
        if bold:
            font.set_bold(True)
        return font
    except Exception:
        return pygame.font.SysFont("arial", size, bold=bold)


# ---------------- 菜单 ----------------
def select_players(screen, models, assets=None):
    """启动菜单：选择红/黑分别为人类还是某个模型。

    models: [(显示名, 模型路径), ...]
    返回 (red_model_path, black_model_path)，其中 None 表示人类；整体返回 None 表示退出。
    """
    font_title = load_font(36, bold=True)
    font_label = load_font(24, bold=True)
    font_btn = load_font(20)
    font_small = load_font(18)

    # 每侧可选项：人类 + 所有可用模型
    options = [("人类", None)] + [(name, path) for name, path in models]

    cx = WINDOW_WIDTH // 2
    btn_w, btn_h = 120, 44
    gap = 8

    def make_row(cy):
        n = len(options)
        total_w = n * btn_w + (n - 1) * gap
        x0 = cx - total_w // 2
        return [pygame.Rect(x0 + i * (btn_w + gap), cy, btn_w, btn_h) for i in range(n)]

    red_rects = make_row(175)
    black_rects = make_row(310)
    start_btn = pygame.Rect(cx - 90, 430, 180, 56)

    red_sel = 0  # 默认人类
    black_sel = 0

    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if start_btn.collidepoint(mx, my):
                    return options[red_sel][1], options[black_sel][1]
                for i, rect in enumerate(red_rects):
                    if rect.collidepoint(mx, my):
                        red_sel = i
                        break
                for i, rect in enumerate(black_rects):
                    if rect.collidepoint(mx, my):
                        black_sel = i
                        break

        screen.fill(COLOR_BG)
        if assets is not None and assets.get("background") is not None:
            bg = assets["background"]
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 110))
            screen.blit(bg, (0, 0))
            screen.blit(overlay, (0, 0))
        title = font_title.render("翻翻棋 FlipZero", True, COLOR_TEXT)
        screen.blit(title, title.get_rect(center=(cx, 80)))

        if not models:
            tip = font_small.render("未找到模型权重，仅有「人类」可选", True, (180, 60, 60))
            screen.blit(tip, tip.get_rect(center=(cx, 130)))

        # 红方
        lab = font_label.render("红方（先手）", True, COLOR_RED)
        screen.blit(lab, lab.get_rect(center=(cx, 145)))
        for i, rect in enumerate(red_rects):
            _draw_choice(screen, rect, options[i][0], i == red_sel, font_btn, True)

        # 黑方
        lab = font_label.render("黑方（后手）", True, COLOR_BLACK)
        screen.blit(lab, lab.get_rect(center=(cx, 280)))
        for i, rect in enumerate(black_rects):
            _draw_choice(screen, rect, options[i][0], i == black_sel, font_btn, True)

        # 当前选择提示
        red_val = options[red_sel][1]
        black_val = options[black_sel][1]
        red_hint = "红方：人类操作" if red_val is None else f"红方：{options[red_sel][0]}（step {checkpoint_step(red_val)}）"
        black_hint = "黑方：人类操作" if black_val is None else f"黑方：{options[black_sel][0]}（step {checkpoint_step(black_val)}）"
        red_hint_surf = font_small.render(red_hint, True, COLOR_RED)
        black_hint_surf = font_small.render(black_hint, True, COLOR_BLACK)
        screen.blit(red_hint_surf, red_hint_surf.get_rect(center=(cx, 240)))
        screen.blit(black_hint_surf, black_hint_surf.get_rect(center=(cx, 375)))

        # 开始
        mx, my = pygame.mouse.get_pos()
        hover = start_btn.collidepoint(mx, my)
        pygame.draw.rect(screen, COLOR_BTN_ACTIVE if hover else COLOR_BTN, start_btn, border_radius=8)
        pygame.draw.rect(screen, COLOR_BOARD_LINE, start_btn, 2, border_radius=8)
        s = font_label.render("开始游戏", True, COLOR_TEXT)
        screen.blit(s, s.get_rect(center=start_btn.center))

        pygame.display.flip()
        clock.tick(FPS)


def _draw_choice(screen, rect, label, active, font, enabled):
    if not enabled:
        color = (200, 200, 200)
    elif active:
        color = COLOR_BTN_ACTIVE
    else:
        color = COLOR_BTN
    pygame.draw.rect(screen, color, rect, border_radius=6)
    pygame.draw.rect(screen, COLOR_BOARD_LINE, rect, 2, border_radius=6)
    text_color = (130, 130, 130) if not enabled else COLOR_TEXT
    s = font.render(label, True, text_color)
    screen.blit(s, s.get_rect(center=rect.center))


# ---------------- 棋盘绘制 ----------------
# 墓地中棋子的展示顺序（按棋力从高到低）
CEMETERY_RED_ORDER = [
    PieceType.RED_KING, PieceType.RED_MANDARIN, PieceType.RED_ELEPHANT,
    PieceType.RED_KNIGHT, PieceType.RED_ROOK, PieceType.RED_CANNON, PieceType.RED_SOLDIER,
]
CEMETERY_BLACK_ORDER = [
    PieceType.BLACK_KING, PieceType.BLACK_MANDARIN, PieceType.BLACK_ELEPHANT,
    PieceType.BLACK_KNIGHT, PieceType.BLACK_ROOK, PieceType.BLACK_CANNON, PieceType.BLACK_SOLDIER,
]
CEM_PIECE_SIZE = 56  # 墓地棋子绘制尺寸
CEM_COLS = 1          # 每种棋子始终展示 1 枚，数量 >=2 时通过右下角角标 xN 展示
CEM_ROW_GAP = 6
CEM_COL_GAP = 6


def compute_captured(env):
    """根据初始 ALL_PIECE_TYPES 与当前 true_board 上仍存在的棋子，计算各种棋子被击杀数量。"""
    initial = {}
    for pt in ALL_PIECE_TYPES:
        initial[pt] = initial.get(pt, 0) + 1
    alive = {}
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            piece = env.true_board[r][c]
            if piece in (PieceType.EMPTY,):
                continue
            alive[piece] = alive.get(piece, 0) + 1
    captured = {}
    for pt, n in initial.items():
        captured[pt] = max(0, n - alive.get(pt, 0))
    return captured


def draw_cemetery(screen, x, title, order, counts, piece_imgs, font_piece, font_small, title_color):
    """在 (x, BOARD_MARGIN) 处绘制宽度为 CEMETERY_WIDTH、高度对齐棋盘的墓地区域。

    每种棋子最多并排显示 CEM_COLS=2 枚；超过则在最后一枚的右下角绘制 "xN" 角标。
    整个墓地按竖直方向居中。
    """
    area_h = BOARD_HEIGHT * CELL_SIZE
    bg_rect = pygame.Rect(x, BOARD_MARGIN, CEMETERY_WIDTH, area_h)
    bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
    bg_surf.fill((0, 0, 0, 60))
    screen.blit(bg_surf, bg_rect.topleft)
    pygame.draw.rect(screen, COLOR_BOARD_LINE, bg_rect, 1)

    # 标题
    title_surf = font_small.render(title, True, title_color)
    screen.blit(title_surf, title_surf.get_rect(midtop=(x + CEMETERY_WIDTH // 2, BOARD_MARGIN + 4)))
    title_h = title_surf.get_height() + 8

    # 行数：仅显示有阵亡的棋子种类
    rows = [pt for pt in order if counts.get(pt, 0) > 0]
    if not rows:
        return

    row_h = CEM_PIECE_SIZE + CEM_ROW_GAP
    total_h = len(rows) * row_h - CEM_ROW_GAP
    avail_h = area_h - title_h
    start_y = BOARD_MARGIN + title_h + max(0, (avail_h - total_h) // 2)

    for i, pt in enumerate(rows):
        n = counts[pt]
        show = min(n, CEM_COLS)
        # 该行总宽度（show 枚棋子）
        row_w = show * CEM_PIECE_SIZE + (show - 1) * CEM_COL_GAP
        row_x = x + (CEMETERY_WIDTH - row_w) // 2
        row_y = start_y + i * row_h

        last_rect = None
        for k in range(show):
            cx = row_x + k * (CEM_PIECE_SIZE + CEM_COL_GAP)
            piece_rect = pygame.Rect(cx, row_y, CEM_PIECE_SIZE, CEM_PIECE_SIZE)
            img = piece_imgs.get(pt)
            if img is not None:
                scaled = pygame.transform.smoothscale(img, (CEM_PIECE_SIZE, CEM_PIECE_SIZE))
                screen.blit(scaled, piece_rect)
            else:
                color = COLOR_RED if is_red(pt) else COLOR_BLACK
                pygame.draw.circle(screen, (250, 240, 220), piece_rect.center, CEM_PIECE_SIZE // 2 - 2)
                pygame.draw.circle(screen, color, piece_rect.center, CEM_PIECE_SIZE // 2 - 2, 2)
                name = PIECE_NAMES.get(pt, "?")
                surf = font_small.render(name, True, color)
                screen.blit(surf, surf.get_rect(center=piece_rect.center))
            last_rect = piece_rect

        # 右下角数字角标：同种棋子阵亡数 >= 2 时显示 xN
        if n >= 2 and last_rect is not None:
            badge_text = f"x{n}"
            badge_surf = font_small.render(badge_text, True, (255, 255, 255))
            pad = 4
            badge_w = badge_surf.get_width() + pad * 2
            badge_h = badge_surf.get_height() + 2
            badge_rect = pygame.Rect(0, 0, badge_w, badge_h)
            badge_rect.bottomright = (last_rect.right, last_rect.bottom)
            pygame.draw.rect(screen, (200, 50, 50), badge_rect, border_radius=6)
            pygame.draw.rect(screen, (255, 255, 255), badge_rect, 1, border_radius=6)
            screen.blit(badge_surf, badge_surf.get_rect(center=badge_rect.center))


def cell_rect(r, c):
    x = BOARD_LEFT + c * CELL_SIZE
    y = BOARD_MARGIN + r * CELL_SIZE
    return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)


def pos_from_mouse(mx, my):
    if not (BOARD_LEFT <= mx < BOARD_LEFT + BOARD_WIDTH * CELL_SIZE):
        return None
    if not (BOARD_MARGIN <= my < BOARD_MARGIN + BOARD_HEIGHT * CELL_SIZE):
        return None
    c = (mx - BOARD_LEFT) // CELL_SIZE
    r = (my - BOARD_MARGIN) // CELL_SIZE
    return int(r), int(c)


def draw_board(screen, env, selected, target_set, last_move_origin, fonts, status_lines, assets=None, animation=None):
    if assets is not None and assets.get("background") is not None:
        screen.blit(assets["background"], (0, 0))
    else:
        screen.fill(COLOR_BG)
    font_piece, font_label, font_small = fonts

    # 棋盘背景
    board_rect = pygame.Rect(BOARD_LEFT, BOARD_MARGIN,
                             BOARD_WIDTH * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE)
    pygame.draw.rect(screen, COLOR_BOARD_LINE, board_rect.inflate(8, 8))

    piece_imgs = assets["pieces"] if assets else {}
    back_img = assets["back"] if assets else None
    tap_img = assets["tap"] if assets else None

    # 左/右两侧墓地：红方阵亡放左侧，黑方阵亡放右侧
    captured = compute_captured(env)
    red_counts = {pt: captured.get(pt, 0) for pt in CEMETERY_RED_ORDER}
    black_counts = {pt: captured.get(pt, 0) for pt in CEMETERY_BLACK_ORDER}

    left_x = BOARD_MARGIN
    right_x = BOARD_LEFT + BOARD_WIDTH * CELL_SIZE + CEMETERY_GAP
    draw_cemetery(screen, left_x, "红方阵亡", CEMETERY_RED_ORDER, red_counts,
                  piece_imgs, font_piece, font_small, COLOR_RED)
    draw_cemetery(screen, right_x, "黑方阵亡", CEMETERY_BLACK_ORDER, black_counts,
                  piece_imgs, font_piece, font_small, COLOR_BLACK)

    anim_from = animation["from"] if animation else None
    anim_to = animation["to"] if animation else None

    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            rect = cell_rect(r, c)
            base_color = COLOR_CELL
            if selected == (r, c):
                base_color = COLOR_CELL_HL
            elif (r, c) in target_set:
                base_color = COLOR_CELL_TARGET
            pygame.draw.rect(screen, base_color, rect)
            pygame.draw.rect(screen, COLOR_BOARD_LINE, rect, 1)

            if last_move_origin == (r, c) and tap_img is not None:
                screen.blit(tap_img, rect)

            piece = env.board[r][c]
            if animation and ((r, c) == anim_from or (r, c) == anim_to):
                piece = PieceType.EMPTY

            if piece == PieceType.UNFLIPPED:
                if back_img is not None:
                    img_rect = back_img.get_rect(center=rect.center)
                    screen.blit(back_img, img_rect)
                else:
                    cx_, cy_ = rect.center
                    pygame.draw.circle(screen, COLOR_UNFLIPPED, (cx_, cy_), CELL_SIZE // 2 - 8)
                    pygame.draw.circle(screen, (60, 40, 20), (cx_, cy_), CELL_SIZE // 2 - 8, 2)
            elif piece == PieceType.EMPTY:
                pass
            else:
                img = piece_imgs.get(piece)
                if img is not None:
                    img_rect = img.get_rect(center=rect.center)
                    screen.blit(img, img_rect)
                else:
                    cx_, cy_ = rect.center
                    color = COLOR_RED if is_red(piece) else COLOR_BLACK
                    pygame.draw.circle(screen, (250, 240, 220), (cx_, cy_), CELL_SIZE // 2 - 8)
                    pygame.draw.circle(screen, color, (cx_, cy_), CELL_SIZE // 2 - 8, 3)
                    name = PIECE_NAMES.get(piece, "?")
                    surf = font_piece.render(name, True, color)
                    screen.blit(surf, surf.get_rect(center=(cx_, cy_)))

    if animation:
        piece = animation["piece"]
        from_rect = cell_rect(*animation["from"])
        to_rect = cell_rect(*animation["to"])
        t = min(1.0, (time.time() - animation["start_time"]) / MOVE_ANIMATION_DURATION)
        x = from_rect.centerx + (to_rect.centerx - from_rect.centerx) * t
        y = from_rect.centery + (to_rect.centery - from_rect.centery) * t
        img = piece_imgs.get(piece)
        if img is not None:
            screen.blit(img, img.get_rect(center=(x, y)))
        else:
            color = COLOR_RED if is_red(piece) else COLOR_BLACK
            pygame.draw.circle(screen, (250, 240, 220), (int(x), int(y)), CELL_SIZE // 2 - 8)
            pygame.draw.circle(screen, color, (int(x), int(y)), CELL_SIZE // 2 - 8, 3)
            name = PIECE_NAMES.get(piece, "?")
            surf = font_piece.render(name, True, color)
            screen.blit(surf, surf.get_rect(center=(x, y)))

    # 侧栏
    side_x = right_x + CEMETERY_WIDTH + BOARD_MARGIN
    y = BOARD_MARGIN
    title = font_label.render("对局信息", True, COLOR_TEXT)
    screen.blit(title, (side_x, y))
    y += 40
    for line, color in status_lines:
        surf = font_small.render(line, True, color)
        screen.blit(surf, (side_x, y))
        y += 28


# ---------------- 主循环 ----------------
def main():
    pygame.init()
    pygame.display.set_caption("翻翻棋 FlipZero")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    assets = load_assets()

    models = discover_models()

    font_piece = load_font(40, bold=True)
    font_label = load_font(22, bold=True)
    font_small = load_font(18)
    fonts = (font_piece, font_label, font_small)

    # 按模型路径缓存 agent，避免每次重选都重新加载模型
    agent_cache = {}

    font_loading = load_font(28, bold=True)

    def _show_loading(red_model, black_model, loaded_box):
        """在后台线程里创建 agent；主线程同时渲染加载画面保持窗口响应。"""
        def _load():
            loaded_box["red"] = get_agent(red_model, agent_cache)
            loaded_box["black"] = get_agent(black_model, agent_cache)
            loaded_box["done"] = True

        t = threading.Thread(target=_load, daemon=True)
        t.start()

        while not loaded_box.get("done"):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None, None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None, None
            screen.fill(COLOR_BG)
            msg = font_loading.render("AI 加载中（启动 worker 进程）...", True, COLOR_TEXT)
            screen.blit(msg, msg.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)))
            pygame.display.flip()
            clock.tick(30)

        t.join(timeout=1.0)
        return loaded_box.get("red"), loaded_box.get("black")

    app_running = True
    while app_running:
        choice = select_players(screen, models, assets)
        if choice is None:
            break
        red_model, black_model = choice

        loaded_box = {}
        red_agent, black_agent = _show_loading(red_model, black_model, loaded_box)
        if red_agent is None and black_agent is None and red_model is not None:
            # 用户在加载期间退出
            break

        result = _play_one_match(screen, clock, fonts, assets, red_agent, black_agent)
        if result == "quit":
            app_running = False
        # result == "menu" 则继续外层循环，重新进入菜单

    pygame.quit()


def _play_one_match(screen, clock, fonts, assets, red_agent, black_agent):
    """进行一局对局。返回 'menu'（回到选择菜单）或 'quit'（退出程序）。

    red_agent / black_agent 为 GuiDeepAgent 或 None（None 表示该侧由人类操作）。
    """
    env = GameEnv()
    action_history = []
    selected = None  # (r,c) 已翻自己的棋子
    target_map = {}  # {(to_r,to_c): Action}
    last_move_origin = None  # 上一步移动/攻击前棋子的原位置
    move_animation = None
    last_ai_action_time = 0.0
    ai_think_delay = 0.3  # 秒，让玩家看清
    # 异步 MCTS：AI 思考在后台线程跑，主线程继续渲染窗口避免卡死
    ai_thread = None
    ai_result_box = [None]
    ai_start_time = 0.0

    def is_ai_turn():
        if env.done:
            return False
        if env.current_player == 0:
            return red_agent is not None
        return black_agent is not None

    def reset_selection():
        nonlocal selected, target_map
        selected = None
        target_map = {}

    def build_targets_for(r, c):
        """根据已选的自己已翻棋子位置，构造目标 -> Action 映射。

        注意：炮不能自由 MOVE，只能通过隔一个棋子的 ATTACK 攻击对方；
        合法动作由环境侧的 MoveGenerator 生成，这里仅做过滤展示。
        """
        legals = env.get_legal_actions()
        piece = env.board[r][c]
        is_cannon = piece in (PieceType.RED_CANNON, PieceType.BLACK_CANNON)
        m = {}
        for a in legals:
            if a.action_type == ActionType.FLIP:
                continue
            if a.from_row != r or a.from_col != c:
                continue
            # 炮：仅保留 ATTACK（防御性，理论上 MoveGenerator 已不会生成炮的 MOVE）
            if is_cannon and a.action_type != ActionType.ATTACK:
                continue
            m[(a.to_row, a.to_col)] = a
        return m

    def find_flip_action(r, c):
        for a in env.get_legal_actions():
            if a.action_type == ActionType.FLIP and a.to_row == r and a.to_col == c:
                return a
        return None

    def own_piece_at(player, r, c):
        p = env.board[r][c]
        if p in (PieceType.EMPTY, PieceType.UNFLIPPED):
            return False
        if player == 0:
            return is_red(p)
        return is_black(p)

    def apply_action(action):
        nonlocal last_ai_action_time, last_move_origin, move_animation
        moving_piece = None
        if action.action_type in (ActionType.MOVE, ActionType.ATTACK):
            last_move_origin = (action.from_row, action.from_col)
            moving_piece = env.board[action.from_row][action.from_col]
        else:
            last_move_origin = None
        action_history.append(action)
        env.step(action)
        if moving_piece is not None:
            move_animation = {
                "piece": moving_piece,
                "from": (action.from_row, action.from_col),
                "to": (action.to_row, action.to_col),
                "start_time": time.time(),
            }
        else:
            move_animation = None
        reset_selection()
        last_ai_action_time = time.time()

    while True:
        if move_animation and time.time() - move_animation["start_time"] >= MOVE_ANIMATION_DURATION:
            move_animation = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                elif event.key == pygame.K_r and env.done:
                    # 终局后按 R 返回菜单（重新选择先后手）
                    return "menu"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if env.done:
                    # 终局后点击任意处返回菜单（R 键的兜底，避免输入法/焦点问题导致 R 无效）
                    return "menu"
                if is_ai_turn() or move_animation:
                    continue
                mx, my = event.pos
                pos = pos_from_mouse(mx, my)
                if pos is None:
                    continue
                r, c = pos
                cell = env.board[r][c]
                player = env.current_player

                # 已选了己方棋子，再次点击：判断是目标 / 切换选中 / 取消
                if selected is not None:
                    if (r, c) in target_map:
                        apply_action(target_map[(r, c)])
                        continue
                    if (r, c) == selected:
                        reset_selection()
                        continue
                    if own_piece_at(player, r, c):
                        selected = (r, c)
                        target_map = build_targets_for(r, c)
                        continue
                    # 点击未翻棋子也可直接翻
                    if cell == PieceType.UNFLIPPED:
                        a = find_flip_action(r, c)
                        if a is not None:
                            apply_action(a)
                        continue
                    reset_selection()
                else:
                    if cell == PieceType.UNFLIPPED:
                        a = find_flip_action(r, c)
                        if a is not None:
                            apply_action(a)
                    elif own_piece_at(player, r, c):
                        selected = (r, c)
                        target_map = build_targets_for(r, c)

        # AI 行动（异步：MCTS 在后台线程跑，主线程继续渲染窗口避免卡死）
        if not env.done and not move_animation and is_ai_turn():
            agent = red_agent if env.current_player == 0 else black_agent
            # 启动一次新的思考
            if (agent is not None and ai_thread is None
                    and time.time() - last_ai_action_time >= ai_think_delay):
                infoset = InfoSet(player=env.current_player, game_env=env,
                                  action_history=action_history, perfect_info=False)
                history_snapshot = list(action_history)
                result_box = [None]

                def _think():
                    try:
                        result_box[0] = agent.act(infoset, env, history_snapshot)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        result_box[0] = None

                ai_result_box = result_box
                ai_start_time = time.time()
                ai_thread = threading.Thread(target=_think, daemon=True)
                ai_thread.start()

            # 思考结束，取出动作
            if ai_thread is not None and not ai_thread.is_alive():
                action = ai_result_box[0]
                ai_thread = None
                ai_result_box = [None]
                if action is None or env.done:
                    last_ai_action_time = time.time()
                else:
                    apply_action(action)

        # 状态文字
        cur = env.current_player
        cur_text = "红方" if cur == 0 else "黑方"
        cur_color = COLOR_RED if cur == 0 else COLOR_BLACK
        red_role = "AI" if red_agent is not None else "人类"
        black_role = "AI" if black_agent is not None else "人类"

        status = [
            (f"红方({red_role}) 分数: {env.red_score}", COLOR_RED),
            (f"黑方({black_role}) 分数: {env.black_score}", COLOR_BLACK),
            (f"和棋计数: {env.peace_steps}/20", COLOR_TEXT),
            (f"未翻棋子: {len(env.unflipped_positions)}", COLOR_TEXT),
        ]
        if env.done:
            if env.winner == 0:
                msg, color = "红方胜！点击任意位置返回菜单", COLOR_RED
            elif env.winner == 1:
                msg, color = "黑方胜！点击任意位置返回菜单", COLOR_BLACK
            else:
                msg, color = "和棋！点击任意位置返回菜单", COLOR_TEXT
            status.append(("", COLOR_TEXT))
            status.append((msg, color))
        else:
            status.append(("", COLOR_TEXT))
            status.append((f"当前: {cur_text}", cur_color))
            if ai_thread is not None:
                elapsed = time.time() - ai_start_time
                status.append((f"AI 思考中... {elapsed:.1f}s / {MCTS_THINK_TIME:.0f}s", COLOR_TEXT))
            elif is_ai_turn():
                status.append(("AI 思考中...", COLOR_TEXT))

        target_set = set(target_map.keys())
        draw_board(screen, env, selected, target_set, last_move_origin, fonts, status, assets, move_animation)
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
