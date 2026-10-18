"""Rank 晋级赛：新保存的模型通过逐级挑战 rank/3 -> rank/2 -> rank/1 决定去留。

规则：
- 新模型与 rank/3 下的模型比赛 10 轮（随机一方先手，之后每轮轮换先手，
  偶数轮保证双方各先手 5 次，抹平先手优势）。
  赢一局 +1 分，平局/输局不加分。得分相同加赛两轮（双方先后手各一次），
  仍相同则算新模型赢。
- 新模型赢 -> 替换 rank/3（删掉旧模型）；输 -> 流程结束。
- 然后 rank/3 与 rank/2 比赛，赢则交换位置；输则结束。
- 最后 rank/2 与 rank/1 比赛，赢则交换位置；输则结束。
- rank 目录为空视为空位，直接晋级。
- rank 目录下的模型保留原文件名（model_step_N.tar），方便查看训练步数。
"""

import os
import re
import glob
import random
import shutil
import logging
from datetime import datetime

import torch

from flipzero.env.env import Env
from flipzero.evaluation.deep_agent import DeepAgent
from flipzero.dmc.models import FlipZeroModel

logger = logging.getLogger(__name__)

# 晋级记录文件（当前工作目录下）
LOG_FILE = 'log.txt'


def _model_step(path):
    """从文件名 model_step_N.tar 提取步数 N，失败返回 0。"""
    if not path:
        return '0'
    m = re.match(r'model_step_(\d+)\.tar$', os.path.basename(path))
    return m.group(1) if m else '0'


def _log_promotion(rank_level, old_path, new_path):
    """按 [timestamp][rank_n][old_x][new_y]model_step_n 格式追加记录到 log.txt。"""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = (
        f"[{ts}][rank_{rank_level}]"
        f"[old_{_model_step(old_path)}][new_{_model_step(new_path)}]"
        f"{os.path.basename(new_path)}"
    )
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except OSError as e:
        logger.warning(f"[Rank] 写入 {LOG_FILE} 失败: {e}")
    logger.info(f"[Rank] 晋级记录: {line}")


class CheckpointAgent(DeepAgent):
    """从 checkpoint（.tar，含 model_state_dict）加载的 agent。"""

    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = FlipZeroModel()
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()


def _play_one_game(agent_a, agent_b, a_first, perfect_info=False):
    """下一局完整对局，返回获胜方（0=agent_a, 1=agent_b, -1=平局）。"""
    env = Env(perfect_info=perfect_info)
    infoset = env.reset()

    while not env.done:
        if (env.current_player == 0) == a_first:
            action = agent_a.act(infoset)
        else:
            action = agent_b.act(infoset)

        if action is None:
            break
        infoset = env.step(action)

    if env.winner == -2:
        return -1
    if env.winner == 0:
        return 0 if a_first else 1
    return 1 if a_first else 0


def _match_new_wins(new_path, old_path, device, rounds=10, perfect_info=False):
    """新模型 vs 旧模型比赛，返回新模型是否获胜（平分加赛一轮，仍平则新模型赢）。"""
    agent_new = CheckpointAgent(new_path, device=device)
    agent_old = CheckpointAgent(old_path, device=device)

    new_first = random.random() < 0.5
    score_new = 0
    score_old = 0

    def play_round():
        nonlocal new_first, score_new, score_old
        winner = _play_one_game(agent_new, agent_old, new_first, perfect_info=perfect_info)
        if winner == 0:
            score_new += 1
        elif winner == 1:
            score_old += 1
        new_first = not new_first  # 下一局轮换先手

    for _ in range(rounds):
        play_round()

    logger.info(f"[Rank] 比分 new={score_new} old={score_old} ({rounds} 轮)")
    if score_new == score_old:
        logger.info("[Rank] 平分，加赛两轮（先后手各一次，抹平先手优势）")
        play_round()
        play_round()
        logger.info(f"[Rank] 加赛后比分 new={score_new} old={score_old}")
        if score_new == score_old:
            logger.info("[Rank] 仍平分，算新模型赢")
            return True

    return score_new > score_old


def _rank_file(rank_dir):
    """返回 rank 目录下的模型文件路径（不存在则 None），并确保目录存在。"""
    if not os.path.exists(rank_dir):
        os.makedirs(rank_dir, exist_ok=True)
        return None
    files = [f for f in glob.glob(os.path.join(rank_dir, '*.tar')) if not f.endswith('.tmp')]
    return files[0] if files else None


def try_promote(new_model_path, rank_root, device='cpu', rounds=10):
    """晋级流程入口。new_model_path 是刚保存的 checkpoint。"""
    if not os.path.exists(new_model_path):
        logger.warning(f"[Rank] 新模型文件不存在: {new_model_path}，跳过晋级赛")
        return

    rank_dirs = [os.path.join(rank_root, str(i)) for i in (1, 2, 3)]

    # 第一步：挑战 rank/3
    r3 = _rank_file(rank_dirs[2])
    if r3 is None:
        logger.info("[Rank] rank/3 为空，新模型直接进入 rank/3")
        incumbent_wins = False
    else:
        logger.info(f"[Rank] 新模型 vs rank/3 ({os.path.basename(r3)})，共 {rounds} 轮")
        incumbent_wins = not _match_new_wins(new_model_path, r3, device, rounds)

    if incumbent_wins:
        logger.info("[Rank] 新模型不敌 rank/3，晋级流程结束")
        return

    # 新模型进入 rank/3（替换原有模型），保留原文件名（model_step_N.tar）方便查看训练步数
    if r3 is not None:
        os.remove(r3)
        logger.info(f"[Rank] 已删除原 rank/3 模型: {r3}")
    dst = os.path.join(rank_dirs[2], os.path.basename(new_model_path))
    _copy_file(new_model_path, dst)
    logger.info(f"[Rank] 新模型已保存到 rank/3: {dst}")
    _log_promotion(3, r3, new_model_path)

    # 第二步、第三步：rank/3 -> rank/2 -> rank/1
    for src_level, dst_level in ((3, 2), (2, 1)):
        src_dir = rank_dirs[src_level - 1]
        dst_dir = rank_dirs[dst_level - 1]
        src_model = _rank_file(src_dir)
        if src_model is None:
            logger.warning(f"[Rank] rank/{src_level} 下没有模型文件，晋级流程结束")
            return
        incumbent = _rank_file(dst_dir)

        if incumbent is None:
            logger.info(f"[Rank] rank/{dst_level} 为空，rank/{src_level} 直接升入")
        else:
            logger.info(f"[Rank] rank/{src_level} vs rank/{dst_level}，共 {rounds} 轮")
            if not _match_new_wins(src_model, incumbent, device, rounds):
                logger.info(f"[Rank] rank/{src_level} 不敌 rank/{dst_level}，晋级流程结束")
                return

        # 交换（或升入空位）：胜者升到 dst，败者降到 src，均保留原文件名
        tmp = os.path.join(src_dir, 'swap.tmp')
        if incumbent is not None:
            _move_file(incumbent, tmp)
        _move_file(src_model, os.path.join(dst_dir, os.path.basename(src_model)))
        if incumbent is not None:
            _move_file(tmp, os.path.join(src_dir, os.path.basename(incumbent)))
        logger.info(f"[Rank] rank/{src_level} 与 rank/{dst_level} 已交换")
        _log_promotion(dst_level, incumbent, src_model)

    logger.info("[Rank] 晋级流程完成")


def _copy_file(src, dst):
    shutil.copy2(src, dst)


def _move_file(src, dst):
    shutil.move(src, dst)
