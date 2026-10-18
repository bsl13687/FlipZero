import torch
import torch.multiprocessing as mp
import numpy as np
import logging

from flipzero.env.env import Env, action2array, X_NO_ACTION_DIM, ACTION_DIM, Z_DIM, HISTORY_LENGTH
from flipzero.dmc.models import Model

logger = logging.getLogger(__name__)


def create_buffers(flags, ctx):
    """创建共享 buffer。每个 buf_idx 恰好持有 T 步连续数据（可能跨 episode 边界）。

    learner 端会一次取出 batch_size 个 idx 拼成 [T, B, ...]，再 flatten 成 [T*B, ...]
    送入网络，等价于 DouZero 的 mini-batch 构造方式。
    """
    T = flags.unroll_length
    buffers = {}
    buffers['x_no_action'] = torch.zeros((flags.num_buffers, T, X_NO_ACTION_DIM), dtype=torch.float32).share_memory_()
    buffers['action'] = torch.zeros((flags.num_buffers, T, ACTION_DIM), dtype=torch.float32).share_memory_()
    buffers['z'] = torch.zeros((flags.num_buffers, T, HISTORY_LENGTH, Z_DIM), dtype=torch.float32).share_memory_()
    buffers['target'] = torch.zeros((flags.num_buffers, T), dtype=torch.float32).share_memory_()
    # done[t] = True 表示该步是某局对弈的最后一步；用于 learner 区分真实终局回报与中间步的广播 target。
    buffers['done'] = torch.zeros((flags.num_buffers, T), dtype=torch.bool).share_memory_()

    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()
    for i in range(flags.num_buffers):
        free_queue.put(i)

    return buffers, free_queue, full_queue


def act(i, device, model, buffers, free_queue, full_queue, flags):
    # 关键性能设置：必须限制每个 actor 的 torch CPU 线程数。
    # torch 默认用满物理核（本机 6），N 个 actor 就会开 6N 个 OpenMP 线程，
    # 在 12 核机器上严重超订；而 actor 每次只做「LSTM + 若干小 Linear」这种
    # 微秒级算子，线程池同步开销远大于计算本身，实测会慢 ~80x。
    torch.set_num_threads(max(1, getattr(flags, 'actor_num_threads', 1)))
    torch.manual_seed(flags.seed + i)
    np.random.seed(flags.seed + i)
    env = Env(perfect_info=False)  # 自对弈与评估统一为不完美信息视角

    buf_x_no_action = buffers['x_no_action']
    buf_action = buffers['action']
    buf_z = buffers['z']
    buf_target = buffers['target']
    buf_done = buffers['done']
    T = flags.unroll_length
    episode_count = flags._episode_count
    training_steps = flags._training_steps
    ep_red_return_sum = flags._ep_red_return_sum
    ep_draw_count = flags._ep_draw_count
    ep_len_sum = flags._ep_len_sum
    ep_finished = flags._ep_finished

    eps0 = flags.exp_epsilon
    eps_min = getattr(flags, 'exp_epsilon_min', eps0)
    anneal_frames = max(1, getattr(flags, 'exp_epsilon_anneal_frames', 1))

    # actor 本地缓冲：跨 episode 持续累积，每凑够 T 步就 flush 出一个 buf_idx。
    local_x = []         # list[np.ndarray (X_NO_ACTION_DIM,)]
    local_action = []    # list[np.ndarray (ACTION_DIM,)]
    local_z = []         # list[np.ndarray (HISTORY_LENGTH, Z_DIM)]
    local_target = []    # list[float]
    local_done = []      # list[bool]

    try:
        while True:
            infoset = env.reset()
            episode_steps = []  # list of (x, action_enc, z, player)

            while not env.done:
                legal_actions = infoset.legal_actions
                if len(legal_actions) == 0:
                    break

                progress = min(1.0, training_steps.value / anneal_frames)
                cur_eps = eps0 + (eps_min - eps0) * progress
                selected_action, action_enc = _select_action(model, infoset, cur_eps, device)

                episode_steps.append((
                    infoset.x_no_action.copy(),
                    action_enc.copy(),
                    infoset.z.copy(),
                    infoset.player,
                ))

                infoset = env.step(selected_action)

            if len(episode_steps) == 0:
                continue

            # episode_return 始终是“红方视角”的胜负：红胜 +1, 黑胜 -1, 和棋 0。
            if env.winner == 0:
                episode_return = 1.0
                is_draw = False
            elif env.winner == 1:
                episode_return = -1.0
                is_draw = False
            else:
                # winner == -2 (peace_steps 触顶) 或其它非终结情况
                episode_return = 0.0
                is_draw = True

            # 上报终局统计（actor 端聚合，避免 done buffer 的赢家偏置）
            with ep_red_return_sum.get_lock():
                ep_red_return_sum.value += episode_return
            if is_draw:
                with ep_draw_count.get_lock():
                    ep_draw_count.value += 1
            with ep_len_sum.get_lock():
                ep_len_sum.value += len(episode_steps)
            with ep_finished.get_lock():
                ep_finished.value += 1

            # 把整局追加到本地缓冲：每步按视角翻转目标；最后一步 done=True。
            n_ep = len(episode_steps)
            for t, (x, a_enc, z, player) in enumerate(episode_steps):
                target = episode_return if player == 0 else -episode_return
                local_x.append(x)
                local_action.append(a_enc)
                local_z.append(z)
                local_target.append(target)
                local_done.append(t == n_ep - 1)

            with episode_count.get_lock():
                episode_count.value += 1

            # 凑够 T 步就 flush。一个 buf_idx 内可能横跨多局（done=True 不止一处也可能没有）。
            while len(local_target) >= T:
                buf_idx = free_queue.get()
                for t in range(T):
                    buf_x_no_action[buf_idx, t] = torch.from_numpy(local_x[t])
                    buf_action[buf_idx, t] = torch.from_numpy(local_action[t])
                    buf_z[buf_idx, t] = torch.from_numpy(local_z[t])
                    buf_target[buf_idx, t] = local_target[t]
                    buf_done[buf_idx, t] = local_done[t]
                full_queue.put(buf_idx)
                del local_x[:T]
                del local_action[:T]
                del local_z[:T]
                del local_target[:T]
                del local_done[:T]
    except Exception:
        # actor 异常退出前留下 traceback；主进程看门狗会检测到进程死亡并停训，
        # 避免 learner 线程永久阻塞在 full_queue.get() 上。
        import traceback
        traceback.print_exc()
        import os
        os._exit(1)


def _select_action(model, infoset, exp_epsilon, device):
    legal_actions = infoset.legal_actions
    action_encodings = infoset.action_encodings  # [N, ACTION_DIM]

    if len(legal_actions) == 1:
        return legal_actions[0], action_encodings[0]

    if np.random.random() < exp_epsilon:
        idx = np.random.randint(len(legal_actions))
        return legal_actions[idx], action_encodings[idx]

    n = len(legal_actions)
    x_no_action = torch.from_numpy(infoset.x_no_action).to(device).unsqueeze(0).expand(n, -1).contiguous()
    z = torch.from_numpy(infoset.z).to(device).unsqueeze(0).expand(n, -1, -1).contiguous()
    actions = torch.from_numpy(action_encodings).to(device)

    with torch.no_grad():
        values = model.get_model()(x_no_action, actions, z, return_value=True)['value']  # [N]
    best_idx = int(torch.argmax(values).item())
    return legal_actions[best_idx], action_encodings[best_idx]


def get_batch(free_queue, full_queue, buffers, flags, lock):
    """从 full_queue 取 batch_size 个 idx，stack 成 [T, B, ...] 后归还槽位。"""
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)  # → [T, B, ...]
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch


def learn(model, optimizer, batch, device, flags):
    """单次 learner step。batch 是 [T, B, ...]，flatten 成 [T*B, ...] 后送网络。"""
    x_no_action = torch.flatten(batch['x_no_action'].to(device), 0, 1)
    action = torch.flatten(batch['action'].to(device), 0, 1)
    z = torch.flatten(batch['z'].to(device), 0, 1)
    target = torch.flatten(batch['target'].to(device), 0, 1)

    result = model.get_model()(x_no_action, action, z, return_value=True)
    logits = result['value']
    loss = ((logits - target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.get_model().parameters(), flags.max_grad_norm)
    optimizer.step()

    return loss.item(), float(grad_norm)


def batch_and_learn(device, model, buffers, free_queue, full_queue, optimizer, flags, lock, queue_lock):
    training_steps = flags._training_steps
    loss_sum = flags._loss_sum
    loss_count = flags._loss_count
    grad_norm_sum = flags._grad_norm_sum
    sync_interval = max(1, getattr(flags, 'actor_sync_interval', 1))
    steps_since_sync = 0
    T = flags.unroll_length
    B = flags.batch_size
    samples_per_step = T * B

    while True:
        batch = get_batch(free_queue, full_queue, buffers, flags, queue_lock)

        # learn 不需要 done 字段（episode 回报由 actor 端统计，避免 winner 偏置）
        learn_batch = {k: v for k, v in batch.items() if k != 'done'}

        with lock:
            loss, grad_norm = learn(model, optimizer, learn_batch, device, flags)
        with loss_sum.get_lock():
            loss_sum.value += loss
        with loss_count.get_lock():
            loss_count.value += 1
        with grad_norm_sum.get_lock():
            grad_norm_sum.value += grad_norm
        with training_steps.get_lock():
            training_steps.value += samples_per_step
        steps_since_sync += 1

        if steps_since_sync >= sync_interval:
            with lock:
                state = model.state_dict()
                for actor_model in getattr(flags, '_actor_models', []):
                    actor_model.load_state_dict(state)
            steps_since_sync = 0
