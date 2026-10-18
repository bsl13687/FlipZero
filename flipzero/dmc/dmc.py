import os
import glob
import time
import logging
import threading
import torch
import torch.multiprocessing as mp

from flipzero.dmc.models import Model
from flipzero.dmc.utils import create_buffers, act, batch_and_learn
from flipzero.evaluation.rank_promotion import try_promote

logger = logging.getLogger(__name__)


def train(flags):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    if not os.path.exists(flags.savedir):
        os.makedirs(flags.savedir)

    checkpoint_dir = os.path.join(flags.savedir, flags.xpid)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    training_device = torch.device(flags.training_device)
    logger.info(f"Training device: {training_device}")

    model = Model(device=training_device)
    logger.info(f"Model created | params={sum(p.numel() for p in model.get_model().parameters()):,} | device={next(model.get_model().parameters()).device}")

    optimizer = torch.optim.RMSprop(
        model.get_model().parameters(),
        lr=flags.learning_rate,
        alpha=flags.alpha,
        momentum=flags.momentum,
        eps=flags.epsilon,
    )

    # 跨进程共享原语（Value / SimpleQueue）必须与 actor 子进程使用同一个
    # start method 的 context。用默认 context（Linux 下为 fork）创建、再交给
    # spawn 子进程使用，子进程首次获取其 SemLock 时会 segfault。
    ctx = mp.get_context('spawn')

    buffers, free_queue, full_queue = create_buffers(flags, ctx)
    if flags.num_buffers < 2 * flags.batch_size:
        logger.warning(
            f"num_buffers={flags.num_buffers} 小于 2*batch_size={2*flags.batch_size}，"
            f"learner 取 batch 时可能阻塞 actor。建议 num_buffers ≥ 2*batch_size。"
        )
    logger.info(f"Buffers created: {flags.num_buffers} buffers, unroll_length={flags.unroll_length}, "
                f"effective_batch=T*B={flags.unroll_length}*{flags.batch_size}={flags.unroll_length*flags.batch_size}")

    actor_models = []
    if flags.actor_device_cpu:
        actor_device = torch.device('cpu')
    else:
        actor_device = torch.device(f'cuda:{flags.gpu_device}')

    for i in range(flags.num_actors):
        actor_model = Model(device=actor_device)
        actor_model.share_memory()
        actor_model.eval()
        actor_models.append(actor_model)
    flags._actor_models = [m.get_model() for m in actor_models]
    logger.info(f"Actor device: {actor_device} | {flags.num_actors} actor models created")

    episode_count = ctx.Value('i', 0)
    training_steps = ctx.Value('i', 0)
    loss_sum = ctx.Value('d', 0.0)
    loss_count = ctx.Value('i', 0)
    grad_norm_sum = ctx.Value('d', 0.0)
    # 由 actor 维护：每局结束累加一次。
    # ep_red_return: 红方视角胜负 ∈ {-1, 0, +1}，均值反映对称性是否健康（应趋近 0）。
    # ep_draw_count: 和棋数；ep_len_sum: 总步数；ep_finished: 完整 episode 数（与上面三个分母一致）。
    ep_red_return_sum = ctx.Value('d', 0.0)
    ep_draw_count = ctx.Value('i', 0)
    ep_len_sum = ctx.Value('i', 0)
    ep_finished = ctx.Value('i', 0)

    flags._episode_count = episode_count
    flags._training_steps = training_steps
    flags._loss_sum = loss_sum
    flags._loss_count = loss_count
    flags._grad_norm_sum = grad_norm_sum
    flags._ep_red_return_sum = ep_red_return_sum
    flags._ep_draw_count = ep_draw_count
    flags._ep_len_sum = ep_len_sum
    flags._ep_finished = ep_finished

    if flags.load_model:
        ep_val = [0]
        step_val = [0]
        _load_checkpoint(checkpoint_dir, model, optimizer, step_val, ep_val)
        for actor_model in actor_models:
            actor_model.load_state_dict(model.state_dict())
        training_steps.value = step_val[0]
        episode_count.value = ep_val[0]
        logger.info(f"Resumed from step {step_val[0]}, episodes {ep_val[0]}")

    lock = threading.Lock()

    actor_processes = []
    for i in range(flags.num_actors):
        p = ctx.Process(
            target=act,
            args=(i, actor_device, actor_models[i], buffers, free_queue, full_queue, flags),
            daemon=True,
        )
        p.start()
        actor_processes.append(p)
    logger.info(f"Started {flags.num_actors} actor processes")

    learner_threads = []
    queue_lock = threading.Lock()  # 仅 learner 线程间使用，不能挂到 flags 上（spawn 子进程会 pickle flags）
    for i in range(flags.num_threads):
        t = threading.Thread(
            target=batch_and_learn,
            args=(training_device, model, buffers, free_queue, full_queue, optimizer, flags, lock, queue_lock),
            daemon=True,
        )
        t.start()
        learner_threads.append(t)
    logger.info(f"Started {flags.num_threads} learner threads")

    logger.info(f"Training started | total_frames={flags.total_frames:,} | save_interval={flags.save_interval:,} | lr={flags.learning_rate}")

    start_time = time.time()
    last_log_time = time.time()
    last_steps = training_steps.value
    last_milestone = 0
    milestone_interval = flags.total_frames // 10
    last_saved_steps = _last_saved_steps(flags)

    try:
        while True:
            time.sleep(flags.print_interval)

            # 看门狗：任一 actor 进程死亡则停训，否则 learner 线程会永久阻塞在队列 get() 上
            dead_actors = [(i, p) for i, p in enumerate(actor_processes) if not p.is_alive()]
            if dead_actors:
                for i, _ in dead_actors:
                    logger.error(f"Actor process {i} died unexpectedly; aborting training.")
                break

            current_frames = training_steps.value
            frames_delta = current_frames - last_steps
            elapsed = time.time() - last_log_time
            fps = frames_delta / elapsed if elapsed > 0 else 0

            cur_loss_count = loss_count.value
            avg_loss = loss_sum.value / cur_loss_count if cur_loss_count > 0 else 0.0
            avg_grad_norm = grad_norm_sum.value / cur_loss_count if cur_loss_count > 0 else 0.0
            cur_finished = ep_finished.value
            avg_ep_return = ep_red_return_sum.value / cur_finished if cur_finished > 0 else 0.0
            draw_rate = ep_draw_count.value / cur_finished if cur_finished > 0 else 0.0
            avg_ep_len = ep_len_sum.value / cur_finished if cur_finished > 0 else 0.0
            progress = current_frames / flags.total_frames * 100
            elapsed_total = time.time() - start_time

            logger.info(
                f"[Train] Frames: {current_frames:,} | Progress: {progress:.2f}% | "
                f"FPS: {fps:.0f} | Episodes: {episode_count.value:,} | "
                f"Loss: {avg_loss:.4f} | GradNorm: {avg_grad_norm:.3f} | "
                f"EpReturn(red): {avg_ep_return:+.3f} | DrawRate: {draw_rate:.2f} | "
                f"EpLen: {avg_ep_len:.1f} (n={cur_finished}) | "
                f"Elapsed: {_format_time(elapsed_total)}"
            )
            loss_sum.value = 0.0
            loss_count.value = 0
            grad_norm_sum.value = 0.0
            ep_red_return_sum.value = 0.0
            ep_draw_count.value = 0
            ep_len_sum.value = 0
            ep_finished.value = 0
            last_log_time = time.time()
            last_steps = current_frames

            current_milestone = current_frames // milestone_interval if milestone_interval > 0 else 0
            if current_milestone > last_milestone and current_frames > 0:
                logger.info(f"=== Milestone: {current_milestone * 10}% reached at frames {current_frames:,} ===")
                last_milestone = current_milestone

            if current_frames >= flags.total_frames:
                logger.info(f"Training complete! Reached total frames {flags.total_frames:,}")
                break

            if current_frames > 0 and current_frames - last_saved_steps >= flags.save_interval:
                with lock:
                    saved_path = _save_checkpoint(checkpoint_dir, model, optimizer, current_frames, episode_count.value, flags)
                _cleanup_old_checkpoints(checkpoint_dir, flags.max_checkpoints)
                last_saved_steps = current_frames
                # 保存后触发 rank 晋级赛：新模型逐级挑战 rank/3 -> rank/2 -> rank/1
                rank_root = os.path.join(flags.savedir, 'rank')
                try_promote(saved_path, rank_root, device=str(training_device))

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")

    with lock:
        final_path = _save_checkpoint(checkpoint_dir, model, optimizer, training_steps.value, episode_count.value, flags)
    logger.info(f"Final checkpoint saved at frames {training_steps.value:,}")
    rank_root = os.path.join(flags.savedir, 'rank')
    try_promote(final_path, rank_root, device=str(training_device))

    for p in actor_processes:
        p.terminate()
    logger.info("All actor processes terminated.")


def _format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h{minutes}m{secs}s"
    elif minutes > 0:
        return f"{minutes}m{secs}s"
    else:
        return f"{secs}s"


def _last_saved_steps(flags):
    checkpoint_dir = os.path.join(flags.savedir, flags.xpid)
    pattern = os.path.join(checkpoint_dir, "model_step_*.tar")
    files = glob.glob(pattern)
    if not files:
        return 0
    steps = []
    for f in files:
        try:
            step = int(os.path.basename(f).replace("model_step_", "").replace(".tar", ""))
            steps.append(step)
        except ValueError:
            pass
    return max(steps) if steps else 0


def _save_checkpoint(checkpoint_dir, model, optimizer, steps, episodes, flags):
    filepath = os.path.join(checkpoint_dir, f"model_step_{steps}.tar")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'steps': steps,
        'episodes': episodes,
    }, filepath)
    logger.info(f"[Checkpoint] Saved: {filepath} (episodes={episodes:,})")
    return filepath


def _load_checkpoint(checkpoint_dir, model, optimizer, training_steps, episode_count):
    pattern = os.path.join(checkpoint_dir, "model_step_*.tar")
    files = glob.glob(pattern)
    if not files:
        logger.warning("No checkpoint found, starting from scratch.")
        return

    latest = max(files, key=lambda f: int(os.path.basename(f).replace("model_step_", "").replace(".tar", "")))
    checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    training_steps[0] = checkpoint['steps']
    episode_count[0] = checkpoint.get('episodes', 0)
    logger.info(f"[Checkpoint] Loaded: {latest} | steps={training_steps[0]:,} | episodes={episode_count[0]:,}")


def _cleanup_old_checkpoints(checkpoint_dir, max_checkpoints):
    pattern = os.path.join(checkpoint_dir, "model_step_*.tar")
    files = glob.glob(pattern)
    if len(files) <= max_checkpoints:
        return

    files_with_steps = []
    for f in files:
        try:
            step = int(os.path.basename(f).replace("model_step_", "").replace(".tar", ""))
            files_with_steps.append((step, f))
        except ValueError:
            pass

    files_with_steps.sort(key=lambda x: x[0])

    to_delete = len(files_with_steps) - max_checkpoints
    for i in range(to_delete):
        os.remove(files_with_steps[i][1])
        logger.info(f"[Checkpoint] Removed old: {files_with_steps[i][1]}")
