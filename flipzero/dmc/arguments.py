import argparse


def str2bool(v):
    """argparse 的 type=bool 对字符串 'False' 会解析成 True，这里显式解析常见写法。"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'布尔值 expected, got {v!r}')


def parser():
    p = argparse.ArgumentParser(description='FlipZero Training')

    p.add_argument('--xpid', type=str, default='flipzero', help='Experiment ID')
    p.add_argument('--savedir', type=str, default='log', help='Save directory')
    p.add_argument('--seed', type=int, default=12345, help='Random seed')
    p.add_argument('--save_interval', type=int, default=500000, help='Save checkpoint every N training steps')
    p.add_argument('--max_checkpoints', type=int, default=10, help='Maximum number of checkpoints to keep')
    p.add_argument('--load_model', action='store_true', default=False, help='Load model from checkpoint')

    p.add_argument('--gpu_device', type=str, default='0', help='GPU device for training')
    p.add_argument('--training_device', type=str, default='cuda:0', help='Device for learner (e.g. cuda:0 or cpu)')
    p.add_argument('--actor_device_cpu', type=str2bool, default=True, help='Actor uses CPU for inference (yes/no/true/false)')
    p.add_argument('--num_actors', type=int, default=5, help='Number of actor processes')
    p.add_argument('--num_buffers', type=int, default=64, help='Number of shared memory buffers')
    p.add_argument('--num_threads', type=int, default=1, help='Number of learner threads (>1 时各线程在 lock 下串行训练，通常无收益)')
    p.add_argument('--actor_sync_interval', type=int, default=50, help='每多少个 mini-batch 把 learner 权重广播给 actors')
    p.add_argument('--actor_num_threads', type=int, default=1, help='每个 actor 进程的 torch CPU 线程数（务必保持很小，默认 1；开大反而更慢）')

    p.add_argument('--total_frames', type=int, default=10_000_000_000, help='Total training frames')
    p.add_argument('--batch_size', type=int, default=32, help='Batch size')
    p.add_argument('--unroll_length', type=int, default=100, help='Unroll length (episode buffer size)')
    p.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    p.add_argument('--exp_epsilon', type=float, default=0.1, help='Initial epsilon for exploration')
    p.add_argument('--exp_epsilon_min', type=float, default=0.01, help='Final epsilon after annealing')
    p.add_argument('--exp_epsilon_anneal_frames', type=int, default=2_000_000, help='训练步数到此值时 epsilon 退火至 exp_epsilon_min')
    p.add_argument('--max_grad_norm', type=float, default=40.0, help='Max gradient norm for clipping')
    p.add_argument('--alpha', type=float, default=0.99, help='RMSprop smoothing constant')
    p.add_argument('--momentum', type=float, default=0, help='RMSprop momentum')
    p.add_argument('--epsilon', type=float, default=1e-5, help='RMSprop epsilon')

    p.add_argument('--print_interval', type=int, default=5, help='Print stats every N seconds')

    flags = p.parse_args()
    return flags
