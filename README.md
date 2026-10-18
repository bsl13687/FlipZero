# FlipZero

## Description
FlipZero is a Deep Monte Carlo (DMC) reinforcement-learning trainer for Chinese dark chess (翻翻棋 / 暗棋), inspired by DouZero. The project is pure Python and depends on `torch`, `numpy`, and `pygame` (for the GUI only).

## Common commands

Install / develop:
```bash
pip install -r requirements.txt
pip install -e .          # registers the `flipzero` package
```

Train (writes checkpoints to `log/<xpid>/model_step_<N>.tar`):
```bash
python train.py --xpid flipzero --num_actors 12 --training_device cuda:0 --load_model 2>&1 | tee train_<xpid>_log.txt
```

Evaluate a trained model against a random opponent:
```bash
python evaluate.py --model_path log/flipzero/model_step_<N>.tar --num_games 1000
```

Run the GUI (human vs. human, human vs. AI, or AI vs. AI; loads the latest `.tar` under `log/flipzero/` automatically):
```bash
python game.py
```


Useful training flags (see `flipzero/dmc/arguments.py` for the full list):
- `--num_actors` (It is best to configure the number of CPU virtual cores), `--num_buffers` (50), `--num_threads` (4) — concurrency knobs
- `--training_device` (`cuda:0` or `cpu`) and `--actor_device_cpu` (actors run on CPU by default)
- `--unroll_length` (100) — episode-buffer chunk size
- `--save_interval` (500000 steps), `--max_checkpoints` (10) — checkpoint rotation
- `--exp_epsilon` (0.01) — random-action probability during self-play

## Architecture

### Training loop (Actor–Learner, `flipzero/dmc/dmc.py`)
- One **learner** model lives on `training_device`. `flags.num_actors` **actor models** share memory (`share_memory_()`) and run in spawned subprocesses (`torch.multiprocessing`, `mp.get_context('spawn')`), each running self-play via `utils.act`.
- Actors pull a free buffer slot from `free_queue`, fill it with a chunk of episode steps (`x_no_action`, `action`, `z`, `target = episode_return`), and push `(buf_idx, n)` onto `full_queue`.
- `flags.num_threads` learner threads consume `full_queue`, slice the buffer into `batch_size` mini-batches, run MSE regression of `value → episode_return`, RMSprop-step the learner, then sync weights back into every actor model under `lock`.
- Shared progress/loss counters are passed via attributes on `flags` (`flags._episode_count`, `_training_steps`, `_loss_sum`, `_loss_count`, `_actor_models`). This is mutable state on the args namespace — read it before relying on `flags` being immutable.
- Checkpoints save `model_state_dict`, `optimizer_state_dict`, `steps`, `episodes`. `_load_checkpoint` picks the file with the largest step number; old ones get pruned to `max_checkpoints`. The GUI/eval path uses `weights_only=True`, so checkpoint files for inference must be loadable that way.

### Model (`flipzero/dmc/models.py`)
- `FlipZeroModel`: `LSTM(Z_DIM → 128)` over the last `HISTORY_LENGTH=10` action encodings, concat with `[x_no_action (530) | action (67)]`, then a 5-layer 512-wide MLP → scalar value.
- The model returns a value for **one (state, action) pair** — there is no policy head. Action selection (in `utils._select_action` and `evaluation.deep_agent.DeepAgent.act`) iterates over every legal action, evaluates each, and picks the argmax. This loop is the per-move bottleneck; if you change `forward`, preserve the ability to be called per-action with batch dim 1.
- Forward handles three input shapes for `z` (4-D `[B,T,H,F]`, 3-D `[B,H,F]`, or already-flat) — keep that flexibility when editing.

### Game / environment
- `flipzero/env/game.py`: `GameEnv` is the rules engine. 8×4 board, 32 pieces, capture by rank with the special soldier-beats-king rule (`can_capture`). `PieceType.UNFLIPPED` is its own state — `FLIP` actions reveal `true_board[r][c]` into `board[r][c]`.
- **Cannon (炮) is special**: it cannot `MOVE`, only `ATTACK`, and only by jumping over **exactly one** non-empty screen piece on the same row/column. `step()` and `_is_valid_cannon_attack` enforce this; `move_generator._add_cannon_attack_actions` is the only generator that emits cannon actions. If you touch movement code, keep these two paths in sync.
- Win conditions (`_check_winner`): opponent score ≤ 0 (win), `peace_steps > MAX_DRAW_STEP=20` (draw, `winner=-2`), or current player has no legal actions (loss for current). Reward is `±1` / `0` only at terminal states — DMC uses Monte Carlo returns, no bootstrapping.
- `flipzero/env/env.py`: feature encoding. Constants there (`X_NO_ACTION_DIM`, `ACTION_DIM`, `Z_DIM`, `HISTORY_LENGTH`, `BOARD_FEATURE_DIM`) are imported by both the model and the buffers — change them in **one** place; everything reshapes from these.
- `Env.__init__(perfect_info=False)` is now the default everywhere (training self-play, `evaluate.py`, rank promotion, GUI) — actors see only flipped pieces plus honest belief features (`belief_features`: normalized counts of each piece type still hidden under unflipped tiles). `--perfect_info` still exists for optional oracle evaluation.

### Inference paths
- Training-time action selection: `flipzero/dmc/utils.py::_select_action` (with ε-greedy on `flags.exp_epsilon`).
- Evaluation: `flipzero/evaluation/deep_agent.py::DeepAgent` (loads raw `state_dict` with `weights_only=True`).
- GUI: `game.py` reuses `FlipZeroModel` directly, loads the latest `model_step_*.tar` from `log/flipzero/`, and calls the same per-action argmax loop.
