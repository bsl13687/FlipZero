"""联机模式：把 FlipZero 模型接到 flip-game-server 上，与其它 AI 自动对弈。

用法（在 FlipZero 目录下）：
    python game.py --online --role join --user-id 2 --user-name test2
    python game.py --online --role create --user-id 2 --user-name test2

服务器 WebSocket 地址默认 ws://127.0.0.1:8010，棋盘为 8x4 翻翻棋。
本文件不依赖 pygame；模型权重默认取 log/flipzero 下最新的 model_step_*.tar。
"""
import argparse
import base64
import glob
import json
import os
import queue
import random
import socket
import struct
import sys
import threading
import time

from flipzero.env.game import (
    PieceType, ActionType, Action, GameEnv,
    BOARD_HEIGHT, BOARD_WIDTH,
)
from flipzero.env.env import InfoSet
from flipzero.evaluation.deep_agent import DeepAgent

CLOSED = object()

# 服务器棋子序号 -> 本项目棋子枚举
SERVER_TO_PY = {}
for _s in range(2, 9):      # 2-8 黑
    SERVER_TO_PY[_s] = _s + 7
for _s in range(9, 16):     # 9-15 红
    SERVER_TO_PY[_s] = _s - 7

# 初始每种棋子的数量（服务器序号）
FULL_COUNTS = {
    2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 5,
    9: 1, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 5,
}

PIECE_NAMES = {
    -1: "暗", 0: "空", 1: "暗",
    2: "黑将", 3: "黑士", 4: "黑象", 5: "黑马", 6: "黑车", 7: "黑炮", 8: "黑卒",
    9: "红帅", 10: "红士", 11: "红象", 12: "红马", 13: "红车", 14: "红炮", 15: "红兵",
}


class WsClient:
    """极简 WebSocket 客户端，仅依赖标准库，只支持 ws://。"""

    def __init__(self, url, user_id):
        raw = url
        if raw.startswith("ws://"):
            raw = raw[len("ws://"):]
        elif raw.startswith("wss://"):
            raise ValueError("当前内置客户端只支持 ws://")
        self.path = "/socket/" + str(user_id)
        if raw.find("/") >= 0:
            idx = raw.find("/")
            self.path = raw[idx:]
            raw = raw[:idx]
        if ":" in raw:
            host, port = raw.rsplit(":", 1)
            self.host, self.port = host, int(port)
        else:
            self.host, self.port = raw, 80
        self.sock = None
        self.rfile = None
        self.queue = queue.Queue()
        self.closed = threading.Event()
        self._write_lock = threading.Lock()

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=5)
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            "GET %s HTTP/1.1\r\n"
            "Host: %s:%d\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Key: %s\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        ) % (self.path, self.host, self.port, key)
        self.sock.sendall(request.encode("utf-8"))

        header = bytearray()
        while not header.endswith(b"\r\n\r\n"):
            b = self.sock.recv(1)
            if not b:
                raise ConnectionError("握手时连接被关闭")
            header.extend(b)
        first_line = bytes(header).split(b"\r\n", 1)[0].decode("latin-1")
        if "101" not in first_line:
            raise ConnectionError("WebSocket 握手失败: %s" % first_line)

        self.rfile = self.sock.makefile("rb")
        t = threading.Thread(target=self._read_loop, name="ws-reader", daemon=True)
        t.start()
        print("[FlipZero] WebSocket 已连接 %s:%d%s" % (self.host, self.port, self.path))

    def _read_loop(self):
        fragments = bytearray()
        try:
            while not self.closed.is_set():
                b1 = self.rfile.read(1)
                if not b1:
                    break
                b2 = self.rfile.read(1)
                if not b2:
                    break
                b1 = b1[0]
                b2 = b2[0]
                fin = bool(b1 & 0x80)
                opcode = b1 & 0x0F
                masked = bool(b2 & 0x80)
                length = b2 & 0x7F
                if length == 126:
                    length = struct.unpack("!H", self._read_exact(2))[0]
                elif length == 127:
                    length = struct.unpack("!Q", self._read_exact(8))[0]
                if length > 16 * 1024 * 1024:
                    raise ValueError("WebSocket 帧过大: %d" % length)
                mask = self._read_exact(4) if masked else None
                payload = bytearray(self._read_exact(length))
                if mask:
                    for i in range(len(payload)):
                        payload[i] ^= mask[i % 4]

                if opcode == 0x8:  # close
                    try:
                        self._send_frame(0x8, bytes(payload))
                    except Exception:
                        pass
                    break
                elif opcode == 0x9:  # ping
                    self._send_frame(0xA, bytes(payload))
                elif opcode == 0xA:  # pong
                    pass
                elif opcode in (0x1, 0x0):  # text / continuation
                    fragments.extend(payload)
                    if fin:
                        self.queue.put(fragments.decode("utf-8"))
                        fragments = bytearray()
        except Exception as exc:
            if not self.closed.is_set():
                print("[FlipZero] WebSocket 读取异常: %s" % exc)
        finally:
            self.closed.set()

    def _read_exact(self, n):
        chunks = []
        remaining = n
        while remaining > 0:
            data = self.rfile.read(remaining)
            if not data:
                raise EOFError("连接已关闭")
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def _send_frame(self, opcode, payload):
        mask = os.urandom(4)
        header = bytearray()
        header.append(0x80 | opcode)
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n <= 0xFFFF:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", n))
        header.extend(mask)
        masked = bytearray(payload)
        for i in range(len(masked)):
            masked[i] ^= mask[i % 4]
        with self._write_lock:
            if self.sock is None:
                return
            self.sock.sendall(bytes(header) + bytes(masked))

    def send_text(self, text):
        self._send_frame(0x1, text.encode("utf-8"))

    def next_message(self, timeout):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return CLOSED if self.closed.is_set() else None

    def close(self):
        self.closed.set()
        try:
            self._send_frame(0x8, b"")
        except Exception:
            pass
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass


class OnlineBot:
    def __init__(self, args):
        self.args = args
        self.ws = WsClient(args.server, args.user_id)
        self.my_id = str(args.user_id)
        self.my_name = args.user_name
        self.opp_id = str(args.opponent_id)
        self.role = args.role
        self.room_id = None
        self.my_color = None
        self.my_player = None
        self.my_turn = False
        self.ready_sent = False
        self.start_sent = False
        self.join_sent = False
        self.done = False
        self.game_finished = False
        self.move_count = 0
        self.connect_time = 0.0
        self.match_start_time = 0.0
        self.board = [[-1] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        self.black_score = 60
        self.red_score = 60
        self.peace_steps = 0
        self.dead_counts = {t: 0 for t in FULL_COUNTS}
        self.dead_raw = (None, None)
        self.history = []
        self.agent = DeepAgent(args.model_path, device=args.device)
        print("[FlipZero] 模型已加载: %s (device=%s)" % (args.model_path, args.device))

    # ---------------- 主流程 ----------------
    def run(self):
        print("[FlipZero] 联机模式启动: server=%s, userId=%s, name=%s, role=%s"
              % (self.args.server, self.my_id, self.my_name, self.role))
        try:
            self.ws.connect()
            self.connect_time = time.time()
            if self.role == "create":
                self.send_create_room()
            self.main_loop()
        except Exception as exc:
            print("[FlipZero] 运行失败: %s" % exc)
            import traceback
            traceback.print_exc()
        finally:
            self.ws.close()
            print("[FlipZero] 已退出")

    def main_loop(self):
        while not self.done:
            msg = self.ws.next_message(1.0)
            if msg is CLOSED:
                print("[FlipZero] 连接已关闭")
                break
            if msg is not None:
                self.handle_message(msg)
                continue
            now = time.time()
            if self.match_start_time == 0 and now - self.connect_time > 120:
                print("[FlipZero] 等待对手/开局超时")
                break
            if self.match_start_time > 0 and now - self.match_start_time > 3600:
                print("[FlipZero] 对局超时")
                break

    def handle_message(self, text):
        if self.args.debug_messages:
            with open(self.args.debug_messages, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        msg = json.loads(text)
        mtype = msg.get("type")
        if mtype == "online-rooms":
            self.on_online_rooms(msg)
        elif mtype == "create-room-success":
            self.room_id = msg["content"]["id"]
            print("[FlipZero] 已创建房间: %s" % self.room_id)
        elif mtype == "join-room-success":
            self.room_id = msg["content"]["id"]
            print("[FlipZero] 已加入房间: %s" % self.room_id)
            self.send_ready()
        elif mtype == "join-room-fail":
            print("[FlipZero] 加入房间失败: %s" % msg.get("content"))
            self.done = True
        elif mtype == "someone-join-room":
            print("[FlipZero] 对手已加入房间")
            self.send_ready()
        elif mtype == "info-ready":
            if self.role == "create" and msg.get("content") is True and not self.start_sent:
                self.start_sent = True
                print("[FlipZero] 对手已准备，开始对局")
                self.send_json({"type": "start-game", "roomId": self.room_id})
        elif mtype == "start-game":
            self.on_start_game(msg["content"])
        elif mtype == "play":
            self.on_play(msg, sync=False)
        elif mtype == "play-sync":
            self.on_play(msg, sync=True)
        elif mtype == "kicked":
            print("[FlipZero] 账号在别处登录，退出")
            self.done = True
        elif mtype in ("client-leave", "become-owner"):
            if self.match_start_time > 0 and not self.game_finished:
                print("[FlipZero] 对手离开，对局中止")
                self.done = True

    def on_online_rooms(self, msg):
        if self.room_id is not None or self.join_sent or self.role != "join":
            return
        rooms = msg.get("content") or []
        for room in rooms:
            if (str(room.get("playerId1")) == self.opp_id
                    and room.get("state") == "waiting"
                    and str(room.get("playerId2", "-1")) == "-1"):
                self.room_id = room["id"]
                self.join_sent = True
                print("[FlipZero] 发现房间 %s，申请加入" % self.room_id)
                self.send_json({
                    "type": "join-room",
                    "roomId": self.room_id,
                    "user": {"id": self.my_id, "name": self.my_name},
                })
                return

    def on_start_game(self, content):
        self.my_color = content.get("color")
        self.my_player = 0 if self.my_color == "red" else 1
        self.game_finished = False
        self.move_count = 0
        self.peace_steps = 0
        self.history = []
        self.dead_counts = {t: 0 for t in FULL_COUNTS}
        self.apply_state(content, is_move=False)
        self.match_start_time = time.time()
        print("[FlipZero] 对局开始，执%s，黑方先行"
              % ("红" if self.my_color == "red" else "黑"))
        self.print_board()
        self.my_turn = self.my_color == "black"
        if self.my_turn:
            self.play()

    def on_play(self, msg, sync):
        from_me = str(msg.get("from")) == self.my_id
        self.apply_state(msg, is_move=True)
        if msg.get("isGameOver"):
            self.finish_game(msg, last_mover_self=sync)
            return
        if not sync:
            # 对手走完，轮到我方
            self.my_turn = True
            self.play()
        else:
            self.my_turn = False

    def apply_state(self, msg, is_move):
        target_was_empty = False
        if is_move:
            tx = int(msg.get("x2", -1))
            ty = int(msg.get("y2", -1))
            if 0 <= ty < BOARD_HEIGHT and 0 <= tx < BOARD_WIDTH:
                target_was_empty = self.board[ty][tx] == 0

        data = msg.get("data")
        if data is not None:
            self.parse_board(data)
        self.black_score = int(msg.get("playerScore1", self.black_score))
        self.red_score = int(msg.get("playerScore2", self.red_score))
        self.parse_dead(msg.get("red_dead_pieces"), msg.get("black_dead_pieces"))

        if is_move:
            x1 = int(msg.get("x1", -1))
            if x1 == -1:
                self.peace_steps = 0
                action_type = ActionType.FLIP
            elif target_was_empty:
                self.peace_steps += 1
                action_type = ActionType.MOVE
            else:
                self.peace_steps = 0
                action_type = ActionType.ATTACK
            self.move_count += 1
            action = Action(
                action_type,
                from_row=int(msg.get("y1", -1)),
                from_col=int(msg.get("x1", -1)),
                to_row=int(msg.get("y2", -1)),
                to_col=int(msg.get("x2", -1)),
            )
            self.history.append(action)
            self.log_move(msg)

    def parse_board(self, data):
        raw = json.loads(data)
        for key, value in raw.items():
            r, c = key.split(",")
            r, c, value = int(r), int(c), int(value)
            self.board[r][c] = value

    def parse_dead(self, red_dead, black_dead):
        self.dead_raw = (red_dead, black_dead)
        self.dead_counts = {t: 0 for t in FULL_COUNTS}
        for text in (red_dead, black_dead):
            if not text:
                continue
            for part in str(text).split(","):
                part = part.strip()
                if not part:
                    continue
                t = int(part)
                if t in self.dead_counts:
                    self.dead_counts[t] += 1

    def log_move(self, msg):
        x1 = int(msg.get("x1", -1))
        y1 = int(msg.get("y1", -1))
        x2 = int(msg.get("x2", -1))
        y2 = int(msg.get("y2", -1))
        who = "我方" if str(msg.get("from")) == self.my_id else "对手"
        if x1 == -1:
            action = "翻 (%d,%d)" % (y2, x2)
        else:
            action = "行棋 (%d,%d) -> (%d,%d)" % (y1, x1, y2, x2)
        print("[FlipZero] 第%d手 %s(%s) %s  黑分=%d 红分=%d 和棋计数=%d"
              % (self.move_count, who, msg.get("color"), action,
                 self.black_score, self.red_score, self.peace_steps))
        if self.args.print_board:
            self.print_board()

    def print_board(self):
        lines = ["[FlipZero] 当前棋盘:"]
        for r in range(BOARD_HEIGHT):
            lines.append("  " + "\t".join(PIECE_NAMES.get(self.board[r][c], "?")
                                          for c in range(BOARD_WIDTH)))
        print("\n".join(lines))

    # ---------------- AI 决策 ----------------
    def build_env(self):
        remaining = dict(FULL_COUNTS)
        unflipped = []
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                t = self.board[r][c]
                if t in remaining:
                    remaining[t] -= 1
                if t == -1:
                    unflipped.append((r, c))
        for t in remaining:
            remaining[t] -= self.dead_counts.get(t, 0)

        hidden = []
        for t, n in remaining.items():
            hidden.extend([SERVER_TO_PY[t]] * max(0, n))
        random.shuffle(hidden)
        # 可见棋盘完全以服务器下发数据为准；暗子类型只用于模型信念特征。
        # 若按 dead_pieces 反推的数量与未翻位置数不一致，按未翻位置裁剪/补齐，
        # 不改变任何已翻开的棋子或走子合法性。
        if len(hidden) > len(unflipped):
            hidden = hidden[:len(unflipped)]
        elif len(hidden) < len(unflipped):
            hidden.extend([PieceType.RED_SOLDIER] * (len(unflipped) - len(hidden)))
        if self.args.debug_messages and len(hidden) != len(unflipped):
            print("[FlipZero] 调试: 暗子推断与未翻位置数不一致 red_dead=%r black_dead=%r"
                  % (self.dead_raw[0], self.dead_raw[1]))

        board = [[PieceType.EMPTY] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        true_board = [[PieceType.EMPTY] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        idx = 0
        unflipped_set = set(unflipped)
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                t = self.board[r][c]
                if t == -1:
                    board[r][c] = PieceType.UNFLIPPED
                    true_board[r][c] = hidden[idx] if idx < len(hidden) else PieceType.EMPTY
                    idx += 1
                elif t == 0:
                    board[r][c] = PieceType.EMPTY
                    true_board[r][c] = PieceType.EMPTY
                else:
                    piece = PieceType(SERVER_TO_PY[t])
                    board[r][c] = piece
                    true_board[r][c] = piece

        env = GameEnv.__new__(GameEnv)
        env.board = board
        env.true_board = true_board
        env.current_player = self.my_player
        env.red_score = self.red_score
        env.black_score = self.black_score
        env.peace_steps = self.peace_steps
        env.unflipped_positions = unflipped_set
        env.done = False
        env.winner = -1
        return env

    def play(self):
        if self.game_finished or self.done:
            return
        try:
            env = self.build_env()
            infoset = InfoSet(player=self.my_player, game_env=env,
                              action_history=self.history, perfect_info=False)
            start = time.time()
            action = self.agent.act(infoset)
            elapsed = (time.time() - start) * 1000
            if action is None:
                print("[FlipZero] 未找到可行走法，认输")
                self.send_json({
                    "type": "surrender", "roomId": self.room_id,
                    "from": self.my_id, "to": self.opp_id,
                })
                self.done = True
                return
            if action.action_type == ActionType.FLIP:
                x1, y1 = -1, -1
                x2, y2 = action.to_col, action.to_row
            else:
                x1, y1 = action.from_col, action.from_row
                x2, y2 = action.to_col, action.to_row
            self.send_json({
                "type": "play", "roomId": self.room_id,
                "from": self.my_id, "to": self.opp_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "color": self.my_color, "type1": -1, "type2": -1,
            })
            self.my_turn = False
            print("[FlipZero] 已思考 %.0fms，发送走法: %r" % (elapsed, action))
        except Exception as exc:
            print("[FlipZero] 计算/发送走法失败: %s" % exc)
            import traceback
            traceback.print_exc()
            self.done = True

    # ---------------- 终局 ----------------
    def finish_game(self, msg, last_mover_self):
        if self.game_finished:
            return
        self.game_finished = True
        draw = bool(msg.get("isDraw"))
        red_id = self.my_id if self.my_color == "red" else self.opp_id
        black_id = self.my_id if self.my_color == "black" else self.opp_id
        if draw:
            win_id = "0"
            result = "和棋"
        elif self.black_score <= 0:
            win_id = red_id
            result = "红方胜"
        elif self.red_score <= 0:
            win_id = black_id
            result = "黑方胜"
        else:
            win_id = self.my_id if last_mover_self else self.opp_id
            result = "%s 胜（对手无棋可走）" % win_id

        print("[FlipZero] 对局结束: %s  黑分=%d 红分=%d 总手数=%d"
              % (result, self.black_score, self.red_score, self.move_count))
        self.print_board()

        # 只让红方发送 winCheck，避免两端重复写数据库
        if self.my_color == "red":
            lose_id = "0" if win_id == "0" else (self.opp_id if win_id == self.my_id else self.my_id)
            self.send_json({
                "type": "winCheck", "roomId": self.room_id,
                "winId": win_id, "loseId": lose_id,
            })
            time.sleep(1.5)
        else:
            time.sleep(3.0)
        self.done = True

    # ---------------- 协议辅助 ----------------
    def send_create_room(self):
        self.send_json({
            "type": "create-room",
            "user": {"id": self.my_id, "name": self.my_name},
        })
        print("[FlipZero] 请求创建房间")

    def send_ready(self):
        if self.ready_sent:
            return
        self.ready_sent = True
        self.send_json({
            "type": "info-ready", "content": True,
            "from": self.my_id, "to": self.opp_id,
        })
        print("[FlipZero] 已准备")

    def send_json(self, data):
        self.ws.send_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))


def find_latest_model():
    base = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(base, "log", "flipzero", "model_step_*.tar"))

    def step(path):
        try:
            return int(os.path.basename(path).replace("model_step_", "").replace(".tar", ""))
        except ValueError:
            return -1

    return max(files, key=step) if files else None


def build_parser():
    parser = argparse.ArgumentParser(description="FlipZero 联机 AI 客户端")
    parser.add_argument("--online", action="store_true", help="启用联机模式")
    parser.add_argument("--server", default="ws://127.0.0.1:8010", help="服务器地址")
    parser.add_argument("--user-id", default="2", help="本账号用户 id")
    parser.add_argument("--user-name", default="test2", help="昵称")
    parser.add_argument("--opponent-id", default="1", help="对手用户 id")
    parser.add_argument("--role", default="join", choices=["create", "join"], help="创建/加入房间")
    parser.add_argument("--model-path", default=None, help="模型权重路径")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="推理设备")
    parser.add_argument("--print-board", action="store_true", help="每步后打印棋盘")
    parser.add_argument("--debug-messages", default=None, help="把收到的原始消息追加写入文件")
    return parser


def main(argv=None):
    args, _ = build_parser().parse_known_args(argv)
    if not args.model_path:
        args.model_path = find_latest_model()
    if not args.model_path:
        raise SystemExit("未找到模型权重，请通过 --model-path 指定")
    OnlineBot(args).run()


if __name__ == "__main__":
    main(sys.argv[1:])
