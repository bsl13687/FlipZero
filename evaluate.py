import os
import argparse
import numpy as np

from flipzero.env.env import Env
from flipzero.evaluation.deep_agent import DeepAgent, RandomAgent


def evaluate(model_path, num_games=1000, device='cpu', perfect_info=False):
    agent = DeepAgent(model_path, device=device)
    random_agent = RandomAgent()

    wins = 0
    draws = 0
    losses = 0

    for _ in range(num_games):
        env = Env(perfect_info=perfect_info)
        infoset = env.reset()

        while not env.done:
            if env.current_player == 0:
                action = agent.act(infoset)
            else:
                action = random_agent.act(infoset)

            if action is None:
                break
            infoset = env.step(action)

        if env.winner == 0:
            wins += 1
        elif env.winner == -2:
            draws += 1
        else:
            losses += 1

    total = wins + draws + losses
    print(f"Evaluation Results ({num_games} games):")
    print(f"  Wins:   {wins} ({wins/total*100:.1f}%)")
    print(f"  Draws:  {draws} ({draws/total*100:.1f}%)")
    print(f"  Losses: {losses} ({losses/total*100:.1f}%)")

    return wins, draws, losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate FlipZero model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_games', type=int, default=1000, help='Number of evaluation games')
    parser.add_argument('--device', type=str, default='cpu', help='Device for inference')
    parser.add_argument('--perfect_info', action='store_true', help='Use perfect information')
    args = parser.parse_args()

    evaluate(args.model_path, args.num_games, args.device, args.perfect_info)
