import os
import sys

from flipzero.dmc.arguments import parser
from flipzero.dmc.dmc import train


def main():
    flags = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(flags.gpu_device)
    train(flags)


if __name__ == '__main__':
    main()
