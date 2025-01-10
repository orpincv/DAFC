# from train import train
from test import test, test_cfar
import torch.multiprocessing as mp


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # train()
    # test()
    test_cfar("CA")
