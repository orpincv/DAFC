from train import train
from test import test, test_cfar, plot_cfar_results
import torch.multiprocessing as mp
import torch

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # train()
    pd_pfa_results, pd_scnr_results = test_cfar("CA")
    plot_cfar_results(pd_pfa_results, pd_scnr_results, "CA")
