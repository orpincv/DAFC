from train import train
from test import test, test_cfar, plot_cfar_results
import torch.multiprocessing as mp
import torch
from dataset import RadarDataset, cfar_collate_fn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # train()
    # print("\nRunning combined range-Doppler tests...")
    test_cfar()
    # pd_pfa_results = torch.load("CA_CFAR_Results/PD_vs_PFA_results.pt")
    # pd_scnr_results = torch.load("CA_CFAR_Results/PD_vs_SCNR_results.pt")
    # plot_cfar_results(pd_pfa_results, pd_scnr_results, "CA")
