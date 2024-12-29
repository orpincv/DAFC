import os

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from cfar import CACFAR, TMCFAR
from dataset import *
from model import DAFCRadarNet


class CombinedRadarTester:
    def __init__(self, range_model, doppler_model, device):
        """Initialize combined radar tester"""
        self.device = device
        self.range_model = range_model.to(device)
        self.doppler_model = doppler_model.to(device)
        # Create the neighborhood kernel for 2D convolution
        self.kernel = torch.ones(1, 1, 3, 3, device=device)
        self.R = generate_range_steering_matrix().to(device)
        self.V = generate_doppler_steering_matrix().to(device)

    def feed_forward(self, loader):
        """Run models once and get all predictions"""
        self.range_model.eval()
        self.doppler_model.eval()

        all_Y_r = []
        all_Y_v = []
        all_X_rv_proj = []
        all_Y_true = []

        with torch.no_grad():
            for X, Y in loader:
                X = X.to(self.device)

                # Get range and Doppler predictions
                Y_r = self.range_model(X)
                Y_v = self.doppler_model(X)

                # Get range-Doppler projection
                Z = torch.abs(self.R.H @ X @ self.V.conj())

                all_Y_r.append(Y_r)
                all_Y_v.append(Y_v)
                all_X_rv_proj.append(Z)
                all_Y_true.append(Y.to(self.device))

        # Concatenate all batches
        Y_r = torch.cat(all_Y_r, dim=0)
        Y_v = torch.cat(all_Y_v, dim=0)
        X_rv_proj = torch.cat(all_X_rv_proj, dim=0)
        Y_true = torch.cat(all_Y_true, dim=0)

        return Y_r, Y_v, X_rv_proj, Y_true

    @staticmethod
    def predict(Y_r, Y_v, X_rv_proj, threshold: float):
        """Apply threshold to predictions to get detections"""
        # Thresholding
        Y_r_binary = (Y_r > threshold).float()
        Y_v_binary = (Y_v > threshold).float()

        # Combine range and Doppler detections
        Y_rv = Y_r.unsqueeze(-1) @ Y_v.unsqueeze(-2)
        Y_rv_binary = Y_r_binary.unsqueeze(-1) @ Y_v_binary.unsqueeze(-2)

        # Final decision matrix
        U = X_rv_proj * Y_rv * Y_rv_binary
        Y_hat = (U / U.max() > threshold).float()

        return Y_hat

    def get_metrics(self, Y_hat, Y_true):
        """Evaluate detection performance for full dataset at once"""
        Y_true_extended = F.conv2d(Y_true.unsqueeze(1).float(), self.kernel, padding=1).squeeze(1)
        Y_true_extended = (Y_true_extended > 0)
        Y_hat_extended = F.conv2d(Y_hat.unsqueeze(1), self.kernel, padding=1).squeeze(1)
        Y_hat_extended = (Y_hat_extended > 0).float()

        # Calculate PFA (excluding target neighborhoods)
        valid_cells = (~Y_true_extended).float()
        pfa = (Y_hat * valid_cells).sum() / valid_cells.sum()

        # Calculate PD (only for frames with targets)
        n_targets = Y_true.sum(dim=(1, 2))
        frames_with_targets = (n_targets > 0)
        if frames_with_targets.any():
            detected = (Y_hat_extended * Y_true).sum(dim=(1, 2))
            pd = (detected[frames_with_targets] / n_targets[frames_with_targets]).mean()
        else:
            pd = torch.tensor(0.0, device=self.device)

        return pd.item(), pfa.item()

    def find_threshold(self, loader, target_pfa):
        """Find threshold for target PFA using binary search"""
        # Get all predictions once
        Y_r, Y_v, X_rv_proj, Y_true = self.feed_forward(loader)

        th = 0.5  # Start at 0.5
        step = 0.5
        cnt = 1
        pfa_res = 1.0
        rel_err = abs(pfa_res - target_pfa) / abs(target_pfa)

        while rel_err >= 0.01 and cnt < 20:
            Y_hat = self.predict(Y_r, Y_v, X_rv_proj, th)
            metrics = self.get_metrics(Y_hat, Y_true)

            pfa_res = metrics[0]
            rel_err = abs(pfa_res - target_pfa) / abs(target_pfa)

            step = step * 0.5
            if pfa_res > target_pfa:
                th += step
            else:
                th -= step

            cnt += 1

        print(f"Found threshold = {th:.4f}, PFA = {pfa_res:.6f} after {cnt} iterations")
        return th, pfa_res, cnt

    def evaluate_pd_pfa(self, nu, scnr=0):
        """Evaluate PD vs PFA for fixed SCNR"""
        # Create dataset
        test_dataset = ConcatDataset([RadarDataset(4096, n_targets=4, random_n_targets=False, nu=nu, scnr=scnr),
                                      RadarDataset(2048, n_targets=0, random_n_targets=False, nu=nu, scnr=scnr)])
        test_loader = DataLoader(test_dataset, batch_size=256, num_workers=2, persistent_workers=True,
                                 pin_memory=torch.cuda.is_available())

        # Get all predictions once
        Y_r, Y_v, X_rv_proj, Y_true = self.feed_forward(test_loader)

        # Store results for different target PFAs
        results = []
        target_pFAs = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3]

        for target_pfa in tqdm(target_pFAs):
            th, pfa_res, cnt = self.find_threshold(test_loader, target_pfa)
            print(f"Found threshold = {th:.4f}, PFA = {pfa_res:.6f}, after {cnt} iterations")

            Y_hat = self.predict(Y_r, Y_v, X_rv_proj, th)
            pd, pfa = self.get_metrics(Y_hat, Y_true)
            results.append((pd, pfa))

        pd_list, pfa_list = zip(*results)
        return np.array(pd_list), np.array(pfa_list)

    def evaluate_pd_scnr(self, nu, target_pfa=5e-4):
        """Evaluate PD vs SCNR for fixed PFA"""
        # First find threshold using a reference dataset (SCNR = 0)
        ref_dataset = ConcatDataset([RadarDataset(4096, n_targets=4, random_n_targets=False, nu=nu, scnr=0),
                                     RadarDataset(2048, n_targets=0, random_n_targets=False, nu=nu)])
        ref_loader = DataLoader(ref_dataset, batch_size=256, num_workers=2, persistent_workers=True,
                                pin_memory=torch.cuda.is_available())

        # Find threshold using binary search
        th, pfa_res, cnt = self.find_threshold(ref_loader, target_pfa)
        print(f"Found threshold = {th:.4f}, PFA = {pfa_res:.6f}, after {cnt} iterations")

        # Now evaluate for different SCNR values
        results = []
        scnr_range = np.arange(-40, 11, 5)

        for scnr in tqdm(scnr_range):
            # Create dataset for this SCNR
            test_dataset = ConcatDataset([RadarDataset(4096, n_targets=4, random_n_targets=False, nu=nu, scnr=scnr),
                                          RadarDataset(2048, n_targets=0, random_n_targets=False, nu=nu)])
            test_loader = DataLoader(test_dataset, batch_size=256, num_workers=2, persistent_workers=True,
                                     pin_memory=torch.cuda.is_available())

            # Get predictions
            Y_r, Y_v, X_rv_proj, Y_true = self.feed_forward(test_loader)

            # Evaluate using found threshold
            Y_hat = self.predict(Y_r, Y_v, X_rv_proj, th)
            pd, pfa = self.get_metrics(Y_hat, Y_true)

            results.append((pd, pfa))

        pd_list, pfa_list = zip(*results)
        return np.array(pd_list), np.array(pfa_list), scnr_range


class CFARTester:
    def __init__(self, detector, device):
        self.device = device
        self.detector = detector.to(self.device)
        self.kernel = torch.ones(1, 1, 3, 3, device=device)

    def feed_forward(self, loader):
        """Get detector output for entire dataset at once"""
        all_rd_maps = []
        all_Y_true = []

        for rd_maps, Y_true in loader:
            all_rd_maps.append(rd_maps)
            all_Y_true.append(Y_true)

        rd_maps = torch.cat(all_rd_maps, dim=0).to(self.device)
        Y_true = torch.cat(all_Y_true, dim=0).to(self.device)

        with torch.no_grad():
            detection_surface = self.detector(rd_maps)

        return detection_surface, Y_true

    def evaluate_metrics(self, detection_surface, Y_true, threshold):
        """Calculate PD and PFA for the entire dataset"""
        Y_hat = (detection_surface > threshold).float()

        Y_true_extended = F.conv2d(Y_true.unsqueeze(1).float(), self.kernel, padding=1).squeeze(1)
        Y_true_extended = (Y_true_extended > 0)
        Y_hat_extended = F.conv2d(Y_hat.unsqueeze(1), self.kernel, padding=1).squeeze(1)
        Y_hat_extended = (Y_hat_extended > 0).float()

        # Calculate PFA (excluding target neighborhoods)
        valid_cells = (~Y_true_extended).float()
        pfa = (Y_hat * valid_cells).sum() / valid_cells.sum()

        # Calculate PD (only for frames with targets)
        n_targets = Y_true.sum(dim=(1, 2))
        frames_with_targets = (n_targets > 0)
        if frames_with_targets.any():
            detected = (Y_hat_extended * Y_true).sum(dim=(1, 2))
            pd = (detected[frames_with_targets] / n_targets[frames_with_targets]).mean()
        else:
            pd = torch.tensor(0.0, device=self.device)

        return pd.item(), pfa.item()

    def find_threshold(self, loader, target_pfa):
        """Find threshold using binary search on entire dataset"""
        detection_surface, Y_true = self.feed_forward(loader)

        # Initialize threshold
        max_val = detection_surface.max()
        min_val = detection_surface.min()
        th = min_val + 0.5 * (max_val - min_val)
        step = 0.5 * (max_val - min_val)

        cnt = 1
        pfa_res = 1.0
        rel_err = abs(pfa_res - target_pfa) / abs(target_pfa)

        while rel_err >= 0.01 and cnt < 30:
            _, pfa_res = self.evaluate_metrics(detection_surface, Y_true, th)
            rel_err = abs(pfa_res - target_pfa) / abs(target_pfa)

            step = step * 0.5
            if pfa_res > target_pfa:
                th += step
            else:
                th -= step
            cnt += 1

        return th, pfa_res, cnt

    def evaluate_pd_pfa(self, nu, scnr=0):
        """Evaluate PD vs PFA curves"""
        dataset = ConcatDataset([RadarDataset(4096, 4, False, nu, scnr),
                                 RadarDataset(2048, 0, False, nu, scnr)])
        loader = DataLoader(dataset, batch_size=256, num_workers=2, persistent_workers=True,
                            pin_memory=torch.cuda.is_available(), collate_fn=cfar_collate_fn)

        detection_surface, Y_true = self.feed_forward(loader)
        target_pfas = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
        results = []

        for target_pfa in tqdm(target_pfas):
            th, pfa_res, cnt = self.find_threshold(loader, target_pfa)
            print(f"Found threshold = {th:.4f}, PFA = {pfa_res:.6f}, after {cnt} iterations")
            pd, pfa = self.evaluate_metrics(detection_surface, Y_true, th)
            results.append((pd, pfa))

        pd_list, pfa_list = zip(*results)
        return np.array(pd_list), np.array(pfa_list)

    def evaluate_pd_scnr(self, nu, target_pfa=5e-4):
        """Evaluate PD vs SCNR curves"""
        # Find threshold using reference dataset
        ref_dataset = ConcatDataset([RadarDataset(4096, 4, False, nu, 0),
                                     RadarDataset(2048, 0, False, nu)])
        ref_loader = DataLoader(ref_dataset, batch_size=256, num_workers=2, persistent_workers=True,
                                pin_memory=torch.cuda.is_available(), collate_fn=cfar_collate_fn)
        threshold, pfa_res, cnt = self.find_threshold(ref_loader, target_pfa)
        print(f"Found threshold = {threshold:.4f}, PFA = {pfa_res:.6f}, after {cnt} iterations")
        scnr_range = np.arange(-40, 11, 5)
        results = []

        for scnr in tqdm(scnr_range):
            test_dataset = ConcatDataset([RadarDataset(4096, 4, False, nu, scnr),
                                          RadarDataset(2048, 0, False, nu)])
            test_loader = DataLoader(test_dataset, batch_size=256, num_workers=2, persistent_workers=True,
                                     pin_memory=torch.cuda.is_available(), collate_fn=cfar_collate_fn)

            detection_surface, Y_true = self.feed_forward(test_loader)
            pd, pfa = self.evaluate_metrics(detection_surface, Y_true, threshold)
            results.append((pd, pfa))

        pd_list, pfa_list = zip(*results)
        return np.array(pd_list), np.array(pfa_list), scnr_range


def save_plot(save_path):
    plots_dir = os.path.join("plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{save_path}.png")
    plt.savefig(plot_path)


def save_results(save_path, pd_pfa, pd_scnr):
    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)
    pd_pfa_path = os.path.join(results_dir, f"{save_path}_pd_pfa.pt")
    pd_scnr_path = os.path.join(results_dir, f"{save_path}_pd_scnr.pt")
    torch.save(pd_pfa, pd_pfa_path)
    torch.save(pd_scnr, pd_scnr_path)

    return


def plot_pd_pfa(results: dict, save_path: str = 'pd_pfa'):
    """Plot PD vs PFA curves for different nu values"""
    plt.figure(figsize=(8, 6))

    for nu in results.keys():
        pd, pfa = results[nu]
        plt.plot(pfa, pd, label=f'ν = {nu}', marker='s', markersize=5)

    plt.xlabel('Probability of False Alarm')
    plt.ylabel('Probability of Detection')
    plt.title('ROC Curves for Different Clutter Conditions')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xscale('log')
    save_plot(save_path)
    plt.close()


def plot_pd_scnr(results: dict, save_path: str = 'pd_scnr'):
    """Plot PD vs SCNR curves for different nu values"""
    plt.figure(figsize=(8, 6))

    for nu in results.keys():
        pd, pfa, scnr = results[nu]
        plt.plot(scnr, pd, label=f'ν = {nu}', marker='s', markersize=5)

    plt.xlabel('SCNR (dB)')
    plt.ylabel('Probability of Detection')
    plt.title('Detection Performance vs SCNR')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_plot(save_path)
    plt.close()


def load_trained_models():
    """Load trained range and Doppler models"""
    range_model = DAFCRadarNet(detection_type="range")
    doppler_model = DAFCRadarNet(detection_type="doppler")

    try:
        range_model.load_state_dict(torch.load('range_model.pt', weights_only=True))
        print("Loaded range model successfully")
    except FileNotFoundError:
        print("Range model not found. Please train the model first.")
        return None, None

    try:
        doppler_model.load_state_dict(torch.load('doppler_model.pt', weights_only=True))
        print("Loaded doppler model successfully")
    except FileNotFoundError:
        print("Doppler model not found. Please train the model first.")
        return None, None

    range_model.eval()
    doppler_model.eval()

    return range_model, doppler_model


def generate_range_steering_matrix(N=64, dR=32, B=50e6, c=3e8):
    rng_res = c / (2 * B)
    r_vals = torch.arange(dR) * rng_res
    n_vals = torch.arange(N)

    phase = -1j * 2 * torch.pi * (2 * B) / (c * N)
    R = torch.exp(phase * torch.outer(n_vals, r_vals))

    return R


def generate_doppler_steering_matrix(K=64, dV=63, fc=9.39e9, T0=1e-3, c=3e8):
    vel_res = c / (2 * fc * K * T0)
    v_vals = torch.linspace(-dV // 2, dV // 2, dV) * vel_res
    k_vals = torch.arange(K)

    phase = -1j * 2 * torch.pi * (2 * fc * T0) / c
    V = torch.exp(phase * torch.outer(k_vals, v_vals))

    return V


def test():
    """Run complete test suite"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    range_model, doppler_model = load_trained_models()
    tester = CombinedRadarTester(range_model, doppler_model, device)
    nu_values = [0.2, 0.5, 1.0]

    # PD vs PFA test (SCNR = 0dB)
    pd_pfa_results = {}
    print("Running PD vs PFA test...")
    for nu in tqdm(nu_values):
        pd, pfa = tester.evaluate_pd_pfa(nu, scnr=0)
        pd_pfa_results[nu] = (pd, pfa)
    plot_pd_pfa(pd_pfa_results)

    # PD vs SCNR test (PFA = 5e-4)
    pd_scnr_results = {}
    print("Running PD vs SCNR test...")
    for nu in tqdm(nu_values):
        pd, pfa, scnr = tester.evaluate_pd_scnr(nu)
        pd_scnr_results[nu] = (pd, pfa, scnr)
    plot_pd_scnr(pd_scnr_results)

    save_results("dafc", pd_pfa_results, pd_scnr_results)

    return pd_pfa_results, pd_scnr_results


def test_cfar(detector="CA"):
    """Run complete test suite"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfar = CACFAR() if detector == "CA" else TMCFAR()
    cfar_tester = CFARTester(cfar, device)
    nu_values = [200.0, 500.0, 1000.0]

    # PD vs PFA test (SCNR = 0dB)
    pd_pfa_results = {}
    print("Running PD vs PFA test...")
    for nu in tqdm(nu_values):
        # Test CFAR
        pd, pfa = cfar_tester.evaluate_pd_pfa(nu, scnr=0)
        pd_pfa_results[nu] = (pd, pfa)
    plot_pd_pfa(pd_pfa_results, detector + "_CFAR_pd_pfa")

    # PD vs SCNR test (PFA = 5e-4)
    pd_scnr_results = {}
    print("Running PD vs SCNR test...")
    for nu in tqdm(nu_values):
        # Test CFAR
        pd, pfa, scnr = cfar_tester.evaluate_pd_scnr(nu)
        pd_scnr_results[nu] = (pd, pfa, scnr)
    plot_pd_scnr(pd_scnr_results, detector + "_CFAR_pd_scnr")

    save_results(detector + "_CFAR", pd_pfa_results, pd_scnr_results)

    return pd_pfa_results, pd_scnr_results
