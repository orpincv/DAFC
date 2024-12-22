import torch
from torch.utils.data import Dataset


class RadarDataset(Dataset):
    def __init__(self, num_samples, n_targets: int = 8, random_n_targets=True, nu=None, scnr=None):
        super().__init__()
        self.num_samples = num_samples
        self.n_targets = n_targets
        self.random_n_targets = random_n_targets
        self.with_targets = n_targets > 0
        self.scnr = scnr
        self.nu = torch.tensor([nu]) if nu is not None else None

        # Parameters from Table II in the paper
        self.N = 64  # Samples per pulse (fast-time)
        self.K = 64  # Pulses per frame (slow-time)
        self.B = 50e6  # Chirp bandwidth (Hz)
        self.T0 = 1e-3  # PRI (s)
        self.fc = 9.39e9  # Carrier frequency (Hz)
        self.fs = 1e-6  # Baseband sample frequency
        self.c = 3e8  # Speed of light (m/s)
        self.CNR = 15  # Clutter-to-noise ratio (dB)
        self.sigma2 = self.N / (2 * 10 ** (self.CNR / 10))
        self.cn_norm = torch.sqrt(self.K * self.N * torch.tensor(self.N // 2 + self.sigma2))

        # Range and Doppler parameters
        self.rng_res = self.c / (2 * self.B)
        self.vel_res = self.c / (2 * self.fc * self.K * self.T0)
        self.r_min, self.r_max = 0, 93  # Range interval (m)
        self.v_min, self.v_max = -(31 * self.vel_res), (31 * self.vel_res)  # Doppler interval (m/s)
        self.vc_min, self.vc_max = self.v_min, self.v_max  # Clutter min/max velocity (m/s)

        # Calculate range and Doppler bins
        self.R = torch.arange(self.r_min, self.r_max + self.rng_res, self.rng_res)
        self.V = torch.linspace(self.v_min, self.v_max, 63)
        self.dR = len(self.R)  # Number of range bins
        self.dV = len(self.V)  # Number of Doppler bins

    def generate_target_signal(self, ranges, velocities, phases, SCNR_dBs):
        """Generate target signal matrix"""
        # Range steering vector
        w_r = (2 * torch.pi * 2 * self.B * ranges) / (self.c * self.N)
        range_steering = torch.exp(-1j * torch.outer(w_r, torch.arange(self.N)))

        # Doppler steering vector
        w_d = (2 * torch.pi * self.T0 * 2 * self.fc * velocities) / self.c
        doppler_steering = torch.exp(-1j * torch.outer(w_d, torch.arange(self.K)))

        # Fast-time x Slow-time matrix
        rd_signal = range_steering.unsqueeze(-1) * doppler_steering.unsqueeze(1)

        # Physical Phase
        tau0 = 2 * ranges.unsqueeze(-1).unsqueeze(-1) / self.c
        physical_phase = ((-1j * 2 * torch.pi * self.fc * tau0) + (torch.pi * self.B * tau0 ** 2)) / (self.N * self.fs)
        rd_signal = rd_signal * torch.exp(physical_phase)

        # Random phase for each target
        rd_signal = rd_signal * torch.exp(1j * phases)

        # Scaling SCNR for each target
        S_norm = torch.linalg.norm(rd_signal, dim=(1, 2)).real
        sig_amp = (10 ** (SCNR_dBs / 20)) * (self.cn_norm / S_norm)

        # Expand sig_amp to have shape (N_range_bins, 1, 1) for broadcasting
        rd_signal = (sig_amp.unsqueeze(-1).unsqueeze(-1) * rd_signal).sum(dim=0)
        return rd_signal

    def generate_clutter(self, nu):
        """Generate K-distributed SIRV clutter using eigendecomposition"""
        # Clutter correlation matrix
        clutter_vel = torch.empty(1).uniform_(self.vc_min, self.vc_max)
        fd = (2 * torch.pi * (2 * self.fc * clutter_vel) / self.c)  # Clutter Doppler shift (m/s)
        sigma_f = 0.05  # From paper
        p, q = torch.meshgrid(torch.arange(self.N), torch.arange(self.K), indexing='ij')

        # Generate complex correlation matrix
        M = torch.exp(-2 * torch.pi ** 2 * sigma_f ** 2 * (p - q) ** 2 - 1j * (p - q) * fd * self.T0)

        # Generate complex normal samples
        z = torch.randn(self.K, self.dR, dtype=torch.cfloat) / torch.sqrt(torch.tensor(2.0))

        # Eigenvalue decomposition of the correlation matrix M
        e, V = torch.linalg.eigh(M)
        e_sqrt = torch.sqrt(torch.maximum(e.real, torch.tensor(0.0)))
        E = torch.diag(e_sqrt)
        A = V @ E.to(V.dtype)
        w_t = A @ z

        # Generate texture component
        s = torch.distributions.Gamma(nu, nu).sample((self.dR,))

        # Scale by texture component
        c_t = (torch.sqrt(s).unsqueeze(0) * w_t.unsqueeze(-1)).squeeze(-1)

        # Convert to fast-time × slow-time representation
        c_r_steer = torch.exp(
            -1j * 2 * torch.pi * torch.outer(torch.arange(self.N), self.R) * (2 * self.B) / (self.c * self.N))
        C = c_r_steer @ c_t.transpose(0, 1)
        return C

    def gen_frame_and_labels(self):
        # Generate Noise
        W = (torch.randn(self.N, self.K, dtype=torch.cfloat) / torch.sqrt(torch.tensor(self.sigma2)))

        # Generate Clutter
        nu = torch.empty(1).uniform_(0.1, 1.5) if self.nu is None else self.nu
        C = self.generate_clutter(nu)
        # C = torch.zeros_like(W)

        # Initialize target signal and label matrices
        S = torch.zeros_like(W)
        rd_label = torch.zeros(self.dR, self.dV)
        if self.with_targets:
            n = torch.randint(1, self.n_targets + 1, (1,)) if self.random_n_targets else self.n_targets
            ranges = torch.empty(n).uniform_(self.r_min, self.r_max)
            velocities = torch.empty(n).uniform_(self.v_min, self.v_max)
            phases = torch.empty(n, 1, 1).uniform_(0, 2 * torch.pi)
            SCNR_dBs = torch.empty(n).uniform_(-5, 10) if self.scnr is None else self.scnr * torch.ones(n)
            S = self.generate_target_signal(ranges, velocities, phases, SCNR_dBs)

            # Create label matrix (Boolean matrix, label[i,j]= 1 if there is a target with range i and Doppler j
            for r, v in zip(ranges, velocities):
                r_bin = torch.argmin(torch.abs(self.R - r))
                v_bin = torch.argmin(torch.abs(self.V - v))
                rd_label[r_bin, v_bin] = True

        # Combine signals
        X = (S + C + W)
        return X, rd_label

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        X, rd_label = self.gen_frame_and_labels()
        return X, rd_label


def preprocess_rd_map(X):
    # Apply 2D inverse FFT
    rd_map = torch.abs(torch.fft.ifft2(X)) ** 2

    # Shift zero-frequency components to the center
    rd_map = torch.fft.fftshift(rd_map, dim=(-2, -1))

    # Extract positive ranges and truncate Doppler bins
    rd_map = rd_map[:, rd_map.shape[-2] // 2:, 1:]

    return rd_map


def cfar_collate_fn(batch):
    """
    Custom collate function that preprocesses radar data during batch creation

    Args:
        batch: List of (X, labels) tuples from RadarDataset

    Returns:
        Tuple of (preprocessed_batch, labels)
    """
    # Separate data and labels
    data, labels = zip(*batch)

    # Stack data and labels into tensors
    data = torch.stack(data)
    labels = torch.stack(labels)

    # Preprocess the stacked data
    with torch.no_grad():
        processed_data = preprocess_rd_map(data)

    return processed_data, labels
