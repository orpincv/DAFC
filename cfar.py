import torch
import torch.nn as nn
import torch.nn.functional as F


class CACFAR(nn.Module):
    def __init__(self, window_size=(9, 15), guard_cells=(3, 3)):
        super().__init__()
        self.window_size = window_size
        self.guard_cells = guard_cells

        # Create single kernel for reference cells
        kernel = torch.ones(1, 1, *window_size)

        # Zero out guard cells region
        center_r, center_d = window_size[0] // 2, window_size[1] // 2
        guard_r, guard_d = guard_cells[0] // 2, guard_cells[1] // 2
        kernel[0, 0, center_r - guard_r:center_r + guard_r + 1, center_d - guard_d:center_d + guard_d + 1] = 0.0

        # Register kernel as non-trainable parameter
        self.register_buffer('kernel', kernel)

        # Pre-calculate number of reference cells
        self.n_ref_cells = float(window_size[0] * window_size[1] - guard_cells[0] * guard_cells[1])

    def forward(self, rd_map):
        # Add channel dimension if needed
        if rd_map.dim() == 3:
            rd_map = rd_map.unsqueeze(1)

        # Calculate reference cells sum using single convolution
        ref_sum = F.conv2d(rd_map, self.kernel, padding='same')

        # Calculate mean of reference cells
        local_mean = ref_sum / self.n_ref_cells

        # Return detection statistic
        return (rd_map / local_mean).squeeze(1)


class TMCFAR(CACFAR):
    def __init__(self, window_size=(9, 15), guard_cells=(3, 3), trim_ratio=0.25):
        super().__init__(window_size, guard_cells)
        self.trim_ratio = trim_ratio

        # Calculate indices for trimming
        self.n_total = window_size[0] * window_size[1]
        self.n_guard = guard_cells[0] * guard_cells[1]
        self.n_ref = self.n_total - self.n_guard
        self.lower_k = int(self.n_ref * trim_ratio)
        self.upper_k = int(self.n_ref * (1 - trim_ratio))

    def forward(self, rd_map):
        # Add channel dimension if needed
        if rd_map.dim() == 3:
            rd_map = rd_map.unsqueeze(1)

        batch_size, _, H, W = rd_map.shape

        # Use unfold with zero padding
        windows = F.unfold(rd_map, kernel_size=self.window_size,
                           padding=(self.window_size[0] // 2, self.window_size[1] // 2))
        windows = windows.view(batch_size, self.n_total, H, W)

        # Use kernel as mask for reference cells
        mask = (self.kernel.view(-1) > 0)
        ref_windows = windows[:, mask]  # Select only reference cells

        # Sort reference cells along the reference cells dimension
        sorted_refs, _ = torch.sort(ref_windows, dim=1)

        # Calculate trimmed mean
        trimmed_mean = torch.mean(sorted_refs[:, self.lower_k:self.upper_k], dim=1)
        trimmed_mean = trimmed_mean.unsqueeze(1)  # Add channel dim back

        # Return detection statistic
        return (rd_map / trimmed_mean).squeeze(1)
