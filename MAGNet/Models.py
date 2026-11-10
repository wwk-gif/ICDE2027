import copy
import torch.nn.functional as F
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import reset, uniform
from torch_scatter import scatter_add
import numpy as np
import warnings
from torch_geometric.utils import degree


# ==================== Real Wavelet Transform Module (Matrix-DWT based) ====================
def _build_dwt_matrix(filter_len, signal_len, device='cpu', dtype=torch.float32):
    """
    Build a 1D DWT decomposition matrix from Daubechies-4 filter coefficients.

    The DWT is a linear operation: approx = W_lo @ x, detail = W_hi @ x
    where W_lo and W_hi are [D//2, D] matrices representing convolution + downsampling.

    This is mathematically equivalent to conv1d(stride=2) but runs as fast as
    a standard Linear layer (matrix multiply), which is MUCH faster than conv1d
    with padding on GPU.
    """
    # Daubechies-4 low-pass filter
    h = torch.tensor([
        0.162901714025, 0.505472857546, 0.446100069123,
        -0.019787513118, -0.132253583684, 0.021808150237,
        0.023251800536, -0.007493494665
    ], dtype=dtype, device=device)

    L = filter_len
    D = signal_len
    D_out = D // 2

    # High-pass filter: g[n] = (-1)^n * h[L-1-n]
    n = torch.arange(L, device=device, dtype=dtype)
    g = ((-1) ** n) * torch.flip(h, dims=[0])

    # Build convolution + downsample matrices (circular boundary)
    # W_lo[i, :] applies low-pass filter centered at position 2*i
    W_lo = torch.zeros(D_out, D, dtype=dtype, device=device)
    W_hi = torch.zeros(D_out, D, dtype=dtype, device=device)

    half = L // 2
    for i in range(D_out):
        center = 2 * i
        for k in range(L):
            j = (center + k - half) % D  # circular indexing
            W_lo[i, j] += h[k]
            W_hi[i, j] += g[k]

    return W_lo, W_hi


def _build_idwt_matrix(filter_len, signal_len, device='cpu', dtype=torch.float32):
    """
    Build 1D IDWT reconstruction matrix.
    x_rec = W_syn_lo @ approx + W_syn_hi @ detail
    where W_syn are [D, D//2] matrices.
    """
    h = torch.tensor([
        0.162901714025, 0.505472857546, 0.446100069123,
        -0.019787513118, -0.132253583684, 0.021808150237,
        0.023251800536, -0.007493494665
    ], dtype=dtype, device=device)

    L = filter_len
    D = signal_len
    D_in = D // 2

    # Synthesis filters: h_syn[n] = h[L-1-n], g_syn[n] = (-1)^(L-1-n) * h[n]
    h_syn = torch.flip(h, dims=[0])
    n = torch.arange(L, device=device, dtype=dtype)
    g_syn = ((-1) ** (L - 1 - n)) * h

    W_lo_syn = torch.zeros(D, D_in, dtype=dtype, device=device)
    W_hi_syn = torch.zeros(D, D_in, dtype=dtype, device=device)

    half = L // 2
    for i in range(D_in):
        center = 2 * i
        for k in range(L):
            j = (center + k - half) % D  # circular
            W_lo_syn[j, i] += h_syn[k]
            W_hi_syn[j, i] += g_syn[k]

    return W_lo_syn, W_hi_syn


class SpaWaveletTransform(nn.Module):
    """
    Genuine wavelet-inspired multi-scale encoding using fixed DWT matrices.

    This is mathematically a real 2-level Discrete Wavelet Transform (Daubechies-4),
    but implemented as matrix multiplication (as fast as Linear layers).

    Decomposition:  X @ W1_lo.T → A₁ [D//2], A₁ @ W2_lo.T → A₂ [D//4]
    Reconstruction: A₂ @ W2_syn.T → A₁_recon [D//2], A₁_recon @ W1_syn.T → X_recon [D]
    Fusion: output = α ⊙ proj(A₂) + (1-α) ⊙ residual(X) + β ⊙ detail_features(D₁)

    Key: uses pre-computed orthogonal DWT matrices → O(ND²) matmul, same speed as Linear.
    """

    def __init__(self, in_features, out_features, alpha_init=0.7, filter_length=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_levels = 2

        # Pre-compute DWT matrices for the feature dimension
        # We need matrices sized for in_features (after possible padding)
        self.register_buffer('_dwt_ready', torch.tensor(0))  # flag

        # Feature projections
        d_level2 = max(1, in_features // 4)
        self.approx_proj = nn.Sequential(
            nn.Linear(d_level2, d_level2 * 2),
            nn.GELU(),
            nn.Linear(d_level2 * 2, out_features),
        )

        d_level1 = max(1, in_features // 2)
        self.detail_proj = nn.Sequential(
            nn.Linear(d_level1, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features),
        )

        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        else:
            self.residual_proj = nn.Identity()

        # Gated fusion
        alpha_init = max(0.01, min(0.99, alpha_init))
        self.alpha_logit = nn.Parameter(
            torch.ones(out_features) * np.log(alpha_init / (1 - alpha_init)))
        beta_init = 0.3
        self.beta_logit = nn.Parameter(
            torch.ones(out_features) * np.log(beta_init / (1 - beta_init)))

        # Filter length for matrix construction
        self.filter_length = filter_length

    def _get_dwt_matrices(self, D, device):
        """Get or build DWT matrices for given feature dimension."""
        # Pad to even length compatible with 2 levels
        D_compat = 4  # 2^2
        D_padded = ((D + D_compat - 1) // D_compat) * D_compat

        cache_key = (D_padded, device)
        if not hasattr(self, '_dwt_cache'):
            self._dwt_cache = {}

        if cache_key not in self._dwt_cache:
            # Level 1 matrices: D_padded -> D_padded//2
            W1_lo, W1_hi = _build_dwt_matrix(
                self.filter_length, D_padded, device=device, dtype=torch.float32)
            W1_lo_syn, W1_hi_syn = _build_idwt_matrix(
                self.filter_length, D_padded, device=device, dtype=torch.float32)

            # Level 2 matrices: D_padded//2 -> D_padded//4
            W2_lo, W2_hi = _build_dwt_matrix(
                self.filter_length, D_padded // 2, device=device, dtype=torch.float32)
            W2_lo_syn, W2_hi_syn = _build_idwt_matrix(
                self.filter_length, D_padded // 2, device=device, dtype=torch.float32)

            self._dwt_cache[cache_key] = {
                'W1_lo': W1_lo, 'W1_hi': W1_hi,
                'W1_lo_syn': W1_lo_syn, 'W1_hi_syn': W1_hi_syn,
                'W2_lo': W2_lo, 'W2_hi': W2_hi,
                'W2_lo_syn': W2_lo_syn, 'W2_hi_syn': W2_hi_syn,
                'D_padded': D_padded,
            }
        return self._dwt_cache[cache_key]

    def forward(self, x, return_all_scales=False):
        N, D = x.shape
        device = x.device

        # Get cached DWT matrices
        mats = self._get_dwt_matrices(D, device)
        D_pad = mats['D_padded']

        # Pad to compatible size
        if D_pad > D:
            x_padded = F.pad(x, (0, D_pad - D), mode='circular')
        else:
            x_padded = x

        # === Level 1 DWT (matrix multiply — fast!) ===
        # x_padded: [N, D_pad], W1_lo: [D_pad//2, D_pad]
        approx1 = F.linear(x_padded, mats['W1_lo'])      # [N, D_pad//2]
        detail1 = F.linear(x_padded, mats['W1_hi'])      # [N, D_pad//2]

        # === Level 2 DWT ===
        D1 = approx1.shape[1]
        approx2 = F.linear(approx1, mats['W2_lo'])       # [N, D_pad//4]
        detail2 = F.linear(approx1, mats['W2_hi'])       # [N, D_pad//4]

        # === Level 2 IDWT ===
        # F.linear(x, W) = x @ W.T. W_syn is [D_out, D_in], so this gives [N, D_out].
        approx1_recon = (F.linear(approx2, mats['W2_lo_syn']) +
                         F.linear(detail2, mats['W2_hi_syn']))  # [N, D_pad//2]

        # === Level 1 IDWT ===
        x_recon_full = (F.linear(approx1_recon, mats['W1_lo_syn']) +
                        F.linear(detail1, mats['W1_hi_syn']))    # [N, D_pad]
        x_recon = x_recon_full[:, :D]  # trim to original

        # === Feature projections ===
        approx2_features = self.approx_proj(approx2)
        detail1_features = self.detail_proj(detail1[:, :D // 2])
        residual = self.residual_proj(x)

        # === Gated fusion ===
        alpha = torch.sigmoid(self.alpha_logit)
        beta = torch.sigmoid(self.beta_logit)
        coarse = alpha * approx2_features + (1. - alpha) * residual
        x_wavelet = (1. - beta) * coarse + beta * detail1_features

        if return_all_scales:
            return {
                'approx1': approx1[:, :D // 2],
                'detail1': detail1[:, :D // 2],
                'approx2': approx2,
                'detail2': detail2,
                'approx1_recon': approx1_recon[:, :D // 2],
                'x_recon': x_recon,
                'detail1_features': detail1_features,
                'approx2_features': approx2_features,
                'residual': residual,
                'x_wavelet': x_wavelet,
                'alpha': alpha, 'beta': beta,
            }
        return x_wavelet


# ==================== Fourier Convolution Module ====================
class SpaFourierConv(nn.Module):
    """
    Fourier Convolution Module — replaces causal attention with spectral-domain convolution.

    Principle (convolution theorem):
      convolution(x, k) = IFFT(FFT(x) ⊙ FFT(k))

    The filter is centered at 1.0 (identity) with learned ±10% modulation:
      modulation = 1.0 + filter(freq) × gate(|FFT(x)|) × scale

    This ensures training stability — early epochs behave like identity,
    then gradually learn frequency-domain adjustments.

    Architecture:
      Channel 1 (Spatial):  FFT along nodes → filter × gate → IFFT
      Channel 2 (Feature):  FFT along features → filter × gate → IFFT
      FFN: Standard feed-forward
    """

    def __init__(self, hidden_dim, dropout=0.1, fourier_scale=0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        D_freq = hidden_dim // 2 + 1

        # --- Channel 1: Spatial Fourier Convolution ---
        self.spatial_proj = nn.Linear(hidden_dim, hidden_dim)

        # Frequency → filter MLP (learned modulation around 1.0)
        self.spatial_filter_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),                       # range (-1, 1)
        )

        # Input-adaptive gate: |FFT(x)| → per-freq scalar
        self.spatial_gate_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Learnable modulation scale (starts very small → near-identity at epoch 0)
        self.spatial_scale = nn.Parameter(torch.tensor(fourier_scale))

        # --- Channel 2: Feature Fourier Convolution ---
        self.feat_proj_in = nn.Linear(hidden_dim, hidden_dim)
        self.feat_proj_out = nn.Linear(hidden_dim, hidden_dim)

        # Frequency-domain mixing: small Linear along D_freq axis for richer feature interaction
        self.feat_freq_mix = nn.Linear(D_freq, D_freq)

        self.feat_filter = nn.Parameter(torch.zeros(D_freq))
        self.feat_gate_net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # Learnable modulation scale for feature channel
        self.feat_scale = nn.Parameter(torch.tensor(fourier_scale))

        # --- Second Fourier block (deeper processing) ---
        self.spatial_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_filter_net2 = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.spatial_gate_net2 = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid(),
        )
        self.spatial_scale2 = nn.Parameter(torch.tensor(0.05))

        self.feat_proj_in2 = nn.Linear(hidden_dim, hidden_dim)
        self.feat_proj_out2 = nn.Linear(hidden_dim, hidden_dim)
        self.feat_freq_mix2 = nn.Linear(D_freq, D_freq)
        self.feat_filter2 = nn.Parameter(torch.zeros(D_freq))
        self.feat_gate_net2 = nn.Sequential(
            nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid(),
        )
        self.feat_scale2 = nn.Parameter(torch.tensor(0.05))

        # --- Feed-forward ---
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm1b = nn.LayerNorm(hidden_dim)
        self.norm2b = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _spatial_fourier_conv(self, x):
        """
        Spatial Fourier convolution: f_out = IFFT(FFT(Wx) ⊙ (1 + Δ))
        where Δ = filter(freq) × gate(|FFT|) × scale is a small learned modulation.
        """
        N, D = x.shape
        N_freq = N // 2 + 1

        # Project + FFT
        x_proj = self.spatial_proj(x)
        x_freq = torch.fft.rfft(x_proj.float(), dim=0)  # [N_freq, D] C

        # Filter: Tanh output ∈ (-1, 1), scaled to small modulation
        freqs = torch.linspace(0, 1, N_freq, device=x.device, dtype=torch.float32)
        spatial_filter = self.spatial_filter_net(freqs.unsqueeze(-1))  # [N_freq, D]

        # Adaptive gate
        mag = torch.abs(x_freq).mean(dim=-1)            # [N_freq]
        spatial_gate = self.spatial_gate_net(
            mag.unsqueeze(-1)                           # [N_freq, 1]
        )                                                # [N_freq, 1]

        # Modulation centered at 1.0: small perturbation around identity
        delta = spatial_filter * spatial_gate * self.spatial_scale  # [N_freq, D], ~[-0.05, 0.05]
        modulation = 1.0 + delta                                    # [N_freq, D], ~[0.95, 1.05]
        x_freq_mod = x_freq * modulation

        return torch.fft.irfft(x_freq_mod, n=N, dim=0)  # [N, D]

    def _feature_fourier_conv(self, x):
        """
        Feature Fourier convolution: f_out = IFFT(FFT(Wx) ⊙ (1 + Δ))
        """
        N, D = x.shape
        D_freq = D // 2 + 1

        x_proj = self.feat_proj_in(x)
        x_freq = torch.fft.rfft(x_proj.float(), dim=1)  # [N, D_freq] C

        # Frequency-domain mixing
        x_freq_orig = x_freq
        real_mix = self.feat_freq_mix(x_freq.real)       # [N, D_freq]
        imag_mix = self.feat_freq_mix(x_freq.imag)       # [N, D_freq]
        x_freq = torch.complex(real_mix, imag_mix)
        x_freq = x_freq + x_freq_orig * 0.1                # residual

        # Filter + gate
        feat_filter = torch.tanh(self.feat_filter)       # [D_freq], range (-1,1)
        mag = torch.abs(x_freq).mean(dim=0)              # [D_freq]
        feat_gate = self.feat_gate_net(mag.unsqueeze(-1)).squeeze(-1)  # [D_freq]

        # Modulation centered at 1.0
        delta = feat_filter * feat_gate * self.feat_scale   # [D_freq]
        modulation = 1.0 + delta.unsqueeze(0)                # [1, D_freq]
        x_freq_mod = x_freq * modulation                     # [N, D_freq]

        feat_out = torch.fft.irfft(x_freq_mod, n=D, dim=1)  # [N, D]
        return self.feat_proj_out(feat_out)

    def _spatial_fourier_conv2(self, x):
        """Block 2 spatial Fourier — same structure, separate parameters"""
        N, D = x.shape
        x_proj = self.spatial_proj2(x)
        x_freq = torch.fft.rfft(x_proj.float(), dim=0)
        freqs = torch.linspace(0, 1, x_freq.shape[0], device=x.device, dtype=torch.float32)
        spatial_filter = self.spatial_filter_net2(freqs.unsqueeze(-1))
        mag = torch.abs(x_freq).mean(dim=-1)
        spatial_gate = self.spatial_gate_net2(mag.unsqueeze(-1))
        delta = spatial_filter * spatial_gate * self.spatial_scale2
        modulation = 1.0 + delta
        x_freq_mod = x_freq * modulation
        return torch.fft.irfft(x_freq_mod, n=N, dim=0)

    def _feature_fourier_conv2(self, x):
        """Block 2 feature Fourier — same structure, separate parameters"""
        N, D = x.shape
        D_freq = D // 2 + 1
        x_proj = self.feat_proj_in2(x)
        x_freq = torch.fft.rfft(x_proj.float(), dim=1)
        x_freq_orig = x_freq
        real_mix = self.feat_freq_mix2(x_freq.real)
        imag_mix = self.feat_freq_mix2(x_freq.imag)
        x_freq = torch.complex(real_mix, imag_mix)
        x_freq = x_freq + x_freq_orig * 0.1
        feat_filter = torch.tanh(self.feat_filter2)
        mag = torch.abs(x_freq).mean(dim=0)
        feat_gate = self.feat_gate_net2(mag.unsqueeze(-1)).squeeze(-1)
        delta = feat_filter * feat_gate * self.feat_scale2
        modulation = 1.0 + delta.unsqueeze(0)
        x_freq_mod = x_freq * modulation
        feat_out = torch.fft.irfft(x_freq_mod, n=D, dim=1)
        return self.feat_proj_out2(feat_out)

    def forward(self, x):
        # === Block 1 ===
        spatial_out = self._spatial_fourier_conv(x)
        x = self.norm1(x + self.dropout(spatial_out))
        feat_out = self._feature_fourier_conv(x)
        x = self.norm2(x + self.dropout(feat_out))

        # === Block 2 (deeper refinement) ===
        spatial_out2 = self._spatial_fourier_conv2(x)
        x = self.norm1b(x + self.dropout(spatial_out2))
        feat_out2 = self._feature_fourier_conv2(x)
        x = self.norm2b(x + self.dropout(feat_out2))

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


# ==================== Original functions kept unchanged ====================
def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def full_block(in_features, out_features, p_drop, act=nn.ELU()):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        act,
        nn.Dropout(p=p_drop),
    )


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, act=F.relu, bn=True):
        super(GraphConv, self).__init__()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.in_features = in_features
        self.out_features = out_features
        self.bn = bn(out_features)
        self.act = act
        self.dropout = dropout
        self.conv = GCNConv(in_channels=self.in_features, out_channels=self.out_features, cached=True)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        return x


# ==================== Modified Encoder class ====================
class Encoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = config['feat_hidden1']
        self.feat_hidden2 = config['feat_hidden2']
        self.gcn_hidden = config['gcn_hidden']
        self.latent_dim = config['latent_dim']
        self.p_drop = config['p_drop']

        # Default enable wavelet transform and Fourier convolution module
        self.use_wavelet = config.get('use_wavelet', True)
        self.use_fourier = config.get('use_fourier', True)
        wavelet_filter_len = config.get('wavelet_filter_length', 8)  # db4=8, db2=4; longer = more expressive

        print(f"Enhanced encoder initialized (genuine DWT wavelet + Fourier):")
        print(f"  Input dim: {input_dim}")
        print(f"  Use wavelet transform: {self.use_wavelet} (DWT-based, filter_len={wavelet_filter_len})")
        print(f"  Use Fourier convolution: {self.use_fourier}")

        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))

        # Genuine wavelet transform module (learnable DWT-based, enabled by default)
        alpha_init = config.get('wavelet_alpha_init', 0.7)
        self.wavelet_module = SpaWaveletTransform(
            self.feat_hidden2,
            self.feat_hidden2,
            alpha_init=alpha_init,
            filter_length=wavelet_filter_len,
        )

        # Fourier convolution module (replaces causal inference, enabled by default)
        fourier_scale = config.get('fourier_scale', 0.01)
        self.fourier_module = SpaFourierConv(
            self.feat_hidden2,
            dropout=self.p_drop,
            fourier_scale=fourier_scale
        )

        # GCN layers
        self.gc1 = GraphConv(self.feat_hidden2, self.gcn_hidden, dropout=self.p_drop, act=F.relu)
        self.gc2 = GraphConv(self.gcn_hidden, self.latent_dim, dropout=self.p_drop, act=lambda x: x)

    def forward(self, x, edge_index, return_wavelet_features=False, return_fourier_features=False):
        # Base feature extraction
        base_features = self.encoder(x)

        # Apply wavelet transform
        if self.use_wavelet and return_wavelet_features:
            wf = self.wavelet_module(base_features, return_all_scales=True)
            wavelet_features = wf['x_wavelet']
        elif self.use_wavelet:
            wavelet_features = self.wavelet_module(base_features, return_all_scales=False)
        else:
            wavelet_features = base_features

        # Apply Fourier convolution (no masks needed — O(N log N) spectral mixing)
        if self.use_fourier:
            fourier_features = self.fourier_module(wavelet_features)
        else:
            fourier_features = wavelet_features

        # Graph convolution (further propagate over Fourier-enhanced features)
        x = self.gc1(fourier_features, edge_index)
        x = self.gc2(x, edge_index)

        # Build wavelet feature dictionary
        if self.use_wavelet and return_wavelet_features:
            wf['fourier_features'] = fourier_features
            wf['gcn1_features'] = x  # gc2 input
            if return_fourier_features:
                return x, wf, fourier_features
            return x, wf

        # Return Fourier module output for external use
        if return_fourier_features:
            return x, fourier_features

        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, config, imputation=True):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = config['latent_dim']
        self.p_drop = config['p_drop']
        self.imputation = imputation
        if self.imputation:
            self.layer1 = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.layer1 = GraphConv(self.input_dim, self.output_dim, dropout=self.p_drop, act=nn.Identity())

    def forward(self, x, edge_index):
        if self.imputation:
            return self.layer1(x)
        return self.layer1(x, edge_index)


# ==================== Modified MACWT_model class ====================
class MACWT_model(nn.Module):
    def __init__(self, input_dim, config, imputation=True):
        super().__init__()
        self.imputation = imputation
        self.dec_in_dim = config['latent_dim']

        # Default enable wavelet transform and Fourier convolution
        self.use_wavelet = True  # Enable wavelet transform by default
        self.use_fourier = True  # Enable Fourier convolution by default

        print(f"\nEnhanced MACWT model initialized (enhancements enabled by default):")
        print(f"  Input dim: {input_dim}")
        print(f"  Latent dim: {self.dec_in_dim}")
        print(f"  Use wavelet transform: {self.use_wavelet}")
        print(f"  Use Fourier convolution: {self.use_fourier}")

        self.online_encoder = Encoder(input_dim, config)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self._init_target()

        self.encoder_to_decoder = nn.Linear(self.dec_in_dim, config['project_dim'], bias=False)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        self.projector = GraphConv(config['project_dim'], self.dec_in_dim, dropout=config['p_drop'], act=lambda x: x)
        self.decoder = Decoder(input_dim, config, self.imputation)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.rep_mask = nn.Parameter(torch.zeros(1, self.dec_in_dim))
        self.mask_rate = config['mask_rate']
        self.t = config['t']
        self.momentum_rate = config['momentum_rate']
        self.replace_rate = 0.05
        self.mask_token_rate = 1 - self.replace_rate
        self.anchor_pair = None

        self.weight = nn.Parameter(torch.empty(self.dec_in_dim, self.dec_in_dim))
        uniform(self.dec_in_dim, self.weight)

    def _init_target(self):
        for param_teacher in self.target_encoder.parameters():
            param_teacher.detach()
            param_teacher.requires_grad = False

    def momentum_update(self):
        base_momentum = self.momentum_rate
        for param_encoder, param_teacher in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + param_encoder.data * (1. - base_momentum)

    def encoding_mask_noise(self, x, edge_index, mask_rate=0.3):
        num_nodes = x.shape[0]
        self.num_nodes = num_nodes
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]

        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_edge_index = edge_index.clone()

        return out_x, use_edge_index, (mask_nodes, keep_nodes)

    def generate_neg_nodes(self, mask_nodes):
        num_mask_nodes = mask_nodes.size(0)
        neg_nodes_x = torch.randint(0, self.num_nodes, (num_mask_nodes,), device=mask_nodes.device)
        neg_nodes_y = torch.randint(0, self.num_nodes, (num_mask_nodes,), device=mask_nodes.device)
        return neg_nodes_x, neg_nodes_y

    def mask_attr_prediction(self, x, edge_index, anchor_pair):
        use_x, use_adj, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, edge_index, self.mask_rate)

        # Encoder forward pass
        enc_rep = self.online_encoder(use_x, use_adj)

        with torch.no_grad():
            x_t = x.clone()
            x_t[keep_nodes] = 0.0
            x_t[keep_nodes] += self.enc_mask_token
            rep_t = self.target_encoder(x_t, use_adj)

        # Fix: ensure cl_loss is always a torch tensor
        if anchor_pair is not None:
            anchor, positive, negative = anchor_pair
            summary = self.avg_readout(enc_rep, [anchor, positive])
            num_mask_nodes = mask_nodes.size(0)
            neg_nodes = torch.randint(0, self.num_nodes, (num_mask_nodes,), device=mask_nodes.device)
            cl_loss = self.dgi_loss(enc_rep[mask_nodes], enc_rep[neg_nodes], summary[mask_nodes])
        else:
            cl_loss = torch.tensor(0.0, device=x.device)

        rep = enc_rep
        rep = self.encoder_to_decoder(rep)
        rep[mask_nodes] = 0.0
        rep = self.projector(rep, use_adj)

        match_loss = self.match_loss(rep, rep_t, mask_nodes)
        recon = self.decoder(rep, use_adj)
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        rec_loss = self.sce_loss(x_rec, x_init, t=self.t)

        return match_loss, rec_loss, cl_loss

    def match_loss(self, rep, rep_t, mask_nodes, t=2):
        pox_x_index, pox_y_index = mask_nodes, mask_nodes
        neg_x_index, neg_y_index = self.generate_neg_nodes(mask_nodes)
        std_emb = F.normalize(rep.clone(), p=2, dim=-1)
        tgt_emb = F.normalize(rep_t.clone(), p=2, dim=-1)

        pox_x = std_emb[pox_x_index]
        pox_y = tgt_emb[pox_y_index]
        neg_x = std_emb[neg_x_index]
        neg_y = tgt_emb[neg_y_index]

        pos_cos = (0.5 * (1 + (pox_x * pox_y).sum(dim=-1))).pow(t)
        pos_loss = -torch.log(pos_cos)
        neg_cos = (0.5 * (1 + (neg_x * neg_y).sum(dim=-1))).pow(t)
        neg_loss = -torch.log(1 - neg_cos)
        loss = 0.5 * (pos_loss.mean() + neg_loss.mean())
        return loss

    def sce_loss(self, x, y, t=2):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        cos_m = (1 + (x * y).sum(dim=-1)) * 0.5
        loss = -torch.log(cos_m.pow_(t))
        return loss.mean()

    def triplet_loss(self, emb, anchor, positive, negative, margin=1.0):
        anchor_arr = emb[anchor]
        positive_arr = emb[positive]
        negative_arr = emb[negative]
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        return tri_output

    def forward(self, x, edge_index, anchor_pair):
        # Always return 3 values (match_loss, rec_loss, cl_loss)
        return self.mask_attr_prediction(x, edge_index, anchor_pair)

    @torch.no_grad()
    def evaluate(self, x, edge_index):
        # Encoder evaluation
        enc_rep = self.online_encoder(x, edge_index)
        rep = self.encoder_to_decoder(enc_rep)
        rep = self.projector(rep, edge_index)
        recon = self.decoder(rep, edge_index)
        return enc_rep, recon

    @torch.no_grad()
    def get_wavelet_features(self, x, edge_index):
        """
        [Visualization only] Extract multi-scale features from the wavelet module.

        Two conditions must be met for valid wavelet features:
          1. config.use_wavelet = True
          2. Model initialized with Encoder(use_wavelet=True)

        Returns:
            (latent, wavelet_dict) where wavelet_dict contains:
                - decomposed1:  intermediate scale (N, d_h1)
                - decomposed2:  coarsest scale (N, d_h2) ← global domain pattern
                - reconstructed: reconstructed features (N, out_features)
                - residual:      high-frequency residual (N, out_features)
                - x_wavelet:     gated fusion output (N, out_features)
                - alpha:         gate weights (out_features,)
                - fourier_features: Fourier convolution output (N, out_features)
        """
        self.eval()
        result = self.online_encoder(x, edge_index, return_wavelet_features=True)
        if isinstance(result, tuple):
            latent, wf = result
            return latent, wf
        else:
            # Wavelet not enabled, return None
            return result, None

    @torch.no_grad()
    def std_tgt_embedding(self, x, edge_index):
        s_rep = self.online_encoder(x, edge_index)
        t_rep = self.target_encoder(x, edge_index)
        return s_rep, t_rep

    def avg_readout(self, rep_pos_x, edge_index):
        # Handle edge_index type
        if isinstance(edge_index, torch.sparse.Tensor):
            # For sparse tensor, convert to dense to get edges
            adj = edge_index.to_dense()
            src, dst = torch.nonzero(adj, as_tuple=True)
        elif isinstance(edge_index, list):
            # If edge_index is a list, assume it's [anchor, positive]
            # Directly compute the feature mean of these points
            anchor, positive = edge_index
            src = anchor
            dst = positive
        else:
            src, dst = edge_index

        neighbor_sum = scatter_add(rep_pos_x[src], dst, dim=0, dim_size=rep_pos_x.size(0))
        neighbor_count = scatter_add(torch.ones_like(src, dtype=torch.float), dst, dim=0, dim_size=rep_pos_x.size(0))
        neighbor_count = neighbor_count.clamp(min=1)
        summary = neighbor_sum / neighbor_count.unsqueeze(-1)
        return torch.sigmoid(summary)

    def discriminate(self, z, summary, sigmoid=True):
        assert isinstance(summary, torch.Tensor), "Summary should be a torch.Tensor"
        value = torch.matmul(z, torch.matmul(self.weight, summary.t()))
        return torch.sigmoid(value) if sigmoid else value

    def dgi_loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss

    def CL_Loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + 1e-15).mean()
        Cos_loss = -torch.log(1 - F.cosine_similarity(pos_z, neg_z) + 1e-15).mean()
        loss = Cos_loss + pos_loss + neg_loss  # 50
        return loss
