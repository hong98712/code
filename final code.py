# ------------------ Imports ------------------
import math
import pandas as pd
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyEMD import EMD
from scipy.fft import dct
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


# ------------------ Configuration ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 6
pred_len = 2
batch_size = 96
epochs = 150
learning_rate = 0.001
patience = 30


# ------------------ Data Loading ------------------
def load_cropwise_normalized_data(filepath):
    df = pd.read_csv(filepath)
    df = df[df['Year'].between(2016, 2023)]
    df['Yield_value'].fillna(df['Yield_value'].mean(), inplace=True)

    scaler_dict = {}
    df_scaled = pd.DataFrame()

    for crop in df['Crop_type'].unique():
        crop_df = df[df['Crop_type'] == crop].copy()
        scaler = MinMaxScaler()
        crop_df['Yield_value'] = scaler.fit_transform(crop_df[['Yield_value']])
        scaler_dict[crop] = scaler
        df_scaled = pd.concat([df_scaled, crop_df], ignore_index=True)

    return df_scaled, scaler_dict


# ------------------ AMD Processing ------------------
def decompose_signal(signal):
    return EMD().emd(signal)


def discrete_wavelet_filter(imf, wavelet='db4'):
    level = pywt.dwt_max_level(len(imf), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(imf, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)


def gwma(signal, window=5):
    weights = np.arange(1, window + 1)
    return np.convolve(signal, weights / weights.sum(), mode='same')


def permutation_entropy(ts, order=3, delay=1):
    n = len(ts)
    if n < order * delay:
        return 0
    permutations = np.array([ts[i:i + order * delay:delay] for i in range(n - order * delay + 1)])
    sorted_idx = np.argsort(permutations, axis=1)
    _, counts = np.unique(sorted_idx, axis=0, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


def compute_permutation_entropy(imfs):
    return np.array([permutation_entropy(imf) for imf in imfs])


def dbscan_filtering(entropies, eps=0.3, min_samples=5):
    entropies = entropies.reshape(-1, 1)
    return DBSCAN(eps=eps, min_samples=min_samples).fit(entropies).labels_


def reconstruct_entropy_mode(imfs, labels, entropies):
    label_entropy = defaultdict(list)
    for i, label in enumerate(labels):
        label_entropy[label].append(entropies[i])

    best_label = min(label_entropy, key=lambda k: np.mean(label_entropy[k]))
    return np.sum([imfs[i] for i in range(len(imfs)) if labels[i] == best_label], axis=0)


def preprocess_and_amd(signal):
    signal = np.nan_to_num(signal)
    imfs = decompose_signal(signal)
    filtered_imfs = [discrete_wavelet_filter(imf) if i < 2 else gwma(imf) for i, imf in enumerate(imfs)]
    entropies = compute_permutation_entropy(filtered_imfs)
    labels = dbscan_filtering(entropies)
    return reconstruct_entropy_mode(filtered_imfs, labels, entropies)


# ------------------ Dataset ------------------
def split_crop_data(df, crop, train_ratio=0.75, val_ratio=0.10):
    crop_df = df[df["Crop_type"] == crop].sort_values(by="Year").reset_index(drop=True)
    total_len = len(crop_df)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    train_df = crop_df[:train_end]
    val_df = crop_df[train_end:val_end]
    test_df = crop_df[val_end:]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class YieldDataset(Dataset):
    def __init__(self, df, crop, seq_len=6, pred_len=2):
        crop_df = df[df['Crop_type'] == crop].sort_values(['Year'])
        values = crop_df['Yield_value'].values
        self.samples = [(values[i:i+seq_len], values[i+seq_len:i+seq_len+pred_len])
                        for i in range(len(values) - seq_len - pred_len + 1)]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def get_dataloader(df, crop, seq_len=6, pred_len=2, shuffle=False):
    dataset = YieldDataset(df, crop, seq_len, pred_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ------------------ Model Components ------------------
class ProbSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, top_u=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.top_u = top_u
        self.scale = 1 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        out = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=dilation, dilation=dilation,
                                   groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2))
        x = self.pool(x).expand_as(x)
        x = self.pointwise(self.depthwise(x))
        return x.transpose(1, 2)


class CosineMixingLayer(nn.Module):
    def forward(self, x):
        x_dct = dct(x.detach().cpu().numpy(), axis=1, norm='ortho')
        return torch.tensor(x_dct, dtype=x.dtype, device=x.device)


class TDIBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn1 = ProbSparseAttention(channels)
        self.conv1 = TemporalConvBlock(channels, dilation=1)
        self.attn2 = ProbSparseAttention(channels)
        self.conv2 = TemporalConvBlock(channels, dilation=2)
        self.mix = CosineMixingLayer()

    def forward(self, x):
        x = self.attn1(x)
        x = self.conv1(x)
        x = self.attn2(x)
        x = self.conv2(x)
        return torch.cat([x, self.mix(x)], dim=2)


class Decoder(nn.Module):
    def __init__(self, input_dim, output_len):
        super().__init__()
        self.attn = ProbSparseAttention(input_dim)
        self.mix = CosineMixingLayer()
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_len))
        self.proj = nn.Linear(input_dim * 2, input_dim)

    def forward(self, x):
        x = self.proj(x)
        attn_out = self.attn(x)
        mix_out = self.mix(x)
        x = torch.cat([attn_out, mix_out], dim=2)
        return self.fc(x.mean(dim=1))


class Temporal_Depthwise_Informer(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len):
        super().__init__()
        self.embedding = nn.Linear(1, input_dim)
        self.encoder = TDIBlock(input_dim)
        self.decoder = Decoder(input_dim, pred_len)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.encoder(x)
        return self.decoder(x)


# ------------------ Metrics ------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-7, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def symmetric_mape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-7))


def nash_sutcliffe_efficiency(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)


# ------------------ Train / Evaluate ------------------
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, crop):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0  # Moved inside function
    model_path = f"best_model_{crop}.pth"

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, criterion)
        scheduler.step(val_loss)
        print(f"[{crop}] Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{crop}] Early stopping at epoch {epoch+1}")
                break


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            total_loss += criterion(model(inputs), targets).item()

    return total_loss / len(loader)


def test_model(model, model_path, loader, crop, scaler_dict):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            preds.append(output.cpu().numpy())
            trues.append(targets.cpu().numpy())

    preds = np.array(preds).reshape(-1, 1)
    trues = np.array(trues).reshape(-1, 1)
    scaler = scaler_dict[crop]
    preds = scaler.inverse_transform(preds).flatten()
    trues = scaler.inverse_transform(trues).flatten()

    return {
        "R2": r2_score(trues, preds),
        "MAE": mean_absolute_error(trues, preds),
        "MSE": mean_squared_error(trues, preds),
        "RMSE": mean_squared_error(trues, preds, squared=False),
        "MAPE": mean_absolute_percentage_error(trues, preds),
        "sMAPE": symmetric_mape(trues, preds),
        "NSE": nash_sutcliffe_efficiency(trues, preds)
    }

# ------------------ Main ------------------
if __name__ == "__main__":
    df_scaled, scaler_dict = load_cropwise_normalized_data("../src/csvfiles/grain_yield.csv")
    results = []

    for crop in df_scaled['Crop_type'].unique():
        crop_mask = df_scaled['Crop_type'] == crop
        signal = df_scaled.loc[crop_mask, 'Yield_value'].values
        processed = preprocess_and_amd(signal)
        processed = np.nan_to_num(processed)

        if len(processed) < seq_len + pred_len:
            print(f"[{crop}] Skipping due to short processed signal.")
            continue

        scaler = MinMaxScaler()
        processed_scaled = scaler.fit_transform(processed.reshape(-1, 1)).flatten()
        df_scaled.loc[crop_mask, 'Yield_value'] = processed_scaled
        scaler_dict[crop] = scaler

        crop_df = df_scaled[df_scaled['Crop_type'] == crop].copy()
        train, val, test = split_crop_data(crop_df, crop)

        model = Temporal_Depthwise_Informer(input_dim=256, seq_len=6, pred_len=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
        criterion = nn.MSELoss()

        train_loader = get_dataloader(train, crop, seq_len, pred_len, shuffle=True)
        val_loader = get_dataloader(val, crop, seq_len, pred_len, shuffle=False)
        test_loader = get_dataloader(test, crop, seq_len, pred_len, shuffle=False)

        train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, crop)
        metrics = test_model(model, f"best_model_{crop}.pth", test_loader, crop, scaler_dict)
        metrics['Crop'] = crop
        results.append(metrics)

    pd.DataFrame(results).to_csv("../src/csvfiles/output.csv", index=False)
