import os
import math
import time
import random
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

#config
SENSOR_COUNT = 21
OP_COUNT = 3
BASE_COLS = ["unit", "cycle"]
OP_COLS = [f"op{i}" for i in range(1, OP_COUNT + 1)]
SENSOR_COLS = [f"s{i}" for i in range(1, SENSOR_COUNT + 1)]
ALL_COLS = BASE_COLS + OP_COLS + SENSOR_COLS


#utils
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_txt(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] > len(ALL_COLS):
        df = df.iloc[:, :len(ALL_COLS)]
    df.columns = ALL_COLS
    return df


def load_data(data_dir, subset):
    train = read_txt(Path(data_dir) / f"train_{subset}.txt")
    test = read_txt(Path(data_dir) / f"test_{subset}.txt")
    rul = pd.read_csv(Path(data_dir) / f"RUL_{subset}.txt", header=None)
    rul.columns = ["RUL_end"]
    return train, test, rul


def add_train_rul(df, cap):
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def add_test_rul(df, rul_df, cap):
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit")

    units = sorted(df["unit"].unique())
    rul_map = {units[i]: int(rul_df.iloc[i, 0]) for i in range(len(units))}

    df["RUL_end"] = df["unit"].map(rul_map)
    df["RUL"] = (df["max_cycle"] - df["cycle"]) + df["RUL_end"]

    df.drop(columns=["max_cycle"], inplace=True)
    df["RUL"] = df["RUL"].clip(upper=cap)

    return df


#eda
def exploratory_analysis(df, results_dir):

    print("\n===== DATASET OVERVIEW =====")
    print("Shape:", df.shape)
    print("Engines:", df["unit"].nunique())

    summary = df.describe()
    summary.to_csv(results_dir / "statistical_summary.csv")

    plt.figure()
    df["RUL"].hist(bins=50)
    plt.title("RUL Distribution")
    plt.xlabel("RUL")
    plt.ylabel("Frequency")
    plt.savefig(results_dir / "rul_distribution.png")
    plt.close()

    sample = df[df["unit"] == df["unit"].iloc[0]]

    plt.figure()
    plt.plot(sample["cycle"], sample["s1"])
    plt.title("Sensor1 Degradation Trend")
    plt.xlabel("Cycle")
    plt.ylabel("Sensor1")
    plt.savefig(results_dir / "sensor_trend.png")
    plt.close()


#sequence builder
def build_sequences(df, features, window, stride):

    X, y = [], []
    df = df.sort_values(["unit", "cycle"])

    for uid, g in df.groupby("unit"):

        g = g.reset_index(drop=True)

        if len(g) < window:
            continue

        feats = g[features].values.astype(np.float32)
        rul = g["RUL"].values.astype(np.float32)

        for i in range(0, len(g) - window + 1, stride):

            X.append(feats[i:i + window])
            y.append(rul[i + window - 1])

    return np.array(X), np.array(y)


#last window
def last_windows(df, features, window):

    X, y, engines = [], [], []

    for uid, g in df.groupby("unit"):

        g = g.reset_index(drop=True)

        feats = g[features].values.astype(np.float32)

        if len(g) < window:
            pad = np.repeat(feats[:1], window - len(g), axis=0)
            win = np.vstack([pad, feats])
        else:
            win = feats[-window:]

        X.append(win)
        y.append(g["RUL"].iloc[-1])
        engines.append(uid)

    return np.array(X), np.array(y), engines


#dataset
class RULDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#lstm model
class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden, layers, dropout):

        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):

        _, (h, _) = self.lstm(x)

        return self.fc(h[-1]).squeeze()


#gru model
class GRUModel(nn.Module):

    def __init__(self, input_size, hidden, layers, dropout):

        super().__init__()

        self.gru = nn.GRU(input_size, hidden, layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):

        _, h = self.gru(x)

        return self.fc(h[-1]).squeeze()


#cnn model
class CNNModel(nn.Module):

    def __init__(self, input_size, hidden):

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, hidden, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):

        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.squeeze(-1)

        return self.fc(x).squeeze()


#metrics
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - ss_res / ss_tot


def nasa_score(y_true, y_pred):

    d = y_pred - y_true
    score = 0

    for di in d:

        if di < 0:
            score += math.exp(-di / 13) - 1
        else:
            score += math.exp(di / 10) - 1

    return score


#train epoch
def train_epoch(model, loader, opt, loss_fn, device):

    model.train()

    total = 0

    for x, y in loader:

        x, y = x.to(device), y.to(device)

        opt.zero_grad()

        pred = model(x)

        loss = loss_fn(pred, y)

        loss.backward()

        opt.step()

        total += loss.item()

    return total / len(loader)


#main
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--subset", type=str, default="FD001")

    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=512)

    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)

    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--cap", type=int, default=125)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    train_df, test_df, rul_df = load_data(args.data_dir, args.subset)

    train_df = add_train_rul(train_df, args.cap)
    test_df = add_test_rul(test_df, rul_df, args.cap)

    exploratory_analysis(train_df, results_dir)

    features = OP_COLS + SENSOR_COLS

    scaler = StandardScaler()

    scaler.fit(train_df[features])

    train_df[features] = scaler.transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    X_train, y_train = build_sequences(train_df, features, args.window, args.stride)

    X_test, y_test, engines = last_windows(test_df, features, args.window)

    loader = DataLoader(RULDataset(X_train, y_train), batch_size=args.batch, shuffle=True)

    models = {
        "LSTM": LSTMModel(len(features), args.hidden, args.layers, args.dropout).to(device),
        "GRU": GRUModel(len(features), args.hidden, args.layers, args.dropout).to(device),
        "CNN": CNNModel(len(features), args.hidden).to(device)
    }

    results = []

    for name, model in models.items():

        print(f"\nTraining {name}")

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        loss_fn = nn.MSELoss()

        history = []

        for epoch in range(args.epochs):

            loss = train_epoch(model, loader, opt, loss_fn, device)

            history.append(loss)

            print(f"{name} Epoch {epoch+1} Loss {loss:.4f}")

        plt.figure()
        plt.plot(history)
        plt.title(f"{name} Training Loss")
        plt.savefig(results_dir / f"{name}_loss.png")
        plt.close()

        model.eval()

        with torch.no_grad():

            preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

        preds = np.clip(preds, 0, args.cap)

        rmse_val = rmse(y_test, preds)
        mae_val = mae(y_test, preds)
        r2_val = r2_score(y_test, preds)
        nasa_val = nasa_score(y_test, preds)

        results.append([name, rmse_val, mae_val, r2_val, nasa_val])

        plt.figure()
        plt.scatter(y_test, preds)
        plt.title(f"{name} Prediction vs True")
        plt.savefig(results_dir / f"{name}_scatter.png")
        plt.close()

        plt.figure()
        plt.hist(preds - y_test, bins=30)
        plt.title(f"{name} Error Distribution")
        plt.savefig(results_dir / f"{name}_error.png")
        plt.close()

    table = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R2", "NASA_Score"])

    print("\n===== MODEL COMPARISON =====")
    print(table.to_string(index=False))

    table.to_csv(results_dir / "model_comparison.csv", index=False)


if __name__ == "__main__":
    main()