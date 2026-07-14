

import os, sys, time, warnings, argparse, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve, auc,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR     = "/Users/kartikeyahazela/research paper/Datasets"
OUTPUT_DIR      = "/Users/kartikeyahazela/research paper/Technical Code/ddos_fl_system/outputs"

CSV_FILES = [
    "DrDoS_DNS.csv", "DrDoS_LDAP.csv", "DrDoS_MSSQL.csv",
    "DrDoS_NetBIOS.csv", "DrDoS_NTP.csv", "DrDoS_SNMP.csv",
    "DrDoS_SSDP.csv", "DrDoS_UDP.csv", "Syn.csv", "TFTP.csv", "UDPLag.csv",
]

# IDs / timestamps / meta cols that must be dropped
DROP_COLS = [
    "Unnamed: 0", "Flow ID", "Source IP", "Destination IP",
    "Timestamp", "SimillarHTTP",
]

SAMPLE_PER_FILE  = 50_000        # rows per CSV file (initial load cap)
CHUNK_SIZE       = 100_000       # chunk size for pd.read_csv
MAX_TOTAL_ROWS   = 200_000       # hard cap AFTER balancing (step 6)
TOP_K_FEATURES   = 30            # features kept after RF importance selection
RF_N_ESTIMATORS  = 200
RF_MAX_DEPTH     = 20
CV_FOLDS         = 5             # cross-validation folds
MLP_EPOCHS       = 20
MLP_BATCH_SIZE   = 2048
MLP_LR           = 1e-3
RANDOM_STATE     = 42
N_FL_CLIENTS     = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  1. DATA LOADING  (chunked + sampled)
# ─────────────────────────────────────────────────────────────────────────────

def load_all_csvs(dataset_dir: str, files: list[str],
                  sample_per_file: int, chunk_size: int) -> pd.DataFrame:
    """
    Load each CSV in chunks, take up to `sample_per_file` rows, concatenate.
    Column names are stripped of surrounding whitespace immediately.
    """
    frames = []
    for idx, fname in enumerate(files, 1):
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [WARN] Not found, skipping: {fname}")
            continue

        print(f"  [{idx}/{len(files)}] Loading {fname} …")
        collected, n = [], 0
        try:
            reader = pd.read_csv(fpath, chunksize=chunk_size,
                                 low_memory=False, on_bad_lines="skip")
            for chunk in reader:
                chunk.columns = chunk.columns.str.strip()   # strip once
                collected.append(chunk)
                n += len(chunk)
                if n >= sample_per_file:
                    break

            file_df = pd.concat(collected, ignore_index=True)
            if len(file_df) > sample_per_file:
                file_df = file_df.sample(n=sample_per_file,
                                         random_state=RANDOM_STATE)
            frames.append(file_df)
            print(f"       ↳ kept {len(file_df):,} rows  |  cols: {file_df.shape[1]}")

        except Exception as exc:
            print(f"  [ERROR] {fname}: {exc}")
        finally:
            del collected
            gc.collect()

    print(f"\n  Concatenating {len(frames)} files …")
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  2. DROP DUPLICATES + CLASS BALANCING
# ─────────────────────────────────────────────────────────────────────────────

def clean_and_balance(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """
    1. Encode Label → 0 / 1 on the raw DataFrame.
    2. Drop exact duplicate rows.
    3. Undersample the majority class (ATTACK) to match minority (BENIGN).
    4. Cap total rows at `max_rows` to keep evaluation realistic.
    5. Shuffle.
    """
    # ── Strip & encode label ───────────────────────────────────────
    LABEL = "Label"
    df[LABEL] = df[LABEL].astype(str).str.strip()
    df[LABEL] = df[LABEL].apply(lambda v: 0 if v.upper() == "BENIGN" else 1)

    before_rows = len(df)
    print(f"\n  Rows before deduplication : {before_rows:,}")

    # ── Drop duplicates ────────────────────────────────────────────
    df = df.drop_duplicates()
    after_dedup = len(df)
    print(f"  Rows after  deduplication : {after_dedup:,}  "
          f"(removed {before_rows - after_dedup:,})")

    # ── Class distribution before balancing ───────────────────────
    n_benign = int((df[LABEL] == 0).sum())
    n_attack = int((df[LABEL] == 1).sum())
    print(f"\n  Before balancing → BENIGN: {n_benign:,}  |  ATTACK: {n_attack:,}")

    df_benign = df[df[LABEL] == 0]
    df_attack = df[df[LABEL] == 1]

    # ── Undersample majority to minority size ─────────────────────
    minority_n = min(n_benign, n_attack)

    df_benign_bal = resample(df_benign, replace=False,
                             n_samples=minority_n, random_state=RANDOM_STATE)
    df_attack_bal = resample(df_attack, replace=False,
                             n_samples=minority_n, random_state=RANDOM_STATE)

    df = pd.concat([df_benign_bal, df_attack_bal], ignore_index=True)

    # ── Hard cap at MAX_TOTAL_ROWS ─────────────────────────────────
    if len(df) > max_rows:
        half = max_rows // 2
        df = pd.concat([
            df[df[LABEL] == 0].sample(n=half, random_state=RANDOM_STATE),
            df[df[LABEL] == 1].sample(n=half, random_state=RANDOM_STATE),
        ], ignore_index=True)
        print(f"  Capped to {max_rows:,} rows to keep evaluation realistic.")

    # ── Shuffle ───────────────────────────────────────────────────
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    n_benign_final = int((df[LABEL] == 0).sum())
    n_attack_final = int((df[LABEL] == 1).sum())
    print(f"  After  balancing → BENIGN: {n_benign_final:,}  |  ATTACK: {n_attack_final:,}")
    print(f"  Final dataset shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  3. PREPROCESSING  (pure feature cleaning — no fitting)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract X and y from the cleaned DataFrame.  NO scaler / selector is
    fitted here — those happen strictly on the train split later.
    """
    df.columns = df.columns.str.strip()

    LABEL = "Label"
    y = df[LABEL].values.astype(np.int64)
    df = df.drop(columns=[LABEL])

    # Drop ID / meta columns
    drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=drop, inplace=True, errors="ignore")

    # Drop non-numeric columns (e.g. IP strings not caught above)
    non_num = df.select_dtypes(exclude="number").columns.tolist()
    if non_num:
        print(f"  Dropping non-numeric cols: {non_num}")
        df.drop(columns=non_num, inplace=True)

    feature_names = df.columns.tolist()
    X = df.values.astype(np.float32)
    del df

    # Replace NaN / ±Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n  Feature matrix → X: {X.shape}  y: {y.shape}")
    return X, y, feature_names


# ─────────────────────────────────────────────────────────────────────────────
#  4. FEATURE SELECTION  (train-only fit)
# ─────────────────────────────────────────────────────────────────────────────

def select_features_on_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    k: int = TOP_K_FEATURES,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Fit a lightweight RandomForest on TRAIN data only to get feature
    importances.  Returns the column indices, names, and importance values
    for the top-k features.
    """
    print(f"\n>>> Feature Selection — top {k} features (fit on train only) …")
    rf_sel = RandomForestClassifier(n_estimators=50, max_depth=10,
                                    n_jobs=-1, random_state=RANDOM_STATE)
    rf_sel.fit(X_train, y_train)
    importances = rf_sel.feature_importances_

    sorted_idx      = np.argsort(importances)[::-1]
    top_idx         = sorted_idx[:k]
    top_features    = [feature_names[i] for i in top_idx]
    top_importances = importances[top_idx]

    print(f"  Selected {k} / {X_train.shape[1]} features")
    return top_idx, top_features, top_importances


# ─────────────────────────────────────────────────────────────────────────────
#  5a. RANDOM FOREST  (with 5-fold CV)
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    print(f"\n>>> Training RandomForestClassifier "
          f"(n={RF_N_ESTIMATORS}, depth={RF_MAX_DEPTH}) …")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    print(f"  Training time : {time.time()-t0:.1f}s")

    # ── 5-fold cross-validation ──────────────────────────────────
    print(f"  Running {CV_FOLDS}-fold cross-validation …")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    cv_scores = cross_val_score(rf, X_train, y_train,
                                cv=cv, scoring="f1", n_jobs=-1)
    print(f"  CV F1 scores  : {cv_scores.round(4)}")
    print(f"  CV F1 mean    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return rf


# ─────────────────────────────────────────────────────────────────────────────
#  5b. PYTORCH MLP
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = MLP_EPOCHS,
    batch_size: int = MLP_BATCH_SIZE,
    lr: float = MLP_LR,
) -> MLP:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n>>> Training PyTorch MLP on {device} "
          f"(epochs={epochs}, batch={batch_size}) …")

    dataset   = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                               torch.tensor(y_train, dtype=torch.long))
    loader    = DataLoader(dataset, batch_size=batch_size,
                           shuffle=True, num_workers=0)
    model     = MLP(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for bX, by in loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print(f"  Epoch {epoch:02d}/{epochs}  loss={epoch_loss/len(loader):.4f}")

    print(f"  MLP training time: {time.time()-t0:.1f}s")
    return model


def predict_mlp(model: MLP, X: np.ndarray) -> np.ndarray:
    """Hard class predictions."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
    return logits.argmax(dim=1).cpu().numpy()


def predict_proba_mlp(model: MLP, X: np.ndarray) -> np.ndarray:
    """Probability of class 1 (ATTACK) for ROC / PR curves."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        probs  = torch.softmax(logits, dim=1)[:, 1]
    return probs.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  6. EVALUATION  (accuracy, precision, recall, F1, AUC-ROC)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> dict:
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    cm    = confusion_matrix(y_true, y_pred)
    roc   = roc_auc_score(y_true, y_prob)

    # Precision-Recall AUC
    pr_p, pr_r, _ = precision_recall_curve(y_true, y_prob)
    pr_auc        = auc(pr_r, pr_p)

    print(f"\n{'='*54}")
    print(f"  Results — {model_name}")
    print(f"{'='*54}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  F1-Score          : {f1:.4f}")
    print(f"  ROC-AUC           : {roc:.4f}")
    print(f"  PR-AUC            : {pr_auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n"
          f"{classification_report(y_true, y_pred, target_names=['BENIGN','ATTACK'], zero_division=0)}")
    print(f"{'='*54}")

    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": roc, "pr_auc": pr_auc,
        "cm": cm, "pr_precision": pr_p, "pr_recall": pr_r,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  7. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["BENIGN", "ATTACK"],
                yticklabels=["BENIGN", "ATTACK"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    slug  = model_name.replace(" ", "_").lower()
    fname = os.path.join(output_dir, f"cm_{slug}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [Plot] Confusion matrix saved → {fname}")


def plot_pr_curve(
    rf_metrics: dict,
    mlp_metrics: dict,
    output_dir: str,
) -> None:
    """Precision-Recall curves for both models on the same axes."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, m, color in [
        ("RandomForest", rf_metrics,  "steelblue"),
        ("PyTorch MLP",  mlp_metrics, "darkorange"),
    ]:
        ax.plot(m["pr_recall"], m["pr_precision"],
                label=f"{name} (AUC={m['pr_auc']:.3f})", color=color, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [Plot] PR curve saved → {fname}")


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    output_dir: str,
    top_n: int = 20,
) -> None:
    n     = min(top_n, len(feature_names))
    names = feature_names[:n]
    vals  = importances[:n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(n), vals[::-1], color="steelblue")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance (RandomForest — train-set fit)")
    plt.tight_layout()
    fname = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [Plot] Feature importance saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
#  8. SAVE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def save_models(
    rf: RandomForestClassifier,
    mlp: MLP,
    scaler: StandardScaler,
    top_idx: np.ndarray,
    output_dir: str,
) -> None:
    joblib.dump(rf,      os.path.join(output_dir, "random_forest.joblib"))
    joblib.dump(scaler,  os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(top_idx, os.path.join(output_dir, "feature_indices.joblib"))
    torch.save(mlp.state_dict(), os.path.join(output_dir, "mlp_model.pt"))
    print(f"\n  [Save] All models saved to → {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
#  BONUS — FEDERATED LEARNING SIMULATION  (FedAvg, 3 clients)
# ─────────────────────────────────────────────────────────────────────────────

def _fedavg(weight_list: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    return [torch.stack(lw).mean(dim=0) for lw in zip(*weight_list)]


def run_federated_simulation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_clients: int = N_FL_CLIENTS,
    rounds: int = 3,
    local_epochs: int = 5,
) -> None:
    print(f"\n{'='*60}")
    print(f"  BONUS — Federated Learning Simulation")
    print(f"  Clients={n_clients}  Rounds={rounds}  LocalEpochs={local_epochs}")
    print(f"{'='*60}")

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = X_train.shape[1]
    splits    = np.array_split(np.random.permutation(len(X_train)), n_clients)
    c_data    = [(X_train[s], y_train[s]) for s in splits]
    global_m  = MLP(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()

    for rnd in range(1, rounds + 1):
        print(f"\n  ── FL Round {rnd}/{rounds} ──")
        all_weights = []
        for cid, (cX, cy) in enumerate(c_data, 1):
            local_m = MLP(input_dim).to(device)
            local_m.load_state_dict(global_m.state_dict())
            opt = torch.optim.Adam(local_m.parameters(), lr=MLP_LR)
            ldr = DataLoader(
                TensorDataset(torch.tensor(cX, dtype=torch.float32),
                              torch.tensor(cy, dtype=torch.long)),
                batch_size=MLP_BATCH_SIZE, shuffle=True, num_workers=0,
            )
            local_m.train()
            for _ in range(local_epochs):
                for bX, by in ldr:
                    bX, by = bX.to(device), by.to(device)
                    opt.zero_grad()
                    criterion(local_m(bX), by).backward()
                    opt.step()
            all_weights.append([p.data.clone() for p in local_m.parameters()])
            print(f"    Client {cid} done — {len(cX):,} samples")

        averaged = _fedavg(all_weights)
        with torch.no_grad():
            for p, w in zip(global_m.parameters(), averaged):
                p.copy_(w)

        preds = predict_mlp(global_m, X_test)
        print(f"    Round {rnd} global → "
              f"Acc={accuracy_score(y_test, preds):.4f}  "
              f"F1={f1_score(y_test, preds, zero_division=0):.4f}")

    print("\n  Final Federated Model Evaluation:")
    fed_preds = predict_mlp(global_m, X_test)
    fed_proba = predict_proba_mlp(global_m, X_test)
    evaluate(y_test, fed_preds, fed_proba, "Federated MLP (FedAvg)")
    torch.save(global_m.state_dict(),
               os.path.join(OUTPUT_DIR, "federated_mlp_model.pt"))
    print(f"  [Save] Federated model → {OUTPUT_DIR}/federated_mlp_model.pt")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DDoS Detection Pipeline v2")
    parser.add_argument("--federated", action="store_true",
                        help="Run federated learning simulation.")
    parser.add_argument("--sample", type=int, default=SAMPLE_PER_FILE,
                        help=f"Rows per CSV file (default: {SAMPLE_PER_FILE})")
    args = parser.parse_args()

    print("=" * 60)
    print("  DDoS Detection Pipeline v2 — CICDDoS2019")
    print("=" * 60)
    t_start = time.time()

    # ── Step 1: Load ─────────────────────────────────────────────
    print("\n>>> Step 1: Loading data …")
    df = load_all_csvs(DATASET_DIR, CSV_FILES, args.sample, CHUNK_SIZE)
    print(f"  Raw combined shape: {df.shape}")

    # ── Step 2: Dedup + Balance ───────────────────────────────────
    print("\n>>> Step 2: Deduplication + Class Balancing …")
    df = clean_and_balance(df, max_rows=MAX_TOTAL_ROWS)

    # ── Step 3: Extract feature matrix ───────────────────────────
    print("\n>>> Step 3: Feature Extraction …")
    X, y, feature_names = extract_features(df)
    del df
    gc.collect()

    # ── Step 4: Train/Test split (FIRST — prevents leakage) ──────
    print("\n>>> Step 4: Train/Test split (80/20 stratified) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    # ── Step 4a: StandardScaler — fit on TRAIN only ───────────────
    print("  Fitting StandardScaler on train split only …")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # ── Step 4b: Feature selection — fit on TRAIN only ────────────
    print("\n>>> Step 4b: Feature Selection …")
    top_idx, top_features, top_importances = select_features_on_train(
        X_train, y_train, feature_names, k=TOP_K_FEATURES,
    )
    X_train_sel = X_train[:, top_idx]
    X_test_sel  = X_test[:, top_idx]
    print(f"  Shape after selection → train: {X_train_sel.shape}  "
          f"test: {X_test_sel.shape}")

    # ── Step 5a: Random Forest ────────────────────────────────────
    print("\n>>> Step 5a: Random Forest …")
    rf       = train_random_forest(X_train_sel, y_train)
    rf_preds = rf.predict(X_test_sel)
    rf_proba = rf.predict_proba(X_test_sel)[:, 1]
    rf_metrics = evaluate(y_test, rf_preds, rf_proba, "RandomForest")

    # ── Step 5b: PyTorch MLP ──────────────────────────────────────
    print("\n>>> Step 5b: PyTorch MLP …")
    mlp       = train_mlp(X_train_sel, y_train)
    mlp_preds = predict_mlp(mlp, X_test_sel)
    mlp_proba = predict_proba_mlp(mlp, X_test_sel)
    mlp_metrics = evaluate(y_test, mlp_preds, mlp_proba, "PyTorch MLP")

    # ── Step 7: Plots ────────────────────────────────────────────
    print("\n>>> Step 7: Generating plots …")
    plot_confusion_matrix(rf_metrics["cm"],  "RandomForest", OUTPUT_DIR)
    plot_confusion_matrix(mlp_metrics["cm"], "PyTorch MLP",  OUTPUT_DIR)
    plot_pr_curve(rf_metrics, mlp_metrics, OUTPUT_DIR)
    plot_feature_importance(top_features, top_importances, OUTPUT_DIR)

    # ── Step 8: Save ─────────────────────────────────────────────
    print("\n>>> Step 8: Saving models …")
    save_models(rf, mlp, scaler, top_idx, OUTPUT_DIR)

    # ── Step 9: Summary ──────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    header = f"  {'Model':<20}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'ROC-AUC':>8}  {'PR-AUC':>7}"
    print(header)
    print(f"  {'-'*70}")
    for name, m in [("RandomForest", rf_metrics), ("PyTorch MLP", mlp_metrics)]:
        print(f"  {name:<20}  {m['accuracy']:>7.4f}  {m['precision']:>7.4f}"
              f"  {m['recall']:>7.4f}  {m['f1']:>7.4f}"
              f"  {m['roc_auc']:>8.4f}  {m['pr_auc']:>7.4f}")
    print(f"{'='*60}")
    print(f"  Total runtime: {elapsed/60:.1f} min")

    # ── Bonus: Federated ─────────────────────────────────────────
    if args.federated:
        run_federated_simulation(X_train_sel, y_train, X_test_sel, y_test)

    print(f"\n  Done. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
