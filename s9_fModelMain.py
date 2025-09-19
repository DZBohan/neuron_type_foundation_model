import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# ---------- Config Zone ----------
DATA_PATH     = "/coh_labs/dits/bzhang/test/hWorldFM2/subset_2523_210_Cerebral-cortex.screened.h5ad"
CHECKPOINT_DIR= "ckpts_2523_210_1"
LOG_DIR       = "logs_2523_210_1"
LOG_FILE      = os.path.join(LOG_DIR, "train.log")
FIG_DIR       = "figs_2523_210_1"

SEED       = 42
USE_LOG1P  = True
NUM_BINS   = 20
BATCH_SIZE = 64
VAL_BATCH  = 128
EPOCHS     = 30
EMB_DIM    = 64
NHEAD      = 4
NLAYERS    = 3
DROPOUT    = 0.1
LR         = 5e-4
WEIGHT_DECAY = 1e-2
VAL_RATIO  = 0.2
MASK_RATIO = 0.15
NONZERO_FRAC = 0.5

# ---------- logging ----------
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logger = logging.getLogger("trainer")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
sh = logging.StreamHandler()
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
fh.setFormatter(fmt)
sh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(sh)

def log(msg):
    logger.info(msg)

# ---------- seeds & device ----------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")
log(f"Using device: {device}")

# ---------- load ----------
adata = sc.read_h5ad(DATA_PATH)
if hasattr(adata, "layers") and ("counts" in adata.layers):
    X_counts = adata.layers["counts"]
    adata.X = X_counts.toarray() if hasattr(X_counts, "toarray") else X_counts
sc.pp.normalize_total(adata, target_sum=1e4)
if USE_LOG1P:
    sc.pp.log1p(adata)

# ---------- split ----------
X_full = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
n_cells, n_genes = X_full.shape
n_val = int(n_cells * VAL_RATIO)
n_train = n_cells - n_val
indices = np.arange(n_cells)
rng = np.random.RandomState(SEED)
rng.shuffle(indices)
train_idx = indices[:n_train]
val_idx = indices[n_train:]
X_train = X_full[train_idx]
X_val   = X_full[val_idx]
log(f"Dataset: {n_cells} cells x {n_genes} genes | train={n_train}, val={n_val}")

# ---------- binning ----------
def compute_bin_edges_per_gene(X, num_bins=NUM_BINS):
    qs = np.linspace(0, 1, num_bins + 1)
    edges_list = []
    for g in range(X.shape[1]):
        e = np.unique(np.quantile(X[:, g], qs))
        if e.size < 2:
            minv, maxv = X[:, g].min(), X[:, g].max()
            if maxv <= minv:
                maxv = minv + 1e-8
            e = np.array([minv, maxv])
        edges_list.append(e)
    return edges_list

def apply_bin_edges(X, edges_list, num_bins=NUM_BINS):
    B = np.zeros_like(X, dtype=int)
    for g in range(X.shape[1]):
        idx = np.searchsorted(edges_list[g], X[:, g], side="right") - 1
        B[:, g] = np.clip(idx, 0, num_bins - 1)
    return B

edges = compute_bin_edges_per_gene(X_train, num_bins=NUM_BINS)
binned_train = apply_bin_edges(X_train, edges, num_bins=NUM_BINS)

# === 新增：保存“分桶边界 + 基因顺序 + 预处理设置”，以便下游/推理复用 ===
np.savez(
    os.path.join(CHECKPOINT_DIR, "binning_meta.npz"),
    edges=np.array(edges, dtype=object),          # 逐基因边界（object 数组）
    gene_names=np.array(adata.var_names),         # 训练时基因顺序
    num_bins=np.int32(NUM_BINS),                  # 分桶数
    target_sum=np.float32(1e4),                   # 归一化用的 target_sum
    use_log1p=np.bool_(USE_LOG1P)                 # 是否做了 log1p
)

binned_val   = apply_bin_edges(X_val,   edges, num_bins=NUM_BINS)

# ---------- dataset ----------
class MaskedBinnedDataset(Dataset):
    def __init__(self, binned_matrix, mask_ratio=MASK_RATIO, nonzero_frac=NONZERO_FRAC, mask_token_id=None, seed=SEED, deterministic=True):
        self.data = binned_matrix
        self.mask_ratio = mask_ratio
        self.nonzero_frac = nonzero_frac
        self.mask_token_id = mask_token_id
        self.deterministic = deterministic
        self.base_seed = seed
    def __len__(self):
        return self.data.shape[0]
    def _rng(self, idx):
        s = (self.base_seed + idx) if self.deterministic else np.random.randint(0, 2**31-1)
        return np.random.RandomState(s)
    def __getitem__(self, idx):
        row = self.data[idx]
        rng = self._rng(idx)
        L = len(row)
        target_k = max(1, int(self.mask_ratio * L))
        nonzero_idx = np.where(row > 0)[0]
        zero_idx    = np.where(row == 0)[0]
        k_nonzero_target = int(np.round(self.nonzero_frac * target_k))
        k_nonzero = min(len(nonzero_idx), k_nonzero_target)
        k_zero    = target_k - k_nonzero
        if k_zero > len(zero_idx):
            deficit = k_zero - len(zero_idx)
            k_zero = len(zero_idx)
            k_nonzero = min(len(nonzero_idx), k_nonzero + deficit)
        mask_nonzero = rng.choice(nonzero_idx, size=k_nonzero, replace=False) if k_nonzero > 0 else np.array([], dtype=int)
        mask_zero    = rng.choice(zero_idx,    size=k_zero,    replace=False) if k_zero    > 0 else np.array([], dtype=int)
        mask_idx = np.concatenate([mask_nonzero, mask_zero])
        input_ids = row.copy()
        labels = np.full_like(row, fill_value=-100, dtype=int)
        input_ids[mask_idx] = self.mask_token_id
        labels[mask_idx] = row[mask_idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

MASK_TOKEN_ID = NUM_BINS
train_dataset = MaskedBinnedDataset(binned_train, mask_token_id=MASK_TOKEN_ID, deterministic=False)
val_dataset   = MaskedBinnedDataset(binned_val,   mask_token_id=MASK_TOKEN_ID, deterministic=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=VAL_BATCH, shuffle=False)

# ---------- model ----------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, n_genes, emb_dim=EMB_DIM, nhead=NHEAD, nlayers=NLAYERS, dropout=DROPOUT, mask_token_id=MASK_TOKEN_ID):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.token_embed = nn.Embedding(vocab_size + 1, emb_dim)
        self.gene_embed  = nn.Embedding(n_genes, emb_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.output = nn.Linear(emb_dim, vocab_size)
    def forward(self, x):
        bsz, seqlen = x.shape
        tok = self.token_embed(x)
        gene_idx = torch.arange(seqlen, device=x.device, dtype=torch.long)
        gene = self.gene_embed(gene_idx).unsqueeze(0).expand(bsz, -1, -1)
        emb = tok + gene
        out = self.transformer(emb)
        logits = self.output(out)
        return logits

model = SimpleTransformer(vocab_size=NUM_BINS, n_genes=n_genes).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

@torch.no_grad()
def eval_metrics(model, loader):
    model.eval()
    total_loss = 0.0
    total_mask = 0
    total_correct = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        B, S, V = logits.shape
        logits_2d = logits.view(-1, V)
        labels_1d = labels.view(-1)
        loss = loss_fn(logits_2d, labels_1d)
        total_loss += loss.item()
        mask_positions = labels_1d != -100
        if mask_positions.any():
            preds = logits_2d.argmax(dim=-1)
            total_correct += (preds[mask_positions] == labels_1d[mask_positions]).sum().item()
            total_mask += mask_positions.sum().item()
    avg_loss = total_loss / max(1, len(loader))
    ppl = float(np.exp(avg_loss))
    acc = total_correct / total_mask if total_mask > 0 else 0.0
    return {"loss": avg_loss, "perplexity": ppl, "masked_acc": acc}

# ---------- training ----------
train_losses, val_losses, val_ppl_list, val_acc_list = [], [], [], []
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        V = logits.shape[-1]
        loss = loss_fn(logits.view(-1, V), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_avg_loss = running_loss / max(1, len(train_loader))
    val_stats = eval_metrics(model, val_loader)

    train_losses.append(train_avg_loss)
    val_losses.append(val_stats["loss"])
    val_ppl_list.append(val_stats["perplexity"])
    val_acc_list.append(val_stats["masked_acc"])

    log(f"Epoch {epoch:02d} | train_loss={train_avg_loss:.4f} | val_loss={val_stats['loss']:.4f} | val_ppl={val_stats['perplexity']:.2f} | val_masked_acc={val_stats['masked_acc']:.3f}")

    if val_stats["loss"] < best_val_loss:
        best_val_loss = val_stats["loss"]
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "hello_model_best.pt"))

torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "hello_model_last.pt"))
log("Saved: checkpoints/hello_model_best.pt (best on val), hello_model_last.pt (last epoch)")

# ---------- quick demo ----------
@torch.no_grad()
def quick_demo_sample(model, dataset, sample_idx=0, topk=3):
    model.eval()
    x, y = dataset[sample_idx]
    logits = model(x.unsqueeze(0))
    probs = torch.softmax(logits[0], dim=-1)
    log("Quick demo on one cell (masked positions only):")
    for i in range(len(x)):
        if y[i].item() != -100:
            topv, topi = probs[i].topk(topk)
            true_bin = y[i].item()
            log(f"Gene {i:03d}: true={true_bin:02d}, top{topk}={topi.tolist()}")

quick_demo_sample(model, val_dataset, sample_idx=0, topk=3)

# ---------- plots ----------
def _save_lineplot(y_list, ylabel, fname):
    plt.figure()
    plt.plot(np.arange(1, len(y_list)+1), y_list)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    log(f"Saved figure: {path}")

_save_lineplot(train_losses, "Train Loss",    "train_loss.png")
_save_lineplot(val_losses,   "Val Loss",      "val_loss.png")
_save_lineplot(val_ppl_list, "Val Perplexity","val_ppl.png")
_save_lineplot(val_acc_list, "Val Masked Acc","val_acc.png")

# ---------- save metrics as CSV ----------
import pandas as pd
metrics_df = pd.DataFrame({
    "epoch": np.arange(1, EPOCHS+1),
    "train_loss": train_losses,
    "val_loss": val_losses,
    "val_perplexity": val_ppl_list,
    "val_masked_acc": val_acc_list,
})
metrics_csv = os.path.join(LOG_DIR, "training_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)
log(f"Saved metrics: {metrics_csv}")
