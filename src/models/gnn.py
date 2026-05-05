"""
src/models/gnn.py
------------------
Week 9: GraphSAGE GNN with Optuna hyperparameter tuning.

Why Optuna for GNN
------------------
Fixed parameters converged too quickly (epoch 47) with test_rmse=11.850.
Optuna searches hidden dimensions, dropout, learning rate, weight decay,
and number of layers — letting the data determine the right architecture.

Stability fixes retained from debugging
----------------------------------------
1. Labels normalised to [0,1] — prevents MSE ~4000 at init
2. Feature clipping at +-5 std devs — eliminates 54-sigma outliers
3. All-NaN column removal before scaling (safety_sidewalk_coverage)
4. Head bias = 0.68 (Walk Score mean / 100)
5. LayerNorm instead of BatchNorm
6. Gradient clipping at 1.0

Usage
-----
    python -m src.models.gnn               # 30 Optuna trials
    python -m src.models.gnn --fast        # 10 trials, 50 epochs
    python -m src.models.gnn --trials 50   # custom trial count
"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import geopandas as gpd
import h3, mlflow
import numpy as np, optuna, pandas as pd
import torch, torch.nn.functional as F
from loguru import logger
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from src.utils.config import cfg
from src.models.utils import load_modeling_data, CENTER_FOLD

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED     = 42
GRAD_CLIP       = 1.0
MAX_EPOCHS      = 200
PATIENCE        = 25
N_TRIALS        = 30
FAST_TRIALS     = 10
FAST_EPOCHS     = 50
WALK_SCORE_MEAN = 68.0


def build_h3_graph(hex_ids):
    hex_to_idx = {hid: i for i, hid in enumerate(hex_ids)}
    valid_set  = set(hex_ids)
    src_list, dst_list = [], []
    for hid in hex_ids:
        src_idx    = hex_to_idx[hid]
        neighbours = set(h3.grid_disk(hid, 1)) - {hid}
        for nb in neighbours:
            if nb in valid_set:
                src_list.append(src_idx)
                dst_list.append(hex_to_idx[nb])
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    logger.info(f"Graph: {len(hex_ids):,} nodes  {edge_index.shape[1]:,} edges  avg_degree={edge_index.shape[1]/len(hex_ids):.1f}")
    return edge_index


def build_graph_data(X, y, folds, hex_ids, device):
    train_np = (folds != CENTER_FOLD).values
    X_clean  = X.copy()
    nan_cols = X_clean.columns[X_clean.isna().all()].tolist()
    if nan_cols:
        X_clean = X_clean.drop(columns=nan_cols)
    train_med = X_clean.iloc[train_np].median()
    X_filled  = X_clean.fillna(train_med).fillna(0.0)
    X_arr     = X_filled.values.astype(np.float32)
    scaler    = StandardScaler()
    scaler.fit(X_arr[train_np])
    X_scaled  = np.clip(scaler.transform(X_arr), -5.0, 5.0)
    X_scaled  = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    x        = torch.tensor(X_scaled,       dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values/100.0, dtype=torch.float32).to(device)
    edge_index = build_h3_graph(hex_ids).to(device)
    train_mask = torch.tensor(((folds != CENTER_FOLD) & (folds != 1)).values, dtype=torch.bool).to(device)
    val_mask   = torch.tensor((folds == 1).values,           dtype=torch.bool).to(device)
    test_mask  = torch.tensor((folds == CENTER_FOLD).values, dtype=torch.bool).to(device)
    return x, y_tensor, edge_index, train_mask, val_mask, test_mask, scaler


class WalkabilityGNN(nn.Module):
    def __init__(self, in_channels, hidden_dims, dropout):
        super().__init__()
        try:
            from torch_geometric.nn import SAGEConv
        except ImportError:
            raise ImportError("pip install torch_geometric")
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_channels] + hidden_dims
        for i in range(len(hidden_dims)):
            self.convs.append(SAGEConv(dims[i], dims[i+1]))
            self.norms.append(nn.LayerNorm(dims[i+1]))
        self.dropout = nn.Dropout(p=dropout)
        self.head    = nn.Linear(hidden_dims[-1], 1)
        nn.init.xavier_uniform_(self.head.weight, gain=0.01)
        nn.init.constant_(self.head.bias, WALK_SCORE_MEAN / 100.0)

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.head(x).squeeze(-1)


def train_epoch(model, opt, x, edge_index, y, mask):
    model.train()
    opt.zero_grad()
    loss = F.mse_loss(model(x, edge_index)[mask], y[mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
    opt.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(model, x, edge_index, y, mask):
    model.eval()
    preds  = model(x, edge_index)[mask].cpu().numpy() * 100.0
    y_true = y[mask].cpu().numpy() * 100.0
    rmse   = float(np.sqrt(mean_squared_error(y_true, preds)))
    ss_res = np.sum((y_true - preds)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2     = float(1 - ss_res/ss_tot) if ss_tot > 0 else 0.0
    return rmse, r2


def train_with_early_stopping(model, x, edge_index, y, train_mask, val_mask,
                               lr, weight_decay, max_epochs=MAX_EPOCHS,
                               patience=PATIENCE, verbose=False):
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=10, factor=0.5)
    best_rmse, best_state, no_improve = float("inf"), None, 0
    history = {"train_loss": [], "val_rmse": []}
    for epoch in range(1, max_epochs+1):
        loss             = train_epoch(model, opt, x, edge_index, y, train_mask)
        val_rmse, val_r2 = evaluate(model, x, edge_index, y, val_mask)
        sched.step(val_rmse)
        history["train_loss"].append(loss)
        history["val_rmse"].append(val_rmse)
        if val_rmse < best_rmse:
            best_rmse  = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if verbose and (epoch % 20 == 0 or epoch == 1):
            logger.info(f"  Epoch {epoch:>4}/{max_epochs}  loss={loss:.4f}  val_rmse={val_rmse:.3f}  val_r2={val_r2:.3f}")
        if no_improve >= patience:
            if verbose:
                logger.info(f"  Early stopping at epoch {epoch}")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_rmse, history


def make_objective(x, y, edge_index, train_mask, val_mask, in_channels, device, max_epochs):
    def objective(trial):
        torch.manual_seed(RANDOM_SEED)
        n_layers    = trial.suggest_int("n_layers", 1, 3)
        hidden_dims = []
        prev        = None
        for i in range(n_layers):
            hi  = prev if prev else 256
            dim = trial.suggest_int(f"hidden_dim_{i}", 16, hi, step=16)
            hidden_dims.append(dim)
            prev = dim
        dropout      = trial.suggest_float("dropout",      0.0,  0.6)
        lr           = trial.suggest_float("lr",           1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        model        = WalkabilityGNN(in_channels, hidden_dims, dropout).to(device)
        _, val_rmse, _ = train_with_early_stopping(
            model, x, edge_index, y, train_mask, val_mask,
            lr=lr, weight_decay=weight_decay,
            max_epochs=max_epochs, patience=PATIENCE,
        )
        return val_rmse
    return objective


def run_gnn_pipeline(n_trials=N_TRIALS, max_epochs=MAX_EPOCHS):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    slug = cfg.city.slug
    cfg.paths.models.mkdir(parents=True, exist_ok=True)
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if hasattr(torch.backends,"mps") and torch.backends.mps.is_available()
              else torch.device("cpu"))
    logger.info(f"Device: {device}")
    
    mlflow.set_experiment("walkability-models")

    with mlflow.start_run(run_name=f"gnn_tuned_{slug}"):
        logger.info("Loading modeling data...")
        X, y, folds, feature_cols = load_modeling_data()
        net    = gpd.read_parquet(str(cfg.paths.processed/f"{slug}_network_features.parquet"))[["h3_index","data_sparse"]]
        labels = pd.read_parquet(str(cfg.paths.labels/f"{slug}_walk_scores.parquet"))[["h3_index","walk_score"]]
        splits = pd.read_parquet(str(cfg.paths.splits/f"{slug}_spatial_cv.parquet"))[["h3_index","fold"]]
        base   = net[net["data_sparse"]==0].merge(labels.dropna(subset=["walk_score"]),on="h3_index").merge(splits,on="h3_index").reset_index(drop=True)
        hex_ids = base["h3_index"].tolist()

        logger.info("Building graph and preprocessing...")
        x, y_tensor, edge_index, train_mask, val_mask, test_mask, scaler = build_graph_data(X, y, folds, hex_ids, device)
        in_channels = x.shape[1]
        logger.info(f"Graph ready: {x.shape[0]:,} nodes  {edge_index.shape[1]:,} edges  feature_dim={in_channels}")
        mlflow.log_params({"model":"gnn_graphsage_tuned","n_nodes":len(hex_ids),"n_features":in_channels,"n_trials":n_trials,"max_epochs":max_epochs})

        logger.info("="*55)
        logger.info(f"GNN OPTUNA TUNING — {n_trials} trials")
        logger.info("="*55)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        t0    = time.perf_counter()
        study.optimize(make_objective(x, y_tensor, edge_index, train_mask, val_mask, in_channels, device, max_epochs), n_trials=n_trials, show_progress_bar=True)
        logger.info(f"Tuning: {(time.perf_counter()-t0)/60:.1f} min  best_val_rmse={study.best_value:.3f}  params={study.best_params}")
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_val_rmse", study.best_value)

        bp          = study.best_params
        n_layers    = bp["n_layers"]
        hidden_dims = [bp[f"hidden_dim_{i}"] for i in range(n_layers)]
        dropout     = bp["dropout"]
        lr          = bp["lr"]
        weight_decay = bp["weight_decay"]
        logger.info(f"Best arch: {in_channels} -> {hidden_dims} -> 1")

        torch.manual_seed(RANDOM_SEED)
        final_model    = WalkabilityGNN(in_channels, hidden_dims, dropout).to(device)
        n_params       = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
        all_train_mask = torch.tensor((folds != CENTER_FOLD).values, dtype=torch.bool).to(device)

        logger.info("="*55)
        logger.info("FINAL MODEL TRAINING")
        logger.info("="*55)
        final_model, best_val_rmse, history = train_with_early_stopping(
            final_model, x, edge_index, y_tensor, all_train_mask, val_mask,
            lr=lr, weight_decay=weight_decay, max_epochs=max_epochs, patience=PATIENCE, verbose=True,
        )
        logger.info(f"Training: {len(history['val_rmse'])} epochs  best_val_rmse={best_val_rmse:.3f}")

        logger.info("="*55)
        logger.info("CENTER FOLD EVALUATION")
        logger.info("="*55)
        train_rmse, train_r2 = evaluate(final_model, x, edge_index, y_tensor, all_train_mask)
        test_rmse,  test_r2  = evaluate(final_model, x, edge_index, y_tensor, test_mask)
        logger.info(f"  Train  RMSE={train_rmse:.3f}  R2={train_r2:.3f}")
        logger.info(f"  Test   RMSE={test_rmse:.3f}  R2={test_r2:.3f}  (center fold)")

        final_metrics = {"train_rmse":train_rmse,"train_r2":train_r2,"test_rmse":test_rmse,"test_r2":test_r2,"n_epochs":len(history["val_rmse"]),"best_val_rmse":best_val_rmse,"n_params":n_params}
        mlflow.log_metrics(final_metrics)

        ens_path = cfg.paths.processed.parent / "predictions_ensemble.parquet"
        if ens_path.exists():
            ens = gpd.read_parquet(str(ens_path))
            ens_c = ens[ens["split"]=="test"]
            ens_rmse = float(np.sqrt(mean_squared_error(ens_c["walk_score"],ens_c["ensemble_score"])))
            logger.info(f"  Ensemble RMSE={ens_rmse:.3f}  GNN improvement: {ens_rmse-test_rmse:+.3f}")
            mlflow.log_metric("improvement_vs_ensemble", ens_rmse-test_rmse)

        model_path = cfg.paths.models / "gnn_best.pt"
        torch.save({"model_state_dict":final_model.state_dict(),"hidden_dims":hidden_dims,"dropout":dropout,"in_channels":in_channels,"feature_cols":feature_cols,"scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist(),"hex_ids":hex_ids,"best_params":bp}, str(model_path))
        logger.info(f"Model saved -> {model_path.name}")
        mlflow.log_artifact(str(model_path))

        final_model.eval()
        with torch.no_grad():
            all_preds = np.clip(final_model(x, edge_index).cpu().numpy()*100.0, 0, 100)
        master  = gpd.read_parquet(str(cfg.paths.processed.parent/"master_features.parquet"))[["h3_index","geometry","centroid_lat","centroid_lng"]]
        pred_df = pd.DataFrame({"h3_index":hex_ids,"gnn_score":all_preds,"walk_score":y.values,"fold":folds.values})
        pred_df["residual"]     = pred_df["walk_score"] - pred_df["gnn_score"]
        pred_df["abs_residual"] = pred_df["residual"].abs()
        result = gpd.GeoDataFrame(master.merge(pred_df,on="h3_index",how="right"), crs=cfg.city.crs)
        out_path = cfg.paths.processed.parent / "predictions_gnn.parquet"
        result.to_parquet(str(out_path))
        logger.info(f"Predictions saved -> {out_path.name}  ({len(result):,} hexes)")

        logger.info("="*55)
        logger.info("WEEK 9 GNN SUMMARY")
        logger.info("="*55)
        logger.info(f"  Architecture : {in_channels} -> {hidden_dims} -> 1")
        logger.info(f"  Parameters   : {n_params:,}")
        logger.info(f"  Optuna trials: {n_trials}")
        logger.info(f"  Best CV RMSE : {study.best_value:.3f}")
        logger.info(f"  Test  RMSE   : {test_rmse:.3f}  (center fold)")
        logger.info(f"  Test  R2     : {test_r2:.3f}")
        logger.info("="*55)
        return final_metrics


if __name__ == "__main__":
    import sys
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/gnn.log", rotation="10 MB", level="DEBUG")

    parser = argparse.ArgumentParser(description="Week 9: GraphSAGE GNN with Optuna tuning.")
    parser.add_argument("--fast",   action="store_true", help=f"{FAST_TRIALS} trials, {FAST_EPOCHS} epochs")
    parser.add_argument("--trials", type=int, default=None, help="Number of Optuna trials")
    args = parser.parse_args()

    if args.trials is not None:
        n_trials, max_epochs = args.trials, MAX_EPOCHS
    elif args.fast:
        n_trials, max_epochs = FAST_TRIALS, FAST_EPOCHS
    else:
        n_trials, max_epochs = N_TRIALS, MAX_EPOCHS

    try:
        results = run_gnn_pipeline(n_trials=n_trials, max_epochs=max_epochs)
        logger.success(f"GNN complete — test RMSE={results['test_rmse']:.3f}  R2={results['test_r2']:.3f}")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal: {e}")
        sys.exit(1)