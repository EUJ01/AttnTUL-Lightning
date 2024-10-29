import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.molnet import GcnNet, MolNet
from models.molnet_lightning import MolLightningModule
from utils.datasets import MolDataModule
from utils.rawprocess import get_data_and_graph


def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description="Moving Object Linking")
    parse.add_argument(
        "--dataset", type=str, default="gowalla-all", help="Dataset for training"
    )
    parse.add_argument(
        "--epochs", type=int, default=80, help="Number of epochs to train"
    )
    parse.add_argument(
        "--read_pkl", type=bool, default=True, help="Read preprocessed input"
    )
    parse.add_argument("--grid_size", type=int, default=120, help="Size of grid")
    parse.add_argument("--batch_size", type=int, default=16, help="Size of batches")
    parse.add_argument(
        "--patience", type=int, default=10, help="Number of early stop patience"
    )
    parse.add_argument(
        "--localGcn_hidden",
        type=int,
        default=512,
        help="Number of local gcn hidden units",
    )
    parse.add_argument(
        "--globalGcn_hidden",
        type=int,
        default=512,
        help="Number of global gcn hidden units",
    )
    parse.add_argument(
        "--gcn_dropout",
        type=float,
        default=0.5,
        help="Dropout rate (1 - keep probability)",
    )
    parse.add_argument(
        "--Attn_Strategy", type=str, default="cos", help="Global Attention Strategy"
    )
    parse.add_argument(
        "--Softmax_Strategy",
        type=str,
        default="complex",
        help="Global Softmax Strategy",
    )
    parse.add_argument(
        "--Pool_Strategy", type=str, default="max", help="Pooling layer Strategy"
    )
    parse.add_argument(
        "--d_model", type=int, default=128, help="Number of point vector dim"
    )
    parse.add_argument(
        "--d_k", type=int, default=64, help="Number of querry vector dim"
    )
    parse.add_argument("--d_v", type=int, default=64, help="Number of key vector dim")
    parse.add_argument(
        "--d_ff", type=int, default=512, help="Number of Feed forward transform dim"
    )
    parse.add_argument("--n_heads", type=int, default=5, help="Number of heads")
    parse.add_argument("--n_layers", type=int, default=2, help="Number of EncoderLayer")

    args = parse.parse_args()
    return args


def main(
    dataset,
    epochs,
    read_pkl,
    batch_size,
    localGcn_hidden,
    globalGcn_hidden,
    gcn_dropout,
    patience,
    d_model,
    d_k,
    d_v,
    d_ff,
    n_heads,
    n_layers,
    Attn_Strategy,
    Softmax_Strategy,
    Pool_Strategy,
    grid_size,
):
    raw_path = "data/" + dataset + ".csv"
    (
        local_feature,
        local_adj,
        global_feature,
        global_adj,
        user_traj_train,
        user_traj_test,
        grid_nums,
        traj_nums,
        user_nums,
        test_nums,
        local_graph,
        global_graph,
    ) = get_data_and_graph(raw_path, read_pkl, grid_size)

    # Model Initialization
    LocalGcnModel = GcnNet(grid_nums, localGcn_hidden, d_model, gcn_dropout)
    GlobalGcnModel = GcnNet(grid_nums, globalGcn_hidden, d_model, gcn_dropout)
    MolModel = MolNet(
        Attn_Strategy,
        Softmax_Strategy,
        Pool_Strategy,
        d_model,
        d_k,
        d_v,
        d_ff,
        n_heads,
        n_layers,
        user_nums,
    )

    # Initialize lightningDataModule, lightningModule, earlystopping
    data_module = MolDataModule(
        user_traj_train,
        user_traj_test,
        test_nums,
        batch_size=batch_size,
        val_test_split=0.5,
    )
    model = MolLightningModule(
        LocalGcnModel,
        GlobalGcnModel,
        MolModel,
        local_feature,
        local_adj,
        global_feature,
        global_adj,
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=patience, verbose=True, mode="min"
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else 0,
        callbacks=[early_stop],
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset=args.dataset,
        epochs=args.epochs,
        read_pkl=args.read_pkl,
        batch_size=args.batch_size,
        localGcn_hidden=args.localGcn_hidden,
        globalGcn_hidden=args.globalGcn_hidden,
        gcn_dropout=args.gcn_dropout,
        patience=args.patience,
        d_model=args.d_model,
        d_k=args.d_k,
        d_v=args.d_v,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        Attn_Strategy=args.Attn_Strategy,
        Softmax_Strategy=args.Softmax_Strategy,
        Pool_Strategy=args.Pool_Strategy,
        grid_size=args.grid_size,
    )
