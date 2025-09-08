import argparse
import logging
import os
import time

import wandb

from dataset import CrossDomain
from model import Model, Perceptor
from train import train
from utils import set_logging, set_seed


def search(args):
    args.search = True

    wandb.init(project="BiGNAS", config=args)
    set_seed(args.seed)
    set_logging()

    logging.info(f"args: {args}")

    dataset = CrossDomain(
        root=args.root,
        categories=args.categories,
        target=args.target,
        use_source=args.use_source,
    )

    data = dataset[0]
    args.num_users = data.num_users
    args.num_source_items = data.num_source_items
    args.num_target_items = data.num_target_items
    logging.info(f"data: {data}")

    DATE_FORMAT = "%Y-%m-%d_%H:%M:%S"
    args.model_path = os.path.join(
        args.model_dir,
        f'{time.strftime(DATE_FORMAT, time.localtime())}_{"_".join(args.categories)}.pt',
    )

    model = Model(args)
    perceptor = Perceptor(args)
    logging.info(f"model: {model}")
    train(model, perceptor, data, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # device & mode settings
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--search", default=False, action="store_true")
    parser.add_argument("--use-meta", default=False, action="store_true")
    parser.add_argument("--use-source", default=False, action="store_true")

    # dataset settings
    parser.add_argument(
        "--categories", type=str, nargs="+", default=["Electronic", "Clothing"]
    )
    parser.add_argument("--target", type=str, default="Clothing")
    parser.add_argument("--root", type=str, default="data/")

    # model settings
    parser.add_argument("--aggr", type=str, default="mean")
    parser.add_argument("--bn", type=bool, default=False)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--model-dir", type=str, default="./save/")

    # supernet settings
    parser.add_argument(
        "--space",
        type=str,
        nargs="+",
        default=["gcn", "gatv2", "sage", "lightgcn", "linear"],
    )
    parser.add_argument("--warm-up", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--T", type=int, default=1)
    parser.add_argument("--entropy", type=float, default=0.0)

    # training settings
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eta-min", type=float, default=0.001)
    parser.add_argument("--T-max", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=15, help="Top-K for hit ratio evaluation") #新增top k


    # meta settings
    parser.add_argument("--meta-interval", type=int, default=50)
    parser.add_argument("--meta-num-layers", type=int, default=2)
    parser.add_argument("--meta-hidden-dim", type=int, default=32)
    parser.add_argument("--meta-batch-size", type=int, default=512)
    parser.add_argument("--conv-lr", type=float, default=1)
    parser.add_argument("--hpo-lr", type=float, default=0.01)
    parser.add_argument("--descent-step", type=int, default=10)
    parser.add_argument("--meta-op", type=str, default="gat")

    # CL 超參數
    parser.add_argument('--ssl_aug_type', type=str, default='edge', choices=['edge', 'node'])
    parser.add_argument('--edge_drop_rate', type=float, default=0.2)
    parser.add_argument('--node_drop_rate', type=float, default=0.2)
    parser.add_argument('--ssl_reg', type=float, default=0.1)   # InfoNCE 權重
    parser.add_argument('--reg', type=float, default=1e-4)      # L2 權重
    parser.add_argument('--nce_temp', type=float, default=0.2)  # InfoNCE 溫度
    parser.add_argument('--hard_ratio', type=float, default=0.1) # 取前10% hard users
    parser.add_argument('--hard_mine_interval', type=int, default=1) # 每幾個 epoch 重算一次
    parser.add_argument('--inject_source', action='store_true')  # 是否在 source 注入
    parser.add_argument('--inject_target', action='store_true')  # 是否在 target 注入
    parser.add_argument('--neg_samples', type=int, default=1)    # BPR 每個正例的負樣本數

    # SGL 初始化與跨域對齊設定
    parser.add_argument("--use-sgl-init", type=bool, default=True,
                        help="是否載入 SGL 匯出的最終(傳播後) embedding 來初始化來源域嵌入")
    parser.add_argument("--sgl-dir", type=str, default="../SGL-Torch/outputs/music_instrument",
                        help="SGL 匯出 user_embeddings_final.npy / item_embeddings_final.npy 的資料夾")
    parser.add_argument("--freeze-src-steps", type=int, default=1000,
                        help="先凍結來源域嵌入的步數，之後自動解凍微調")

    parser.add_argument("--use-align", type=bool, default=True,
                        help="是否啟用跨域使用者 embedding 對齊損失(L2)")
    parser.add_argument("--align-weight", type=float, default=1e-2,
                        help="對齊損失權重")
    parser.add_argument("--detach-source", type=bool, default=True,
                        help="對齊時是否對來源域 embedding 做 detach，將其當作錨點")

    # 新增 source_item_top_ratio 超參數
    parser.add_argument('--source-item-top-ratio', type=float, default=0.1)
    
    # 新增target_embedding向量
    parser.add_argument(
        "--target-item-embedding-path",
        type=str,
        default=None,
        help="Path to the pre-trained target item embedding (npy file)."
    )
    
    args = parser.parse_args()
    search(args)
