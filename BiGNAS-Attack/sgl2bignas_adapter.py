# sgl2bignas_adapter.py
import logging
from typing import Optional

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"expected np.ndarray or torch.Tensor, got {type(x)}")


def load_sgl_final_embeddings(sgl_dir: str):
    """
    載入 SGL 訓練後的最終（傳播後）embedding，並做 L2 normalize。
    回傳 numpy，避免後續混用 dtype/device。

    Args:
        sgl_dir: SGL 模型存放目錄（以 '/' 結尾或不結尾皆可）

    Returns:
        E_s_u_np: numpy [U_s, d_s]
        E_s_i_np: numpy [I_s, d_s]
    """
    sgl_dir = sgl_dir if sgl_dir.endswith("/") else sgl_dir + "/"
    try:
        E_s_u = np.load(sgl_dir + "user_embeddings_final.npy")
        E_s_i = np.load(sgl_dir + "item_embeddings_final.npy")

        # L2 normalize（單位向量）
        E_s_u = E_s_u / (np.linalg.norm(E_s_u, axis=1, keepdims=True) + 1e-12)
        E_s_i = E_s_i / (np.linalg.norm(E_s_i, axis=1, keepdims=True) + 1e-12)

        logging.info(
            f"Loaded SGL embeddings: users={E_s_u.shape}, items={E_s_i.shape}"
        )
        return E_s_u, E_s_i
    except FileNotFoundError as e:
        logging.error(f"SGL final embeddings not found in {sgl_dir}")
        logging.error(
            "Please ensure SGL training completed and export_final_embeddings() was called"
        )
        raise e


def _maybe_remap_by_index(
    emb_mat_np: np.ndarray,
    index_map: Optional[np.ndarray],
    take_k: Optional[int] = None,
) -> np.ndarray:
    """
    依照 index_map 重排；若未提供 index_map，則以切片前 k 筆方式對齊。
    index_map: 長度 = 目標側的 N，index_map[j] = 對應到 SGL 的 row id。
               例如 user_index_map[j] = sgl_user_id_of_bignas_user_j
    take_k: 若不重排時要切的筆數（通常等於目標側表大小）
    """
    if index_map is not None:
        assert index_map.ndim == 1, "index_map must be 1D"
        assert index_map.max() < emb_mat_np.shape[0], "index_map out of range"
        return emb_mat_np[index_map]
    if take_k is None:
        return emb_mat_np
    return emb_mat_np[:take_k]


def _project_if_needed(E_np: np.ndarray, d_target: int, device: torch.device) -> np.ndarray:
    """
    若 SGL 維度 != BiGNAS 維度，臨時用一個 Linear 做投影（不掛在 model 上）。
    回傳 numpy。
    """
    d_src = E_np.shape[1]
    if d_src == d_target:
        return E_np
    proj = torch.nn.Linear(d_src, d_target, bias=False).to(device)
    with torch.no_grad():
        E_t = torch.from_numpy(E_np).to(device=device, dtype=torch.float32)
        E_proj = proj(E_t).cpu().numpy()
    logging.info(f"Projected embeddings from dim {d_src} -> {d_target}")
    return E_proj


def init_bignas_source_from_sgl(
    model: torch.nn.Module,
    data,
    E_s_u_np: np.ndarray,
    E_s_i_np: np.ndarray,
    device: torch.device,
    freeze_steps: int = 1000,
    user_index_map: Optional[np.ndarray] = None,
    source_item_index_map: Optional[np.ndarray] = None,
):
    """
    用 SGL 的 user/item 最終向量初始化 BiGNAS 的 user / source_item embedding。

    Args:
        model: 你的 BiGNAS 模型，需有 user_embedding, source_item_embedding
        data: 你的 PyG Data，需有 num_users, num_source_items
        E_s_u_np: SGL user 最終向量（numpy, [U_s, d_s]）
        E_s_i_np: SGL item 最終向量（numpy, [I_s, d_s]）
        device: 目標裝置
        freeze_steps: 先凍來源 item 幾步，配合 step_unfreeze_if_ready() 使用
        user_index_map: 若你有 raw id 對齊，傳 numpy 的 index map 進來會先重排
        source_item_index_map: 同上，對來源 item 做重排

    Returns:
        已覆寫權重並設定好錨點/凍結計數器的 model
    """
    # 1) 對齊維度
    d_target = int(model.user_embedding.embedding_dim)
    E_s_u_np = _project_if_needed(E_s_u_np, d_target, device)
    E_s_i_np = _project_if_needed(E_s_i_np, d_target, device)

    # 2) 依照 index_map 或切片對齊尺寸
    n_users = int(data.num_users)
    n_src_items = int(data.num_source_items)
    E_u_use = _maybe_remap_by_index(E_s_u_np, user_index_map, take_k=n_users)
    E_i_use = _maybe_remap_by_index(E_s_i_np, source_item_index_map, take_k=n_src_items)

    if E_u_use.shape[0] < n_users:
        logging.warning(
            f"SGL users ({E_u_use.shape[0]}) < BiGNAS users ({n_users}). "
            f"Only first {E_u_use.shape[0]} users will be initialized."
        )
    if E_i_use.shape[0] < n_src_items:
        logging.warning(
            f"SGL items ({E_i_use.shape[0]}) < BiGNAS source items ({n_src_items}). "
            f"Only first {E_i_use.shape[0]} items will be initialized."
        )

    # 3) 覆寫 BiGNAS 權重（dtype/device 對齊）
    with torch.no_grad():
        # user
        u_weight = model.user_embedding.weight
        u_tensor = torch.from_numpy(E_u_use).to(device=u_weight.device, dtype=u_weight.dtype)
        model.user_embedding.weight[: u_tensor.size(0)].copy_(u_tensor)

        # source item
        s_weight = model.source_item_embedding.weight
        i_tensor = torch.from_numpy(E_i_use).to(device=s_weight.device, dtype=s_weight.dtype)
        model.source_item_embedding.weight[: i_tensor.size(0)].copy_(i_tensor)

    # 4) 註冊錨點（不存進 checkpoint）
    model.register_buffer(
        "_sgl_user_anchor",
        torch.from_numpy(E_u_use).to(device=device, dtype=torch.float32),
        persistent=False,
    )

    # 5) 凍結來源 item，等待若干步後再解凍
    model._sgl_freeze_counter = int(freeze_steps)
    for p in model.source_item_embedding.parameters():
        p.requires_grad = False
    # 若也想凍 user：取消下面註解
    # for p in model.user_embedding.parameters():
    #     p.requires_grad = False

    logging.info(
        f"SGL init done. users_init={E_u_use.shape}, items_init={E_i_use.shape}, "
        f"freeze_src_steps={freeze_steps}"
    )
    return model


def step_unfreeze_if_ready(model: torch.nn.Module):
    """
    放在訓練 loop 每個 iteration 開頭呼叫。
    當 _sgl_freeze_counter 減到 0 時，解凍先前凍住的來源 item（以及你想解的其他參數）。
    """
    if hasattr(model, "_sgl_freeze_counter") and model._sgl_freeze_counter is not None:
        if model._sgl_freeze_counter > 0:
            model._sgl_freeze_counter -= 1
        if model._sgl_freeze_counter == 0:
            for p in model.source_item_embedding.parameters():
                p.requires_grad = True
            # 若有凍 user，也在這裡解凍
            # for p in model.user_embedding.parameters():
            #     p.requires_grad = True
            logging.info("Unfroze SGL-initialized source embeddings")
            model._sgl_freeze_counter = None


def load_user_alignment(csv_path: str):
    """
    載入跨域使用者對應表（如果你有的話）
    CSV: source_uid,target_uid

    Returns:
        src_u_idx (LongTensor), tgt_u_idx (LongTensor)
    """
    try:
        pairs = np.loadtxt(csv_path, delimiter=",", dtype=np.int64, skiprows=1)
        src_u = torch.from_numpy(pairs[:, 0]).long()
        tgt_u = torch.from_numpy(pairs[:, 1]).long()
        logging.info(f"Loaded {len(src_u)} user alignments from {csv_path}")
        return src_u, tgt_u
    except Exception as e:
        logging.warning(f"Failed to load user alignment: {e}")
        return None, None
