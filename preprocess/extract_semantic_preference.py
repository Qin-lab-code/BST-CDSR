import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from llama import Llama  # from meta-llama/llama repo

BASE_PROMPT = (
    "You will act as a time-aware preference interpreter. "
    "Please extract the time-aware semantic preferences from the following interaction sequence:\n"
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="生成用户时间感知嵌入")
    parser.add_argument("--interaction_file", type=str, default='Food-Kitchen/traindata_new.txt',
                        help="用户交互序列文件路径")
    parser.add_argument("--output_embedding", type=str, default='output',
                        help="输出embedding文件路径")
    return parser.parse_args()


def load_item_titles(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index2title: Dict[int, str] = {}
    for item in data:
        if "index" in item and "title" in item:
            try:
                idx = int(item["index"])
                index2title[idx] = str(item["title"])
            except (ValueError, TypeError):
                continue
    return index2title


def parse_interaction_line(
        line: str,
) -> Tuple[Optional[str], List[Tuple[int, str]]]:
    line = line.strip()
    if not line:
        return None, []

    parts = line.split("\t")
    if len(parts) < 3:
        return None, []

    user_id = parts[0].strip()
    interactions_parts = parts[2:]

    interactions: List[Tuple[int, str]] = []
    for seg in interactions_parts:
        seg = seg.strip()
        if not seg:
            continue
        seg = seg.strip("|")
        fields = seg.split("|")
        if len(fields) < 3:
            continue
        item_str, ts_str, time_str = fields[0], fields[1], fields[2]
        try:
            item_id = int(item_str)
        except ValueError:
            continue
        # 只取日期
        date_part = time_str.split(" ")[0]  # "2012-10-31"
        interactions.append((item_id, date_part))

    interactions_sorted = sort_interactions_by_time(interactions)

    return user_id, interactions_sorted


def sort_interactions_by_time(interactions: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    def get_date_key(interaction: Tuple[int, str]) -> datetime:
        _, date_str = interaction
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return datetime.min

    return sorted(interactions, key=get_date_key)


def split_domains(
        interactions: List[Tuple[int, str]],
        threshold: int,
):
    seq_a = []
    seq_b = []

    for item_id, date_str in interactions:
        if item_id < threshold:
            seq_a.append((item_id, date_str))
        else:
            seq_b.append((item_id, date_str))

    removed_a = seq_a[-1] if len(seq_a) > 0 else None
    removed_b = seq_b[-1] if len(seq_b) > 0 else None

    if removed_a:
        seq_a = seq_a[:-1]
    if removed_b:
        seq_b = seq_b[:-1]

    seq_mixed = list(interactions)  # copy

    if removed_a:
        for i in range(len(seq_mixed) - 1, -1, -1):
            if seq_mixed[i] == removed_a:
                seq_mixed.pop(i)
                break

    if removed_b:
        for i in range(len(seq_mixed) - 1, -1, -1):
            if seq_mixed[i] == removed_b:
                seq_mixed.pop(i)
                break

    return seq_a, seq_b, seq_mixed


def seq_to_text_for_domain(
        seq: List[Tuple[int, str]],
        domain: str,
        index2title_a: Dict[int, str],
        index2title_b: Dict[int, str],
        threshold: int,
) -> str:
    pieces: List[str] = []
    prev_date: Optional[datetime] = None

    for i, (item_id, date_str) in enumerate(seq):
        try:
            curr_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            curr_date = None

        if domain == "A":
            dom_token = "[A_DOMAIN]"
            idx = item_id
            title = index2title_a.get(idx, f"Unknown A item {item_id}")
        elif domain == "B":
            dom_token = "[B_DOMAIN]"
            idx = item_id - threshold
            title = index2title_b.get(idx, f"Unknown B item {item_id}")
        elif domain == "mixed":
            if item_id >= threshold:
                dom_token = "[B_DOMAIN]"
                idx = item_id - threshold
                title = index2title_b.get(idx, f"Unknown B item {item_id}")
            else:
                dom_token = "[A_DOMAIN]"
                idx = item_id
                title = index2title_a.get(idx, f"Unknown A item {item_id}")
        else:
            dom_token = "[UNK_DOMAIN]"
            title = f"Unknown item {item_id}"

        clean_title = str(title).replace("\t", " ").replace("\n", " ")
        clean_title = "[" + clean_title + "]"

        if i == 0:
            pieces.append(dom_token)
            pieces.append(clean_title)
        else:
            if prev_date is not None and curr_date is not None:
                gap_days = (curr_date - prev_date).days
                if gap_days < 0:
                    gap_days = 0
            else:
                gap_days = 0

            if gap_days <= 1:
                gap_token = "[GAP_LE_1]"
            elif 2 <= gap_days <= 7:
                gap_token = "[GAP_2_7]"
            elif 8 <= gap_days <= 15:
                gap_token = "[GAP_8_15]"
            elif 16 <= gap_days <= 30:
                gap_token = "[GAP_16_30]"
            elif 31 <= gap_days <= 60:
                gap_token = "[GAP_31_60]"
            else:  # > 60
                gap_token = "[GAP_GT_60]"

            pieces.append(gap_token)
            pieces.append(dom_token)
            pieces.append(clean_title)

        prev_date = curr_date

    return " ".join(pieces)


def load_embed_dim_from_params(ckpt_dir: str) -> int:
    params_path = os.path.join(ckpt_dir, "params.json")
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    dim = int(params.get("dim", 4096))
    print(f"[INFO] Loaded model hidden dim = {dim} from {params_path}")
    return dim


def encode_batch_to_tokens(
        texts: List[str],
        tokenizer,
        max_seq_len: int,
        device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    token_lists: List[List[int]] = []
    last_indices: List[int] = []

    for s in texts:
        ids = tokenizer.encode(s, bos=True, eos=True)
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
        token_lists.append(ids)

    max_len = max(len(ids) for ids in token_lists)
    pad_id = getattr(tokenizer, "pad_id", None)
    if pad_id is None or pad_id < 0:
        pad_id = getattr(tokenizer, "eos_id", 0)

    tokens = torch.full(
        (len(token_lists), max_len),
        pad_id,
        dtype=torch.long,
        device=device,
    )

    for i, ids in enumerate(token_lists):
        length = len(ids)
        tokens[i, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        last_indices.append(length - 1)

    return tokens, last_indices


def run_llama_hidden_embeddings(
        generator: Llama,
        seq_texts: List[str],
        embed_dim: int,
        batch_size: int,
        max_len: int,
) -> List[List[float]]:
    model = generator.model
    tokenizer = generator.tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_embeddings: List[List[float]] = []

    total = len(seq_texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = seq_texts[start:end]

        tokens, last_indices = encode_batch_to_tokens(
            batch_texts, tokenizer, max_len, device
        )

        hidden_container = {}

        def hook_fn(module, input, output):
            hidden_container["h"] = input[0]

        handle = model.norm.register_forward_hook(hook_fn)

        with torch.inference_mode():
            _ = model(tokens, start_pos=0)

        handle.remove()

        if "h" not in hidden_container:
            raise RuntimeError("Failed to capture hidden states from model.norm hook.")

        h = hidden_container["h"]  # [B, T, dim]
        bsz, seqlen, dim = h.shape
        if dim != embed_dim:
            print(f"[WARN] hidden dim from model ({dim}) != embed_dim ({embed_dim})")

        h_cpu = h.detach().cpu().float().numpy()

        for i in range(bsz):
            idx = last_indices[i]
            if idx < 0 or idx >= seqlen:
                raise RuntimeError(
                    f"Invalid last index {idx} for sequence length {seqlen}"
                )
            vec = h_cpu[i, idx, :].tolist()
            all_embeddings.append(vec)

        if end % 5000 == 0:
            print(f"[LLaMA] Processed {end}/{total} sequences...")

    return all_embeddings


def main():
    args = parse_arguments()

    ITEM_JSON_A_PATH = "Food-Kitchen/Aitems.json"
    ITEM_JSON_B_PATH = "Food-Kitchen/Bitems.json"

    INTERACTION_FILE_PATH = args.interaction_file
    OUTPUT_EMBEDDING_PATH = args.output_embedding
    DOMAIN_SPLIT_THRESHOLD = 29207

    CKPT_DIR = "../llama-2-7b/llama-2-7b"
    TOKENIZER_PATH = "../llama-2-7b/tokenizer.model"

    MAX_SEQ_LEN = 1024  # 输入序列最大长度（token 数）
    BATCH_SIZE = 8  # 按用户 batch 并行大小

    print(f"开始处理文件: {INTERACTION_FILE_PATH}")
    print(f"输出文件: {OUTPUT_EMBEDDING_PATH}")

    embed_dim = load_embed_dim_from_params(CKPT_DIR)

    print("Loading A-domain item titles...")
    index2title_a = load_item_titles(ITEM_JSON_A_PATH)
    print(f"Loaded {len(index2title_a)} A-domain items.")

    print("Loading B-domain item titles...")
    index2title_b = load_item_titles(ITEM_JSON_B_PATH)
    print(f"Loaded {len(index2title_b)} B-domain items.")

    user_ids_a: List[str] = []
    seq_texts_a: List[str] = []

    user_ids_b: List[str] = []
    seq_texts_b: List[str] = []

    user_ids_mixed: List[str] = []
    seq_texts_mixed: List[str] = []

    print("Reading interaction file and building domain sequences...")
    with open(INTERACTION_FILE_PATH, "r", encoding="utf-8") as fin:
        line_count = 0
        valid_users = 0
        for line in fin:
            line_count += 1
            user_id, interactions = parse_interaction_line(line)
            if user_id is None or not interactions:
                continue

            seq_a, seq_b, seq_mixed = split_domains(
                interactions,
                threshold=DOMAIN_SPLIT_THRESHOLD,
            )

            # A 域
            if seq_a:
                seq_text_a = seq_to_text_for_domain(
                    seq_a, "A",
                    index2title_a=index2title_a,
                    index2title_b=index2title_b,
                    threshold=DOMAIN_SPLIT_THRESHOLD,
                )
                seq_text_a = BASE_PROMPT + seq_text_a
                user_ids_a.append(user_id)
                seq_texts_a.append(seq_text_a)

            # B 域
            if seq_b:
                seq_text_b = seq_to_text_for_domain(
                    seq_b, "B",
                    index2title_a=index2title_a,
                    index2title_b=index2title_b,
                    threshold=DOMAIN_SPLIT_THRESHOLD,
                )
                seq_text_b = BASE_PROMPT + seq_text_b
                user_ids_b.append(user_id)
                seq_texts_b.append(seq_text_b)

            # 混合域
            if seq_mixed:
                seq_text_m = seq_to_text_for_domain(
                    seq_mixed, "mixed",
                    index2title_a=index2title_a,
                    index2title_b=index2title_b,
                    threshold=DOMAIN_SPLIT_THRESHOLD,
                )
                seq_text_m = BASE_PROMPT + seq_text_m
                user_ids_mixed.append(user_id)
                seq_texts_mixed.append(seq_text_m)

            valid_users += 1

    print(f"Total sequences: A={len(seq_texts_a)}, B={len(seq_texts_b)}, Mixed={len(seq_texts_mixed)}")

    print("Building LLaMA generator...")
    generator = Llama.build(
        ckpt_dir=CKPT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=BATCH_SIZE,
    )
    print("LLaMA generator is ready.")

    # ----- A 域 -----
    if seq_texts_a:
        print("Running LLaMA hidden embeddings for A-domain sequences...")
        emb_list_a = run_llama_hidden_embeddings(
            generator, seq_texts_a, embed_dim, batch_size=BATCH_SIZE, max_len=MAX_SEQ_LEN
        )
        emb_a = np.array(emb_list_a, dtype=np.float32)
        user_ids_a_arr = np.array(user_ids_a, dtype=object)
    else:
        emb_a = np.zeros((0, embed_dim), dtype=np.float32)
        user_ids_a_arr = np.array([], dtype=object)

    # ----- B 域 -----
    if seq_texts_b:
        print("Running LLaMA hidden embeddings for B-domain sequences...")
        emb_list_b = run_llama_hidden_embeddings(
            generator, seq_texts_b, embed_dim, batch_size=BATCH_SIZE, max_len=MAX_SEQ_LEN
        )
        emb_b = np.array(emb_list_b, dtype=np.float32)
        user_ids_b_arr = np.array(user_ids_b, dtype=object)
    else:
        emb_b = np.zeros((0, embed_dim), dtype=np.float32)
        user_ids_b_arr = np.array([], dtype=object)

    # ----- 混合域 -----
    if seq_texts_mixed:
        print("Running LLaMA hidden embeddings for Mixed-domain sequences...")
        emb_list_m = run_llama_hidden_embeddings(
            generator, seq_texts_mixed, embed_dim, batch_size=BATCH_SIZE, max_len=MAX_SEQ_LEN
        )
        emb_m = np.array(emb_list_m, dtype=np.float32)
        user_ids_mixed_arr = np.array(user_ids_mixed, dtype=object)
    else:
        emb_m = np.zeros((0, embed_dim), dtype=np.float32)
        user_ids_mixed_arr = np.array([], dtype=object)

    print(f"Saving embeddings to {OUTPUT_EMBEDDING_PATH} ...")
    np.savez(
        OUTPUT_EMBEDDING_PATH,
        user_ids_a=user_ids_a_arr,
        emb_a=emb_a,
        user_ids_b=user_ids_b_arr,
        emb_b=emb_b,
        user_ids_mixed=user_ids_mixed_arr,
        emb_mixed=emb_m,
        embed_dim=np.array([embed_dim], dtype=np.int32),
    )
    print("All done.")


if __name__ == "__main__":
    main()
