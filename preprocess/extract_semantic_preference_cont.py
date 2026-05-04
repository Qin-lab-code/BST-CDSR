# -*- coding: utf-8 -*-
import json
import os
import argparse
import datetime
import typing

import numpy as np
import torch
import random

from llama import Llama  # from meta-llama/llama repo

BASE_PROMPT = (
    "You will act as a time-aware preference interpreter. "
    "Please extract the time-aware semantic preferences from the following interaction sequence:\n"
)

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def parse_arguments():
    parser = argparse.ArgumentParser(description="生成小扰动/大扰动用户时间感知嵌入")
    parser.add_argument("--interaction_file", type=str, default='Food-Kitchen/traindata_new.txt',
                        help="用户交互序列文件路径")
    parser.add_argument("--output_small_embedding", type=str, default='output1/train_embeddings.npz',
                        help="小扰动输出 embedding 文件路径")
    parser.add_argument("--output_big_embedding", type=str, default='output2/train_embeddings.npz',
                        help="大扰动输出 embedding 文件路径")
    return parser.parse_args()

def load_item_titles(path: str) -> typing.Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index2title: typing.Dict[int, str] = {}
    for item in data:
        if "index" in item and "title" in item:
            try:
                idx = int(item["index"])
                index2title[idx] = str(item["title"])
            except (ValueError, TypeError):
                continue
    return index2title


def sort_interactions_by_time(
        interactions: typing.List[typing.Tuple[int, str]]
) -> typing.List[typing.Tuple[int, str]]:
    def get_date_key(interaction: typing.Tuple[int, str]) -> datetime.datetime:
        _, date_str = interaction
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return datetime.datetime.min

    return sorted(interactions, key=get_date_key)


def parse_interaction_line(
        line: str,
) -> typing.Tuple[typing.Optional[str], typing.List[typing.Tuple[int, str]]]:
    line = line.strip()
    if not line:
        return None, []

    parts = line.split("\t")
    if len(parts) < 3:
        return None, []

    user_id = parts[0].strip()
    interactions_parts = parts[2:]

    interactions: typing.List[typing.Tuple[int, str]] = []
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
        date_part = time_str.split(" ")[0]
        interactions.append((item_id, date_part))

    interactions_sorted = sort_interactions_by_time(interactions)

    return user_id, interactions_sorted


def split_domains(
        interactions: typing.List[typing.Tuple[int, str]],
        threshold: int,
):
    seq_a: typing.List[typing.Tuple[int, str]] = []
    seq_b: typing.List[typing.Tuple[int, str]] = []

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
        seq: typing.List[typing.Tuple[int, str]],
        domain: str,
        index2title_a: typing.Dict[int, str],
        index2title_b: typing.Dict[int, str],
        threshold: int,
) -> str:
    pieces: typing.List[str] = []
    prev_date: typing.Optional[datetime.datetime] = None

    for i, (item_id, date_str) in enumerate(seq):
        try:
            curr_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
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


GAP_SHORT = {"[GAP_LE_1]", "[GAP_2_7]"}
GAP_MID   = {"[GAP_8_15]", "[GAP_16_30]"}
GAP_LONG  = {"[GAP_31_60]", "[GAP_GT_60]"}


def gap_bucket(tok: str) -> str:
    if tok in GAP_SHORT:
        return "S"
    if tok in GAP_MID:
        return "M"
    if tok in GAP_LONG:
        return "L"
    return "OTHER"


def choose_neighbor_gap(token: str) -> str:
    if token in GAP_SHORT:
        cand = list(GAP_SHORT)
        if token in cand:
            cand.remove(token)
        if not cand:
            cand = list(GAP_SHORT)
        return random.choice(cand)
    if token in GAP_MID:
        cand = list(GAP_MID)
        if token in cand:
            cand.remove(token)
        if not cand:
            cand = list(GAP_MID)
        return random.choice(cand)
    if token in GAP_LONG:
        cand = list(GAP_LONG)
        if token in cand:
            cand.remove(token)
        if not cand:
            cand = list(GAP_LONG)
        return random.choice(cand)
    return token


def choose_opposite_gap(token: str) -> str:
    b = gap_bucket(token)
    if b == "S":
        return random.choice(list(GAP_LONG))
    if b == "L":
        return random.choice(list(GAP_SHORT))
    if b == "M":
        return random.choice(list(GAP_SHORT | GAP_LONG))
    return token


def build_small_and_big_version(
    seq_text: str,
    rho_small: float = 0.25,
    rho_big: float = 0.5,
    min_big_frac_over_small: float = 2.5,
):
    tokens = seq_text.split()
    gap_indices = [i for i, t in enumerate(tokens) if gap_bucket(t) in {"S", "M", "L"}]
    n = len(gap_indices)
    if n == 0:
        return seq_text, seq_text

    max_small = max(1, int(round(rho_small * n)))
    small_k = random.randint(1, max_small)
    small_indices = random.sample(gap_indices, k=min(small_k, n))
    small_tokens = tokens[:]
    for idx in small_indices:
        small_tokens[idx] = choose_neighbor_gap(small_tokens[idx])
    small_seq = " ".join(small_tokens)
    small_ratio = len(small_indices) / n

    target_min_big = int(max(1, int(round(min_big_frac_over_small * small_ratio * n))))
    max_big = max(1, int(round(rho_big * n)))
    big_k = max(target_min_big, max_small)
    big_k = min(big_k, n, max_big)
    big_indices = random.sample(gap_indices, k=big_k)
    big_tokens = tokens[:]
    for idx in big_indices:
        big_tokens[idx] = choose_opposite_gap(big_tokens[idx])
    big_seq = " ".join(big_tokens)

    return small_seq, big_seq


def load_embed_dim_from_params(ckpt_dir: str) -> int:
    params_path = os.path.join(ckpt_dir, "params.json")
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    dim = int(params.get("dim", 4096))
    print(f"[INFO] Loaded model hidden dim = {dim} from {params_path}")
    return dim


def encode_batch_to_tokens(
        texts: typing.List[str],
        tokenizer,
        max_seq_len: int,
        device: torch.device,
) -> typing.Tuple[torch.Tensor, typing.List[int]]:
    token_lists: typing.List[typing.List[int]] = []
    last_indices: typing.List[int] = []

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
        seq_texts: typing.List[str],
        embed_dim: int,
        batch_size: int,
        max_len: int,
) -> np.ndarray:
    model = generator.model
    tokenizer = generator.tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_embeddings: typing.List[typing.List[float]] = []

    total = len(seq_texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = seq_texts[start:end]

        tokens, last_indices = encode_batch_to_tokens(
            batch_texts, tokenizer, max_len, device
        )

        hidden_container: typing.Dict[str, torch.Tensor] = {}

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

    return np.array(all_embeddings, dtype=np.float32)

def main():
    args = parse_arguments()

    ITEM_JSON_A_PATH = "Food-Kitchen/Aitems.json"
    ITEM_JSON_B_PATH = "Food-Kitchen/Bitems.json"

    INTERACTION_FILE_PATH = args.interaction_file
    OUTPUT_SMALL_PATH = args.output_small_embedding
    OUTPUT_BIG_PATH = args.output_big_embedding

    DOMAIN_SPLIT_THRESHOLD = 29207

    CKPT_DIR = "../llama-2-7b/llama-2-7b"
    TOKENIZER_PATH = "../llama-2-7b/tokenizer.model"
    MAX_SEQ_LEN = 1024
    BATCH_SIZE = 8

    print(f"Interaction file: {INTERACTION_FILE_PATH}")
    print(f"Small-perturb embedding output: {OUTPUT_SMALL_PATH}")
    print(f"Big-perturb embedding output:   {OUTPUT_BIG_PATH}")

    embed_dim = load_embed_dim_from_params(CKPT_DIR)

    print("Loading A-domain item titles...")
    index2title_a = load_item_titles(ITEM_JSON_A_PATH)
    print(f"Loaded {len(index2title_a)} A-domain items.")
    print("Loading B-domain item titles...")
    index2title_b = load_item_titles(ITEM_JSON_B_PATH)
    print(f"Loaded {len(index2title_b)} B-domain items.")

    user_ids_a_small: typing.List[str] = []
    seq_texts_a_small: typing.List[str] = []
    user_ids_b_small: typing.List[str] = []
    seq_texts_b_small: typing.List[str] = []
    user_ids_m_small: typing.List[str] = []
    seq_texts_m_small: typing.List[str] = []

    user_ids_a_big: typing.List[str] = []
    seq_texts_a_big: typing.List[str] = []
    user_ids_b_big: typing.List[str] = []
    seq_texts_b_big: typing.List[str] = []
    user_ids_m_big: typing.List[str] = []
    seq_texts_m_big: typing.List[str] = []

    print("Reading interaction file and building small/big domain sequences...")
    with open(INTERACTION_FILE_PATH, "r", encoding="utf-8") as fin:
        line_count = 0
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
                base_text_a = seq_to_text_for_domain(
                    seq_a, "A",
                    index2title_a=index2title_a,
                    index2title_b=index2title_b,
                    threshold=DOMAIN_SPLIT_THRESHOLD,
                )
                small_a, big_a = build_small_and_big_version(base_text_a, 0.25, 0.5, 2.5)

                small_a = BASE_PROMPT + small_a
                big_a   = BASE_PROMPT + big_a

                user_ids_a_small.append(user_id)
                seq_texts_a_small.append(small_a)
                user_ids_a_big.append(user_id)
                seq_texts_a_big.append(big_a)

            # B 域
            if seq_b:
                base_text_b = seq_to_text_for_domain(
                    seq_b, "B",
                    index2title_a=index2title_a,
                    index2title_b=index2title_b,
                    threshold=DOMAIN_SPLIT_THRESHOLD,
                )
                small_b, big_b = build_small_and_big_version(base_text_b, 0.2, 0.5, 2.5)

                small_b = BASE_PROMPT + small_b
                big_b   = BASE_PROMPT + big_b

                user_ids_b_small.append(user_id)
                seq_texts_b_small.append(small_b)
                user_ids_b_big.append(user_id)
                seq_texts_b_big.append(big_b)

            # 混合域
            if seq_mixed:
                base_text_m = seq_to_text_for_domain(
                    seq_mixed, "mixed",
                    index2title_a=index2title_a,
                    index2title_b=index2title_b,
                    threshold=DOMAIN_SPLIT_THRESHOLD,
                )
                small_m, big_m = build_small_and_big_version(base_text_m, 0.25, 0.5, 2.5)

                small_m = BASE_PROMPT + small_m
                big_m   = BASE_PROMPT + big_m

                user_ids_m_small.append(user_id)
                seq_texts_m_small.append(small_m)
                user_ids_m_big.append(user_id)
                seq_texts_m_big.append(big_m)

    print(f"Sequences count (small/big) - "
          f"A: {len(seq_texts_a_small)}/{len(seq_texts_a_big)}, "
          f"B: {len(seq_texts_b_small)}/{len(seq_texts_b_big)}, "
          f"Mixed: {len(seq_texts_m_small)}/{len(seq_texts_m_big)}")

    print("Building LLaMA generator...")
    generator = Llama.build(
        ckpt_dir=CKPT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=BATCH_SIZE,
    )
    print("LLaMA generator is ready.")

    print("Running LLaMA embeddings for SMALL perturbation...")
    emb_a_small = run_llama_hidden_embeddings(generator, seq_texts_a_small, embed_dim, BATCH_SIZE, MAX_SEQ_LEN) \
        if seq_texts_a_small else np.zeros((0, embed_dim), dtype=np.float32)
    emb_b_small = run_llama_hidden_embeddings(generator, seq_texts_b_small, embed_dim, BATCH_SIZE, MAX_SEQ_LEN) \
        if seq_texts_b_small else np.zeros((0, embed_dim), dtype=np.float32)
    emb_m_small = run_llama_hidden_embeddings(generator, seq_texts_m_small, embed_dim, BATCH_SIZE, MAX_SEQ_LEN) \
        if seq_texts_m_small else np.zeros((0, embed_dim), dtype=np.float32)

    print(f"Saving SMALL perturbation embeddings to {OUTPUT_SMALL_PATH} ...")
    np.savez(
        OUTPUT_SMALL_PATH,
        user_ids_a=np.array(user_ids_a_small, dtype=object),
        emb_a=emb_a_small,
        user_ids_b=np.array(user_ids_b_small, dtype=object),
        emb_b=emb_b_small,
        user_ids_mixed=np.array(user_ids_m_small, dtype=object),
        emb_mixed=emb_m_small,
        embed_dim=np.array([embed_dim], dtype=np.int32),
    )

    print("Running LLaMA embeddings for BIG perturbation...")
    emb_a_big = run_llama_hidden_embeddings(generator, seq_texts_a_big, embed_dim, BATCH_SIZE, MAX_SEQ_LEN) \
        if seq_texts_a_big else np.zeros((0, embed_dim), dtype=np.float32)
    emb_b_big = run_llama_hidden_embeddings(generator, seq_texts_b_big, embed_dim, BATCH_SIZE, MAX_SEQ_LEN) \
        if seq_texts_b_big else np.zeros((0, embed_dim), dtype=np.float32)
    emb_m_big = run_llama_hidden_embeddings(generator, seq_texts_m_big, embed_dim, BATCH_SIZE, MAX_SEQ_LEN) \
        if seq_texts_m_big else np.zeros((0, embed_dim), dtype=np.float32)

    print(f"Saving BIG perturbation embeddings to {OUTPUT_BIG_PATH} ...")
    np.savez(
        OUTPUT_BIG_PATH,
        user_ids_a=np.array(user_ids_a_big, dtype=object),
        emb_a=emb_a_big,
        user_ids_b=np.array(user_ids_b_big, dtype=object),
        emb_b=emb_b_big,
        user_ids_mixed=np.array(user_ids_m_big, dtype=object),
        emb_mixed=emb_m_big,
        embed_dim=np.array([embed_dim], dtype=np.int32),
    )

    print("All done.")


if __name__ == "__main__":
    main()
