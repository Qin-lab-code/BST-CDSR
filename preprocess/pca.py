import os
import argparse
import numpy as np
from sklearn.decomposition import PCA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", type=str, required=True,
                        help="训练集语义 embedding 的 npz 路径")
    parser.add_argument("--val_npz", type=str, required=True,
                        help="验证集语义 embedding 的 npz 路径")
    parser.add_argument("--test_npz", type=str, required=True,
                        help="测试集语义 embedding 的 npz 路径")
    parser.add_argument("--out_dir", type=str, default="./pca_out",
                        help="保存压缩后 npz 和 PCA 参数的目录")
    parser.add_argument("--pca_dim", type=int, default=512,
                        help="PCA 输出维度 K（例如 512/768），后面再接 adapter 到 256")
    parser.add_argument("--max_samples", type=int, default=1000000,
                        help="用于拟合 PCA 的最大样本数（超过则随机采样）")
    return parser.parse_args()


def load_npz(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = np.load(path)
    for key in ["emb_a", "emb_b", "emb_mixed"]:
        if key not in data:
            raise KeyError(f"{path} 中缺少键：{key}")
    return data


def fit_shared_pca(emb_list, pca_dim, max_samples):
    print(">>> Concatenating train embeddings for PCA fit ...")
    all_train = np.concatenate(emb_list, axis=0)  # [N_total, 4096]
    n_samples = all_train.shape[0]
    print(f"Total train samples for PCA: {n_samples}")

    if n_samples > max_samples:
        print(f"Sampling {max_samples} examples out of {n_samples} for PCA...")
        idx = np.random.choice(n_samples, size=max_samples, replace=False)
        all_train = all_train[idx]

    print(f"Fitting PCA to dimension {pca_dim} ...")
    pca = PCA(
        n_components=pca_dim,
        svd_solver="randomized",  # 更快，适合高维
        random_state=42,
    )
    pca.fit(all_train)

    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA fitted. Explained variance ratio sum: {explained:.4f}")

    return pca


def transform_split(pca, in_data, split_name, out_path):
    print(f">>> Transforming {split_name} ...")

    emb_a = in_data["emb_a"]
    emb_b = in_data["emb_b"]
    emb_m = in_data["emb_mixed"]

    print(f"  {split_name} emb_a shape: {emb_a.shape}")
    print(f"  {split_name} emb_b shape: {emb_b.shape}")
    print(f"  {split_name} emb_mixed shape: {emb_m.shape}")

    emb_a_pca = pca.transform(emb_a).astype("float32")
    emb_b_pca = pca.transform(emb_b).astype("float32")
    emb_m_pca = pca.transform(emb_m).astype("float32")

    np.savez(
        out_path,
        emb_a=emb_a_pca,
        emb_b=emb_b_pca,
        emb_mixed=emb_m_pca,
    )
    print(f"  Saved PCA-compressed {split_name} to: {out_path}")


def save_pca_params(pca, out_path):
    np.savez(
        out_path,
        mean=pca.mean_.astype("float32"),
        components=pca.components_.astype("float32"),
        explained_variance=pca.explained_variance_.astype("float32"),
        explained_variance_ratio=pca.explained_variance_ratio_.astype("float32"),
    )
    print(f">>> Saved PCA parameters to: {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Loading train npz ===")
    train_data = load_npz(args.train_npz)
    emb_a_train = train_data["emb_a"]
    emb_b_train = train_data["emb_b"]
    emb_m_train = train_data["emb_mixed"]

    pca = fit_shared_pca(
        emb_list=[emb_a_train, emb_b_train, emb_m_train],
        pca_dim=args.pca_dim,
        max_samples=args.max_samples,
    )

    # 保存 PCA 参数
    pca_param_path = os.path.join(args.out_dir, "shared_pca_params.npz")
    save_pca_params(pca, pca_param_path)

    # train
    train_out = os.path.join(args.out_dir, "train_semantic_pca.npz")
    transform_split(pca, train_data, "train", train_out)

    # val
    print("=== Loading val npz ===")
    val_data = load_npz(args.val_npz)
    val_out = os.path.join(args.out_dir, "val_semantic_pca.npz")
    transform_split(pca, val_data, "val", val_out)

    # test
    print("=== Loading test npz ===")
    test_data = load_npz(args.test_npz)
    test_out = os.path.join(args.out_dir, "test_semantic_pca.npz")
    transform_split(pca, test_data, "test", test_out)

    print("=== All done. ===")


if __name__ == "__main__":
    main()
