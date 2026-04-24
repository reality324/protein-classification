#!/usr/bin/env python3
"""Quick ESM2 training script - train XGBoost on pre-computed ESM2 features"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms import ClassifierRegistry

def main():
    print("=" * 60)
    print("ESM2 Feature Training - Quick Version")
    print("=" * 60)

    # Load ESM2 features
    features_dir = Path("data/processed/esm2_aligned")
    labels_parquet = Path("data/datasets/train_subset.parquet")

    print(f"\n[1] Loading ESM2 features from {features_dir}")
    X_train = np.load(features_dir / "train_features.npy")
    X_val = np.load(features_dir / "val_features.npy")
    X_test = np.load(features_dir / "test_features.npy")
    print(f"    Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Load labels
    print(f"\n[2] Loading labels from {labels_parquet}")
    df = pd.read_parquet(labels_parquet)

    # Extract EC columns
    ec_cols = [c for c in df.columns if c.startswith('ec_')]

    # Same random split as feature extraction
    indices = np.arange(len(df))
    np.random.seed(42)
    np.random.shuffle(indices)

    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Extract EC main class (first digit, 0-indexed)
    def get_main_class(onehot_labels):
        """Extract main EC class from one-hot encoding. EC1.x -> 0, EC2.x -> 1, etc."""
        result = []
        for row in onehot_labels:
            idx = np.argmax(row)  # Find which column has 1
            col_name = ec_cols[idx]
            main_class = int(col_name.split('_')[1].split('.')[0]) - 1  # EC1.x -> 0
            result.append(main_class)
        return np.array(result)

    y_train = get_main_class(df[ec_cols].values[train_idx])
    y_val = get_main_class(df[ec_cols].values[val_idx])
    y_test = get_main_class(df[ec_cols].values[test_idx])

    # Filter samples that have valid EC labels (not all zeros)
    train_mask = y_train < 10  # Keep only EC1-7 (first digit 1-7, 0-indexed 0-6)
    val_mask = y_val < 10
    test_mask = y_test < 10

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    print(f"    After filtering (EC1-7): Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    print(f"    Classes: {np.unique(y_train)}")

    # Train Random Forest (faster than GradientBoosting)
    print(f"\n[3] Training Random Forest...")
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate
    print(f"\n[4] Evaluation...")
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f"    Accuracy: {acc:.4f}")
    print(f"    F1 (micro): {f1_micro:.4f}")
    print(f"    F1 (macro): {f1_macro:.4f}")

    # Save results
    output_dir = Path("results/test/models/esm2_rf")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[5] Saving results to {output_dir}...")

    # Save model
    import pickle
    with open(output_dir / "rf_best.pt", 'wb') as f:
        pickle.dump(clf, f)

    # Save metrics
    metrics = {
        "encoding": "esm2",
        "algorithm": "rf",
        "accuracy": float(acc),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "feature_dim": X_train.shape[1],
        "num_classes": len(np.unique(y_train)),
    }
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[6] Done! Results saved to {output_dir}")
    print("\n" + "=" * 60)
    print(f"ESM2 + Random Forest Results:")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1 (micro): {f1_micro:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
