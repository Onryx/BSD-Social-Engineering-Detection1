import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from baselines import (
    ContentEmbeddingBaseline,
    NetworkIsolationForestBaseline,
    calculate_metrics,
)
from bsd_detection import BSDDetector
from synthetic_generator import generate_corpus, load_config, save_corpus_summary


def score_sequence(detector, sequence):
    temporal_score = detector.calculate_temporal_score(sequence["timestamps_by_channel"])
    content_score = detector.calculate_content_similarity(sequence["messages"])
    behavioral_score = detector.calculate_behavioral_score(sequence["roles"])
    network_score = detector.calculate_network_score(
        edges=sequence["edges"],
        shared_targets=sequence["shared_targets"],
        total_targets=sequence["total_targets"],
    )

    final_score, label_name = detector.classify(
        temporal_score,
        content_score,
        behavioral_score,
        network_score,
    )

    prediction = 1 if label_name == "COORDINATED_ATTACK" else 0

    return {
        "T": temporal_score,
        "C": content_score,
        "B": behavioral_score,
        "N": network_score,
        "score": final_score,
        "prediction": prediction,
        "label_name": label_name,
    }


def make_folds(sequences, n_folds, seed):
    labels = np.array([sequence["label"] for sequence in sequences])

    splitter = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    fold_assignments = {}

    for fold_id, (_, test_index) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        for index in test_index:
            fold_assignments[index] = fold_id

    return fold_assignments


def save_folds(sequences, fold_assignments, output_path="folds.csv"):
    rows = []

    for index, sequence in enumerate(sequences):
        rows.append(
            {
                "sequence_id": sequence["sequence_id"],
                "label": sequence["label"],
                "fold": fold_assignments[index],
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)


def evaluate_bsd(sequences, fold_assignments, config):
    detector = BSDDetector(
        weights=config["weights"],
        threshold=config["threshold"],
        model_name=config["model_name"],
    )

    rows = []

    for index, sequence in enumerate(sequences):
        scores = score_sequence(detector, sequence)

        rows.append(
            {
                "sequence_id": sequence["sequence_id"],
                "fold": fold_assignments[index],
                "true_label": sequence["label"],
                "predicted_label": scores["prediction"],
                "predicted_name": scores["label_name"],
                "bsd_score": scores["score"],
                "T": scores["T"],
                "C": scores["C"],
                "B": scores["B"],
                "N": scores["N"],
                "scenario": sequence["scenario"],
                "channels": "|".join(sequence["channels"]),
            }
        )

    dataframe = pd.DataFrame(rows)
    return dataframe


def evaluate_baselines(sequences, fold_assignments, config):
    labels = np.array([sequence["label"] for sequence in sequences])
    fold_ids = sorted(set(fold_assignments.values()))

    all_rows = []

    for fold_id in fold_ids:
        train_indices = [index for index, fold in fold_assignments.items() if fold != fold_id]
        test_indices = [index for index, fold in fold_assignments.items() if fold == fold_id]

        train_sequences = [sequences[index] for index in train_indices]
        test_sequences = [sequences[index] for index in test_indices]

        y_train = labels[train_indices]
        y_test = labels[test_indices]

        content_baseline = ContentEmbeddingBaseline(
            model_name=config["model_name"],
            random_state=config["random_seed"],
        )
        content_baseline.fit(train_sequences, y_train)
        content_predictions = content_baseline.predict(test_sequences)
        content_probabilities = content_baseline.predict_proba(test_sequences)

        network_baseline = NetworkIsolationForestBaseline(
            contamination=0.1,
            random_state=config["random_seed"],
        )
        network_baseline.fit(train_sequences)
        network_predictions = network_baseline.predict(test_sequences)
        network_probabilities = network_baseline.predict_proba(test_sequences)

        for local_index, sequence_index in enumerate(test_indices):
            all_rows.append(
                {
                    "sequence_id": sequences[sequence_index]["sequence_id"],
                    "fold": fold_id,
                    "true_label": int(y_test[local_index]),
                    "content_baseline_pred": int(content_predictions[local_index]),
                    "content_baseline_score": float(content_probabilities[local_index]),
                    "network_baseline_pred": int(network_predictions[local_index]),
                    "network_baseline_score": float(network_probabilities[local_index]),
                }
            )

    return pd.DataFrame(all_rows)


def summarize_results(bsd_predictions, baseline_predictions):
    y_true = bsd_predictions["true_label"].to_numpy()
    y_bsd = bsd_predictions["predicted_label"].to_numpy()

    bsd_metrics = calculate_metrics(y_true, y_bsd)

    merged = bsd_predictions.merge(
        baseline_predictions,
        on=["sequence_id", "fold", "true_label"],
        how="left",
    )

    content_metrics = calculate_metrics(
        merged["true_label"].to_numpy(),
        merged["content_baseline_pred"].to_numpy(),
    )

    network_metrics = calculate_metrics(
        merged["true_label"].to_numpy(),
        merged["network_baseline_pred"].to_numpy(),
    )

    summary = pd.DataFrame(
        [
            {"method": "BSD", **bsd_metrics},
            {"method": "ContentEmbeddingBaseline", **content_metrics},
            {"method": "NetworkIsolationForest", **network_metrics},
        ]
    )

    return summary, merged


def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main():
    config = load_config("config.json")

    np.random.seed(config["random_seed"])

    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    sequences = generate_corpus(config)
    save_corpus_summary(sequences, "data/corpus_summary.csv")

    fold_assignments = make_folds(
        sequences=sequences,
        n_folds=config["n_folds"],
        seed=config["random_seed"],
    )

    save_folds(sequences, fold_assignments, "folds.csv")

    bsd_predictions = evaluate_bsd(sequences, fold_assignments, config)
    baseline_predictions = evaluate_baselines(sequences, fold_assignments, config)

    summary, merged_predictions = summarize_results(
        bsd_predictions=bsd_predictions,
        baseline_predictions=baseline_predictions,
    )

    merged_predictions.to_csv("results/sequence_predictions.csv", index=False)
    summary.to_csv("results/metrics_summary.csv", index=False)

    save_json(config, "results/config_used.json")

    print("Experiment complete.")
    print("Saved folds.csv")
    print("Saved data/corpus_summary.csv")
    print("Saved results/sequence_predictions.csv")
    print("Saved results/metrics_summary.csv")
    print()
    print(summary)


if __name__ == "__main__":
    main()
