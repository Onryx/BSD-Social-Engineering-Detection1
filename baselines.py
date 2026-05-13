import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ContentEmbeddingBaseline:
    """
    Lightweight content baseline.

    This uses Sentence-BERT embeddings with logistic regression.
    It is a practical repository baseline for text-only comparison.
    If exact fine-tuned BERT is required, add a separate transformers-based
    training script.
    """

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", random_state=42):
        self.model = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(max_iter=1000, random_state=random_state)

    def _embed_sequences(self, sequences):
        texts = [" ".join(sequence["messages"]) for sequence in sequences]
        return self.model.encode(texts, convert_to_numpy=True)

    def fit(self, train_sequences, y_train):
        x_train = self._embed_sequences(train_sequences)
        self.classifier.fit(x_train, y_train)

    def predict(self, test_sequences):
        x_test = self._embed_sequences(test_sequences)
        return self.classifier.predict(x_test)

    def predict_proba(self, test_sequences):
        x_test = self._embed_sequences(test_sequences)
        return self.classifier.predict_proba(x_test)[:, 1]


class NetworkIsolationForestBaseline:
    """
    Graph-only Isolation Forest baseline using simple network features.
    """

    def __init__(self, contamination=0.1, random_state=42):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
        )

    @staticmethod
    def extract_network_features(sequence):
        edges = set(tuple(edge) for edge in sequence["edges"])
        nodes = set()

        for source, target in edges:
            nodes.add(source)
            nodes.add(target)

        node_count = len(nodes)
        edge_count = len(edges)

        if node_count <= 1:
            density = 0.0
        else:
            density = edge_count / (node_count * (node_count - 1))

        total_targets = sequence.get("total_targets", 0)
        if total_targets <= 0:
            overlap = 0.0
        else:
            overlap = sequence.get("shared_targets", 0) / total_targets

        channel_count = len(sequence.get("channels", []))
        message_count = len(sequence.get("messages", []))

        return [density, overlap, channel_count, message_count]

    def _features(self, sequences):
        return np.array([self.extract_network_features(sequence) for sequence in sequences])

    def fit(self, train_sequences):
        x_train = self._features(train_sequences)
        self.model.fit(x_train)

    def predict(self, test_sequences):
        x_test = self._features(test_sequences)

        raw_predictions = self.model.predict(x_test)

        return np.array([1 if value == -1 else 0 for value in raw_predictions])

    def predict_proba(self, test_sequences):
        x_test = self._features(test_sequences)
        anomaly_score = -self.model.decision_function(x_test)

        min_score = anomaly_score.min()
        max_score = anomaly_score.max()

        if max_score == min_score:
            return np.zeros_like(anomaly_score)

        return (anomaly_score - min_score) / (max_score - min_score)


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": false_positive_rate(y_true, y_pred),
    }


def false_positive_rate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))

    denominator = false_positives + true_negatives

    if denominator == 0:
        return 0.0

    return false_positives / denominator
