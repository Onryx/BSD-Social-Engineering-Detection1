import itertools
import numpy as np
from sentence_transformers import SentenceTransformer, util


class BSDDetector:
    """
    Behavioral Synchronization Detection (BSD).

    Implements the paper's four runtime score families:
    T = temporal synchronization
    C = content similarity
    B = behavioral role-sequence similarity
    N = network structure

    Cohen's kappa is deliberately not used as a runtime feature.
    It belongs only to annotation-quality reporting.
    """

    def __init__(
        self,
        weights=None,
        threshold=0.70,
        model_name="sentence-transformers/all-mpnet-base-v2",
        epsilon=1e-6,
    ):
        self.weights = weights or {
            "T": 0.35,
            "C": 0.20,
            "B": 0.25,
            "N": 0.20,
        }
        self.threshold = threshold
        self.epsilon = epsilon
        self.model = SentenceTransformer(model_name)

    def calculate_temporal_score(self, timestamps_by_channel):
        """
        Calculates temporal score T.

        Input:
            timestamps_by_channel can be either:
            1. list of timestamps for a single-channel sequence
            2. dict such as:
               {
                   "email": [1714900000, 1714900120],
                   "sms": [1714900060, 1714900180]
               }

        Paper logic:
            T_reg = max(0, 1 - sigma_delta / (mean_delta + epsilon))
            T = (0.55*T_reg + 0.45*I_ab*max(0, r_ab)) / (0.55 + 0.45*I_ab)

        For one-channel sequences, r_ab is not imputed.
        """

        if isinstance(timestamps_by_channel, dict):
            all_timestamps = []
            channel_intervals = []

            for timestamps in timestamps_by_channel.values():
                sorted_times = sorted(timestamps)
                all_timestamps.extend(sorted_times)

                if len(sorted_times) >= 2:
                    channel_intervals.append(np.diff(sorted_times))

            channel_count = len([v for v in timestamps_by_channel.values() if len(v) > 0])
        else:
            all_timestamps = sorted(timestamps_by_channel)
            channel_intervals = [np.diff(all_timestamps)] if len(all_timestamps) >= 2 else []
            channel_count = 1

        if len(all_timestamps) < 2:
            return 0.0

        intervals = np.diff(sorted(all_timestamps))
        mean_delta = float(np.mean(intervals))
        sigma_delta = float(np.std(intervals))

        t_reg = max(0.0, 1.0 - sigma_delta / (mean_delta + self.epsilon))
        t_reg = float(np.clip(t_reg, 0.0, 1.0))

        if channel_count < 2:
            return t_reg

        positive_correlations = []

        for a, b in itertools.combinations(channel_intervals, 2):
            min_len = min(len(a), len(b))

            if min_len < 2:
                continue

            a = np.asarray(a[:min_len], dtype=float)
            b = np.asarray(b[:min_len], dtype=float)

            if np.std(a) == 0 or np.std(b) == 0:
                continue

            r_ab = float(np.corrcoef(a, b)[0, 1])
            positive_correlations.append(max(0.0, r_ab))

        mean_positive_r = float(np.mean(positive_correlations)) if positive_correlations else 0.0
        temporal_score = (0.55 * t_reg + 0.45 * mean_positive_r) / (0.55 + 0.45)

        return float(np.clip(temporal_score, 0.0, 1.0))

    def calculate_content_similarity(self, messages):
        """
        Calculates content score C using Sentence-BERT embeddings.

        Mean cosine similarity is mapped from [-1, 1] to [0, 1]:
            C = (1 + mean_cosine) / 2
        """

        if len(messages) < 2:
            return 0.0

        embeddings = self.model.encode(messages, convert_to_tensor=True)
        similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

        mask = np.ones(similarity_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, False)

        mean_cosine = float(similarity_matrix[mask].mean())
        content_score = (1.0 + mean_cosine) / 2.0

        return float(np.clip(content_score, 0.0, 1.0))

    def calculate_behavioral_score(self, observed_roles, templates=None):
        """
        Calculates behavioral role-sequence score B using normalized
        Levenshtein distance against known campaign templates.

        Example roles:
            IC = InitialContact
            TB = TrustBuilding
            AC = AuthorityClaim
            UI = UrgencyInduction
            AR = ActionRequest
        """

        if not observed_roles:
            return 0.0

        templates = templates or [
            ["IC", "TB", "AC", "UI", "AR"],
            ["IC", "TB", "UI", "AR"],
            ["IC", "TB", "TB", "AC", "UI", "AR"],
            ["IC", "AC", "UI", "AR"],
            ["IC", "TB", "AR"],
            ["IC", "UI", "AR"],
        ]

        distances = []
        for template in templates:
            distance = self._levenshtein_distance(observed_roles, template)
            normalizer = max(len(observed_roles), len(template), 1)
            distances.append(distance / normalizer)

        best_distance = min(distances)
        behavioral_score = 1.0 - best_distance

        return float(np.clip(behavioral_score, 0.0, 1.0))

    def calculate_network_score(self, edges, shared_targets=0, total_targets=0):
        """
        Calculates network score N.

        Input:
            edges: list of directed interaction tuples, e.g.
                   [("actor1", "victim"), ("actor2", "victim")]
            shared_targets: count of targets contacted by more than one actor
            total_targets: total distinct targets

        N = mean(density, overlap, clustering)
        """

        if not edges:
            return 0.0

        simple_edges = set(edges)
        nodes = set()

        for source, target in simple_edges:
            nodes.add(source)
            nodes.add(target)

        node_count = len(nodes)
        edge_count = len(simple_edges)

        if node_count <= 1:
            density = 0.0
        else:
            density = edge_count / (node_count * (node_count - 1))

        if total_targets <= 0:
            overlap = 0.0
        else:
            overlap = shared_targets / total_targets

        clustering = self._mean_local_clustering(nodes, simple_edges)

        network_score = (density + overlap + clustering) / 3.0
        return float(np.clip(network_score, 0.0, 1.0))

    def calculate_bsd_score(self, temporal_score, content_score, behavioral_score, network_score):
        """
        Final BSD score:
            S = w1*T + w2*C + w3*B + w4*N

        No kappa term is included.
        """

        score = (
            self.weights["T"] * temporal_score
            + self.weights["C"] * content_score
            + self.weights["B"] * behavioral_score
            + self.weights["N"] * network_score
        )

        return float(np.clip(score, 0.0, 1.0))

    def classify(self, temporal_score, content_score, behavioral_score, network_score):
        score = self.calculate_bsd_score(
            temporal_score,
            content_score,
            behavioral_score,
            network_score,
        )

        label = "COORDINATED_ATTACK" if score >= self.threshold else "BENIGN"
        return score, label

    @staticmethod
    def _levenshtein_distance(a, b):
        rows = len(a) + 1
        cols = len(b) + 1

        dp = [[0] * cols for _ in range(rows)]

        for i in range(rows):
            dp[i][0] = i

        for j in range(cols):
            dp[0][j] = j

        for i in range(1, rows):
            for j in range(1, cols):
                substitution_cost = 0 if a[i - 1] == b[j - 1] else 1

                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + substitution_cost,
                )

        return dp[-1][-1]

    @staticmethod
    def _mean_local_clustering(nodes, directed_edges):
        undirected_neighbors = {node: set() for node in nodes}

        for source, target in directed_edges:
            undirected_neighbors[source].add(target)
            undirected_neighbors[target].add(source)

        coefficients = []

        for node, neighbors in undirected_neighbors.items():
            degree = len(neighbors)

            if degree < 2:
                coefficients.append(0.0)
                continue

            possible_links = degree * (degree - 1) / 2
            actual_links = 0

            for u, v in itertools.combinations(neighbors, 2):
                if (u, v) in directed_edges or (v, u) in directed_edges:
                    actual_links += 1

            coefficients.append(actual_links / possible_links)

        return float(np.mean(coefficients)) if coefficients else 0.0


if __name__ == "__main__":
    detector = BSDDetector()

    timestamps_by_channel = {
        "email": [1714900000, 1714900120, 1714900240],
        "sms": [1714900060, 1714900180, 1714900300],
    }

    messages = [
        "Please verify your bank account immediately.",
        "Your account requires urgent verification.",
        "Security alert: verify your banking credentials now.",
    ]

    roles = ["IC", "TB", "AC", "UI", "AR"]

    edges = [
        ("actor_email", "target_employee"),
        ("actor_sms", "target_employee"),
        ("actor_voice", "target_employee"),
        ("actor_email", "actor_sms"),
    ]

    temporal_score = detector.calculate_temporal_score(timestamps_by_channel)
    content_score = detector.calculate_content_similarity(messages)
    behavioral_score = detector.calculate_behavioral_score(roles)
    network_score = detector.calculate_network_score(
        edges=edges,
        shared_targets=1,
        total_targets=1,
    )

    final_score, label = detector.classify(
        temporal_score,
        content_score,
        behavioral_score,
        network_score,
    )

    print("BSD Framework Detection Results")
    print(f"T score: {temporal_score:.4f}")
    print(f"C score: {content_score:.4f}")
    print(f"B score: {behavioral_score:.4f}")
    print(f"N score: {network_score:.4f}")
    print(f"Synchronization Score S: {final_score:.4f}")
    print(f"Status: {label}")



