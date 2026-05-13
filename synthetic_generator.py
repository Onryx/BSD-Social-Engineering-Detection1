import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


ATTACK_MESSAGES = {
    "IC": [
        "Hello, I am contacting you about a pending account update.",
        "Good day, we need to confirm a recent request on your profile.",
        "This is a quick follow-up about your organization account.",
    ],
    "TB": [
        "We have worked with your department before and need to confirm the details.",
        "This should only take a moment and helps us keep your access current.",
        "The request is routine and relates to an existing service record.",
    ],
    "AC": [
        "This request has been approved by management.",
        "Your supervisor asked that this be completed today.",
        "The finance office authorized this verification step.",
    ],
    "UI": [
        "Please complete this immediately to avoid service interruption.",
        "This is urgent and must be handled before the close of business.",
        "A delay may temporarily restrict your account access.",
    ],
    "AR": [
        "Open the verification form and confirm your credentials.",
        "Send the requested code so we can complete the update.",
        "Confirm the account details using the secure link provided.",
    ],
}

BENIGN_MESSAGES = [
    "Please find the meeting notes attached for review.",
    "Can we move the project discussion to tomorrow afternoon?",
    "I have updated the document with the requested changes.",
    "Thanks for the clarification. I will follow up with the team.",
    "The schedule looks fine from my side.",
    "Please confirm whether the report should include the new figures.",
    "I will share the draft once the remaining comments are resolved.",
    "The department meeting has been moved to next week.",
    "Kindly review the agenda before the call.",
    "The file has been received and archived.",
]

SCENARIO_TEMPLATES = {
    "BEC": ["IC", "TB", "AC", "UI", "AR"],
    "spear_phishing": ["IC", "TB", "UI", "AR"],
    "pretexting": ["IC", "TB", "TB", "AC", "UI", "AR"],
    "authority_impersonation": ["IC", "AC", "UI", "AR"],
}

CHANNELS = ["email", "sms", "voice", "social"]


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def choose_scenario(config):
    names = list(config["attack_priors"].keys())
    probabilities = list(config["attack_priors"].values())
    return random.choices(names, weights=probabilities, k=1)[0]


def make_attack_sequence(sequence_id, config):
    gen = config["synthetic_generation"]

    scenario = choose_scenario(config)
    base_roles = SCENARIO_TEMPLATES[scenario]

    step_count = random.randint(gen["attack_step_min"], gen["attack_step_max"])

    roles = []
    while len(roles) < step_count:
        roles.extend(base_roles)
    roles = roles[:step_count]

    channel_count = random.randint(gen["attack_channel_min"], gen["attack_channel_max"])
    channels = random.sample(CHANNELS, channel_count)

    actor_count = random.randint(gen["attack_actor_min"], gen["attack_actor_max"])
    actors = [f"actor_{i}" for i in range(actor_count)]

    start_time = 1714900000 + sequence_id * 10000
    base_gap = random.randint(240, 900)

    timestamps_by_channel = {channel: [] for channel in channels}
    messages = []
    edges = []

    for step, role in enumerate(roles):
        channel = channels[step % channel_count]
        actor = actors[step % actor_count]

        timestamp = start_time + step * base_gap + random.randint(-60, 60)
        timestamps_by_channel[channel].append(timestamp)

        message = random.choice(ATTACK_MESSAGES[role])
        messages.append(message)

        target = "target_employee"
        edges.append((actor, target))

        if step > 0:
            previous_actor = actors[(step - 1) % actor_count]
            edges.append((previous_actor, actor))

    overlap_ratio = random.uniform(
        gen["attack_target_overlap_min"],
        gen["attack_target_overlap_max"],
    )

    return {
        "sequence_id": f"seq_{sequence_id:04d}",
        "label": 1,
        "class_name": "attack",
        "scenario": scenario,
        "timestamps_by_channel": timestamps_by_channel,
        "messages": messages,
        "roles": roles,
        "edges": edges,
        "shared_targets": 1,
        "total_targets": max(1, round(1 / overlap_ratio)),
        "channels": channels,
    }


def make_benign_sequence(sequence_id, config):
    gen = config["synthetic_generation"]

    step_count = random.randint(gen["benign_step_min"], gen["benign_step_max"])
    channel_count = random.randint(gen["benign_channel_min"], gen["benign_channel_max"])

    channels = random.sample(CHANNELS[:2], channel_count)
    actors = [f"employee_{i}" for i in range(random.randint(2, 5))]

    start_time = 1714900000 + sequence_id * 10000
    timestamps_by_channel = {channel: [] for channel in channels}

    messages = []
    roles = []
    edges = []

    benign_roles = ["IC", "TB", "AR"]

    for step in range(step_count):
        channel = channels[step % channel_count]
        actor = actors[step % len(actors)]
        recipient = actors[(step + 1) % len(actors)]

        timestamp = start_time + step * random.randint(1800, 7200) + random.randint(-300, 300)
        timestamps_by_channel[channel].append(timestamp)

        messages.append(random.choice(BENIGN_MESSAGES))
        roles.append(random.choice(benign_roles))
        edges.append((actor, recipient))

    overlap_ratio = random.uniform(
        gen["benign_target_overlap_min"],
        gen["benign_target_overlap_max"],
    )

    total_targets = 4
    shared_targets = int(round(overlap_ratio * total_targets))

    return {
        "sequence_id": f"seq_{sequence_id:04d}",
        "label": 0,
        "class_name": "benign",
        "scenario": "benign_professional_exchange",
        "timestamps_by_channel": timestamps_by_channel,
        "messages": messages,
        "roles": roles,
        "edges": edges,
        "shared_targets": shared_targets,
        "total_targets": total_targets,
        "channels": channels,
    }


def generate_corpus(config):
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    sequences = []

    for index in range(config["n_attack"]):
        sequences.append(make_attack_sequence(index, config))

    for index in range(config["n_attack"], config["n_attack"] + config["n_benign"]):
        sequences.append(make_benign_sequence(index, config))

    random.shuffle(sequences)
    return sequences


def flatten_for_csv(sequences):
    rows = []

    for sequence in sequences:
        rows.append(
            {
                "sequence_id": sequence["sequence_id"],
                "label": sequence["label"],
                "class_name": sequence["class_name"],
                "scenario": sequence["scenario"],
                "channels": "|".join(sequence["channels"]),
                "roles": "|".join(sequence["roles"]),
                "message_count": len(sequence["messages"]),
                "edge_count": len(sequence["edges"]),
                "shared_targets": sequence["shared_targets"],
                "total_targets": sequence["total_targets"],
            }
        )

    return pd.DataFrame(rows)


def save_corpus_summary(sequences, output_path="data/corpus_summary.csv"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = flatten_for_csv(sequences)
    dataframe.to_csv(output_path, index=False)


if __name__ == "__main__":
    config = load_config()
    corpus = generate_corpus(config)
    save_corpus_summary(corpus)
    print(f"Generated {len(corpus)} sequences.")
    print("Saved data/corpus_summary.csv")
