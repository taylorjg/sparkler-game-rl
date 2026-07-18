"""Export a Stable-Baselines3 MlpPolicy to browser-friendly JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch as th
from stable_baselines3 import PPO


def tensor_to_list(tensor: th.Tensor) -> list[list[float]] | list[float]:
    values = tensor.detach().cpu().tolist()
    if isinstance(values, list) and values and isinstance(values[0], list):
        return values
    return values


def detect_activation(policy_net: th.nn.Sequential) -> str:
    for module in policy_net:
        if isinstance(module, th.nn.Tanh):
            return "tanh"
        if isinstance(module, th.nn.ReLU):
            return "relu"
    raise RuntimeError("Could not detect hidden-layer activation in policy_net")


def export_policy(model_path: Path, output_path: Path) -> None:
    model = PPO.load(model_path)
    policy_net = model.policy.mlp_extractor.policy_net
    action_net = model.policy.action_net

    linear_layers = [module for module in policy_net if isinstance(module, th.nn.Linear)]
    if len(linear_layers) < 1:
        raise RuntimeError("Expected at least one Linear layer in policy_net")

    activation = detect_activation(policy_net)
    hidden_layers = [layer.out_features for layer in linear_layers]
    weights: dict[str, list[list[float]] | list[float]] = {}
    biases: dict[str, list[list[float]] | list[float]] = {}

    for index, layer in enumerate(linear_layers):
        weights[f"layer{index}"] = tensor_to_list(layer.weight)
        biases[f"layer{index}"] = tensor_to_list(layer.bias)

    weights["action"] = tensor_to_list(action_net.weight)
    biases["action"] = tensor_to_list(action_net.bias)

    payload = {
        "observationSize": int(model.observation_space.shape[0]),
        "actionSize": int(model.action_space.n),
        "hiddenLayers": hidden_layers,
        "activation": activation,
        "weights": weights,
        "biases": biases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Exported policy to {output_path}")
    print(f"Observation size: {payload['observationSize']}")
    print(f"Hidden layers: {payload['hiddenLayers']}")
    print(f"Action size: {payload['actionSize']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SB3 policy weights to JSON")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/sparkler_bc"),
        help="Path to SB3 model (without .zip extension)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/sparkler_bc.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()
    export_policy(args.model, args.output)


if __name__ == "__main__":
    main()
