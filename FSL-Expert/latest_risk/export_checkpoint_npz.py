"""Export a reviewed PyTorch state dictionary for dependency-light APP inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{key: value.detach().cpu().numpy() for key, value in state.items()})
    print(output.resolve())


if __name__ == "__main__":
    main()
