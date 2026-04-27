# NOTE: written but not hardware-verified — smoke-test before use
"""
prepare_model.py
----------------
Downloads pretrained ResNet-18 weights from torchvision and exports them as an
ONNX model at:

    model_repository/resnet18_onnx/1/model.onnx

Run this once on the host before `docker compose up`.  The ONNX binary is
excluded from git (see .gitignore); this script is the canonical way to
reproduce it.

Usage:
    python prepare_model.py

Requirements:
    torch, torchvision, onnx  (see requirements.txt)
"""

import pathlib
import sys

import torch
import torchvision.models as models


OUTPUT_PATH = pathlib.Path(__file__).parent / "model_repository" / "resnet18_onnx" / "1" / "model.onnx"
OPSET_VERSION = 17  # ONNX opset; ONNX Runtime 1.17+ supports up to opset 20


def export_resnet18(output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading pretrained ResNet-18 weights from torchvision …")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    # Dummy input: batch=1, 3-channel 224×224 image (ImageNet standard)
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting ONNX to: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=OPSET_VERSION,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            # Allow variable batch size at runtime (Triton dynamic batching)
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        export_params=True,
        do_constant_folding=True,
    )

    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"Done. File size: {file_size_mb:.1f} MB")
    print()
    print("Next steps:")
    print("  docker compose up -d")
    print("  python client.py")


if __name__ == "__main__":
    if OUTPUT_PATH.exists():
        print(f"Model already exists at {OUTPUT_PATH}. Delete it to re-export.")
        sys.exit(0)
    export_resnet18(OUTPUT_PATH)
