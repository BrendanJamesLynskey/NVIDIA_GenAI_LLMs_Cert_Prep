# NOTE: written but not hardware-verified — smoke-test before use
"""
client.py
---------
Python client for the ResNet-18 Triton demo.

Sends a single random image tensor (or a real JPEG if --image is supplied)
to the Triton gRPC endpoint, prints the top-5 ImageNet class indices, and
reports round-trip latency.

Usage:
    python client.py [--server localhost:8001] [--image path/to/image.jpg]

Requirements:
    tritonclient[grpc], numpy  (see requirements.txt)
    Triton server must be running: docker compose up -d
"""

import argparse
import time

import numpy as np
import tritonclient.grpc as grpcclient


# --------------------------------------------------------------------------- #
# ImageNet class index → label (top-20 for quick reference)
# Full list: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
IMAGENET_LABELS = {
    258: "Samoyed",
    259: "Pomeranian",
    260: "chow",
    261: "keeshond",
    262: "Brabancon griffon",
    281: "tabby cat",
    282: "tiger cat",
    283: "Persian cat",
    284: "Siamese cat",
    285: "Egyptian cat",
}


def preprocess_random() -> np.ndarray:
    """Return a random NCHW FP32 tensor of shape (1, 3, 224, 224)."""
    rng = np.random.default_rng(seed=42)
    img = rng.uniform(0.0, 1.0, (1, 3, 224, 224)).astype(np.float32)
    # Normalise to ImageNet statistics (mean / std per channel)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    return (img - mean) / std


def preprocess_image(path: str) -> np.ndarray:
    """Load a JPEG/PNG, resize to 224×224, return NCHW FP32."""
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("Pillow is required for --image: pip install Pillow")

    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0  # HWC [0, 1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[np.newaxis]  # NCHW


def run_inference(server: str, image_path: str | None) -> None:
    print(f"Triton server: {server}")

    client = grpcclient.InferenceServerClient(url=server, verbose=False)

    # ------------------------------------------------------------------ #
    # Check server / model readiness
    if not client.is_server_ready():
        raise SystemExit("Triton server is not ready. Is `docker compose up` running?")

    model_name = "resnet18_onnx"
    model_version = "1"

    if not client.is_model_ready(model_name, model_version):
        raise SystemExit(
            f"Model {model_name} v{model_version} is not ready. "
            "Run prepare_model.py first, then restart docker compose."
        )

    meta = client.get_model_metadata(model_name, model_version)
    print(f"Model: {meta.name}  version: {model_version}  state: READY")

    # ------------------------------------------------------------------ #
    # Prepare input
    if image_path:
        input_data = preprocess_image(image_path)
        print(f"Loaded image from: {image_path}")
    else:
        input_data = preprocess_random()
        print("Using synthetic random input (pass --image <path> for a real image)")

    print(f"Sent input shape: {input_data.shape}  dtype: FP32")

    infer_input = grpcclient.InferInput("input", input_data.shape, "FP32")
    infer_input.set_data_from_numpy(input_data)

    infer_output = grpcclient.InferRequestedOutput("output")

    # ------------------------------------------------------------------ #
    # Send request and time it
    t0 = time.perf_counter()
    response = client.infer(
        model_name=model_name,
        model_version=model_version,
        inputs=[infer_input],
        outputs=[infer_output],
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------ #
    # Parse response
    logits = response.as_numpy("output")[0]  # shape (1000,)
    top5_indices = np.argsort(logits)[::-1][:5]
    top5_labels = [IMAGENET_LABELS.get(int(i), f"class {i}") for i in top5_indices]

    print(f"Top-5 classes: {list(top5_indices)}")
    print(f"Top-5 labels:  {top5_labels}")
    print(f"Round-trip latency: {latency_ms:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton ResNet-18 demo client")
    parser.add_argument("--server", default="localhost:8001", help="Triton gRPC endpoint")
    parser.add_argument("--image", default=None, help="Path to a JPEG/PNG image (optional)")
    args = parser.parse_args()
    run_inference(args.server, args.image)
