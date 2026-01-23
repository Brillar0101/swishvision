"""
Fine-tune RF-DETR S on basketball player detection dataset.

RF-DETR is a transformer-based object detection model that provides
state-of-the-art performance. This script fine-tunes it on the
basketball-player-detection-3-ycjdo dataset from Roboflow.

Usage:
    python train_rfdetr.py --epochs 50 --batch-size 4

The trained model will be saved to: output/rfdetr_basketball/
"""
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)


def download_dataset(
    workspace: str = "roboflow-jvuqo",
    project: str = "basketball-player-detection-3-ycjdo",
    version: int = 4,
    output_dir: str = "datasets",
) -> str:
    """
    Download dataset from Roboflow in COCO format.

    Args:
        workspace: Roboflow workspace
        project: Project name
        version: Dataset version
        output_dir: Directory to save dataset

    Returns:
        Path to downloaded dataset
    """
    from roboflow import Roboflow

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY environment variable not set")

    print(f"Downloading dataset: {workspace}/{project}/{version}")

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download("coco", location=output_dir)

    return dataset.location


def train_rfdetr(
    dataset_id: str = "basketball-player-detection-3-ycjdo/4",
    epochs: int = 50,
    batch_size: int = 4,
    image_size: int = 560,
    output_dir: str = "output/rfdetr_basketball",
    resume: str = None,
    dataset_dir: str = None,
):
    """
    Fine-tune RF-DETR S on basketball dataset.

    Args:
        dataset_id: Roboflow dataset ID (workspace/project/version)
        epochs: Number of training epochs
        batch_size: Batch size for training
        image_size: Input image size
        output_dir: Directory to save trained model
        resume: Path to checkpoint to resume from
        dataset_dir: Pre-downloaded dataset directory (optional)
    """
    try:
        from rfdetr import RFDETRBase
    except ImportError:
        print("Error: rfdetr package not installed.")
        print("Install it with: pip install rfdetr")
        return None

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RF-DETR Fine-tuning for Basketball Player Detection")
    print("=" * 60)
    print(f"Dataset: {dataset_id}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Download dataset if not provided
    if dataset_dir is None:
        print("\nDownloading dataset from Roboflow...")
        # Parse dataset_id: "project/version" or "workspace/project/version"
        parts = dataset_id.split("/")
        if len(parts) == 2:
            # Default workspace for basketball dataset
            workspace = "roboflow-jvuqo"
            project = parts[0]
            version = int(parts[1])
        elif len(parts) == 3:
            workspace = parts[0]
            project = parts[1]
            version = int(parts[2])
        else:
            raise ValueError(f"Invalid dataset_id format: {dataset_id}")

        dataset_dir = download_dataset(
            workspace=workspace,
            project=project,
            version=version,
            output_dir=str(output_path / "dataset"),
        )
    else:
        print(f"\nUsing pre-downloaded dataset: {dataset_dir}")

    print(f"Dataset location: {dataset_dir}")

    # Initialize model
    print("\nInitializing RF-DETR S model...")
    model = RFDETRBase()

    # Train
    print("\nStarting training...")

    model.train(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=image_size,
        project=str(output_path),
        resume=resume,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {output_path}")
    print("=" * 60)

    return str(output_path)


def export_model(checkpoint_path: str, output_path: str = None):
    """
    Export trained model for inference.

    Args:
        checkpoint_path: Path to trained checkpoint
        output_path: Path to save exported model
    """
    try:
        from rfdetr import RFDETRBase
    except ImportError:
        print("Error: rfdetr package not installed.")
        return None

    if output_path is None:
        output_path = Path(checkpoint_path).parent / "exported"

    print(f"Exporting model from: {checkpoint_path}")
    print(f"Output: {output_path}")

    model = RFDETRBase()
    model.load(checkpoint_path)

    # Export to ONNX for faster inference
    model.export(format="onnx", output=str(output_path))

    print(f"Exported model saved to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune RF-DETR S on basketball player detection"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="basketball-player-detection-3-ycjdo/4",
        help="Roboflow dataset ID",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4, reduce if OOM)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=560,
        help="Input image size (default: 560)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/rfdetr_basketball",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export model from checkpoint path (skips training)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Pre-downloaded dataset directory (skips download)",
    )

    args = parser.parse_args()

    if args.export:
        export_model(args.export)
    else:
        train_rfdetr(
            dataset_id=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            output_dir=args.output,
            resume=args.resume,
            dataset_dir=args.dataset_dir,
        )
