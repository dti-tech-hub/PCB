"""Download real PCB YOLO26 weights (reproducible, no fake weights).

Source: betty0/pcb-defect-detection (Hugging Face, AGPL-3.0).
Classes: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper.
Run: python scripts/download_weights.py
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "betty0/pcb-defect-detection"
FILENAME = "best.pt"
DEST = Path(__file__).resolve().parent.parent / "models" / "pcb_yolo26.pt"


def download(dest: Path = DEST) -> Path:
    """Download weights to models/pcb_yolo26.pt.

    Args:
        dest: Destination path.

    Returns:
        Destination path.
    """
    src = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(src).read_bytes())
    print(f"Saved {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return dest


if __name__ == "__main__":
    download()
