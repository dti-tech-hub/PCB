"""Verify sample images through the real app pipeline (uses real weights)."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "uploads" / "samples"


def summarize(name: str, body: str) -> None:
    """Print status, error and verdict for one response body.

    Args:
        name: Sample name.
        body: Response HTML body.
    """
    err = re.findall(r'<div class="error">(.*?)</div>', body)
    if err:
        print(f"{name}: REJECTED -> {err[0]}")
        return
    verdict = re.search(r'<strong class="(\w+)">\s*(\w+)\s*</strong>', body)
    method = re.search(r"<strong>Method:</strong>\s*(\w+)", body)
    count = re.search(r"Detections \((\d+)\)", body)
    classes = re.findall(r"<li>(\w+) (\d\.\d\d)", body)
    print(
        f"{name}: {verdict.group(2) if verdict else '?'} | method={method.group(1) if method else '?'}"
        f" | dets={count.group(1) if count else '?'} | {classes[:4]}"
    )


def main() -> None:
    """Upload every sample and print the pipeline outcome."""
    client = TestClient(create_app(), raise_server_exceptions=False)
    for path in sorted(SAMPLES.glob("*.png")):
        with path.open("rb") as fh:
            resp = client.post("/upload", files={"file": (path.name, fh, "image/png")})
        summarize(path.name, resp.text)


if __name__ == "__main__":
    main()
