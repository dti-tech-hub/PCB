"""Quick live test - large image with defects (SAHI path)."""

from pathlib import Path
import numpy as np, cv2
from fastapi.testclient import TestClient
from app.main import create_app

c = TestClient(create_app())

def large_with_defects_png_bytes():
    """Create a large (1600x1600) sharp test image WITH defects."""
    img = np.full((1600, 1600, 3), (0, 130, 0), dtype=np.uint8)
    # Add grid lines for texture
    for i in range(0, 1600, 100):
        img[i, :] = 255        # horizontal line
        img[:, i] = 255        # vertical line
    # Add a clear short defect (blue rectangle)
    img[500:600, 600:700] = (255, 0, 0)       # short (blue in CLASS_COLORS)
    # Add a spur defect
    img[1100:1150, 1200:1280] = (255, 0, 255) # spur (magenta)
    # Add noise for sharpness
    import random as _rng
    for _ in range(300):
        x = _rng.randint(0, 1599)
        y = _rng.randint(0, 1599)
        cv2.circle(img, (x, y), 2, (0, 0, 0), -1)
    ok, b = cv2.imencode('.png', img)
    assert ok
    return bytes(b)

r = c.post('/upload', files={'file': ('defects.png', large_with_defects_png_bytes(), 'image/png')})
import re as re_mod
matches = re_mod.findall(r'<div class="error"[^>]*>(.*?)</div>', r.text)
print('error divs:', matches)
body_start = r.text.find('<body>')
body_end = r.text.find('</body>')
if body_start != -1 and body_end != -1:
    body = r.text[body_start:body_end+7]
    verdict_match = re_mod.search(r'<strong class="([^"]+)">([^<]+)</strong>', body)
    verdict_class = verdict_match.group(1) if verdict_match else None
    verdict_text = verdict_match.group(2) if verdict_match else None
    print(f'Verdict class: {verdict_class}, text: {verdict_text}')
    det_match = re_mod.search(r'Detections \(([0-9]+)\)</h3>', body)
    if det_match:
        print(f'Detections count: {det_match.group(1)}')
    det_lines = re_mod.findall(r'<li>([^<]+)</li>', body)
    print(f'Detection lines (first 6): {det_lines[:6]}')
    # Check method
    method_match = re_mod.search(r'Method: ([a-z]+)', body)
    print(f'Method: {method_match.group(1) if method_match else "N/A"}')
    print('Body excerpt:', body[:450])