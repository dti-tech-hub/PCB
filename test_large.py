"""Quick live test - large image SAHI path."""

from pathlib import Path
import numpy as np, cv2
from fastapi.testclient import TestClient
from app.main import create_app

c = TestClient(create_app())

def large_png_bytes():
    """Create a large test image (1600x1600) that exceeds LARGE_IMAGE_MAX_SIDE=1280."""
    img = np.full((1600, 1600, 3), (0, 130, 0), dtype=np.uint8)
    # Add some rectangular defects
    img[200:250, 400:450] = (0, 0, 255)      # missing_hole region
    img[700:750, 200:300] = (0, 165, 255)    # mouse_bite region
    ok, b = cv2.imencode('.png', img)
    assert ok
    return bytes(b)

r = c.post('/upload', files={'file': ('large.png', large_png_bytes(), 'image/png')})
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
    print(f'Detection lines: {det_lines[:5]}')
    print('Body excerpt:', body[:350])