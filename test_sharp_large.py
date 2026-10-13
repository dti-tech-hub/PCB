"""Quick live test - large sharp image."""

from pathlib import Path
import numpy as np, cv2, random
from fastapi.testclient import TestClient
from app.main import create_app

c = TestClient(create_app())

def sharp_large_png_bytes():
    """Create a large (1600x1600) sharp test image with edges."""
    img = np.full((1600, 1600, 3), (0, 130, 0), dtype=np.uint8)
    # Add grid lines for texture
    for i in range(0, 1600, 100):
        img[i, :] = 255        # horizontal line
        img[:, i] = 255        # vertical line
    # Add some rectangular defects
    img[300:400, 400:500] = (0, 0, 255)      # missing_hole (red)
    img[900:1000, 200:350] = (0, 165, 255)   # mouse_bite (orange)
    # Add some noise for sharpness
    for _ in range(200):
        x = random.randint(0, 1599)
        y = random.randint(0, 1599)
        cv2.circle(img, (x, y), 2, (0, 0, 0), -1)
    ok, b = cv2.imencode('.png', img)
    assert ok
    return bytes(b)

r = c.post('/upload', files={'file': ('large.png', sharp_large_png_bytes(), 'image/png')})
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
    print(f'Detection lines: {det_lines[:6]}')
    print('Body excerpt:', body[:400])