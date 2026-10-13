"""Quick live test script - sharp image."""

from pathlib import Path
import numpy as np, cv2
from fastapi.testclient import TestClient
from app.main import create_app

c = TestClient(create_app())

def sharp_png_bytes(w=400, h=400):
    """Create a sharp test image with edges."""
    img = np.full((h,w,3), (0,128,0), dtype=np.uint8)
    # Add a sharp white rectangle
    img[100:300, 100:300] = 255
    # Add some grid lines for texture
    for i in range(0, h, 50):
        img[i, :] = 255
    for i in range(0, w, 50):
        img[:, i] = 255
    ok,b = cv2.imencode('.png',img)
    assert ok
    return bytes(b)

r = c.post('/upload', files={'file': ('test.png', sharp_png_bytes(), 'image/png')})
import re
matches = re.findall(r'<div class="error"[^>]*>(.*?)</div>', r.text)
print('error divs:', matches)
for keyword in ['blurry', 'PCB', 'quality', 'too small', 'Invalid', 'Error']:
    if keyword.lower() in r.text.lower():
        idx = r.text.lower().find(keyword.lower())
        print(f'Found "{keyword}" at index {idx}: ...{r.text[max(0,idx-20):idx+40]}...')
print('Full status:', r.status_code)
# Print just the body text (skip HTML boilerplate)
body_start = r.text.find('<body>')
body_end = r.text.find('</body>')
if body_start != -1 and body_end != -1:
    print('Body:', r.text[body_start:body_end+7])