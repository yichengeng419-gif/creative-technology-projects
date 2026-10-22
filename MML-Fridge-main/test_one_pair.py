import cv2
import numpy as np
from pathlib import Path

before_path = Path("data/event_sequences/insert/apple_insert_001_before.jpg")
after_path = Path("data/event_sequences/insert/apple_insert_001_after.jpg")

img_before = cv2.imread(str(before_path))
img_after = cv2.imread(str(after_path))

if img_before is None:
    raise FileNotFoundError(f"Cannot read: {before_path}")
if img_after is None:
    raise FileNotFoundError(f"Cannot read: {after_path}")

gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

gray_before = cv2.GaussianBlur(gray_before, (7, 7), 0)
gray_after = cv2.GaussianBlur(gray_after, (7, 7), 0)

diff = cv2.absdiff(gray_before, gray_after)
_, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

kernel = np.ones((7, 7), np.uint8)
mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=2)

contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result = img_after.copy()

largest_area = 0
largest_box = None

for contour in contours:
    area = cv2.contourArea(contour)
    if area < 1500:
        continue
    x, y, w, h = cv2.boundingRect(contour)
    if area > largest_area:
        largest_area = area
        largest_box = (x, y, w, h)

if largest_box:
    x, y, w, h = largest_box
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
    print("Largest box:", largest_box, "Area:", largest_area)
else:
    print("No significant change detected")

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

cv2.imwrite(str(output_dir / "apple_diff.jpg"), diff)
cv2.imwrite(str(output_dir / "apple_mask.jpg"), mask)
cv2.imwrite(str(output_dir / "apple_result.jpg"), result)

print("Saved:")
print(output_dir / "apple_diff.jpg")
print(output_dir / "apple_mask.jpg")
print(output_dir / "apple_result.jpg")