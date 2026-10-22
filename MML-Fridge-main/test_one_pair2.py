import cv2
import numpy as np
from pathlib import Path

# ===== Input image paths =====
before_path = Path("data/event_sequences/insert/milk_insert_000_before.jpg")
after_path = Path("data/event_sequences/insert/milk_insert_000_after.jpg")

# ===== Read images =====
img_before = cv2.imread(str(before_path))
img_after = cv2.imread(str(after_path))

if img_before is None:
    raise FileNotFoundError(f"Cannot read: {before_path}")
if img_after is None:
    raise FileNotFoundError(f"Cannot read: {after_path}")

h, w = img_before.shape[:2]

# ===== Define ROI (lower shelf center area) =====
# You can adjust these later if needed
x1 = int(w * 0.15)
x2 = int(w * 0.90)
y1 = int(h * 0.40)
y2 = int(h * 0.95)

roi_before = img_before[y1:y2, x1:x2]
roi_after = img_after[y1:y2, x1:x2]

# ===== Convert to grayscale =====
gray_before = cv2.cvtColor(roi_before, cv2.COLOR_BGR2GRAY)
gray_after = cv2.cvtColor(roi_after, cv2.COLOR_BGR2GRAY)

# ===== Gaussian blur to reduce noise =====
gray_before = cv2.GaussianBlur(gray_before, (7, 7), 0)
gray_after = cv2.GaussianBlur(gray_after, (7, 7), 0)

# ===== Absolute difference =====
diff = cv2.absdiff(gray_before, gray_after)

# ===== Threshold =====
_, thresh = cv2.threshold(diff, 55, 255, cv2.THRESH_BINARY)

# ===== Morphological processing =====
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=1)

# ===== Find contours =====
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result = img_after.copy()
all_boxes = img_after.copy()

largest_area = 0
largest_box = None

roi_h, roi_w = gray_after.shape

for contour in contours:
    area = cv2.contourArea(contour)
    if area < 800:
        continue

    x, y, bw, bh = cv2.boundingRect(contour)

    # Reject very large boxes that cover nearly the whole ROI
    if bw > roi_w * 0.85 and bh > roi_h * 0.85:
        continue

    # Draw all candidate boxes in blue on full image
    full_x1 = x1 + x
    full_y1 = y1 + y
    full_x2 = full_x1 + bw
    full_y2 = full_y1 + bh
    cv2.rectangle(all_boxes, (full_x1, full_y1), (full_x2, full_y2), (255, 0, 0), 2)

    if area > largest_area:
        largest_area = area
        largest_box = (x, y, bw, bh)

# Draw ROI on result image
cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 0), 2)

# Draw largest box in green
if largest_box:
    x, y, bw, bh = largest_box

    full_x1 = x1 + x
    full_y1 = y1 + y
    full_x2 = full_x1 + bw
    full_y2 = full_y1 + bh

    cv2.rectangle(result, (full_x1, full_y1), (full_x2, full_y2), (0, 255, 0), 3)

    cv2.putText(
        result,
        f"INSERT candidate area={int(largest_area)}",
        (full_x1, max(30, full_y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    print("Largest box in ROI:", largest_box, "Area:", largest_area)
else:
    cv2.putText(
        result,
        "No significant change detected",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    print("No significant change detected")

# ===== Save outputs =====
output_dir = Path("output2")
output_dir.mkdir(exist_ok=True)

cv2.imwrite(str(output_dir / "milk000_01_before.jpg"), img_before)
cv2.imwrite(str(output_dir / "milk000_02_after.jpg"), img_after)
cv2.imwrite(str(output_dir / "milk000_03_roi_before.jpg"), roi_before)
cv2.imwrite(str(output_dir / "milk000_04_roi_after.jpg"), roi_after)
cv2.imwrite(str(output_dir / "milk000_05_diff.jpg"), diff)
cv2.imwrite(str(output_dir / "milk000_06_thresh.jpg"), thresh)
cv2.imwrite(str(output_dir / "milk000_07_mask.jpg"), mask)
cv2.imwrite(str(output_dir / "milk000_08_all_boxes.jpg"), all_boxes)
cv2.imwrite(str(output_dir / "milk000_09_result.jpg"), result)

print("\nSaved files:")
for name in [
    "milk000_01_before.jpg",
    "milk000_02_after.jpg",
    "milk000_03_roi_before.jpg",
    "milk000_04_roi_after.jpg",
    "milk000_05_diff.jpg",
    "milk000_06_thresh.jpg",
    "milk000_07_mask.jpg",
    "milk000_08_all_boxes.jpg",
    "milk000_09_result.jpg",
]:
    print(output_dir / name)