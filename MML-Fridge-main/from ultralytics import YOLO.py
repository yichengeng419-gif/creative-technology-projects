from ultralytics import YOLO
import cv2

# 1. 加载模型（第一次运行会自动下载）
model = YOLO('yolov8n.pt')

# 2. 执行识别 (这里可以换成你手拍的食物图片路径)
# 我们先用一张自带的图测试，或者你放一张 'apple.jpg' 在目录下
results = model.predict(source='Food/apple.jpg', save=True, conf=0.5)

# 3. 打印识别到的结果，模拟“入库”前的逻辑提取
print("--- 识别结果摘要 ---")
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = r.names[cls_id]
        conf = float(box.conf[0])
        print(f"检测到物体: {label} | 置信度: {conf:.2f}")

        # 模拟导师建议的“逻辑映射”
        if label in ['apple', 'orange', 'broccoli']:
            print(f"提示：已自动识别出生鲜 [{label}]，准备计算保鲜期...")
        elif label == 'bottle':
            print("提示：检测到瓶状物，需手动确认是牛奶还是果汁。")
