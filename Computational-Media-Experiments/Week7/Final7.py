import cv2
import numpy as np
from dorothy import Dorothy 

dot = Dorothy()

class MyOutput():
    def __init__(self):
        #Method:Defining the "Perception System"
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=18, detectShadows=False)
        self.kernel_erosion = np.ones((3, 3), np.uint8)
        self.kernel_dilation = np.ones((6, 6), np.uint8)
        self.mosaic_block_size = 8
        self.trail_alpha = 0.82
        dot.start_loop(self.setup, self.draw)

    def setup(self):
        #Method:Camera input + image preprocessing
        self.camera = cv2.VideoCapture(0)
        self.prev_frame = None

    def draw(self):
        success, frame = self.camera.read()
        if not success:
            dot.background(dot.black)
            return
        #Method:Generate motion mask
        frame = cv2.resize(frame, (dot.width, dot.height))
        fgmask = self.fgbg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.erode(fgmask, self.kernel_erosion, iterations=1)
        motion_mask = cv2.dilate(motion_mask, self.kernel_dilation, iterations=2)
        motion_mask = cv2.GaussianBlur(motion_mask, (25, 25), 0)

        if self.prev_frame is None:
            self.prev_frame = frame.copy()
        #Method:Generate Trail Memory
        trail = cv2.addWeighted(self.prev_frame, self.trail_alpha, frame, 1 - self.trail_alpha, 0)
        self.prev_frame = trail.copy()
        #Method:Colour Mapping + Colour Grading
        colorized_mask = cv2.applyColorMap(motion_mask, cv2.COLORMAP_OCEAN)
        blended = cv2.addWeighted(trail, 0.6, colorized_mask, 0.4, 0)

        hsv = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h = (h + 15) % 180
        s = cv2.add(s, 25)
        v = cv2.add(v, 15)
        hsv_shifted = cv2.merge([h, s, v])
        blended = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)

        alpha, beta = 1.15, 10
        blended = cv2.convertScaleAbs(blended, alpha=alpha, beta=beta)
        #Method:Simulates the camera's "auto white balance offset" or "visual perception filter"
        overlay_color = (180, 210, 255)
        tint_layer = np.full_like(blended, overlay_color, dtype=np.uint8)
        final_blended = cv2.addWeighted(blended, 0.85, tint_layer, 0.15, 0)

        cv2.imshow("DynamicMosaic — Cold Flow", final_blended)
        if cv2.waitKey(1) & 0xFF == 27:
            self.camera.release()
            cv2.destroyAllWindows()
            exit()

MyOutput()
