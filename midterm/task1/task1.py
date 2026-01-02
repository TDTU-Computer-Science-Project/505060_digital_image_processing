import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

class FrameProcessor:
    def __init__(self):
        self.lower_red1 = np.array([0, 0, 0])
        self.upper_red1 = np.array([5, 255, 255])
        self.lower_red2 = np.array([113, 0, 50])
        self.upper_red2 = np.array([179, 200, 210])
        self.lower_blue = np.array([100, 100, 90])
        self.upper_blue = np.array([130, 255, 255])
        self.error_lower_red = np.array([100, 0, 30])
        self.error_upper_red = np.array([113, 250, 150])

    def preprocess(self, img, clip, tile, k_size, sigma, gam=True):
        def contrast_limited_adaptive_histogram_equalization(channel, clip, tile):
            return cv.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(channel)

        def blurr(img, k_size, sigma):
            return cv.GaussianBlur(img, k_size, sigma)
        
        def adjust_gamma(img, gamma):
            invGamma = 1 / gamma
            table = np.array([((i / 255) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
            return cv.LUT(img, table)
        
        img_gm = adjust_gamma(img, gamma=2) if gam else img
        blurr = cv.GaussianBlur(img_gm, (5, 5), 0)
        hsv = cv.cvtColor(blurr, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        v_clahe = contrast_limited_adaptive_histogram_equalization(v, clip, tile)
        hsv = cv.merge([h, s, v_clahe])
        return hsv

    def merge_masks(self, hsv_img, morphology=True):
        MORPH_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        DILATE_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

        def clean_mask(mask, kernel=MORPH_KERNEL, iterations=1):
            m = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=iterations)
            m = cv.morphologyEx(m, cv.MORPH_CLOSE, kernel, iterations=iterations)
            return m
        
        mask_red = cv.bitwise_or(
            cv.inRange(hsv_img, self.lower_red1, self.upper_red1),
            cv.inRange(hsv_img, self.lower_red2, self.upper_red2)
        )
        mask_error_red = cv.inRange(hsv_img, self.error_lower_red, self.error_upper_red)
        mask_blue = cv.inRange(hsv_img, self.lower_blue, self.upper_blue)
        if morphology:
            mask_red = clean_mask(mask_red, MORPH_KERNEL, iterations=1)
            mask_blue = clean_mask(mask_blue, MORPH_KERNEL, iterations=1)
            mask_red = cv.dilate(mask_red, DILATE_KERNEL, iterations=1)
            mask_blue = cv.dilate(mask_blue, DILATE_KERNEL, iterations=1)
        return mask_red, mask_blue

    def contour_process(self, img, mask_red, mask_blue, epsilon, min_area):
        masks = [mask_red, mask_blue]
        output_contour = []
        for mask in masks:
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            output_contour.append(contours)
            
            # Retrival part
            for cnt in contours:
                area = cv.contourArea(cnt)
                if area < min_area:
                    continue
                
                perimeter = cv.arcLength(cnt, True)
                approximate = cv.approxPolyDP(cnt, epsilon * perimeter, True)
                if len(approximate) == 3:
                    pts = [approximate[i][0] for i in range(3)]
                    d1 = np.linalg.norm(pts[0] - pts[1])
                    d2 = np.linalg.norm(pts[1] - pts[2])
                    d3 = np.linalg.norm(pts[2] - pts[0])

                    average = (d1 + d2 + d3) / 3
                    deviation = 0.2 * average

                    if abs(d1 - average) < deviation and abs(d2 - average) < deviation and abs(d3 - average) < deviation:
                        x, y, w, h = cv.boundingRect(approximate)
                        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                (x_c, y_c), radius = cv.minEnclosingCircle(cnt)
                radius = float(radius)
                circle_area = np.pi * (radius ** 2) if radius > 0 else 1.0
                circularity = area / circle_area
                if circularity > 0.67:
                    center = (int(x_c), int(y_c))
                    r = int(radius)
                    x0, y0 = max(0, center[0] - r), max(0, center[1] - r)
                    x1, y1 = center[0] + r, center[1] + r
                    cv.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    continue
                
        return img

    def pipeline_full_process(self, img, clip=3.0, tile=(8, 8), k_size=5, sigma=0, gamma=True, epsilon=0.03, min_area=600):
        preprocessed = self.preprocess(img, clip, tile, k_size, sigma, gamma)
        mask_red, mask_blue = self.merge_masks(preprocessed)
        result = self.contour_process(img, mask_red, mask_blue, epsilon, min_area)
        return result


INPUT_VIDEO = "task1.mp4"
OUTPUT_VIDEO = "task1_output.mp4"

cap = cv.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {INPUT_VIDEO}")

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS) or 25.0
print(f"[INFO] fps={fps}, size=({frame_width},{frame_height})")

fourcc = cv.VideoWriter_fourcc(*'mp4v')  # dùng mp4v để tránh cảnh báo XVID->mp4
out = cv.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))
if not out.isOpened():
    raise RuntimeError("Cannot open VideoWriter. Kiểm tra codec/đường dẫn đầu ra.")

frame_idx = 0

frame_processor = FrameProcessor()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = frame.copy()

    img_processed = frame_processor.pipeline_full_process(img)
    out.write(img_processed)

    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"[INFO] processed {frame_idx} frames...")

# Cleanup
cap.release()
out.release()
cv.destroyAllWindows()
print("[INFO] Done. Output saved to:", OUTPUT_VIDEO)
