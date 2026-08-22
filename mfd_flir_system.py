import cv2
import numpy as np
import math
import time
import sys
import traceback
from PIL import Image, ImageDraw, ImageFont

def get_tactical_font(size):
    """Loads a monospaced tactical font, falling back gracefully if not found."""
    for font_name in ["consola.ttf", "consolas.ttf", "cour.ttf", "courier.ttf", "msgothic.ttc"]:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
    return ImageFont.load_default()

def create_green_ir_lut():
    """Generates a military green-phosphor thermal lookup table."""
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        r = int(i * 0.15)
        g = int(min(255, i * 1.05 + 10))
        b = int(i * 0.25)
        lut[i, 0] = [b, g, r]
    return lut

def apply_thermal_filter(frame, mode='GREEN', polarity='WHT'):
    """Applies FLIR sensor simulation (Green Thermal vs Black & White Thermal)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    if polarity == 'BLK':
        enhanced = cv2.bitwise_not(enhanced)

    if mode == 'GREEN':
        lut = create_green_ir_lut()
        thermal_frame = cv2.LUT(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), lut)
    else:
        thermal_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return thermal_frame, enhanced

def apply_digital_zoom(frame, zoom_factor):
    """Applies real digital zoom by cropping and scaling the video feed."""
    if zoom_factor <= 1.0:
        return frame
    
    h, w = frame.shape[:2]
    crop_w = int(w / zoom_factor)
    crop_h = int(h / zoom_factor)

    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

class OpticalFlowFLIRTracker:
    """Lucas-Kanade Optical Flow + Thermal Centroid Hybrid Tracker for ultra-fast objects."""
    def __init__(self):
        self.prev_gray = None
        self.pts = None
        self.bbox = None  # (x, y, w, h)
        
        # LK Optical Flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

    def init(self, gray_frame, bbox):
        x, y, w, h = [int(v) for v in bbox]
        self.bbox = (x, y, w, h)
        self.prev_gray = gray_frame.copy()
        self._extract_features(gray_frame)
        return self.pts is not None and len(self.pts) > 0

    def _extract_features(self, gray_frame):
        x, y, w, h = self.bbox
        img_h, img_w = gray_frame.shape[:2]
        
        # Create mask over current target box
        mask = np.zeros_like(gray_frame)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        mask[y1:y2, x1:x2] = 255

        # Detect corner features inside locked region
        p = cv2.goodFeaturesToTrack(
            gray_frame, 
            mask=mask, 
            maxCorners=35, 
            qualityLevel=0.08, 
            minDistance=3
        )
        if p is not None:
            self.pts = p.reshape(-1, 1, 2)
        else:
            self.pts = None

    def update(self, gray_frame):
        if self.prev_gray is None or self.bbox is None:
            return False, self.bbox

        x, y, w, h = self.bbox
        img_h, img_w = gray_frame.shape[:2]

        # 1. Primary Vector Optical Flow Tracking
        if self.pts is not None and len(self.pts) >= 2:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray_frame, self.pts, None, **self.lk_params
            )

            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = self.pts[st == 1]

                if len(good_new) >= 2:
                    # Calculate median translation motion vector
                    dx = np.median(good_new[:, 0] - good_old[:, 0])
                    dy = np.median(good_new[:, 1] - good_old[:, 1])

                    # Shift bounding box
                    new_x = int(np.clip(x + dx, 0, img_w - w))
                    new_y = int(np.clip(y + dy, 0, img_h - h))
                    self.bbox = (new_x, new_y, w, h)

                    # Update keypoints & re-seed features if low
                    self.pts = good_new.reshape(-1, 1, 2)
                    if len(self.pts) < 12:
                        self._extract_features(gray_frame)

                    self.prev_gray = gray_frame.copy()
                    return True, self.bbox

        # 2. Thermal Centroid / Hotspot Fallback for sudden high-speed jerks
        search_pad = max(w, h)
        sx1, sy1 = max(0, x - search_pad), max(0, y - search_pad)
        sx2, sy2 = min(img_w, x + w + search_pad), min(img_h, y + h + search_pad)
        
        search_roi = gray_frame[sy1:sy2, sx1:sx2]
        if search_roi.size > 0:
            _, thresh = cv2.threshold(search_roi, 180, 255, cv2.THRESH_BINARY)
            M = cv2.moments(thresh)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + sx1
                cy = int(M["m01"] / M["m00"]) + sy1
                
                new_x = int(np.clip(cx - w // 2, 0, img_w - w))
                new_y = int(np.clip(cy - h // 2, 0, img_h - h))
                self.bbox = (new_x, new_y, w, h)
                
                self.prev_gray = gray_frame.copy()
                self._extract_features(gray_frame)
                return True, self.bbox

        self.prev_gray = gray_frame.copy()
        return False, self.bbox

class CalibratedSpeedTracker:
    """Calculates target velocity in Knots with jitter suppression."""
    def __init__(self, pixel_scale_factor=1.4, deadband_px=1.2):
        self.history = {}
        self.scale = pixel_scale_factor
        self.deadband = deadband_px

    def update_and_get_speed(self, target_id, box):
        cx = box[0] + box[2] / 2.0
        cy = box[1] + box[3] / 2.0
        current_time = time.perf_counter()

        if target_id in self.history:
            last_cx, last_cy, last_time, smoothed_speed = self.history[target_id]
            dt = current_time - last_time
            
            if dt > 0.005:
                dist_px = math.hypot(cx - last_cx, cy - last_cy)
                if dist_px < self.deadband:
                    inst_speed = 0.0
                else:
                    inst_speed_px_sec = dist_px / dt
                    inst_speed = inst_speed_px_sec * self.scale
                
                alpha = 0.25 if inst_speed > 0 else 0.45
                new_speed = (alpha * inst_speed) + ((1.0 - alpha) * smoothed_speed)
            else:
                new_speed = smoothed_speed
        else:
            new_speed = 0.0

        self.history[target_id] = (cx, cy, current_time, new_speed)
        
        if new_speed < 2.0:
            return 0
        return int(round(new_speed))

def detect_auto_targets(gray_frame, primary_bbox, min_area=100, max_targets=4):
    """Detects secondary target hotspots on a downscaled frame."""
    small_gray = cv2.resize(gray_frame, (0, 0), fx=0.5, fy=0.5)
    _, thresh = cv2.threshold(small_gray, 205, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    auto_targets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < 8000:
            x, y, w, h = cv2.boundingRect(cnt)
            x, y, w, h = x * 2, y * 2, w * 2, h * 2

            if primary_bbox is not None:
                px, py, pw, ph = primary_bbox
                if abs((x + w/2) - (px + pw/2)) < (w + pw)/2 and abs((y + h/2) - (py + ph/2)) < (h + ph)/2:
                    continue
            auto_targets.append((x, y, w, h))
            if len(auto_targets) >= max_targets:
                break
    return auto_targets

def draw_mfd_reticle(frame, cx, cy, is_locked, primary_bbox, width, height, hud_color):
    """Draws ATFLIR reticle and locking box."""
    thick = max(2, int(width * 0.0018))
    gap = int(width * 0.02)
    arm_len = int(width * 0.065)
    
    cv2.line(frame, (int(cx - gap - arm_len), int(cy)), (int(cx - gap), int(cy)), hud_color, thick)
    cv2.line(frame, (int(cx + gap), int(cy)), (int(cx + gap + arm_len), int(cy)), hud_color, thick)
    cv2.line(frame, (int(cx), int(cy - gap - arm_len)), (int(cx), int(cy - gap)), hud_color, thick)
    cv2.line(frame, (int(cx), int(cy + gap)), (int(cx), int(cy + gap + arm_len)), hud_color, thick)
    
    tick_len = int(height * 0.015)
    cv2.line(frame, (int(cx - gap - arm_len), int(cy - tick_len)), (int(cx - gap - arm_len), int(cy + tick_len)), hud_color, thick)
    cv2.line(frame, (int(cx + gap + arm_len), int(cy - tick_len)), (int(cx + gap + arm_len), int(cy + tick_len)), hud_color, thick)
    cv2.line(frame, (int(cx - tick_len), int(cy - gap - arm_len)), (int(cx + tick_len), int(cy - gap - arm_len)), hud_color, thick)
    cv2.line(frame, (int(cx - tick_len), int(cy + gap + arm_len)), (int(cx + tick_len), int(cy + gap + arm_len)), hud_color, thick)

    if is_locked and primary_bbox is not None:
        p1 = (int(primary_bbox[0]), int(primary_bbox[1]))
        p2 = (int(primary_bbox[0] + primary_bbox[2]), int(primary_bbox[1] + primary_bbox[3]))
        cv2.rectangle(frame, p1, p2, hud_color, thick)
    else:
        center_box_sz = int(width * 0.018)
        cv2.rectangle(frame, (int(cx - center_box_sz), int(cy - center_box_sz)), 
                             (int(cx + center_box_sz), int(cy + center_box_sz)), hud_color, 1)

def render_mfd_hud_overlay(frame, width, height, thermal_mode, zoom_label, polarity, 
                           is_locked, primary_bbox, auto_targets, speed_tracker, font, font_sm):
    """Draws screen-edge OSB readouts and status metrics."""
    hud_color_bgr = (50, 255, 50) if thermal_mode == 'GREEN' else (240, 240, 240)
    hud_color_rgb = (50, 255, 50, 255) if thermal_mode == 'GREEN' else (240, 240, 240, 255)
    
    for i, box in enumerate(auto_targets):
        bx, by, bw, bh = box
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), hud_color_bgr, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    mode_text = "IR-GRN" if thermal_mode == "GREEN" else "IR-B&W"

    # Top Row
    draw.text((int(width * 0.05), int(height * 0.03)), "OPR", font=font, fill=hud_color_rgb)
    draw.text((int(width * 0.14), int(height * 0.03)), f"WFOV\n{zoom_label}", font=font, fill=hud_color_rgb)
    draw.text((int(width * 0.44), int(height * 0.03)), mode_text, font=font, fill=hud_color_rgb)
    
    rtcl_x, rtcl_y = int(width * 0.58), int(height * 0.03)
    draw.rectangle([rtcl_x - 4, rtcl_y - 2, rtcl_x + 48, rtcl_y + 22], outline=hud_color_rgb, width=2)
    draw.text((rtcl_x, rtcl_y), "RTCL", font=font, fill=hud_color_rgb)
    draw.text((int(width * 0.46), int(height * 0.08)), "1° R", font=font, fill=hud_color_rgb)

    soi_x, soi_y = int(width * 0.72), int(height * 0.05)
    draw.polygon([(soi_x, soi_y - 6), (soi_x + 6, soi_y), (soi_x, soi_y + 6), (soi_x - 6, soi_y)], outline=hud_color_rgb, width=2)

    # Coordinates
    coords_x = int(width * 0.65)
    draw.text((coords_x, int(height * 0.09)), "N 36°47.30'", font=font_sm, fill=hud_color_rgb)
    draw.text((coords_x, int(height * 0.12)), "W115°26.99'", font=font_sm, fill=hud_color_rgb)
    draw.text((coords_x, int(height * 0.15)), "ELEV 3472 FT", font=font_sm, fill=hud_color_rgb)
    draw.text((coords_x - int(width*0.06), int(height * 0.18)), "GRID 11S PA 383725", font=font_sm, fill=hud_color_rgb)

    # Left Side
    draw.text((int(width * 0.03), int(height * 0.28)), "ZOOM", font=font, fill=hud_color_rgb)
    draw.text((int(width * 0.03), int(height * 0.32)), zoom_label, font=font, fill=hud_color_rgb)
    draw.text((int(width * 0.03), int(height * 0.48)), "-12°", font=font, fill=hud_color_rgb)
    draw.text((int(width * 0.03), int(height * 0.62)), "FOCS\n0", font=font, fill=hud_color_rgb)

    # Right Side
    draw.text((int(width * 0.90), int(height * 0.48)), "VVSL", font=font, fill=hud_color_rgb)
    draw.text((int(width * 0.88), int(height * 0.68)), "LST\n1688\n1688\nLTD/R", font=font_sm, fill=hud_color_rgb)
    draw.text((int(width * 0.88), int(height * 0.82)), "01", font=font, fill=hud_color_rgb)
    draw.text((int(width * 0.88), int(height * 0.88)), "SETUP", font=font, fill=hud_color_rgb)

    # Bottom Row
    draw.text((int(width * 0.05), int(height * 0.82)), "ADV-\n298\nM 0.71", font=font_sm, fill=hud_color_rgb)
    draw.text((int(width * 0.18), int(height * 0.88)), polarity, font=font, fill=hud_color_rgb)

    alg_x, alg_y = int(width * 0.35), int(height * 0.88)
    draw.rectangle([alg_x - 4, alg_y - 2, alg_x + 38, alg_y + 22], outline=hud_color_rgb, width=2)
    draw.text((alg_x, alg_y), "ALG", font=font, fill=hud_color_rgb)

    draw.text((int(width * 0.48), int(height * 0.88)), "1811", font=font, fill=hud_color_rgb)
    draw.text((int(width * 0.58), int(height * 0.88)), "LST", font=font, fill=hud_color_rgb)
    draw.text((int(width * 0.68), int(height * 0.83)), "23150", font=font, fill=hud_color_rgb)
    draw.text((int(width * 0.68), int(height * 0.88)), "DCLTR", font=font, fill=hud_color_rgb)

    if is_locked and primary_bbox is not None:
        spd = speed_tracker.update_and_get_speed(0, primary_bbox)
        draw.text((int(primary_bbox[0]), max(0, int(primary_bbox[1] - height * 0.03))), 
                  f"SPD: {spd} KTS", font=font_sm, fill=hud_color_rgb)

    for i, box in enumerate(auto_targets):
        bx, by, bw, bh = box
        spd = speed_tracker.update_and_get_speed(i + 1, box)
        draw.text((bx, max(0, by - int(height * 0.025))), f"SPD:{spd}", font=font_sm, fill=hud_color_rgb)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("\n[ERROR] Camera index 0 could not be opened.")
        return

    window_name = 'Tactical MFD Green/BW FLIR System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    ret, frame = cap.read()
    if not ret: return
    height, width = frame.shape[:2]

    tactical_font = get_tactical_font(max(16, int(height * 0.026)))
    tactical_font_sm = get_tactical_font(max(12, int(height * 0.020)))
    
    speed_tracker = CalibratedSpeedTracker(pixel_scale_factor=1.4, deadband_px=1.2)

    target_x, target_y = width // 2, height // 2
    slew_speed = int(width * 0.016)
    
    thermal_mode = "GREEN"
    polarity = "WHT"
    
    zoom_factors = [1.0, 2.0, 4.0]
    zoom_labels = ["Z1.0", "Z2.0", "Z4.0"]
    zoom_idx = 0
    
    tracker = None
    is_locked = False
    primary_bbox = None
    lock_size = int(width * 0.07)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        zoomed_frame = apply_digital_zoom(frame, zoom_factors[zoom_idx])
        display_frame, gray_frame = apply_thermal_filter(zoomed_frame, mode=thermal_mode, polarity=polarity)
        
        # High-Speed Optical Flow Tracker Update
        if is_locked and tracker is not None:
            success, bbox_out = tracker.update(gray_frame)
            if success:
                primary_bbox = [int(v) for v in bbox_out]
                target_x = int(primary_bbox[0] + (primary_bbox[2] / 2))
                target_y = int(primary_bbox[1] + (primary_bbox[3] / 2))
            else:
                is_locked = False
                tracker = None
                primary_bbox = None

        auto_targets = detect_auto_targets(gray_frame, primary_bbox)

        key = cv2.waitKeyEx(1)
        
        if key in [27, ord('q'), ord('Q')]:
            break
            
        # Slew Controls
        elif key in [2490368, 65362, ord('w'), ord('W')]:
            target_y = max(0, target_y - slew_speed)
            is_locked = False; tracker = None
        elif key in [2621440, 65364, ord('s'), ord('S')]:
            target_y = min(height, target_y + slew_speed)
            is_locked = False; tracker = None
        elif key in [2424832, 65361, ord('a'), ord('A')]:
            target_x = max(0, target_x - slew_speed)
            is_locked = False; tracker = None
        elif key in [2555904, 65363, ord('d'), ord('D')]:
            target_x = min(width, target_x + slew_speed)
            is_locked = False; tracker = None
            
        # Mode Controls
        elif key in [ord('t'), ord('T')]:
            thermal_mode = "BW" if thermal_mode == "GREEN" else "GREEN"
        elif key in [ord('p'), ord('P')]:
            polarity = "BLK" if polarity == "WHT" else "WHT"
        elif key in [ord('z'), ord('Z')]:
            zoom_idx = (zoom_idx + 1) % len(zoom_factors)
            
        # Target Lock Key ('L' or Spacebar)
        elif key in [ord('l'), ord('L'), ord(' '), 108, 76]:
            x1 = max(0, min(width - 20, int(target_x - lock_size // 2)))
            y1 = max(0, min(height - 20, int(target_y - lock_size // 2)))
            w = max(10, min(width - x1, lock_size))
            h = max(10, min(height - y1, lock_size))
            primary_bbox = (x1, y1, w, h)

            new_tracker = OpticalFlowFLIRTracker()
            if new_tracker.init(gray_frame, primary_bbox):
                tracker = new_tracker
                is_locked = True
            else:
                tracker = None
                is_locked = False

        hud_color = (50, 255, 50) if thermal_mode == 'GREEN' else (240, 240, 240)
        draw_mfd_reticle(display_frame, target_x, target_y, is_locked, primary_bbox, width, height, hud_color)
        
        display_frame = render_mfd_hud_overlay(
            display_frame, width, height, thermal_mode, zoom_labels[zoom_idx], polarity,
            is_locked, primary_bbox, auto_targets, speed_tracker,
            tactical_font, tactical_font_sm
        )

        cv2.imshow(window_name, display_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 60)
        print(" SCRIPT ERROR DETAILED BELOW")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        input("\nPress ENTER to close...")