from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# Drawing settings
ERASE_COLOR = (0, 0, 0)
BRUSH_THICKNESS = 7
ERASER_THICKNESS = 40
SMOOTHING_ALPHA = 0.3
GESTURE_HYSTERESIS_FRAMES = 3
TRANSFORM_HYSTERESIS_FRAMES = 3
COLOR_LOCK_HYSTERESIS_FRAMES = 2
MIN_STROKE_STEP = 2
MAX_SEGMENT_LENGTH = 14
FINGER_UP_MARGIN = 0.015
RGB_HUE_STEP = 3
LASER_HALO_THICKNESS_DELTA = 5
LASER_HALO_BRIGHTNESS = 0.55
GLOW_SIGMA_SOFT = 10
GLOW_SIGMA_WIDE = 24
MIN_TRANSFORM_DISTANCE = 35.0
MIN_SCALE = 0.35
MAX_SCALE = 3.2
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_NAME = "Air Drawing Studio"
EXPORTS_DIR = Path("exports")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


class AirDrawer:
    def __init__(self) -> None:
        self.canvas: Optional[np.ndarray] = None
        self.prev_points: Dict[str, Tuple[int, int]] = {}
        self.smoothed_points: Dict[str, Tuple[float, float]] = {}
        self.hand_hues: Dict[str, int] = {"Left": 15, "Right": 100}

        self.active_draw_hands = 0
        self.active_erase_hands = 0

        self.is_transforming = False
        self.transform_on_frames = 0
        self.transform_off_frames = 0
        self.transform_base_canvas: Optional[np.ndarray] = None
        self.transform_anchor_center: Optional[Tuple[float, float]] = None
        self.transform_anchor_distance = 1.0
        self.transform_anchor_angle = 0.0
        self.transform_live_center: Optional[Tuple[float, float]] = None
        self.transform_live_scale = 1.0
        self.transform_live_angle_deg = 0.0
        self.transform_live_points: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

        self.color_locked = False
        self.color_lock_on_frames = 0
        self.color_lock_off_frames = 0
        self.locked_color: Optional[Tuple[int, int, int]] = None

    def _update_debounced_state(
        self,
        candidate_state: bool,
        current_state: bool,
        on_frames: int,
        off_frames: int,
        hysteresis_frames: int,
    ) -> Tuple[bool, int, int]:
        if candidate_state:
            on_frames += 1
            off_frames = 0
            if not current_state and on_frames >= hysteresis_frames:
                current_state = True
        else:
            off_frames += 1
            on_frames = 0
            if current_state and off_frames >= hysteresis_frames:
                current_state = False

        return current_state, on_frames, off_frames

    def _smooth(self, hand_id: str, point: Tuple[int, int]) -> Tuple[int, int]:
        current = self.smoothed_points.get(hand_id)
        if current is None:
            self.smoothed_points[hand_id] = (float(point[0]), float(point[1]))
        else:
            sx, sy = current
            self.smoothed_points[hand_id] = (
                sx * (1.0 - SMOOTHING_ALPHA) + point[0] * SMOOTHING_ALPHA,
                sy * (1.0 - SMOOTHING_ALPHA) + point[1] * SMOOTHING_ALPHA,
            )
        smoothed = self.smoothed_points[hand_id]
        return int(smoothed[0]), int(smoothed[1])

    def _next_rgb_color(self, hand_id: str) -> Tuple[int, int, int]:
        base_hue = self.hand_hues.get(hand_id, 45)
        next_hue = (base_hue + RGB_HUE_STEP) % 180
        self.hand_hues[hand_id] = next_hue
        hsv_color = np.array([[[next_hue, 255, 255]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])

    def _get_draw_color(self, hand_id: str) -> Tuple[int, int, int]:
        if self.color_locked:
            if self.locked_color is None:
                self.locked_color = self._next_rgb_color(hand_id)
            return self.locked_color

        color = self._next_rgb_color(hand_id)
        self.locked_color = color
        return color

    def update_color_lock(self, color_lock_candidate: bool) -> None:
        self.color_locked, self.color_lock_on_frames, self.color_lock_off_frames = self._update_debounced_state(
            candidate_state=color_lock_candidate,
            current_state=self.color_locked,
            on_frames=self.color_lock_on_frames,
            off_frames=self.color_lock_off_frames,
            hysteresis_frames=COLOR_LOCK_HYSTERESIS_FRAMES,
        )

    def _draw_interpolated_stroke(
        self,
        hand_id: str,
        start: Tuple[int, int],
        end: Tuple[int, int],
        thickness: int,
    ) -> None:
        if self.canvas is None:
            return

        distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        segments = max(1, int(distance // MAX_SEGMENT_LENGTH) + 1)

        prev = start
        for idx in range(1, segments + 1):
            t = idx / segments
            point = (
                int(start[0] + (end[0] - start[0]) * t),
                int(start[1] + (end[1] - start[1]) * t),
            )
            color = self._get_draw_color(hand_id)
            halo_color = (
                int(color[0] * LASER_HALO_BRIGHTNESS),
                int(color[1] * LASER_HALO_BRIGHTNESS),
                int(color[2] * LASER_HALO_BRIGHTNESS),
            )
            cv2.line(
                self.canvas,
                prev,
                point,
                halo_color,
                thickness + LASER_HALO_THICKNESS_DELTA,
                cv2.LINE_AA,
            )
            cv2.line(self.canvas, prev, point, color, thickness, cv2.LINE_AA)
            prev = point

    @staticmethod
    def _midpoint(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[float, float]:
        return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

    @staticmethod
    def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    @staticmethod
    def _angle(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(np.arctan2(b[1] - a[1], b[0] - a[0]))

    def _reset_hand_tracks(self) -> None:
        self.prev_points.clear()
        self.smoothed_points.clear()

    def _start_transform(self, hand_a: dict, hand_b: dict) -> None:
        if self.canvas is None:
            return

        index_a = hand_a["index_tip"]
        index_b = hand_b["index_tip"]
        distance = max(self._distance(index_a, index_b), MIN_TRANSFORM_DISTANCE)

        self.transform_base_canvas = self.canvas.copy()
        self.transform_anchor_center = self._midpoint(index_a, index_b)
        self.transform_anchor_distance = distance
        self.transform_anchor_angle = self._angle(index_a, index_b)
        self.transform_live_center = self.transform_anchor_center
        self.transform_live_scale = 1.0
        self.transform_live_angle_deg = 0.0
        self.transform_live_points = (index_a, index_b)
        self._reset_hand_tracks()

    def _apply_transform(self, hand_a: dict, hand_b: dict) -> None:
        if (
            self.canvas is None
            or self.transform_base_canvas is None
            or self.transform_anchor_center is None
        ):
            return

        index_a = hand_a["index_tip"]
        index_b = hand_b["index_tip"]
        current_center = self._midpoint(index_a, index_b)
        current_distance = max(self._distance(index_a, index_b), MIN_TRANSFORM_DISTANCE)
        current_angle = self._angle(index_a, index_b)

        scale = float(np.clip(current_distance / self.transform_anchor_distance, MIN_SCALE, MAX_SCALE))
        angle_delta_deg = float(np.degrees(current_angle - self.transform_anchor_angle))
        tx = current_center[0] - self.transform_anchor_center[0]
        ty = current_center[1] - self.transform_anchor_center[1]

        self.transform_live_center = current_center
        self.transform_live_scale = scale
        self.transform_live_angle_deg = angle_delta_deg
        self.transform_live_points = (index_a, index_b)

        transform = cv2.getRotationMatrix2D(self.transform_anchor_center, angle_delta_deg, scale)
        transform[0, 2] += tx
        transform[1, 2] += ty

        self.canvas = cv2.warpAffine(
            self.transform_base_canvas,
            transform,
            (self.transform_base_canvas.shape[1], self.transform_base_canvas.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def update_transform(self, hands_data: List[dict]) -> bool:
        transform_candidate = (
            len(hands_data) >= 2
            and all(hand["index_up"] and hand["middle_up"] for hand in hands_data[:2])
        )

        self.is_transforming, self.transform_on_frames, self.transform_off_frames = self._update_debounced_state(
            candidate_state=transform_candidate,
            current_state=self.is_transforming,
            on_frames=self.transform_on_frames,
            off_frames=self.transform_off_frames,
            hysteresis_frames=TRANSFORM_HYSTERESIS_FRAMES,
        )

        if not self.is_transforming:
            self.transform_base_canvas = None
            self.transform_anchor_center = None
            self.transform_live_center = None
            self.transform_live_scale = 1.0
            self.transform_live_angle_deg = 0.0
            self.transform_live_points = None
            return False

        if len(hands_data) < 2:
            return True

        if self.transform_base_canvas is None or self.transform_anchor_center is None:
            self._start_transform(hands_data[0], hands_data[1])
        self._apply_transform(hands_data[0], hands_data[1])
        return True

    def update_canvas(
        self,
        frame_shape: Tuple[int, int, int],
        hands_data: List[dict],
    ) -> None:
        if self.canvas is None:
            self.canvas = np.zeros(frame_shape, dtype=np.uint8)

        if self.is_transforming:
            return

        self.active_draw_hands = 0
        self.active_erase_hands = 0

        color_lock_candidate = any(hand["ring_up"] for hand in hands_data)
        self.update_color_lock(color_lock_candidate)

        visible_hands = set()
        for hand in hands_data:
            hand_id = hand["hand_id"]
            visible_hands.add(hand_id)

            draw_candidate = hand["draw_candidate"]
            erase_candidate = hand["erase_candidate"]
            if not (draw_candidate or erase_candidate):
                self.prev_points.pop(hand_id, None)
                self.smoothed_points.pop(hand_id, None)
                continue

            if draw_candidate:
                self.active_draw_hands += 1
            if erase_candidate:
                self.active_erase_hands += 1

            draw_point = self._smooth(hand_id, hand["index_tip"])
            prev_point = self.prev_points.get(hand_id)

            if prev_point is None:
                self.prev_points[hand_id] = draw_point
                continue

            thickness = ERASER_THICKNESS if erase_candidate else BRUSH_THICKNESS
            if erase_candidate:
                cv2.line(self.canvas, prev_point, draw_point, ERASE_COLOR, thickness)
            else:
                distance = float(
                    np.hypot(
                        draw_point[0] - prev_point[0],
                        draw_point[1] - prev_point[1],
                    )
                )
                if distance >= MIN_STROKE_STEP:
                    self._draw_interpolated_stroke(hand_id, prev_point, draw_point, thickness)

            self.prev_points[hand_id] = draw_point

        stale_ids = [hid for hid in self.prev_points.keys() if hid not in visible_hands]
        for hid in stale_ids:
            self.prev_points.pop(hid, None)
            self.smoothed_points.pop(hid, None)

    def clear_canvas(self) -> None:
        if self.canvas is not None:
            self.canvas[:] = 0

    def save_canvas(self, output_dir: Path) -> Optional[Path]:
        if self.canvas is None:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"air_drawing_{time.strftime('%Y%m%d_%H%M%S')}.png"
        file_path = output_dir / file_name
        saved = cv2.imwrite(str(file_path), self.canvas)
        return file_path if saved else None

    def render_overlay(
        self,
        frame: np.ndarray,
        fps: float,
        tracking_confidence: float,
        status_text: str = "",
    ) -> np.ndarray:
        if self.canvas is None:
            return frame

        glow_soft = cv2.GaussianBlur(self.canvas, (0, 0), GLOW_SIGMA_SOFT)
        glow_wide = cv2.GaussianBlur(self.canvas, (0, 0), GLOW_SIGMA_WIDE)

        output = cv2.addWeighted(frame, 0.62, glow_wide, 0.48, 0)
        output = cv2.addWeighted(output, 1.0, glow_soft, 0.82, 0)
        output = cv2.addWeighted(output, 1.0, self.canvas, 1.0, 0)

        panel = output.copy()
        cv2.rectangle(panel, (8, 8), (390, 86), (18, 18, 24), -1)
        output = cv2.addWeighted(panel, 0.30, output, 0.70, 0)

        mode_text = "TRANSFORM" if self.is_transforming else "DRAW"
        color_lock_text = "LOCKED" if self.color_locked else "AUTO"

        cv2.putText(
            output,
            "AIR DRAWING STUDIO",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (130, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"Mode {mode_text} | DrawHands {self.active_draw_hands} | EraseHands {self.active_erase_hands}",
            (16, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 245, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"Color {color_lock_text} | FPS {fps:.1f} | Track {tracking_confidence:.2f}",
            (16, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (210, 255, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            "Each hand: index draw, index+middle erase | 2 hands index+middle: move/rotate/zoom | Ring: color lock | C: clear | S: save | F: fullscreen",
            (16, output.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 220, 180),
            1,
            cv2.LINE_AA,
        )

        if status_text:
            status_panel = output.copy()
            cv2.rectangle(status_panel, (16, output.shape[0] - 72), (540, output.shape[0] - 38), (16, 28, 20), -1)
            output = cv2.addWeighted(status_panel, 0.42, output, 0.58, 0)
            cv2.putText(
                output,
                status_text,
                (24, output.shape[0] - 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (180, 255, 210),
                1,
                cv2.LINE_AA,
            )

        if self.is_transforming and self.transform_live_points and self.transform_live_center:
            hand_a, hand_b = self.transform_live_points
            cx, cy = int(self.transform_live_center[0]), int(self.transform_live_center[1])

            cv2.line(output, hand_a, hand_b, (255, 180, 70), 2, cv2.LINE_AA)
            cv2.circle(output, hand_a, 10, (90, 255, 220), 2, cv2.LINE_AA)
            cv2.circle(output, hand_b, 10, (90, 255, 220), 2, cv2.LINE_AA)
            cv2.drawMarker(
                output,
                (cx, cy),
                (255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

            gizmo_panel = output.copy()
            cv2.rectangle(gizmo_panel, (output.shape[1] - 270, 8), (output.shape[1] - 8, 64), (14, 20, 30), -1)
            output = cv2.addWeighted(gizmo_panel, 0.35, output, 0.65, 0)
            cv2.putText(
                output,
                f"Transform Scale {self.transform_live_scale:.2f}x",
                (output.shape[1] - 262, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (190, 250, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                f"Rotate {self.transform_live_angle_deg:+.1f} deg",
                (output.shape[1] - 262, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (190, 250, 255),
                1,
                cv2.LINE_AA,
            )

        return output


def ensure_hand_landmarker_model() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print("Downloading hand model (one-time setup)...")
        urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def finger_is_up(lm, tip_idx: int, pip_idx: int) -> bool:
    return lm[tip_idx].y < (lm[pip_idx].y - FINGER_UP_MARGIN)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permission and connection.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    model_path = ensure_hand_landmarker_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        num_hands=2,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
        running_mode=vision.RunningMode.VIDEO,
    )

    drawer = AirDrawer()
    frame_index = 0
    prev_frame_time = time.perf_counter()
    smoothed_fps = 0.0
    fullscreen_mode = False
    read_failures = 0
    status_message = ""
    status_until = 0.0

    with vision.HandLandmarker.create_from_options(options) as hand_landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                read_failures += 1
                fail_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
                cv2.putText(
                    fail_frame,
                    "Camera signal lost. Reconnecting...",
                    (40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (120, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    fail_frame,
                    "Check camera permission or close other apps using the webcam.",
                    (40, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (190, 240, 220),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    fail_frame,
                    f"Retry {read_failures} | Press Q or Esc to quit",
                    (40, 146),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (200, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, fail_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            read_failures = 0

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = frame_index * 33
            frame_index += 1
            result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            current_time = time.perf_counter()
            dt = max(current_time - prev_frame_time, 1e-6)
            prev_frame_time = current_time
            instant_fps = 1.0 / dt
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else (smoothed_fps * 0.9 + instant_fps * 0.1)

            tracking_confidence = 0.0
            if result.handedness:
                hand_scores = [float(hand[0].score) for hand in result.handedness if hand]
                if hand_scores:
                    tracking_confidence = sum(hand_scores) / len(hand_scores)

            if result.hand_landmarks:
                h, w, _ = frame.shape
                hands_data = []

                for i, lm in enumerate(result.hand_landmarks[:2]):
                    hand_id = f"Hand{i}"
                    if result.handedness and i < len(result.handedness) and result.handedness[i]:
                        hand_id = str(result.handedness[i][0].category_name)

                    index_tip = (int(lm[8].x * w), int(lm[8].y * h))

                    index_up = finger_is_up(lm, 8, 6)
                    middle_up = finger_is_up(lm, 12, 10)
                    ring_up = finger_is_up(lm, 16, 14)
                    draw_candidate = index_up and not middle_up
                    erase_candidate = index_up and middle_up

                    hands_data.append(
                        {
                            "hand_id": hand_id,
                            "index_tip": index_tip,
                            "index_up": index_up,
                            "middle_up": middle_up,
                            "ring_up": ring_up,
                            "draw_candidate": draw_candidate,
                            "erase_candidate": erase_candidate,
                            "landmarks": lm,
                        }
                    )

                drawer.update_transform(hands_data)
                drawer.update_canvas(
                    frame_shape=frame.shape,
                    hands_data=hands_data,
                )

                for hand_data in hands_data:
                    for landmark in hand_data["landmarks"]:
                        point = (int(landmark.x * w), int(landmark.y * h))
                        cv2.circle(frame, point, 2, (0, 200, 255), -1)
                    cv2.circle(frame, hand_data["index_tip"], 8, (0, 255, 255), -1)
            else:
                drawer.update_transform([])
                drawer.active_draw_hands = 0
                drawer.active_erase_hands = 0

            current_time = time.perf_counter()
            if current_time >= status_until:
                status_message = ""

            output = drawer.render_overlay(
                frame,
                smoothed_fps,
                tracking_confidence,
                status_text=status_message,
            )
            cv2.imshow(WINDOW_NAME, output)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                drawer.clear_canvas()
                status_message = "Canvas cleared"
                status_until = time.perf_counter() + 1.5
            if key == ord("s"):
                saved_path = drawer.save_canvas(EXPORTS_DIR)
                if saved_path is not None:
                    status_message = f"Saved: {saved_path.as_posix()}"
                else:
                    status_message = "Save failed"
                status_until = time.perf_counter() + 2.0
            if key == ord("f"):
                fullscreen_mode = not fullscreen_mode
                window_mode = cv2.WINDOW_FULLSCREEN if fullscreen_mode else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, window_mode)
                if not fullscreen_mode:
                    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
                status_message = "Fullscreen on" if fullscreen_mode else "Fullscreen off"
                status_until = time.perf_counter() + 1.5

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
