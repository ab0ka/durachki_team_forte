"""MediaPipe Pose (Apache-2.0) — opt-in бэкенд. Выход ЯВНО перекладывается из 33 landmark-ов
MediaPipe BlazePose в порядок COCO-17 (MP33_TO_COCO17); прочие 16 точек MediaPipe отбрасываются.
visibility MediaPipe — не то же, что score RTMPose: умножается на mediapipe.visibility_scale.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import PoseBackendError

# COCO-17 индекс → индекс landmark-а MediaPipe Pose (33 точки, BlazePose topology)
MP33_TO_COCO17: tuple[int, ...] = (
    0,   # 0 nose            ← 0 NOSE
    2,   # 1 left_eye        ← 2 LEFT_EYE
    5,   # 2 right_eye       ← 5 RIGHT_EYE
    7,   # 3 left_ear        ← 7 LEFT_EAR
    8,   # 4 right_ear       ← 8 RIGHT_EAR
    11,  # 5 left_shoulder   ← 11 LEFT_SHOULDER
    12,  # 6 right_shoulder  ← 12 RIGHT_SHOULDER
    13,  # 7 left_elbow      ← 13 LEFT_ELBOW
    14,  # 8 right_elbow     ← 14 RIGHT_ELBOW
    15,  # 9 left_wrist      ← 15 LEFT_WRIST
    16,  # 10 right_wrist    ← 16 RIGHT_WRIST
    23,  # 11 left_hip       ← 23 LEFT_HIP
    24,  # 12 right_hip      ← 24 RIGHT_HIP
    25,  # 13 left_knee      ← 25 LEFT_KNEE
    26,  # 14 right_knee     ← 26 RIGHT_KNEE
    27,  # 15 left_ankle     ← 27 LEFT_ANKLE
    28,  # 16 right_ankle    ← 28 RIGHT_ANKLE
)


def mp33_to_coco17(landmarks_xyv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(33, 3) [x, y, visibility] → (17, 2) xy, (17,) score — в порядке COCO-17."""
    lm = np.asarray(landmarks_xyv, dtype=np.float32)
    if lm.shape != (33, 3):
        raise ValueError(f"ожидались 33 landmark-а MediaPipe, получено {lm.shape}")
    idx = np.array(MP33_TO_COCO17)
    return lm[idx, :2].copy(), np.clip(lm[idx, 2], 0.0, 1.0)


class MediaPipePose:
    name = "mediapipe"

    def __init__(self, cfg: dict[str, Any]) -> None:
        try:
            import mediapipe as mp  # noqa: F401
            self._mp_pose = mp.solutions.pose
        except Exception as exc:  # noqa: BLE001
            raise PoseBackendError(f"mediapipe недоступен: {exc}") from exc
        mc = cfg.get("backend", {}).get("mediapipe", {})
        self.pose = self._mp_pose.Pose(static_image_mode=True,
                                       model_complexity=int(mc.get("model_complexity", 1)),
                                       min_detection_confidence=float(mc.get("min_detection_confidence", 0.3)))
        self.pad = float(cfg.get("backend", {}).get("rtmpose", {}).get("padding", 1.25))
        self.vis_scale = float(mc.get("visibility_scale", 1.0))
        self.batch_size = 1

    def infer(self, items: Sequence[tuple[np.ndarray, np.ndarray]], meta: Any = None) -> list[tuple[np.ndarray, np.ndarray]]:
        import cv2

        out = []
        for frame, box in items:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = (float(v) for v in box)
            cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * self.pad, (y2 - y1) * self.pad
            ox, oy = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
            ex, ey = int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2))
            crop = frame[oy:ey, ox:ex]
            kp = np.zeros((17, 2), np.float32)
            sc = np.zeros(17, np.float32)
            if crop.size:
                res = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    lm = np.array([[p.x, p.y, p.visibility] for p in res.pose_landmarks.landmark], np.float32)
                    xy, sc = mp33_to_coco17(lm)
                    kp = np.stack([ox + xy[:, 0] * crop.shape[1], oy + xy[:, 1] * crop.shape[0]], axis=1)
                    sc = np.clip(sc * self.vis_scale, 0.0, 1.0)
            out.append((kp.astype(np.float32), sc.astype(np.float32)))
        return out
