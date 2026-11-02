# src/qf_neuroshield/data/frame_sampler.py
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List

class VideoFrameSampler:
    def __init__(self, num_frames: int = 8, image_size: int = 224, crop_face: bool = True):
        self.num_frames = num_frames
        self.image_size = image_size
        self.crop_face = crop_face
        self.face_detector = None

        if self.crop_face:
            # Check for CascadeClassifier in cv2 or cv2.objdetect submodule
            classifier_cls = getattr(cv2, "CascadeClassifier", None)
            if classifier_cls is None and hasattr(cv2, "objdetect"):
                classifier_cls = getattr(cv2.objdetect, "CascadeClassifier", None)

            if classifier_cls is not None and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                detector = classifier_cls(cascade_path)
                if not detector.empty():
                    self.face_detector = detector

    def _extract_face(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w, _ = frame_rgb.shape

        if self.face_detector is not None:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                # Select the largest detected face
                x, y, fw, fh = max(faces, key=lambda rect: rect[2] * rect[3])
                margin_x = int(fw * 0.15)
                margin_y = int(fh * 0.15)
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(w, x + fw + margin_x)
                y2 = min(h, y + fh + margin_y)
                return frame_rgb[y1:y2, x1:x2]

        # Robust Fallback: square center crop
        min_dim = min(h, w)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        return frame_rgb[start_y:start_y + min_dim, start_x:start_x + min_dim]

    def sample(self, video_path: str | Path) -> List[Image.Image]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video contains 0 frames: {video_path}")

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        sampled_images = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret:
                if sampled_images:
                    sampled_images.append(sampled_images[-1].copy())
                else:
                    sampled_images.append(Image.new("RGB", (self.image_size, self.image_size)))
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            cropped_rgb = self._extract_face(frame_rgb)
            
            pil_img = Image.fromarray(cropped_rgb).resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            sampled_images.append(pil_img)

        cap.release()
        return sampled_images