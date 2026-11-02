import sys
import time
import json
from collections import deque
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import onnxruntime as ort

from src.qf_neuroshield.models.vit_detector import ViTDeepfakeDetector
from src.qf_neuroshield.training.calibration import apply_calibration
from src.qf_neuroshield.models.anti_hallucination import AntiHallucinationEngine

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("==========================================================================")
    print("🚀 QF-NeuroShield v2.0: Anti-Hallucination Live Edge Detector")
    print("   [ViT + 2D-FFT Spectrum + rPPG Blood-Pulse + Helmholtz Kalman Gate]")
    print("==========================================================================")

    model = ViTDeepfakeDetector()
    ckpt = Path("checkpoints/federated_global_vit.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    snn_onnx_path = Path("checkpoints/onnx/neuromorphic_spiking_head.onnx")
    ort_sess = ort.InferenceSession(str(snn_onnx_path), providers=["CPUExecutionProvider"]) if snn_onnx_path.exists() else None

    calib_path = Path("checkpoints/calibration_params.json")
    calib_params = json.loads(calib_path.read_text()).get("federated", {"optimal_threshold": 0.67, "scale": 6.0}) if calib_path.exists() else {"optimal_threshold": 0.67, "scale": 6.0}
    # Use moderate scale to prevent extreme sigmoid saturation on unseen camera angles
    calib_params["scale"] = min(calib_params.get("scale", 6.0), 6.0)

    stabilizer = AntiHallucinationEngine(buffer_len=30)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    face_detector = None
    classifier_cls = getattr(cv2, "CascadeClassifier", None) or getattr(getattr(cv2, "objdetect", None), "CascadeClassifier", None)
    if classifier_cls and hasattr(cv2, "data"):
        det = classifier_cls(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not det.empty():
            face_detector = det

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access Mac webcam.")
        return

    frame_buffer = deque(maxlen=4)
    frame_count = 0
    current_label = "CALIBRATING BIO-PULSE..."
    current_conf = 0.0
    onnx_latency_ms = 0.0
    box_color = (255, 200, 0)
    metrics_info = {"spectral_anomaly": 0.0, "rppg_liveness": 0.8, "ood_uncertainty": 0.2, "motion_shift": 0.0}

    while True:
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fx1, fy1, fx2, fy2 = (w - min(h, w)//2)//2, (h - min(h, w)//2)//2, (w + min(h, w)//2)//2, (h + min(h, w)//2)//2
        if face_detector is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                fx1, fy1, fx2, fy2 = max(0, x-15), max(0, y-15), min(w, x+fw+15), min(h, y+fh+15)

        face_crop = frame_rgb[fy1:fy2, fx1:fx2]
        if face_crop.size > 0:
            pil_img = Image.fromarray(face_crop).resize((224, 224), Image.Resampling.BILINEAR)
            frame_buffer.append(transform(pil_img))

        if len(frame_buffer) == 4 and frame_count % 4 == 0 and face_crop.size > 0:
            clip_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)
            with torch.no_grad():
                feats = model.extract_video_features(clip_tensor)
                logits_t = model.classifier(feats)[0]
                logits_np = logits_t.cpu().numpy()
                raw_fake_prob = float(torch.softmax(logits_t, dim=0)[1].item())

                if ort_sess is not None:
                    t_onnx = time.perf_counter()
                    _ = ort_sess.run(None, {"video_features": feats.cpu().numpy()})
                    onnx_latency_ms = (time.perf_counter() - t_onnx) * 1000.0

            cal_vit_prob = apply_calibration(raw_fake_prob, calib_params)
            metrics_info = stabilizer.fuse_and_stabilize(logits_np, cal_vit_prob, face_crop, feats[0].cpu().numpy())
            stab_fake_prob = metrics_info["stabilized_fake_prob"]

            if stab_fake_prob >= 0.55:
                current_label = "FAKE / MANIPULATED"
                current_conf = stab_fake_prob * 100.0
                box_color = (0, 0, 255)
            else:
                current_label = "REAL / AUTHENTIC"
                current_conf = (1.0 - stab_fake_prob) * 100.0
                box_color = (0, 255, 100)

        frame_count += 1
        fps = 1.0 / max(time.perf_counter() - t_start, 1e-4)

        # Draw Bounding Box & Scientific Telemetry HUD
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), box_color, 3)
        cv2.rectangle(frame, (0, 0), (w, 115), (15, 23, 42), -1)

        cv2.putText(frame, f"QF-NeuroShield v2 | {current_label} ({current_conf:.1f}%)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, box_color, 2)
        cv2.putText(frame, f"rPPG Bio-Pulse: {metrics_info['rppg_liveness']*100:.0f}% | 2D-FFT Spectral Anomaly: {metrics_info['spectral_anomaly']*100:.0f}% | OOD Gate: {metrics_info['ood_uncertainty']:.2f}",
                    (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 220), 1)
        cv2.putText(frame, f"Kalman Pose-Stabilized | SNN ONNX: {onnx_latency_ms:.3f} ms | FPS: {fps:.1f} | Press 'q' to quit",
                    (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 230, 255), 1)

        cv2.imshow("QF-NeuroShield Live Edge Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
