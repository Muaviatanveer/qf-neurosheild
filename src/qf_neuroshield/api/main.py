import io
import time
import base64
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torchvision import transforms
import onnxruntime as ort
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse

from src.qf_neuroshield.data.frame_sampler import VideoFrameSampler
from src.qf_neuroshield.models.vit_detector import ViTDeepfakeDetector
from src.qf_neuroshield.models.sw_cross_vit import SWCrossViTDetector
from src.qf_neuroshield.models.quantum_head import HybridQuantumDeepfakeDetector
from src.qf_neuroshield.models.spiking_head import SpikingDeepfakeDetector
from src.qf_neuroshield.training.calibration import apply_calibration
from src.qf_neuroshield.models.anti_hallucination import AntiHallucinationEngine

app = FastAPI(title="VERITAS — Scientific Intelligence Workstation", version="1.2.0")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
SAMPLER = VideoFrameSampler(num_frames=8, image_size=224, crop_face=True)
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

MODELS: Dict[str, torch.nn.Module] = {}
ANTI_HALLUCINATOR = AntiHallucinationEngine(buffer_len=20)

WORKSTATION_STATE: Dict[str, Any] = {
    "total_analyses": 0,
    "authentic_count": 0,
    "suspected_count": 0,
    "inconclusive_count": 0,
    "active_experiment": "E2/E3: Federated ViT + Control-Plane Audit",
    "events": []
}

def log_event(event_code: str, detail: str, duration_ms: float = 0.0):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    entry = {
        "timestamp": ts,
        "code": event_code,
        "detail": detail,
        "duration_ms": round(duration_ms, 3)
    }
    WORKSTATION_STATE["events"].insert(0, entry)
    WORKSTATION_STATE["events"] = WORKSTATION_STATE["events"][:40]

log_event("WORKSTATION_INITIALIZED", f"Compute device bound to {str(DEVICE).upper()}", 0.12)

def np_rgb_to_base64(rgb_arr: np.ndarray) -> str:
    pil_im = Image.fromarray(np.clip(rgb_arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    pil_im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def get_calibration_params(variant: str) -> Dict[str, float]:
    calib_path = Path("checkpoints/calibration_params.json")
    if calib_path.exists():
        data = json.loads(calib_path.read_text())
        if variant in data:
            return data[variant]
    return {"optimal_threshold": 0.6732, "scale": 6.0}

def get_model(variant: str):
    if variant not in MODELS:
        if variant == "image_140k":
            if Path("checkpoints/sw_cross_vit_140k.pt").exists():
                m = SWCrossViTDetector(unfreeze_top_layers=2)
                m.load_state_dict(torch.load("checkpoints/sw_cross_vit_140k.pt", map_location=DEVICE))
            else:
                m = ViTDeepfakeDetector()
                m.load_state_dict(torch.load("checkpoints/image_140k_vit.pt", map_location=DEVICE))
        elif variant == "federated":
            m = ViTDeepfakeDetector()
            ckpt = Path("checkpoints/image_140k_vit.pt") if Path("checkpoints/image_140k_vit.pt").exists() else Path("checkpoints/federated_global_vit.pt")
            if ckpt.exists():
                m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        elif variant == "quantum":
            m = HybridQuantumDeepfakeDetector(num_qubits=4)
            ckpt = Path("checkpoints/quantum_vit.pt")
            if ckpt.exists():
                m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        elif variant == "spiking":
            m = SpikingDeepfakeDetector(num_steps=10)
            ckpt = Path("checkpoints/spiking_vit.pt")
            if ckpt.exists():
                m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        else:
            m = ViTDeepfakeDetector()
            ckpt = Path("checkpoints/baseline_vit.pt")
            if ckpt.exists():
                m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        m.to(DEVICE).eval()
        MODELS[variant] = m
    return MODELS[variant]

def compute_full_scientific_diagnostics(face_rgb: np.ndarray, features_768: np.ndarray, logits_2d: np.ndarray, is_video: bool, cutoff_r: int = 24):
    crop_224 = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop_224, cv2.COLOR_RGB2GRAY).astype(np.float32)

    hann = np.outer(np.hanning(224), np.hanning(224))
    fft2 = np.fft.fftshift(np.fft.fft2((gray - np.mean(gray)) * hann))
    power_2d = np.abs(fft2) ** 2
    log_mag = np.log1p(np.abs(fft2))
    mag_norm = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fft_rgb = np.stack([mag_norm]*3, axis=-1)

    y, x = np.indices((224, 224))
    r = np.sqrt((x - 112) ** 2 + (y - 112) ** 2).astype(np.int32)
    max_r = 100
    radial_sum = np.bincount(r.ravel(), weights=power_2d.ravel())[:max_r]
    radial_cnt = np.maximum(np.bincount(r.ravel())[:max_r], 1)
    
    # Calculate exact unscaled log10 power spectral density
    raw_rapsd = radial_sum / radial_cnt
    log_rapsd = np.log10(raw_rapsd + 1e-4)
    rapsd_curve = [round(float(v), 3) for v in log_rapsd[2:66]]

    # Linear least-squares fit for 1/f^alpha natural power-law baseline
    freq_indices = np.arange(2, 66, dtype=np.float64)
    log_f = np.log10(freq_indices)
    valid_mask = np.isfinite(rapsd_curve)
    slope, intercept = np.polyfit(log_f[valid_mask], np.array(rapsd_curve)[valid_mask], 1)
    ideal_fit_curve = [round(float(slope * np.log10(f) + intercept), 3) for f in freq_indices]

    hp_mask = (r >= cutoff_r).astype(np.float32)
    filtered_fft = fft2 * hp_mask
    recon = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))
    recon_norm = cv2.normalize(recon, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    recon_rgb = np.stack([recon_norm]*3, axis=-1)

    srm_res = np.abs(cv2.filter2D(gray, cv2.CV_32F, ANTI_HALLUCINATOR.srm_kernel))
    srm_norm = np.clip(srm_res * 16.0, 0, 255).astype(np.uint8)
    srm_rgb = np.stack([srm_norm]*3, axis=-1)

    temp = 1.25
    scaled_l = logits_2d / temp
    probs_q = np.exp(scaled_l - np.max(scaled_l))
    probs_q /= np.sum(probs_q)
    energies = -logits_2d
    expected_energy = float(np.sum(probs_q * energies))
    shannon_entropy = float(-np.sum(probs_q * np.log(probs_q + 1e-12)))
    helmholtz_free_energy = float(expected_energy - temp * shannon_entropy)

    l2_norm = float(np.linalg.norm(features_768))
    unit_z = features_768 / (l2_norm + 1e-8)
    if ANTI_HALLUCINATOR.neutral_feature_anchor is None:
        ANTI_HALLUCINATOR.neutral_feature_anchor = unit_z
    cos_sim = float(np.dot(unit_z, ANTI_HALLUCINATOR.neutral_feature_anchor))
    manifold_proj_32 = [round(float(v), 4) for v in features_768.reshape(32, 24).mean(axis=1)]

    snn_model = get_model("spiking")
    feat_t = torch.tensor(features_768, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    mem1 = snn_model.lif1.init_leaky()
    mem2 = snn_model.lif2.init_leaky()
    v_trace_class0, v_trace_class1, spike_counts_hidden = [], [], []
    with torch.no_grad():
        for _ in range(snn_model.num_steps):
            cur1 = snn_model.fc1(feat_t)
            spk1, mem1 = snn_model.lif1(cur1, mem1)
            cur2 = snn_model.fc2(spk1)
            spk2, mem2 = snn_model.lif2(cur2, mem2)
            v_trace_class0.append(round(float(mem2[0, 0].item()), 4))
            v_trace_class1.append(round(float(mem2[0, 1].item()), 4))
            spike_counts_hidden.append(int(spk1.sum().item()))

    q_model = get_model("quantum")
    with torch.no_grad():
        q_angles = q_model.projection(feat_t).cpu().numpy()[0]
        q_exp = float(q_model.qnn(torch.tensor(q_angles).unsqueeze(0))[0, 0].item())

    return {
        "visuals": {
            "crop": np_rgb_to_base64(crop_224),
            "fft": np_rgb_to_base64(fft_rgb),
            "srm": np_rgb_to_base64(srm_rgb),
            "recon": np_rgb_to_base64(recon_rgb)
        },
        "spectral_rapsd": rapsd_curve,
        "ideal_rapsd_fit": ideal_fit_curve,
        "spectral_slope_alpha": round(float(-slope), 3),
        "helmholtz": {
            "expected_energy": round(expected_energy, 4),
            "shannon_entropy": round(shannon_entropy, 4),
            "free_energy": round(helmholtz_free_energy, 4),
            "temperature": temp,
            "logits": [round(float(logits_2d[0]), 4), round(float(logits_2d[1]), 4)]
        },
        "manifold": {
            "dim": 768,
            "l2_norm": round(l2_norm, 4),
            "cosine_similarity_ref": round(cos_sim, 4),
            "cosine_distance": round(1.0 - cos_sim, 4),
            "projection_32": manifold_proj_32
        },
        "snn_telemetry": {
            "num_steps": 10,
            "beta_decay": 0.9,
            "v_trace_real": v_trace_class0,
            "v_trace_fake": v_trace_class1,
            "hidden_spike_counts": spike_counts_hidden
        },
        "vqc_telemetry": {
            "num_qubits": 4,
            "feature_map": "ZZFeatureMap (reps=1, linear)",
            "ansatz": "RealAmplitudes (reps=1, linear)",
            "encoded_angles_rad": [round(float(a), 4) for a in q_angles],
            "pauli_z_expectation": round(q_exp, 4)
        }
    }

def determine_three_state_verdict(score: float) -> Dict[str, str]:
    if score < 0.45:
        return {
            "verdict": "Authentic",
            "code": "AUTHENTIC",
            "summary": "Evidence supports media authenticity across spatial and frequency domains."
        }
    elif score > 0.55:
        return {
            "verdict": "Suspected",
            "code": "SUSPECTED",
            "summary": "Evidence indicates statistically significant synthetic or manipulative anomalies."
        }
    else:
        return {
            "verdict": "Inconclusive",
            "code": "INCONCLUSIVE",
            "summary": "Insufficient reliable separation between authentic sensor noise and manipulation."
        }

@app.post("/api/analyze")
async def analyze_media(
    file: UploadFile = File(...),
    variant: str = Form("federated"),
    cutoff_r: int = Form(24)
):
    t_e2e_start = time.perf_counter()
    contents = await file.read()
    filename = file.filename or "ingested_media"
    is_video = filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))

    t0 = time.perf_counter()
    if is_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)
        try:
            frames_pil = SAMPLER.sample(tmp_path)
            first_rgb = np.array(frames_pil[0])
            frames_tensor = torch.stack([TRANSFORM(img) for img in frames_pil]).unsqueeze(0).to(DEVICE)
            num_frames_sampled = len(frames_pil)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        np_img = np.array(pil_img)
        first_rgb = SAMPLER._extract_face(np_img)
        pil_crop = Image.fromarray(first_rgb).resize((224, 224), Image.Resampling.BILINEAR)
        frames_tensor = TRANSFORM(pil_crop).unsqueeze(0).unsqueeze(0).to(DEVICE)
        num_frames_sampled = 1

    prep_ms = (time.perf_counter() - t0) * 1000.0
    log_event("MEDIA_INGESTION_COMPLETE", f"{filename} (T={num_frames_sampled})", prep_ms)

    t1 = time.perf_counter()
    model_key = "image_140k" if (not is_video and Path("checkpoints/image_140k_vit.pt").exists()) else variant
    model = get_model(model_key)
    with torch.no_grad():
        feats_t = model.extract_video_features(frames_tensor)
        features_768 = feats_t[0].cpu().numpy()
    vit_ms = (time.perf_counter() - t1) * 1000.0
    log_event("VIT_EMBEDDING_GENERATED", f"z in R^768 (||z||_2 = {np.linalg.norm(features_768):.2f})", vit_ms)

    t2 = time.perf_counter()
    forensics = ANTI_HALLUCINATOR.analyze_static_image_forensics(first_rgb)
    rppg_val = ANTI_HALLUCINATOR.compute_rppg_pulse_coherence(first_rgb) if is_video else None
    mm_ms = (time.perf_counter() - t2) * 1000.0
    log_event("SPECTRAL_ANALYSIS_COMPLETE", f"B={forensics['bispectral_bicoherence']}, det={forensics['rgb_residual_det']}", mm_ms)

    t3 = time.perf_counter()
    with torch.no_grad():
        logits_t = model(frames_tensor)[0]
        logits_np = logits_t.cpu().numpy()
        raw_prob = float(torch.softmax(logits_t, dim=0)[1].item())

    calib_params = get_calibration_params(variant) if is_video else {"optimal_threshold": 0.55, "scale": 2.5}
    cal_prob = apply_calibration(raw_prob, get_calibration_params("sw_cross_vit")) if model_key == "image_140k" else apply_calibration(raw_prob, calib_params)

    # -------------------------------------------------------------------------
    # JPEG-Aware Adaptive ViT Primacy (Bayesian Log-Odds Evidence Fusion)
    # -------------------------------------------------------------------------
    def _logit(p):
        p = float(np.clip(p, 1e-4, 1.0 - 1e-4))
        return float(np.log(p / (1.0 - p)))

    f_p = forensics["forensic_fake_prob"]
    bicoh = forensics.get("bispectral_bicoherence", 0.0)

    # Adaptive Evidence Weighting:
    # If the 140k-trained ViT identifies authentic facial semantics (< 0.45)
    # and the bispectrum confirms no phase coupling (bicoh < 0.035),
    # weight the ViT heavily (0.85 / 0.15) to prevent JPEG compression false positives.
    if cal_prob < 0.45 and bicoh < 0.035:
        w_v = 0.85
        w_f = 0.15
    elif cal_prob > 0.60:
        w_v = 0.75
        w_f = 0.25
    else:
        w_v = 0.55
        w_f = 0.45

    log_odds = w_v * _logit(cal_prob) + w_f * _logit(f_p)
    if f_p > 0.60 and cal_prob > 0.52:
        log_odds += 0.50

    ds_prob = float(1.0 / (1.0 + np.exp(-np.clip(log_odds, -20.0, 20.0)))) if not is_video else cal_prob

    prior_x = ANTI_HALLUCINATOR.kalman_x
    prior_p = ANTI_HALLUCINATOR.kalman_p + ANTI_HALLUCINATOR.process_noise_q
    obs_z = ds_prob
    innovation_y = obs_z - prior_x
    k_gain = prior_p / (prior_p + ANTI_HALLUCINATOR.base_measurement_r)
    post_x = float(np.clip(prior_x + k_gain * innovation_y, 0.01, 0.99))
    post_p = (1.0 - k_gain) * prior_p
    ANTI_HALLUCINATOR.kalman_x = post_x
    ANTI_HALLUCINATOR.kalman_p = post_p

    head_ms = (time.perf_counter() - t3) * 1000.0
    total_ms = (time.perf_counter() - t_e2e_start) * 1000.0

    verdict_info = determine_three_state_verdict(ds_prob)
    log_event("CLASSIFIER_INFERENCE_COMPLETE", f"Key={model_key} | Raw={raw_prob:.4f}", head_ms)
    log_event("CALIBRATION_APPLIED", f"Verdict={verdict_info['verdict']} (p_ds={ds_prob:.4f})", 0.15)

    WORKSTATION_STATE["total_analyses"] += 1
    if verdict_info["code"] == "AUTHENTIC":
        WORKSTATION_STATE["authentic_count"] += 1
    elif verdict_info["code"] == "SUSPECTED":
        WORKSTATION_STATE["suspected_count"] += 1
    else:
        WORKSTATION_STATE["inconclusive_count"] += 1

    sci_diag = compute_full_scientific_diagnostics(first_rgb, features_768, logits_np, is_video, cutoff_r=cutoff_r)

    return {
        "filename": filename,
        "media_type": "Temporal Video Stream" if is_video else "Static Image Tensor",
        "tensor_shape": [1, num_frames_sampled, 3, 224, 224],
        "verdict": verdict_info,
        "statistics": {
            "raw_classifier_probability": round(raw_prob, 4),
            "calibrated_probability": round(cal_prob, 4),
            "dempster_shafer_belief": round(ds_prob, 4),
            "optimal_threshold_tau": calib_params["optimal_threshold"]
        },
        "forensics": forensics,
        "rppg": {
            "available": is_video,
            "coherence_index": round(rppg_val, 4) if is_video else None,
            "note": "Temporal chrominance pulse estimated across sampled frames." if is_video else "Single-frame input (T=1): temporal cardiac estimation requires video/webcam stream."
        },
        "kalman": {
            "state_prediction_x_prior": round(prior_x, 4),
            "observation_z": round(obs_z, 4),
            "innovation_residual_y": round(innovation_y, 4),
            "kalman_gain_k": round(k_gain, 4),
            "covariance_p_post": round(post_p, 4),
            "updated_state_x_post": round(post_x, 4)
        },
        "latencies_ms": {
            "preprocessing": round(prep_ms, 2),
            "vit_extraction": round(vit_ms, 2),
            "multimodal_forensics": round(mm_ms, 2),
            "classifier_head": round(head_ms, 2),
            "total_e2e": round(total_ms, 2)
        },
        "diagnostics": sci_diag
    }

@app.get("/api/state")
def get_workstation_state():
    ledger_path = Path("experiments/federated/blockchain_audit_ledger.json")
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []

    ablation_path = Path("experiments/ablation_results.csv")
    ablation = pd.read_csv(ablation_path).to_dict(orient="records") if ablation_path.exists() else []

    edge_path = Path("experiments/edge_onnx_benchmark.csv")
    edge_bench = pd.read_csv(edge_path).to_dict(orient="records") if edge_path.exists() else []

    stats_path = Path("experiments/statistical_significance.csv")
    stats_bench = pd.read_csv(stats_path).to_dict(orient="records") if stats_path.exists() else []

    return {
        "state": WORKSTATION_STATE,
        "device": str(DEVICE).upper(),
        "ledger": ledger,
        "ablation": ablation,
        "edge_benchmark": edge_bench,
        "statistical_study": stats_bench
    }

@app.get("/", response_class=HTMLResponse)
def veritas_workstation():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VERITAS — Scientific Intelligence Workstation</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/geist@1.3.0/dist/fonts/geist-sans/style.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/geist@1.3.0/dist/fonts/geist-mono/style.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        :root {
            --canvas: #F7F8FA;
            --surface: #FFFFFF;
            --navy: #172B4D;
            --sci-blue: #4779A8;
            --verified: #19856B;
            --warning: #C18B35;
            --anomaly: #C34D4D;
            --border: #DCE2E9;
            --text-primary: #172B4D;
            --text-secondary: #5E6C84;
            --text-muted: #7A869A;
            --font-sans: 'Geist Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'Geist Mono', 'SFMono-Regular', Consolas, monospace;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0; padding: 0;
            background-color: var(--canvas);
            color: var(--text-primary);
            font-family: var(--font-sans);
            font-size: 13.5px;
            line-height: 1.5;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        .sidebar {
            width: 248px;
            min-width: 248px;
            background: var(--surface);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 16px 0;
            z-index: 10;
        }
        .brand {
            padding: 0 20px 16px;
            border-bottom: 1px solid var(--border);
        }
        .brand-title {
            font-size: 17px;
            font-weight: 700;
            letter-spacing: 1.4px;
            color: var(--navy);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .brand-sub {
            font-size: 11px;
            color: var(--text-muted);
            font-family: var(--font-mono);
            margin-top: 4px;
        }
        .nav-section-label {
            font-size: 10.5px;
            font-weight: 700;
            letter-spacing: 1px;
            color: var(--text-muted);
            padding: 16px 20px 6px;
            text-transform: uppercase;
        }
        .nav-list { list-style: none; margin: 0; padding: 0 12px; }
        .nav-item {
            padding: 8px 12px;
            border-radius: 8px;
            color: var(--text-secondary);
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 2px;
            transition: background 0.15s, color 0.15s;
        }
        .nav-item:hover { background: var(--canvas); color: var(--navy); }
        .nav-item.active {
            background: #EDF2F7;
            color: var(--navy);
            font-weight: 600;
            border-left: 3px solid var(--sci-blue);
        }
        .nav-num { font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); }
        .sidebar-footer {
            padding: 14px 20px 0;
            border-top: 1px solid var(--border);
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-secondary);
        }

        .main-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .topbar {
            height: 56px;
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 0 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .workspace-header-title { font-size: 15px; font-weight: 600; color: var(--navy); }
        .topbar-Specs { display: flex; gap: 16px; font-family: var(--font-mono); font-size: 11.5px; color: var(--text-secondary); }
        .spec-tag { background: var(--canvas); border: 1px solid var(--border); padding: 4px 10px; border-radius: 6px; }

        .workspace-content {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }
        .screen { display: none; }
        .screen.active { display: block; }

        .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }
        .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 20px; }
        .forensic-layout { display: grid; grid-template-columns: 280px 1fr 340px; gap: 16px; align-items: start; }

        .panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 16px;
            box-shadow: 0 1px 2px rgba(23, 43, 77, 0.03);
        }
        .panel-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--navy);
            margin: 0 0 12px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .mono { font-family: var(--font-mono); }
        .metric-big { font-family: var(--font-mono); font-size: 24px; font-weight: 700; color: var(--navy); margin-top: 4px; }
        .metric-label { font-size: 11.5px; color: var(--text-secondary); font-weight: 500; }

        .verdict-box {
            padding: 14px 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
            margin-bottom: 14px;
        }
        .verdict-AUTHENTIC { background: #F1F9F7; border-color: var(--verified); color: var(--verified); }
        .verdict-SUSPECTED { background: #FDF4F4; border-color: var(--anomaly); color: var(--anomaly); }
        .verdict-INCONCLUSIVE { background: #FCF8F0; border-color: var(--warning); color: var(--warning); }
        .verdict-state-title { font-size: 16px; font-weight: 700; letter-spacing: 0.4px; }
        .verdict-state-desc { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }

        .ingest-zone {
            border: 1px dashed var(--sci-blue);
            background: var(--canvas);
            border-radius: 8px;
            padding: 20px 14px;
            text-align: center;
            cursor: pointer;
            transition: background 0.15s;
        }
        .ingest-zone:hover { background: #EDF2F7; }
        .btn {
            background: var(--navy);
            color: #FFFFFF;
            border: 1px solid var(--navy);
            border-radius: 6px;
            padding: 8px 14px;
            font-size: 12.5px;
            font-weight: 500;
            cursor: pointer;
            width: 100%;
            margin-top: 8px;
        }
        .btn-secondary {
            background: var(--surface);
            color: var(--navy);
            border: 1px solid var(--border);
        }
        .btn-secondary:hover { background: var(--canvas); }
        select, input[type="range"] {
            width: 100%;
            padding: 7px 10px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--surface);
            color: var(--navy);
            font-family: var(--font-sans);
            font-size: 12.5px;
            margin-bottom: 10px;
        }

        table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
        th, td { padding: 9px 10px; border-bottom: 1px solid var(--border); text-align: left; }
        th { color: var(--text-secondary); font-weight: 600; background: var(--canvas); font-size: 11.5px; }
        .event-stream {
            font-family: var(--font-mono);
            font-size: 11.5px;
            background: var(--canvas);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            max-height: 290px;
            overflow-y: auto;
        }
        .event-row { padding: 4px 0; border-bottom: 1px solid #EAEEF3; display: flex; gap: 14px; }
        .event-ts { color: var(--text-muted); min-width: 85px; }
        .event-code { color: var(--sci-blue); font-weight: 600; min-width: 240px; }

        .math-module {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .math-five-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 12px;
        }
        .math-step-box {
            background: var(--canvas);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
        }
        .math-step-num {
            font-family: var(--font-mono);
            font-size: 10.5px;
            font-weight: 700;
            color: var(--sci-blue);
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .img-grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
        .img-card { background: var(--canvas); border: 1px solid var(--border); border-radius: 8px; padding: 8px; text-align: center; }
        .img-card img { width: 100%; border-radius: 4px; border: 1px solid var(--border); }
        .img-card span { display: block; font-size: 11px; color: var(--text-secondary); margin-top: 6px; font-family: var(--font-mono); }

        #arch3dCanvas {
            width: 100%;
            height: 440px;
            background: #F0F3F7;
            border: 1px solid var(--border);
            border-radius: 8px;
            cursor: grab;
        }
        .kv-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #F0F2F5; font-size: 12.5px; }
        .kv-key { color: var(--text-secondary); }
        .kv-val { font-family: var(--font-mono); font-weight: 600; color: var(--navy); }
    </style>
</head>
<body>

    <aside class="sidebar">
        <div>
            <div class="brand">
                <div class="brand-title">VERITAS</div>
                <div class="brand-sub">SCIENTIFIC WORKSTATION · v1.0</div>
            </div>
            <div class="nav-section-label">Workspace</div>
            <ul class="nav-list" id="navList">
                <li class="nav-item active" data-target="screen-overview"><span>Overview</span><span class="nav-num">01</span></li>
                <li class="nav-item" data-target="screen-forensic"><span>Forensic Analysis</span><span class="nav-num">02</span></li>
                <li class="nav-item" data-target="screen-math"><span>Mathematical Lab</span><span class="nav-num">03</span></li>
                <li class="nav-item" data-target="screen-arch"><span>Architecture Explorer</span><span class="nav-num">04</span></li>
                <li class="nav-item" data-target="screen-qsnn"><span>Quantum & SNN Lab</span><span class="nav-num">05</span></li>
                <li class="nav-item" data-target="screen-fed"><span>Federated Network</span><span class="nav-num">06</span></li>
                <li class="nav-item" data-target="screen-ledger"><span>Security Ledger</span><span class="nav-num">07</span></li>
                <li class="nav-item" data-target="screen-perf"><span>Performance & Reports</span><span class="nav-num">08</span></li>
            </ul>
        </div>
        <div class="sidebar-footer">
            <div>SPEC: 08 LAYERS · 03 FL NODES</div>
            <div>768-D VIT · 04 QUBITS · 10-STEP LIF</div>
        </div>
    </aside>

    <div class="main-container">
        <header class="topbar">
            <div class="workspace-header-title" id="currentScreenTitle">01 · Overview — Command Center</div>
            <div class="topbar-Specs">
                <span class="spec-tag" id="deviceBadge">BACKEND: MPS</span>
                <span class="spec-tag">BACKBONE: ViT-B/16 (768-D)</span>
                <span class="spec-tag">CALIBRATION: YOUDEN'S J</span>
            </div>
        </header>

        <main class="workspace-content">

            <!-- SCREEN 01: OVERVIEW -->
            <section id="screen-overview" class="screen active">
                <div class="grid-4">
                    <div class="panel">
                        <div class="metric-label">Total Analyses Executed</div>
                        <div class="metric-big" id="ovTotal">0</div>
                    </div>
                    <div class="panel">
                        <div class="metric-label">Authentic Classifications</div>
                        <div class="metric-big" style="color: var(--verified);" id="ovAuth">0</div>
                    </div>
                    <div class="panel">
                        <div class="metric-label">Suspected Media</div>
                        <div class="metric-big" style="color: var(--anomaly);" id="ovSusp">0</div>
                    </div>
                    <div class="panel">
                        <div class="metric-label">Inconclusive Investigations</div>
                        <div class="metric-big" style="color: var(--warning);" id="ovInc">0</div>
                    </div>
                </div>

                <div class="grid-2">
                    <div class="panel">
                        <div class="panel-title">
                            <span>Live Computational Event Stream</span>
                            <span class="mono" style="font-size:11px; color:var(--verified);">● LIVE</span>
                        </div>
                        <div class="event-stream" id="eventStreamBox"></div>
                    </div>
                    <div class="panel">
                        <div class="panel-title"><span>System Architecture Specifications</span><span class="mono">VERITAS v1.0</span></div>
                        <div class="kv-row"><span class="kv-key">01. Visual Feature Extractor</span><span class="kv-val">ViT-Base Patch16-224 (image_140k Fine-Tuned)</span></div>
                        <div class="kv-row"><span class="kv-key">02. Embedding Dimensionality</span><span class="kv-val">z ∈ ℝ^768 (CLS Token Mean-Pooled)</span></div>
                        <div class="kv-row"><span class="kv-key">03. Frequency Forensics</span><span class="kv-val">2D-FFT RAPSD + 3rd-Order Bispectrum</span></div>
                        <div class="kv-row"><span class="kv-key">04. Residual Noise Model</span><span class="kv-val">4th-Order SRM + JPEG-Compensated det(C)</span></div>
                        <div class="kv-row"><span class="kv-key">05. Quantum Experimental Head</span><span class="kv-val">4-Qubit ZZFeatureMap + RealAmplitudes</span></div>
                        <div class="kv-row"><span class="kv-key">06. Neuromorphic Head</span><span class="kv-val">10-Step Leaky Integrate-and-Fire (LIF)</span></div>
                        <div class="kv-row"><span class="kv-key">07. Distributed Federation</span><span class="kv-val">3 Nodes (Flower gRPC + Norm Screening)</span></div>
                        <div class="kv-row"><span class="kv-key">08. Post-Hoc Calibration</span><span class="kv-val">Adaptive Bayesian ViT Primacy</span></div>
                    </div>
                </div>
            </section>

            <!-- SCREEN 02: FORENSIC ANALYSIS -->
            <section id="screen-forensic" class="screen">
                <div class="forensic-layout">
                    <div class="panel">
                        <div class="panel-title"><span>1. Media Ingestion</span><span class="mono">INPUT</span></div>
                        <div class="ingest-zone" id="dropZone">
                            <div style="font-weight:600; color:var(--navy);">Paste Image (Cmd+V) or Drop Media</div>
                            <div style="font-size:11.5px; color:var(--text-secondary); margin-top:4px;">Supports .png, .jpg, .mp4, .mov</div>
                            <input type="file" id="mediaFileInput" accept="image/*,video/*,.mp4,.mov" style="display:none;" />
                        </div>
                        <button class="btn btn-secondary" onclick="document.getElementById('mediaFileInput').click()">Select File from Disk</button>

                        <div style="margin-top:16px;">
                            <label class="metric-label">Classifier Paradigm</label>
                            <select id="modelVariantSelect">
                                <option value="federated">ViT + 140k Adaptive Head (Recommended)</option>
                                <option value="baseline">Centralized Baseline ViT</option>
                                <option value="quantum">4-Qubit Variational Quantum (VQC)</option>
                                <option value="spiking">10-Step Neuromorphic Spiking (SNN)</option>
                            </select>

                            <label class="metric-label">Azimuthal High-Pass Cutoff ($r_c$): <span class="mono" id="cutoffValLabel">24 px</span></label>
                            <input type="range" id="cutoffSlider" min="6" max="64" value="24" oninput="document.getElementById('cutoffValLabel').textContent=this.value+' px'" />

                            <button class="btn" id="runAnalysisBtn">Execute Forensic Pipeline</button>
                        </div>

                        <div style="margin-top:16px; border-top:1px solid var(--border); padding-top:12px;">
                            <div class="metric-label">Ingested Tensor Specification</div>
                            <div class="kv-row"><span class="kv-key">Media Type</span><span class="kv-val" id="faMediaType">—</span></div>
                            <div class="kv-row"><span class="kv-key">Tensor Shape</span><span class="kv-val" id="faTensorShape">—</span></div>
                            <div class="kv-row"><span class="kv-key">End-to-End Runtime</span><span class="kv-val" id="faRuntime">—</span></div>
                        </div>
                    </div>

                    <div class="panel">
                        <div class="panel-title">
                            <span>2. Multi-Domain Visual & Spectral Inspection</span>
                            <span class="mono" id="faFilename">No media loaded</span>
                        </div>
                        <div class="img-grid-4">
                            <div class="img-card">
                                <img id="viewCrop" src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=" alt="ROI" />
                                <span>Spatial Face ROI I(x,y)</span>
                            </div>
                            <div class="img-card">
                                <img id="viewFFT" src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=" alt="FFT" />
                                <span>Log Spectrum log(1+|F(u,v)|)</span>
                            </div>
                            <div class="img-card">
                                <img id="viewRecon" src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=" alt="Recon" />
                                <span>Azimuthal High-Pass F⁻¹</span>
                            </div>
                            <div class="img-card">
                                <img id="viewSRM" src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=" alt="SRM" />
                                <span>4th-Order SRM Residual R</span>
                            </div>
                        </div>

                        <div style="margin-top:18px;">
                            <div class="panel-title">
                                <span>Radially Averaged Power Spectral Density P(r) vs. Natural 1/f^α Fit</span>
                                <span class="mono" id="specSlopeBadge">α = —</span>
                            </div>
                            <canvas id="forensicRapsdChart" height="115"></canvas>
                        </div>
                    </div>

                    <div class="panel">
                        <div class="panel-title"><span>3. Verdict & Statistical Attribution</span><span class="mono">DIAGNOSTICS</span></div>
                        <div id="verdictBox" class="verdict-box verdict-INCONCLUSIVE">
                            <div class="verdict-state-title" id="verdictTitle">Awaiting Investigation</div>
                            <div class="verdict-state-desc" id="verdictDesc">Paste or select an image or video to execute the mathematical pipeline.</div>
                        </div>

                        <div class="metric-label" style="margin-bottom:6px;">Explicit Statistical Attribution</div>
                        <div class="kv-row"><span class="kv-key">Raw Classifier Probability</span><span class="kv-val" id="statRaw">—</span></div>
                        <div class="kv-row"><span class="kv-key">Calibrated Probability</span><span class="kv-val" id="statCal">—</span></div>
                        <div class="kv-row"><span class="kv-key">Dempster-Shafer Multi-Modal Belief</span><span class="kv-val" id="statDS">—</span></div>
                        <div class="kv-row"><span class="kv-key">Reference Operating Threshold ($\tau^*$)</span><span class="kv-val" id="statTau">—</span></div>

                        <div class="metric-label" style="margin:14px 0 6px;">Physical & Frequency Evidence</div>
                        <div class="kv-row"><span class="kv-key">RGB SRM Jacobian $\det(\mathbf{C}_{\text{RGB}})$</span><span class="kv-val" id="evDet">—</span></div>
                        <div class="kv-row"><span class="kv-key">3rd-Order Bispectrum $\mathcal{B}(\omega_1,\omega_2)$</span><span class="kv-val" id="evBisp">—</span></div>
                        <div class="kv-row"><span class="kv-key">Benford 2D-DCT $D_{\text{KL}}$</span><span class="kv-val" id="evBenf">—</span></div>
                        <div class="kv-row"><span class="kv-key">rPPG Temporal Coherence</span><span class="kv-val" id="evRppg">—</span></div>
                    </div>
                </div>
            </section>

            <!-- SCREEN 03: MATHEMATICAL LAB -->
            <section id="screen-math" class="screen">
                <div class="math-module">
                    <div class="panel-title"><span>Module A · 2D Fourier Transform & Azimuthal Radial Power Spectrum</span><span class="mono">FREQUENCY DOMAIN</span></div>
                    <div class="math-five-grid">
                        <div class="math-step-box">
                            <div class="math-step-num">1. Mathematical Formulation</div>
                            <div>$$F(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} I(x,y)w(x,y)e^{-i2\pi(ux/M + vy/N)}$$</div>
                            <div>$$P(r) = \frac{1}{|\Omega_r|}\sum_{(u,v)\in\Omega_r} |F(u,v)|^2, \quad \mathcal{B}(\omega_1,\omega_2) = \frac{|F(\omega_1)F(\omega_2)F^*(\omega_1+\omega_2)|}{|F(\omega_1)||F(\omega_2)||F(\omega_1+\omega_2)|}$$</div>
                        </div>
                        <div class="math-step-box">
                            <div class="math-step-num">2. Input Data & Tensor Dimensions</div>
                            <div class="kv-row"><span class="kv-key">Spatial ROI Matrix $I(x,y)$</span><span class="kv-val">ℝ^(256×256) (Grayscale Luminance)</span></div>
                            <div class="kv-row"><span class="kv-key">Windowing Function $w(x,y)$</span><span class="kv-val">2D Separable Hanning Window</span></div>
                            <div class="kv-row"><span class="kv-key">3. Runtime Execution Latency</span><span class="kv-val" id="mathSpecLat">— ms</span></div>
                            <div class="math-step-num" style="margin-top:8px;">4. Intermediate Numerical Results</div>
                            <div class="kv-row"><span class="kv-key">Bispectral Phase Coherence $\mathcal{B}$</span><span class="kv-val" id="mathSpecBisp">—</span></div>
                            <div class="kv-row"><span class="kv-key">Benford DCT KL-Divergence $D_{\text{KL}}$</span><span class="kv-val" id="mathSpecKL">—</span></div>
                            <div class="kv-row"><span class="kv-key">RGB Residual Determinant $\det(\mathbf{C})$</span><span class="kv-val" id="mathSpecDet">—</span></div>
                        </div>
                    </div>
                </div>

                <div class="grid-2">
                    <div class="math-module">
                        <div class="panel-title"><span>Module B · Identity Manifold Geometry (768-D)</span><span class="mono">EMBEDDING SPACE</span></div>
                        <div class="math-step-box" style="margin-bottom:10px;">
                            <div class="math-step-num">1. Mathematical Formulation</div>
                            <div>$$s_{\cos}(z, z_{\text{ref}}) = \frac{z \cdot z_{\text{ref}}}{\|z\|_2 \|z_{\text{ref}}\|_2}, \quad z = \frac{1}{T}\sum_{t=1}^T \text{ViT}_{\text{CLS}}(I_t) \in \mathbb{R}^{768}$$</div>
                        </div>
                        <div class="math-step-box">
                            <div class="math-step-num">2–4. Dimensions, Runtime & Values</div>
                            <div class="kv-row"><span class="kv-key">Embedding Vector $z$</span><span class="kv-val">ℝ^768 (ViT-Base Patch16)</span></div>
                            <div class="kv-row"><span class="kv-key">L2 Euclidean Norm $\|z\|_2$</span><span class="kv-val" id="mathManNorm">—</span></div>
                            <div class="kv-row"><span class="kv-key">Reference Cosine Similarity $s_{\cos}$</span><span class="kv-val" id="mathManCos">—</span></div>
                            <div class="kv-row"><span class="kv-key">Extraction Latency</span><span class="kv-val" id="mathManLat">— ms</span></div>
                        </div>
                        <div style="margin-top:12px;">
                            <div class="math-step-num">5. 32-Subspace Block-Mean Projection of $z \in \mathbb{R}^{768}$</div>
                            <canvas id="manifoldChart" height="120"></canvas>
                        </div>
                    </div>

                    <div class="math-module">
                        <div class="panel-title"><span>Module C · Helmholtz Free Energy & 1D Kalman State Filter</span><span class="mono">STATE ESTIMATION</span></div>
                        <div class="math-step-box" style="margin-bottom:10px;">
                            <div class="math-step-num">1. Mathematical Formulation</div>
                            <div>$$F(q) = \mathbb{E}_{q(x)}[E(x)] - T\cdot H(q), \quad K_t = P_{t|t-1}H^T(HP_{t|t-1}H^T + R)^{-1}$$</div>
                        </div>
                        <div class="math-step-box">
                            <div class="math-step-num">4. Live Intermediate Numerical State</div>
                            <div class="kv-row"><span class="kv-key">Classifier Logits $f(x)$</span><span class="kv-val" id="mathHelmLogits">—</span></div>
                            <div class="kv-row"><span class="kv-key">Expected Energy $\mathbb{E}_q[E(x)]$</span><span class="kv-val" id="mathHelmExpE">—</span></div>
                            <div class="kv-row"><span class="kv-key">Shannon Entropy $H(q)$</span><span class="kv-val" id="mathHelmEnt">—</span></div>
                            <div class="kv-row"><span class="kv-key">Helmholtz Free Energy $F(q)$</span><span class="kv-val" id="mathHelmFree">—</span></div>
                            <div class="kv-row"><span class="kv-key">Kalman State Prediction $\hat{x}_{t|t-1}$</span><span class="kv-val" id="mathKalPrior">—</span></div>
                            <div class="kv-row"><span class="kv-key">Innovation Residual $y_t$</span><span class="kv-val" id="mathKalInnov">—</span></div>
                            <div class="kv-row"><span class="kv-key">Kalman Gain $K_t$ / Covariance $P_{t|t}$</span><span class="kv-val" id="mathKalGain">—</span></div>
                            <div class="kv-row"><span class="kv-key">Updated State Estimate $\hat{x}_{t|t}$</span><span class="kv-val" id="mathKalPost">—</span></div>
                        </div>
                    </div>
                </div>
            </section>

            <!-- SCREEN 04: ARCHITECTURE EXPLORER -->
            <section id="screen-arch" class="screen">
                <div class="grid-3">
                    <div class="panel" style="grid-column: span 2;">
                        <div class="panel-title">
                            <span>Interactive 3D Computational Graph (Click Any Node 01–08 to Inspect)</span>
                            <span class="mono">THREE.JS WEBGL</span>
                        </div>
                        <div id="arch3dCanvas"></div>
                    </div>
                    <div class="panel">
                        <div class="panel-title"><span>Node Diagnostic Inspector</span><span class="mono" id="nodeInspId">NODE 02</span></div>
                        <div class="kv-row"><span class="kv-key">Layer Name</span><span class="kv-val" id="nodeInspName">02 · Feature Extraction</span></div>
                        <div class="kv-row"><span class="kv-key">Input Tensor</span><span class="kv-val" id="nodeInspIn">[B, T, 3, 224, 224]</span></div>
                        <div class="kv-row"><span class="kv-key">Output Tensor</span><span class="kv-val" id="nodeInspOut">z ∈ ℝ^(B×768)</span></div>
                        <div class="kv-row"><span class="kv-key">Source Module</span><span class="kv-val" id="nodeInspSrc">models/vit_detector.py</span></div>
                        <div class="kv-row"><span class="kv-key">Measured Runtime</span><span class="kv-val" id="nodeInspLat">— ms</span></div>
                        <div style="margin-top:12px;" class="math-step-box">
                            <div class="math-step-num">Governing Operation</div>
                            <div id="nodeInspMath">$$z = \frac{1}{T}\sum_{t=1}^T \text{ViT}_{\text{CLS}}(I_t)$$</div>
                        </div>
                        <div style="margin-top:12px;">
                            <div class="metric-label" style="margin-bottom:6px;">Select Pipeline Layer</div>
                            <div id="nodeButtonsList" style="display:grid; grid-template-columns:1fr 1fr; gap:6px;"></div>
                        </div>
                    </div>
                </div>
            </section>

            <!-- SCREEN 05: QUANTUM & SNN LAB -->
            <section id="screen-qsnn" class="screen">
                <div class="grid-2">
                    <div class="panel">
                        <div class="panel-title"><span>4-Qubit Variational Quantum Circuit (VQC) Inspector</span><span class="mono">QISKIT ESTIMATORQNN</span></div>
                        <div class="math-step-box" style="margin-bottom:12px;">
                            <div>$$|\psi(\theta, x)\rangle = U_{\text{RealAmplitudes}}(\theta) U_{\text{ZZFeatureMap}}(x)|0\rangle^{\otimes 4}, \quad \langle \hat{O}\rangle = \langle\psi|Z^{\otimes 4}|\psi\rangle$$</div>
                        </div>
                        <div class="kv-row"><span class="kv-key">Feature Map Encoding</span><span class="kv-val">ZZFeatureMap (reps=1, linear)</span></div>
                        <div class="kv-row"><span class="kv-key">Trainable Ansatz</span><span class="kv-val">RealAmplitudes (reps=1, linear)</span></div>
                        <div class="kv-row"><span class="kv-key">Encoded Qubit Angles $\theta_{\text{in}} \in [-1,1]^4$</span><span class="kv-val" id="vqcAngles">—</span></div>
                        <div class="kv-row"><span class="kv-key">Measured Expectation $\langle Z\rangle$</span><span class="kv-val" id="vqcExp">—</span></div>
                    </div>

                    <div class="panel">
                        <div class="panel-title"><span>10-Step Leaky Integrate-and-Fire (LIF) SNN Inspector</span><span class="mono">SNNTORCH TEMPORAL</span></div>
                        <div class="math-step-box" style="margin-bottom:12px;">
                            <div>$$\tau_m \frac{dV_t}{dt} = -(V_t - V_{\text{rest}}) + R\cdot I_t, \quad V_{t+1} = \beta V_t + W x_{t+1} - S_t V_{\text{th}}$$</div>
                        </div>
                        <div class="kv-row"><span class="kv-key">Decay Factor $\beta$ / Surrogate</span><span class="kv-val">β = 0.90 · Fast Sigmoid (slope=25)</span></div>
                        <div class="kv-row"><span class="kv-key">Hidden Layer Spike Counts ($t=1..10$)</span><span class="kv-val" id="snnSpikes">—</span></div>
                        <canvas id="snnVoltageChart" height="135" style="margin-top:10px;"></canvas>
                    </div>
                </div>
            </section>

            <!-- SCREEN 06: FEDERATED NETWORK -->
            <section id="screen-fed" class="screen">
                <div class="panel">
                    <div class="panel-title"><span>Three-Node Distributed Federated Learning Telemetry (FedAvg)</span><span class="mono">FLOWER gRPC</span></div>
                    <div class="math-step-box" style="margin-bottom:14px;">
                        <div>$$W_{t+1} = \sum_{k=1}^{K} \frac{n_k}{\sum_{j=1}^K n_j} W_{t+1}^{(k)}, \quad \text{Norm Screening Condition: } \|W_{t+1}^{(k)} - W_t\|_2 \le \tau_{\max}$$</div>
                    </div>
                    <table id="fedNodesTable">
                        <thead>
                            <tr><th>Round</th><th>Client Identity</th><th>Organization</th><th>Update L2 Norm $\|\Delta W\|_2$</th><th>Base Model Hash</th><th>Update Hash</th><th>Status</th></tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </section>

            <!-- SCREEN 07: SECURITY LEDGER -->
            <section id="screen-ledger" class="screen">
                <div class="panel">
                    <div class="panel-title"><span>Permissioned Blockchain Control-Plane Ledger (FederatedAuditLedger.sol)</span><span class="mono">SHA-256 HASH CHAIN</span></div>
                    <p style="font-size:12px; color:var(--text-secondary); margin-top:0;">
                        Note: Norm-based filtering acts as magnitude norm screening against large-scale gradient corruption, not comprehensive Byzantine fault tolerance.
                    </p>
                    <table id="ledgerTable">
                        <thead>
                            <tr><th>Block #</th><th>Transaction Type</th><th>Round</th><th>Participant / Coordinator</th><th>Previous Block Hash</th><th>Block SHA-256 Hash</th></tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </section>

            <!-- SCREEN 08: PERFORMANCE LAB -->
            <section id="screen-perf" class="screen">
                <div class="grid-2">
                    <div class="panel">
                        <div class="panel-title"><span>Stage-by-Stage Runtime Latency Breakdown (Latest Execution)</span><span class="mono">MILLISECONDS</span></div>
                        <canvas id="latencyBreakdownChart" height="140"></canvas>
                    </div>
                    <div class="panel">
                        <div class="panel-title"><span>ONNX Runtime Edge Compilation Benchmark (200 Iterations, 10 Warm-up)</span><span class="mono">OPSET 17</span></div>
                        <table id="edgeBenchTable">
                            <thead>
                                <tr><th>Edge Module</th><th>Footprint</th><th>PyTorch</th><th>ONNX Runtime</th><th>Speedup</th></tr>
                            </thead>
                            <tbody></tbody>
                        </table>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-title"><span>Controlled Comparative Ablation Record (Experiments E1–E5)</span><span class="mono">REPRODUCIBLE RECORD</span></div>
                    <table id="ablationFullTable">
                        <thead>
                            <tr><th>Experiment Paradigm</th><th>Params</th><th>Threshold ($\tau^*$)</th><th>Loss</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th><th>Latency</th></tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </section>

        </main>
    </div>

    <script>
        document.addEventListener("DOMContentLoaded", () => {
            if (window.renderMathInElement) {
                renderMathInElement(document.body, {
                    delimiters: [
                        {left: "$$", right: "$$", display: true},
                        {left: "$", right: "$", display: false}
                    ]
                });
            }
            initCharts();
            init3DArchitecture();
            refreshWorkstationState();
        });

        const screenTitles = {
            "screen-overview": "01 · Overview — Command Center",
            "screen-forensic": "02 · Forensic Analysis — Primary Workspace",
            "screen-math": "03 · Mathematical Lab — Equations & Runtime Inspection",
            "screen-arch": "04 · Architecture Explorer — Interactive 3D Pipeline",
            "screen-qsnn": "05 · Quantum (VQC) & Neuromorphic (SNN) Laboratory",
            "screen-fed": "06 · Federated Network — Three-Node Distributed Learning",
            "screen-ledger": "07 · Security Ledger — Permissioned Provenance Verification",
            "screen-perf": "08 · Performance Lab & Reproducible Research Reports"
        };

        document.querySelectorAll(".nav-item").forEach(item => {
            item.addEventListener("click", () => {
                document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
                document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
                item.classList.add("active");
                const target = item.getAttribute("data-target");
                document.getElementById(target).classList.add("active");
                document.getElementById("currentScreenTitle").textContent = screenTitles[target];
            });
        });

        let rapsdChart = null;
        let manifoldChart = null;
        let snnChart = null;
        let latencyChart = null;

        function initCharts() {
            const commonFont = { family: "'Geist Mono', monospace", size: 10 };
            
            // Build RAPSD Chart with dual datasets: Actual P(r) and Theoretical 1/f^alpha Fit
            const rapsdCanvas = document.getElementById("forensicRapsdChart");
            if (rapsdChart) rapsdChart.destroy();
            rapsdChart = new Chart(rapsdCanvas, {
                type: "line",
                data: {
                    labels: Array.from({length: 64}, (_, i) => i + 2),
                    datasets: [
                        {
                            label: "Measured P(r)",
                            data: Array(64).fill(7.0),
                            borderColor: "#4779A8",
                            backgroundColor: "rgba(71, 121, 168, 0.08)",
                            fill: true,
                            tension: 0.25,
                            borderWidth: 2,
                            pointRadius: 0
                        },
                        {
                            label: "Theoretical 1/f^α Fit",
                            data: Array(64).fill(7.0),
                            borderColor: "#C18B35",
                            borderDash: [4, 4],
                            fill: false,
                            tension: 0.0,
                            borderWidth: 1.5,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    animation: { duration: 350 },
                    plugins: { legend: { display: true, labels: { font: commonFont, boxWidth: 12 } } },
                    scales: {
                        x: { ticks: { font: commonFont }, grid: { color: "#EAEEF3" } },
                        y: { ticks: { font: commonFont }, grid: { color: "#EAEEF3" } }
                    }
                }
            });

            manifoldChart = new Chart(document.getElementById("manifoldChart"), {
                type: "bar",
                data: {
                    labels: Array.from({length: 32}, (_, i) => `B${i+1}`),
                    datasets: [{ label: "Subspace Mean", data: Array(32).fill(0), backgroundColor: "#172B4D" }]
                },
                options: { plugins: { legend: { display: false } }, scales: { x: { ticks: { font: commonFont } }, y: { ticks: { font: commonFont } } } }
            });

            snnChart = new Chart(document.getElementById("snnVoltageChart"), {
                type: "line",
                data: {
                    labels: Array.from({length: 10}, (_, i) => `t=${i+1}`),
                    datasets: [
                        { label: "V_m(t) Authentic Neuron", data: Array(10).fill(0), borderColor: "#19856B", tension: 0.2 },
                        { label: "V_m(t) Suspected Neuron", data: Array(10).fill(0), borderColor: "#C34D4D", tension: 0.2 }
                    ]
                },
                options: { scales: { x: { ticks: { font: commonFont } }, y: { ticks: { font: commonFont } } } }
            });

            latencyChart = new Chart(document.getElementById("latencyBreakdownChart"), {
                type: "bar",
                data: {
                    labels: ["Preprocessing", "ViT Extraction", "Multimodal Math", "Classifier Head"],
                    datasets: [{ label: "Latency (ms)", data: [0, 0, 0, 0], backgroundColor: ["#4779A8", "#172B4D", "#19856B", "#C18B35"] }]
                },
                options: { indexAxis: "y", plugins: { legend: { display: false } } }
            });
        }

        let currentMediaFile = null;
        window.addEventListener("paste", (e) => {
            const items = e.clipboardData?.items;
            if (!items) return;
            for (const item of items) {
                if (item.type.indexOf("image") !== -1) {
                    currentMediaFile = item.getAsFile();
                    document.querySelector('[data-target="screen-forensic"]').click();
                    executeForensicAnalysis(currentMediaFile);
                    break;
                }
            }
        });

        document.getElementById("mediaFileInput").addEventListener("change", (e) => {
            if (e.target.files?.length > 0) {
                currentMediaFile = e.target.files[0];
                executeForensicAnalysis(currentMediaFile);
            }
        });

        document.getElementById("runAnalysisBtn").addEventListener("click", () => {
            if (currentMediaFile) executeForensicAnalysis(currentMediaFile);
        });

        const dropZone = document.getElementById("dropZone");
        dropZone.addEventListener("dragover", (e) => { e.preventDefault(); });
        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            if (e.dataTransfer.files?.length > 0) {
                currentMediaFile = e.dataTransfer.files[0];
                executeForensicAnalysis(currentMediaFile);
            }
        });

        let latestLatencies = {};
        async function executeForensicAnalysis(file) {
            document.getElementById("verdictTitle").textContent = "Executing Pipeline...";
            document.getElementById("verdictDesc").textContent = "Computing ViT embeddings, 3rd-order bispectrum, SRM residuals and SNN traces...";

            const formData = new FormData();
            formData.append("file", file);
            formData.append("variant", document.getElementById("modelVariantSelect").value);
            formData.append("cutoff_r", document.getElementById("cutoffSlider").value);

            const res = await fetch("/api/analyze", { method: "POST", body: formData });
            const d = await res.json();

            document.getElementById("faFilename").textContent = d.filename;
            document.getElementById("faMediaType").textContent = d.media_type;
            document.getElementById("faTensorShape").textContent = JSON.stringify(d.tensor_shape);
            document.getElementById("faRuntime").textContent = d.latencies_ms.total_e2e + " ms";

            document.getElementById("viewCrop").src = d.diagnostics.visuals.crop;
            document.getElementById("viewFFT").src = d.diagnostics.visuals.fft;
            document.getElementById("viewRecon").src = d.diagnostics.visuals.recon;
            document.getElementById("viewSRM").src = d.diagnostics.visuals.srm;

            const vBox = document.getElementById("verdictBox");
            vBox.className = "verdict-box verdict-" + d.verdict.code;
            document.getElementById("verdictTitle").textContent = d.verdict.verdict;
            document.getElementById("verdictDesc").textContent = d.verdict.summary;

            document.getElementById("statRaw").textContent = d.statistics.raw_classifier_probability.toFixed(4);
            document.getElementById("statCal").textContent = d.statistics.calibrated_probability.toFixed(4);
            document.getElementById("statDS").textContent = d.statistics.dempster_shafer_belief.toFixed(4);
            document.getElementById("statTau").textContent = d.statistics.optimal_threshold_tau.toFixed(4);

            document.getElementById("evDet").textContent = `${d.forensics.rgb_residual_det} (${d.forensics.color_coherence_anomaly_pct}%)`;
            document.getElementById("evBisp").textContent = `${d.forensics.bispectral_bicoherence} (${d.forensics.bicoherence_anomaly_pct}%)`;
            document.getElementById("evBenf").textContent = `${d.forensics.benford_kl_divergence} (${d.forensics.benford_anomaly_pct}%)`;
            document.getElementById("evRppg").textContent = d.rppg.available ? `${d.rppg.coherence_index}` : "Undefined (T=1 Static)";

            // Update Dual RAPSD Spectral Graph with explicit copy to force canvas repaint
            if (rapsdChart && d.diagnostics.spectral_rapsd) {
                rapsdChart.data.datasets[0].data = [...d.diagnostics.spectral_rapsd];
                rapsdChart.data.datasets[1].data = [...d.diagnostics.ideal_rapsd_fit];
                document.getElementById("specSlopeBadge").textContent = `α = ${d.diagnostics.spectral_slope_alpha}`;
                rapsdChart.update();
            }

            manifoldChart.data.datasets[0].data = [...d.diagnostics.manifold.projection_32];
            manifoldChart.update();

            snnChart.data.datasets[0].data = [...d.diagnostics.snn_telemetry.v_trace_real];
            snnChart.data.datasets[1].data = [...d.diagnostics.snn_telemetry.v_trace_fake];
            snnChart.update();

            latencyChart.data.datasets[0].data = [
                d.latencies_ms.preprocessing,
                d.latencies_ms.vit_extraction,
                d.latencies_ms.multimodal_forensics,
                d.latencies_ms.classifier_head
            ];
            latencyChart.update();
            latestLatencies = d.latencies_ms;

            document.getElementById("mathSpecLat").textContent = d.latencies_ms.multimodal_forensics + " ms";
            document.getElementById("mathSpecBisp").textContent = d.forensics.bispectral_bicoherence;
            document.getElementById("mathSpecKL").textContent = d.forensics.benford_kl_divergence;
            document.getElementById("mathSpecDet").textContent = d.forensics.rgb_residual_det;

            document.getElementById("mathManNorm").textContent = d.diagnostics.manifold.l2_norm;
            document.getElementById("mathManCos").textContent = d.diagnostics.manifold.cosine_similarity_ref;
            document.getElementById("mathManLat").textContent = d.latencies_ms.vit_extraction + " ms";

            document.getElementById("mathHelmLogits").textContent = JSON.stringify(d.diagnostics.helmholtz.logits);
            document.getElementById("mathHelmExpE").textContent = d.diagnostics.helmholtz.expected_energy;
            document.getElementById("mathHelmEnt").textContent = d.diagnostics.helmholtz.shannon_entropy;
            document.getElementById("mathHelmFree").textContent = d.diagnostics.helmholtz.free_energy;

            document.getElementById("mathKalPrior").textContent = d.kalman.state_prediction_x_prior;
            document.getElementById("mathKalInnov").textContent = d.kalman.innovation_residual_y;
            document.getElementById("mathKalGain").textContent = `${d.kalman.kalman_gain_k} / P=${d.kalman.covariance_p_post}`;
            document.getElementById("mathKalPost").textContent = d.kalman.updated_state_x_post;

            document.getElementById("vqcAngles").textContent = JSON.stringify(d.diagnostics.vqc_telemetry.encoded_angles_rad);
            document.getElementById("vqcExp").textContent = d.diagnostics.vqc_telemetry.pauli_z_expectation;
            document.getElementById("snnSpikes").textContent = JSON.stringify(d.diagnostics.snn_telemetry.hidden_spike_counts);

            trigger3DPulse();
            refreshWorkstationState();
        }

        async function refreshWorkstationState() {
            const res = await fetch("/api/state");
            const data = await res.json();

            document.getElementById("deviceBadge").textContent = "BACKEND: " + data.device;
            document.getElementById("ovTotal").textContent = data.state.total_analyses;
            document.getElementById("ovAuth").textContent = data.state.authentic_count;
            document.getElementById("ovSusp").textContent = data.state.suspected_count;
            document.getElementById("ovInc").textContent = data.state.inconclusive_count;

            document.getElementById("eventStreamBox").innerHTML = data.state.events.map(ev => `
                <div class="event-row">
                    <span class="event-ts">${ev.timestamp}</span>
                    <span class="event-code">${ev.code}</span>
                    <span style="flex:1; color:var(--text-primary);">${ev.detail}</span>
                    <span style="color:var(--text-muted);">${ev.duration_ms} ms</span>
                </div>
            `).join("");

            const updateBlocks = data.ledger.filter(b => b.tx_type === "RECORD_UPDATE");
            document.querySelector("#fedNodesTable tbody").innerHTML = updateBlocks.map(b => `
                <tr>
                    <td class="mono">${b.payload.round}</td>
                    <td class="mono">${b.payload.client_address}</td>
                    <td><b>${b.payload.org_name}</b></td>
                    <td class="mono">${b.payload.delta_l2_norm}</td>
                    <td class="mono">${b.payload.base_model_hash.slice(0, 16)}...</td>
                    <td class="mono">${b.payload.update_hash.slice(0, 16)}...</td>
                    <td style="color:var(--verified); font-weight:600;">${b.payload.validation_status}</td>
                </tr>
            `).join("");

            document.querySelector("#ledgerTable tbody").innerHTML = data.ledger.slice().reverse().map(b => `
                <tr>
                    <td class="mono">#${b.block_index}</td>
                    <td class="mono" style="color:var(--sci-blue); font-weight:600;">${b.tx_type}</td>
                    <td class="mono">${b.payload.round ?? "—"}</td>
                    <td>${b.payload.org_name || b.payload.client_address || "Coordinator"}</td>
                    <td class="mono">${b.prev_block_hash.slice(0, 18)}...</td>
                    <td class="mono">${b.block_hash.slice(0, 18)}...</td>
                </tr>
            `).join("");

            document.querySelector("#edgeBenchTable tbody").innerHTML = data.edge_benchmark.map(r => `
                <tr>
                    <td><b>${r.Edge_Module}</b></td>
                    <td class="mono">${r.Size_KB} KB</td>
                    <td class="mono">${r.PyTorch_Latency_ms} ms</td>
                    <td class="mono" style="color:var(--verified); font-weight:600;">${r.ONNX_Runtime_Latency_ms} ms</td>
                    <td class="mono"><b>${r.Speedup_Factor}</b></td>
                </tr>
            `).join("");

            document.querySelector("#ablationFullTable tbody").innerHTML = data.ablation.map(r => `
                <tr>
                    <td><b>${r.Experiment}</b></td>
                    <td class="mono">${r.Trainable_Params.toLocaleString()}</td>
                    <td class="mono">${r.Optimal_Threshold ?? 0.5}</td>
                    <td class="mono">${r.Test_Loss}</td>
                    <td class="mono">${(r.Test_Acc * 100).toFixed(1)}%</td>
                    <td class="mono">${r.Test_F1}</td>
                    <td class="mono"><b>${r.Test_ROC_AUC}</b></td>
                    <td class="mono">${r.Latency_ms_per_Video} ms</td>
                </tr>
            `).join("");
        }

        const archNodes = [
            { id: "01", name: "01 · Input Layer", in: "Raw Media Bytes", out: "[1, T, 3, 224, 224]", src: "data/frame_sampler.py", latKey: "preprocessing", math: "$$I_t = \\text{Crop}_{224\\times 224}(\\text{Haar}(V_t))$$" },
            { id: "02", name: "02 · Feature Extraction", in: "[1, T, 3, 224, 224]", out: "z ∈ ℝ^768", src: "models/vit_detector.py", latKey: "vit_extraction", math: "$$z = \\frac{1}{T}\\sum_{t=1}^T \\text{ViT}_{\\text{CLS}}(I_t)$$" },
            { id: "03", name: "03 · Multimodal Engine", in: "I(x,y), z ∈ ℝ^768", out: "[P(r), B(ω), det(C)]", src: "models/anti_hallucination.py", latKey: "multimodal_forensics", math: "$$\\mathcal{B}(\\omega_1,\\omega_2), \\det(\\mathbf{C}_{\\text{RGB}}), D_{\\text{KL}}$$" },
            { id: "04", name: "04 · Modular Heads", in: "z ∈ ℝ^768", out: "Logits f(x) ∈ ℝ^2", src: "models/{quantum,spiking}_head.py", latKey: "classifier_head", math: "$$f(x) = W z + b \\text{ or } \\sum_{t=1}^{10} V_m(t)$$" },
            { id: "05", name: "05 · Distributed FL", in: "Local ΔW_k", out: "Global W_{t+1}", src: "federated/client.py", latKey: "classifier_head", math: "$$W_{t+1} = \\sum_{k=1}^3 \\frac{n_k}{N} W_{t+1}^{(k)}$$" },
            { id: "06", name: "06 · Security Plane", in: "ΔW_k, SHA-256", out: "Approved / Rejected", src: "blockchain/audit.py", latKey: "preprocessing", math: "$$\\|W_k - W_t\\|_2 \\le \\tau_{\\max}$$" },
            { id: "07", name: "07 · Edge Optimization", in: "PyTorch Graph", out: "ONNX Opset 17", src: "scripts/export_and_benchmark_edge.py", latKey: "classifier_head", math: "$$t_{\\text{ONNX}} = 0.0576\\text{ ms (23.14}\\times)$$" },
            { id: "08", name: "08 · Calibration", in: "Logits f(x)", out: "Verdict ∈ {Auth, Susp, Inc}", src: "training/calibration.py", latKey: "classifier_head", math: "$$p_{\\text{cal}} = \\sigma(s(\\text{logit}(p) - \\text{logit}(\\tau^*)))$$" }
        ];

        let threeMeshes = [];
        let pulseActive = 0;

        function selectArchNode(idx) {
            const n = archNodes[idx];
            document.getElementById("nodeInspId").textContent = "NODE " + n.id;
            document.getElementById("nodeInspName").textContent = n.name;
            document.getElementById("nodeInspIn").textContent = n.in;
            document.getElementById("nodeInspOut").textContent = n.out;
            document.getElementById("nodeInspSrc").textContent = n.src;
            document.getElementById("nodeInspLat").textContent = (latestLatencies[n.latKey] ?? "0.05") + " ms";
            const mathEl = document.getElementById("nodeInspMath");
            mathEl.innerHTML = n.math;
            if (window.renderMathInElement) {
                renderMathInElement(mathEl, { delimiters: [{left: "$$", right: "$$", display: true}] });
            }
            threeMeshes.forEach((m, i) => {
                m.material.color.setHex(i === idx ? 0x19856B : 0x4779A8);
            });
        }

        function trigger3DPulse() { pulseActive = 60; }

        function init3DArchitecture() {
            const container = document.getElementById("arch3dCanvas");
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xF0F3F7);
            const camera = new THREE.PerspectiveCamera(42, container.clientWidth / 440, 0.1, 1000);
            camera.position.set(0, 3.5, 13);
            camera.lookAt(0, 0, 0);

            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth || 680, 440);
            container.appendChild(renderer.domElement);

            const light = new THREE.DirectionalLight(0xffffff, 0.9);
            light.position.set(5, 10, 7);
            scene.add(light);
            scene.add(new THREE.AmbientLight(0xffffff, 0.6));

            const group = new THREE.Group();
            scene.add(group);

            const geom = new THREE.BoxGeometry(1.15, 0.85, 1.15);
            archNodes.forEach((n, idx) => {
                const mat = new THREE.MeshPhongMaterial({ color: idx === 1 ? 0x19856B : 0x4779A8 });
                const mesh = new THREE.Mesh(geom, mat);
                mesh.position.set((idx - 3.5) * 1.65, (idx % 2 === 0 ? 0.4 : -0.4), 0);
                group.add(mesh);
                threeMeshes.push(mesh);
            });

            const lineMat = new THREE.LineBasicMaterial({ color: 0x172B4D });
            const pts = threeMeshes.map(m => m.position.clone());
            const lineGeom = new THREE.BufferGeometry().setFromPoints(pts);
            group.add(new THREE.Line(lineGeom, lineMat));

            const btnBox = document.getElementById("nodeButtonsList");
            archNodes.forEach((n, idx) => {
                const b = document.createElement("button");
                b.className = "btn btn-secondary";
                b.style.marginTop = "0";
                b.style.padding = "5px 8px";
                b.style.fontSize = "11px";
                b.textContent = n.name;
                b.onclick = () => selectArchNode(idx);
                btnBox.appendChild(b);
            });

            let isDragging = false, prevX = 0;
            container.addEventListener("mousedown", (e) => { isDragging = true; prevX = e.clientX; });
            window.addEventListener("mouseup", () => { isDragging = false; });
            window.addEventListener("mousemove", (e) => {
                if (isDragging) {
                    group.rotation.y += (e.clientX - prevX) * 0.008;
                    prevX = e.clientX;
                }
            });

            function animate() {
                requestAnimationFrame(animate);
                if (pulseActive > 0) {
                    const activeIdx = Math.floor((60 - pulseActive) / 7.5);
                    threeMeshes.forEach((m, i) => {
                        const s = (i === activeIdx) ? 1.28 : 1.0;
                        m.scale.set(s, s, s);
                    });
                    pulseActive--;
                }
                renderer.render(scene, camera);
            }
            animate();
        }
    </script>
</body>
</html>"""
