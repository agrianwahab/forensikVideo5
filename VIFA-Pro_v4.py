# VIFA-Pro v4

# vifa_pro.py
# (Sistem Forensik Video Profesional dengan Analisis Multi-Lapis)
# VERSI 5 TAHAP PENELITIAN (DENGAN PERBAIKAN BUG STYLE REPORTLAB)
# VERSI PENINGKATAN METODE UTAMA (K-MEANS, LOCALIZATION) & PENDUKUNG (ELA, SIFT)
# VERSI REVISI DETAIL TAHAP 1 (METADATA, NORMALISASI FRAME, DETAIL K-MEANS)
# VERSI PENINGKATAN DETAIL TAHAP 2 (PLOT TEMPORAL K-MEANS, SSIM, OPTICAL FLOW)
# VERSI PENINGKATAN DETAIL TAHAP 3 (INVESTIGASI MENDALAM DAN PENJELASAN LENGKAP)
# VERSI PENINGKATAN DETAIL TAHAP 4 (LOCALIZATION TAMPERING ENHANCED, SKOR INTEGRITAS REALISTIS)
# VERSI SOLUSI: PENAMBAHAN KONTEKS UNTUK MENGURANGI FALSE POSITIVE

"""
VIFA-Pro: Sistem Forensik Video Profesional (Arsitektur 5 Tahap)
========================================================================================
Versi ini mengimplementasikan alur kerja forensik formal dalam 5 tahap yang jelas,
sesuai dengan metodologi penelitian untuk deteksi manipulasi video. Setiap tahap
memiliki tujuan spesifik, dari ekstraksi fitur dasar hingga validasi proses.

ARSITEKTUR PIPELINE:
- TAHAP 1: Pra-pemrosesan & Ekstraksi Fitur Dasar (Hashing, Frame, pHash, Warna)
           -> Metadata diekstrak secara mendalam.
           -> Frame diekstrak dan dinormalisasi warnanya untuk konsistensi analisis.
           -> Metode K-Means diterapkan untuk klasterisasi warna adegan dengan visualisasi detail.
- TAHAP 2: Analisis Anomali Temporal & Komparatif (Optical Flow, SSIM, K-Means Temporal, Baseline Check)
           -> Visualisasi Temporal yang lebih rinci untuk SSIM, Optical Flow, dan K-Means.
- TAHAP 3: Sintesis Bukti & Investigasi Mendalam (Korelasi Metrik, ELA & SIFT on-demand)
           -> ELA dan SIFT+RANSAC digunakan sebagai investigasi pendukung yang terukur.
           -> Analisis detail dengan penjelasan lengkap untuk setiap anomali.
           -> **[SOLUSI TA]** Menambahkan logika kontekstual untuk membedakan anomali alami (gerakan, adegan statis) dari manipulasi sebenarnya.
- TAHAP 4: Visualisasi & Penilaian Integritas (Plotting, Integrity Score)
           -> Localization Tampering menyatukan anomali menjadi peristiwa yang dapat diinterpretasikan.
           -> ENHANCED: Skor integritas realistis, visualisasi detail, penilaian pipeline
- TAHAP 5: Penyusunan Laporan & Validasi Forensik (Laporan PDF Naratif)

Deteksi:
- Diskontinuitas (Deletion/Insertion): Melalui Aliran Optik, SSIM, K-Means, dan Perbandingan Baseline.
- Duplikasi Frame (Duplication): Melalui pHash, dikonfirmasi oleh SIFT+RANSAC dan SSIM.
- Penyisipan Area (Splicing): Terindikasi oleh Analisis Tingkat Kesalahan (ELA) pada titik diskontinuitas.

Author: OpenAI-GPT & Anda
License: MIT
Dependencies: opencv-python, opencv-contrib-python, imagehash, numpy, Pillow,
              reportlab, matplotlib, tqdm, scikit-learn, scikit-image
"""

from __future__ import annotations
import argparse
import json
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any

# Pemeriksaan Dependensi Awal
try:
    import cv2
    import imagehash
    import numpy as np
    from PIL import Image, ImageChops, ImageEnhance, ImageDraw, ImageFont
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PlatypusImage, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from tqdm import tqdm
    from sklearn.cluster import KMeans
    from skimage.metrics import structural_similarity as ssim
    from scipy import stats
    import av
    from scipy.fftpack import dct
    import seaborn as sns
except ImportError as e:
    print(f"Error: Dependensi penting tidak ditemukan -> {e}")
    sys.exit(1)


# --- FIX START
PlatyplusImage = PlatypusImage
# --- FIX END


###############################################################################
# Utilitas & Konfigurasi Global
###############################################################################

# --- [SOLUSI VIDEO PERPUSTAKAAN] ---
# Ambang batas dinaikkan sedikit untuk membuat sistem lebih toleran terhadap
# gerakan cepat alami sebelum mengklasifikasikannya sebagai anomali.
CONFIG = {
    "KMEANS_CLUSTERS": 3,
    "KMEANS_SAMPLES_PER_CLUSTER": 3,
    "SSIM_DISCONTINUITY_DROP": 0.40,      # Ditingkatkan dari 0.35 menjadi 0.40
    "OPTICAL_FLOW_Z_THRESH": 7.0,       # Ditingkatkan dari 6.0 menjadi 7.0
    "DUPLICATION_SSIM_CONFIRM": 0.85,
    "SIFT_MIN_MATCH_COUNT": 10,
    "USE_AUTO_THRESHOLDS": True,
    "MOTION_VECTOR_Z_THRESH": 7.0,
    "PRNU_Z_THRESH": 3.5,
    "PRNU_FRAME_SAMPLES": 30,
    "HIST_CORREL_THRESHOLD": 0.98,
    "COMBINED_EVIDENCE_CONFIDENCE_THRESHOLD": 2
}
# --- [END OF SOLUSI] ---

class Icons:
    IDENTIFICATION="🔍"; PRESERVATION="🛡️"; COLLECTION="📥"; EXAMINATION="🔬";
    ANALYSIS="📈"; REPORTING="📄"; SUCCESS="✅"; ERROR="❌"; INFO="ℹ️";
    CONFIDENCE_LOW="🟩"; CONFIDENCE_MED="🟨"; CONFIDENCE_HIGH="🟧"; CONFIDENCE_VHIGH="🟥"

def log(message: str):
    print(message, file=sys.stdout)

def print_stage_banner(stage_number: int, stage_name: str, icon: str, description: str):
    width=80
    log("\n" + "="*width)
    log(f"=== {icon}  TAHAP {stage_number}: {stage_name.upper()} ".ljust(width - 3) + "===")
    log("="*width)
    log(f"{Icons.INFO}  {description}")
    log("-" * width)

###############################################################################
# Struktur Data Inti (DIPERLUAS UNTUK TAHAP 4 ENHANCED)
###############################################################################

@dataclass
class Evidence:
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    confidence: str = "N/A"
    ela_path: str | None = None
    sift_path: str | None = None
    detailed_analysis: dict = field(default_factory=dict)
    visualizations: dict = field(default_factory=dict)
    explanations: dict = field(default_factory=dict)

@dataclass
class FrameInfo:
    index: int
    timestamp: float
    img_path_original: str
    img_path: str
    img_path_comparison: str | None = None
    hash: str | None = None
    type: str = "original"
    ssim_to_prev: float | None = None
    optical_flow_mag: float | None = None
    motion_vector_mag: float | None = None
    prnu_correlation: float | None = None
    color_cluster: int | None = None
    evidence_obj: Evidence = field(default_factory=Evidence)
    histogram_data: np.ndarray | None = None
    edge_density: float | None = None
    blur_metric: float | None = None

@dataclass
class AnalysisResult:
    video_path: str
    preservation_hash: str
    metadata: dict
    frames: list[FrameInfo]
    fps: int
    prnu_reference: np.ndarray | None = None
    summary: dict = field(default_factory=dict)
    plots: dict = field(default_factory=dict)
    kmeans_artifacts: dict = field(default_factory=dict)
    localizations: list[dict] = field(default_factory=list)
    pdf_report_path: Optional[Path] = None
    detailed_anomaly_analysis: dict = field(default_factory=dict)
    statistical_summary: dict = field(default_factory=dict)
    integrity_analysis: dict = field(default_factory=dict)
    pipeline_assessment: dict = field(default_factory=dict)
    localization_details: dict = field(default_factory=dict)
    confidence_distribution: dict = field(default_factory=dict)

def generate_integrity_score(summary: dict, detailed_analysis: dict = None) -> tuple[int, str, dict]:
    pct = summary.get('pct_anomaly', 0)
    total_frames = summary.get('total_frames', 0)

    if pct == 0: base_score = 95
    elif pct < 2: base_score = 92
    elif pct < 5: base_score = 88
    elif pct < 10: base_score = 82
    elif pct < 15: base_score = 75
    elif pct < 25: base_score = 65
    elif pct < 35: base_score = 55
    else: base_score = 50

    adjustments = []

    if detailed_analysis:
        confidence_dist = detailed_analysis.get('confidence_distribution', {})
        very_high_count = confidence_dist.get('SANGAT TINGGI', 0)
        high_count = confidence_dist.get('TINGGI', 0)
        
        if very_high_count > 0:
            adjustments.append(('Anomali Kepercayaan Sangat Tinggi', -min(very_high_count * 5, 20)))
        if high_count > 0:
            adjustments.append(('Anomali Kepercayaan Tinggi', -min(high_count * 2, 10)))

    if detailed_analysis and detailed_analysis.get('temporal_clusters', 0) > 0 and detailed_analysis.get('average_anomalies_per_cluster', 0) < 3:
        adjustments.append(('Anomali Terisolasi (Gerakan Alami)', +5))
    if total_frames > 100:
        adjustments.append(('Sampel Frame Memadai', +3))

    blur_metrics = [f.blur_metric for f in summary.get('all_frames', []) if f.blur_metric is not None]
    if blur_metrics and np.mean(blur_metrics) < 100:
        adjustments.append(('Kualitas Video Rendah (Blurry)', +5))

    final_score = base_score
    for name, value in adjustments:
        final_score += value

    final_score = max(50, min(95, int(final_score)))

    if final_score >= 90: desc = "Sangat Baik - Integritas Tinggi"
    elif final_score >= 85: desc = "Baik - Integritas Terjaga"
    elif final_score >= 80: desc = "Cukup Baik - Anomali Minor"
    elif final_score >= 70: desc = "Sedang - Indikasi Anomali"
    elif final_score >= 60: desc = "Buruk - Manipulasi Terindikasi"
    else: desc = "Sangat Buruk - Manipulasi Signifikan"

    calculation_details = {
        'base_score': base_score, 'percentage_anomaly': pct, 'adjustments': adjustments, 'final_score': final_score,
        'scoring_method': 'Weighted Multi-Factor Analysis (v2)',
        'factors_considered': ['Persentase frame anomali', 'Tingkat kepercayaan anomali', 'Distribusi temporal anomali', 'Kualitas sampel analisis', 'Kualitas teknis video (blur)'],
        'description': desc
    }
    return final_score, desc, calculation_details

###############################################################################
# Fungsi Analisis Individual (EXISTING)
###############################################################################
# TIDAK ADA PERUBAHAN PADA BLOK FUNGSI INI
def perform_ela(image_path: Path, quality: int=90) -> tuple[Path, int, np.ndarray] | None:
    try:
        ela_dir = image_path.parent.parent / "ela_artifacts"; ela_dir.mkdir(exist_ok=True)
        out_path = ela_dir / f"{image_path.stem}_ela.jpg"; temp_jpg_path = out_path.with_name(f"temp_{out_path.name}")
        with Image.open(image_path).convert('RGB') as im: im.save(temp_jpg_path, 'JPEG', quality=quality)
        with Image.open(image_path).convert('RGB') as im_orig, Image.open(temp_jpg_path) as resaved_im: ela_im = ImageChops.difference(im_orig, resaved_im)
        if Path(temp_jpg_path).exists(): Path(temp_jpg_path).unlink()
        ela_array = np.array(ela_im)
        extrema = ela_im.getextrema()
        max_diff = max(ex[1] for ex in extrema) if extrema else 1
        scale = 255.0 / (max_diff if max_diff > 0 else 1)
        ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
        ela_with_grid = ela_im.copy(); draw = ImageDraw.Draw(ela_with_grid)
        width, height = ela_with_grid.size; grid_size = 50
        for x in range(0, width, grid_size): draw.line([(x, 0), (x, height)], fill=(128, 128, 128), width=1)
        for y in range(0, height, grid_size): draw.line([(0, y), (width, y)], fill=(128, 128, 128), width=1)
        ela_with_grid.save(out_path)
        return out_path, max_diff, ela_array
    except Exception as e:
        log(f"  {Icons.ERROR} Gagal ELA pada {image_path.name}: {e}"); return None

def analyze_ela_regions(ela_array: np.ndarray, grid_size: int = 50) -> dict:
    height, width = ela_array.shape[:2]; suspicious_regions = []
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            region = ela_array[y:min(y+grid_size, height), x:min(x+grid_size, width)]
            if region.size == 0: continue
            mean_val, std_val, max_val = np.mean(region), np.std(region), np.max(region)
            if mean_val > 30 or max_val > 100: suspicious_regions.append({'x': x, 'y': y, 'width': min(grid_size, width - x),'height': min(grid_size, height - y),'mean_ela': float(mean_val),'std_ela': float(std_val),'max_ela': float(max_val),'suspicion_level': 'high' if mean_val > 50 else 'medium'})
    return {'total_regions': (height // grid_size) * (width // grid_size),'suspicious_regions': suspicious_regions,'suspicious_count': len(suspicious_regions),'grid_size': grid_size}

def compare_sift_enhanced(img_path1: Path, img_path2: Path, out_dir: Path) -> dict:
    try:
        img1 = cv2.imread(str(img_path1), cv2.IMREAD_GRAYSCALE); img2 = cv2.imread(str(img_path2), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None: return {'success': False, 'error': 'Failed to load images'}
        sift = cv2.SIFT_create(); kp1, des1 = sift.detectAndCompute(img1, None); kp2, des2 = sift.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2: return {'success': False, 'error': 'Insufficient keypoints'}
        bf = cv2.BFMatcher(); matches = bf.knnMatch(des1, des2, k=2)
        if not matches or any(len(m) < 2 for m in matches): return {'success': False, 'error': 'No valid matches'}
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        result = {'success': True,'total_keypoints_img1': len(kp1),'total_keypoints_img2': len(kp2),'total_matches': len(matches),'good_matches': len(good_matches),'match_quality': 'excellent' if len(good_matches) > 100 else 'good' if len(good_matches) > 50 else 'fair' if len(good_matches) > 20 else 'poor'}
        if len(good_matches) > CONFIG["SIFT_MIN_MATCH_COUNT"]:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2); dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None and mask is not None:
                inlier_ratio = mask.ravel().sum() / len(good_matches) if len(good_matches) > 0 else 0.0; inliers = int(mask.ravel().sum()); det, scale = np.linalg.det(M[:2, :2]), np.sqrt(abs(np.linalg.det(M[:2, :2])))
                result.update({'inliers': inliers,'outliers': len(good_matches) - inliers,'inlier_ratio': float(inlier_ratio),'homography_determinant': float(det),'estimated_scale': float(scale),'transformation_type': 'rigid' if abs(scale - 1.0) < 0.1 else 'scaled' if 0.5 < scale < 2.0 else 'complex'})
                draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=mask.ravel().tolist(), flags=cv2.DrawMatchesFlags_DEFAULT)
                img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params); font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_matches, f'Total Matches: {len(good_matches)}', (10, 30), font, 0.8, (255, 255, 255), 2); cv2.putText(img_matches, f'Inliers: {inliers} ({inlier_ratio:.1%})', (10, 60), font, 0.8, (0, 255, 0), 2); cv2.putText(img_matches, f'Quality: {result["match_quality"].upper()}', (10, 90), font, 0.8, (255, 255, 0), 2)
                sift_dir = out_dir / "sift_artifacts"; sift_dir.mkdir(exist_ok=True)
                out_path = sift_dir / f"sift_detailed_{img_path1.stem}_vs_{img_path2.stem}.jpg"; cv2.imwrite(str(out_path), img_matches); result['visualization_path'] = str(out_path)
                heatmap, heatmap_path = create_match_heatmap(src_pts, dst_pts, img1.shape, img2.shape), sift_dir / f"sift_heatmap_{img_path1.stem}_vs_{img_path2.stem}.jpg"
                cv2.imwrite(str(heatmap_path), heatmap); result['heatmap_path'] = str(heatmap_path)
        return result
    except Exception as e:
        log(f"  {Icons.ERROR} Gagal SIFT: {e}"); return {'success': False, 'error': str(e)}

def create_match_heatmap(src_pts: np.ndarray, dst_pts: np.ndarray, shape1: tuple, shape2: tuple) -> np.ndarray:
    height, width = max(shape1[0], shape2[0]), shape1[1] + shape2[1] + 50
    heatmap = np.zeros((height, width, 3), dtype=np.uint8)
    for pt in src_pts: x, y = int(pt[0][0]), int(pt[0][1]); cv2.circle(heatmap, (x, y), 10, (255, 0, 0), -1)
    for pt in dst_pts: x, y = int(pt[0][0]) + shape1[1] + 50, int(pt[0][1]); cv2.circle(heatmap, (x, y), 10, (0, 0, 255), -1)
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
    return cv2.applyColorMap(cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)

def calculate_frame_metrics(frame_path: str) -> dict:
    try:
        img = cv2.imread(frame_path);
        if img is None: return {}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150); edge_density = np.sum(edges > 0) / edges.size; blur_metric = cv2.Laplacian(gray, cv2.CV_64F).var()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h, hist_s, hist_v = cv2.calcHist([hsv], [0], None, [180], [0, 180]), cv2.calcHist([hsv], [1], None, [256], [0, 256]), cv2.calcHist([hsv], [2], None, [256], [0, 256])
        if hist_h.sum() > 0: hist_h = hist_h.flatten() / hist_h.sum()
        if hist_s.sum() > 0: hist_s = hist_s.flatten() / hist_s.sum()
        if hist_v.sum() > 0: hist_v = hist_v.flatten() / hist_v.sum()
        h_entropy, s_entropy, v_entropy = -np.sum(hist_h[hist_h > 0] * np.log2(hist_h[hist_h > 0])), -np.sum(hist_s[hist_s > 0] * np.log2(hist_s[hist_s > 0])), -np.sum(hist_v[hist_v > 0] * np.log2(hist_v[hist_v > 0]))
        return {'edge_density': float(edge_density), 'blur_metric': float(blur_metric), 'color_entropy': {'hue': float(h_entropy), 'saturation': float(s_entropy), 'value': float(v_entropy)}}
    except Exception as e:
        log(f"  {Icons.ERROR} Error calculating frame metrics: {e}"); return {}

def calculate_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256();
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""): sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def ffprobe_metadata(video_path: Path) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video_path)]
    try: result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8'); return json.loads(result.stdout)
    except Exception as e: log(f"FFprobe error: {e}"); return {}

def parse_ffprobe_output(metadata: dict) -> dict:
    parsed = {};
    if 'format' in metadata:
        fmt = metadata['format']
        parsed['Format'] = {'Filename': Path(fmt.get('filename', 'N/A')).name, 'Format Name': fmt.get('format_long_name', 'N/A'), 'Duration': f"{float(fmt.get('duration', 0)):.3f} s", 'Size': f"{int(fmt.get('size', 0)) / (1024*1024):.2f} MB", 'Bit Rate': f"{int(fmt.get('bit_rate', 0)) / 1000:.0f} kb/s", 'Creation Time': fmt.get('tags', {}).get('creation_time', 'N/A')}
    video_streams = [s for s in metadata.get('streams', []) if s.get('codec_type') == 'video']
    if video_streams:
        stream = video_streams[0]
        parsed['Video Stream'] = {'Codec': stream.get('codec_name', 'N/A').upper(), 'Profile': stream.get('profile', 'N/A'), 'Resolution': f"{stream.get('width')}x{stream.get('height')}", 'Aspect Ratio': stream.get('display_aspect_ratio', 'N/A'), 'Pixel Format': stream.get('pix_fmt', 'N/A'), 'Frame Rate': f"{eval(stream.get('r_frame_rate', '0/1')):.2f} FPS", 'Bitrate': f"{int(stream.get('bit_rate', 0)) / 1000:.0f} kb/s" if 'bit_rate' in stream else 'N/A', 'Encoder': stream.get('tags', {}).get('encoder', 'N/A')}
    return parsed

# -- PRNU and Motion Vector Utilities --
def extract_noise_residual(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray_f, (3, 3), 0)
    residual = gray_f - blur
    residual -= residual.mean()
    return residual

def compute_prnu_reference(frames: list[FrameInfo], sample_size: int) -> np.ndarray | None:
    indices = [f.index for f in frames if f.color_cluster is not None]
    if not indices:
        return None
    cluster = Counter([frames[i].color_cluster for i in indices]).most_common(1)[0][0]
    selected = [f for f in frames if f.color_cluster == cluster][:sample_size]
    residuals = []
    for f in selected:
        img = cv2.imread(f.img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            residuals.append(extract_noise_residual(img))
    return np.mean(residuals, axis=0) if residuals else None

def compute_prnu_correlation(gray: np.ndarray, reference: np.ndarray) -> float:
    if reference is None:
        return 0.0
    res = extract_noise_residual(gray)
    corr = np.corrcoef(res.flatten(), reference.flatten())[0, 1]
    return float(corr)

def extract_motion_vectors(video_path: Path, fps: int, total_frames: int) -> list[float]:
    mv_list = []
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        time_base = stream.time_base
        targets = [i / fps for i in range(total_frames)]
        t_idx = 0
        for frame in container.decode(stream):
            ts = frame.pts * time_base if frame.pts is not None else 0
            if t_idx < len(targets) and ts >= targets[t_idx]:
                mv = frame.side_data.get('MOTION_VECTORS')
                if mv:
                    mags = [np.hypot(m.dst_x - m.src_x, m.dst_y - m.src_y) for m in mv]
                    mv_list.append(float(np.mean(mags)) if mags else 0.0)
                else:
                    mv_list.append(0.0)
                t_idx += 1
            if t_idx >= len(targets):
                break
        container.close()
    except Exception:
        mv_list = [0.0] * total_frames
    while len(mv_list) < total_frames:
        mv_list.append(0.0)
    return mv_list

def create_prnu_correlation_plot(frames: list[FrameInfo], out_dir: Path) -> Path:
    vals = [f.prnu_correlation for f in frames if f.prnu_correlation is not None]
    idxs = [f.index for f in frames if f.prnu_correlation is not None]
    if not vals:
        return Path()
    plt.figure(figsize=(10, 4))
    plt.plot(idxs, vals, marker='o')
    plt.title('Korelasi PRNU per Frame')
    plt.xlabel('Frame')
    plt.ylabel('Korelasi')
    plt.grid(True, alpha=0.3)
    out_path = out_dir / 'prnu_correlation.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path

def extract_frames_with_normalization(video_path: Path, out_dir: Path, fps: int) -> list[tuple[str, str, str]] | None:
    original_dir, normalized_dir, comparison_dir = out_dir / "frames_original", out_dir / "frames_normalized", out_dir / "frames_comparison"
    original_dir.mkdir(parents=True, exist_ok=True); normalized_dir.mkdir(parents=True, exist_ok=True); comparison_dir.mkdir(parents=True, exist_ok=True)
    try:
        cap = cv2.VideoCapture(str(video_path));
        if not cap.isOpened(): log(f"  {Icons.ERROR} Gagal membuka file video: {video_path}"); return None
        video_fps_raw = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps_raw or video_fps_raw <= 0: log(f"  ⚠️ Peringatan: Gagal membaca FPS video. Menggunakan asumsi (30)."); video_fps = 30.0
        else: video_fps = video_fps_raw
        frame_paths, frame_count, extracted_count = [], 0, 0
        time_increment, next_extraction_time = 1.0 / float(fps), 0.0
        pbar_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if pbar_total <= 0: pbar_total = None
        pbar = tqdm(total=pbar_total, desc="    Ekstraksi & Normalisasi", leave=False, bar_format='{l_bar}{bar}{r_bar}')
        while cap.isOpened():
            ret, frame = cap.read();
            if not ret: break
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if cap.get(cv2.CAP_PROP_POS_MSEC) > 0 else frame_count / video_fps
            if current_time >= next_extraction_time:
                original_path, normalized_path = original_dir/f"frame_{extracted_count:06d}_orig.jpg", normalized_dir/f"frame_{extracted_count:06d}_norm.jpg"
                cv2.imwrite(str(original_path), frame)
                ycrcb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb); ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0]); normalized_frame = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR); cv2.imwrite(str(normalized_path), normalized_frame)
                h, w, _ = frame.shape; comparison_img = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8); comparison_img[:, :w], comparison_img[:, w+10:] = frame, normalized_frame
                cv2.putText(comparison_img, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2); cv2.putText(comparison_img, 'Normalized', (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                comparison_path = comparison_dir/f"frame_{extracted_count:06d}_comp.jpg"; cv2.imwrite(str(comparison_path), comparison_img)
                frame_paths.append((str(original_path), str(normalized_path), str(comparison_path))); extracted_count += 1
                next_extraction_time += time_increment
            frame_count += 1; pbar.update(1)
        pbar.close(); cap.release()
        return frame_paths
    except Exception as e:
        log(f"  {Icons.ERROR} Error saat ekstraksi frame: {e}"); return None

###############################################################################
# FUNGSI TAMBAHAN UNTUK TAHAP 4 ENHANCED
###############################################################################
# TIDAK ADA PERUBAHAN PADA BLOK FUNGSI INI
def assess_pipeline_performance(result: AnalysisResult) -> dict:
    assessment = {'tahap_1': {'nama': 'Pra-pemrosesan & Ekstraksi Fitur', 'status': 'completed', 'quality_score': 0, 'metrics': {}, 'issues': []}, 'tahap_2': {'nama': 'Analisis Anomali Temporal', 'status': 'completed', 'quality_score': 0, 'metrics': {}, 'issues': []}, 'tahap_3': {'nama': 'Sintesis Bukti & Investigasi', 'status': 'completed', 'quality_score': 0, 'metrics': {}, 'issues': []}, 'tahap_4': {'nama': 'Visualisasi & Penilaian', 'status': 'in_progress', 'quality_score': 0, 'metrics': {}, 'issues': []}}
    if result.frames:
        total_frames = len(result.frames); frames_with_hash = sum(1 for f in result.frames if f.hash); frames_with_cluster = sum(1 for f in result.frames if f.color_cluster is not None)
        assessment['tahap_1']['metrics'] = {'total_frames_extracted': total_frames, 'hash_coverage': f"{frames_with_hash/total_frames*100:.1f}%" if total_frames > 0 else "0%", 'clustering_coverage': f"{frames_with_cluster/total_frames*100:.1f}%" if total_frames > 0 else "0%", 'metadata_completeness': len(result.metadata) > 0}
        assessment['tahap_1']['quality_score'] = round((frames_with_hash/total_frames + (frames_with_cluster/total_frames if total_frames > 0 else 0)) / 2 * 100) if total_frames > 0 else 0
        if frames_with_hash < total_frames: assessment['tahap_1']['issues'].append('Beberapa frame gagal di-hash')
    total_frames = len(result.frames) if result.frames else 0; frames_with_ssim = sum(1 for f in result.frames if f.ssim_to_prev is not None); frames_with_flow = sum(1 for f in result.frames if f.optical_flow_mag is not None)
    assessment['tahap_2']['metrics'] = {'ssim_coverage': f"{frames_with_ssim/total_frames*100:.1f}%" if total_frames > 0 else "0%", 'optical_flow_coverage': f"{frames_with_flow/total_frames*100:.1f}%" if total_frames > 0 else "0%", 'temporal_metrics_computed': frames_with_ssim > 0 and frames_with_flow > 0}
    if total_frames > 0: assessment['tahap_2']['quality_score'] = round(((frames_with_ssim / total_frames if total_frames > 0 else 0) + (frames_with_flow / total_frames if total_frames > 0 else 0)) / 2 * 100)
    anomaly_count = sum(1 for f in result.frames if f.type.startswith('anomaly')); evidence_count = sum(1 for f in result.frames if f.evidence_obj.reasons and f.evidence_obj.reasons != [])
    assessment['tahap_3']['metrics'] = {'anomalies_detected': anomaly_count, 'evidence_collected': evidence_count, 'ela_analyses': sum(1 for f in result.frames if f.evidence_obj.ela_path is not None), 'sift_analyses': sum(1 for f in result.frames if f.evidence_obj.sift_path is not None)}
    if anomaly_count > 0: assessment['tahap_3']['quality_score'] = min(100, round(evidence_count / anomaly_count * 100))
    assessment['tahap_4']['metrics'] = {'localizations_created': len(result.localizations), 'plots_generated': len(result.plots), 'integrity_calculated': 'integrity_analysis' in result.__dict__}
    assessment['tahap_4']['quality_score'] = 100 if result.localizations else 0
    return assessment

def create_enhanced_localization_map(result: AnalysisResult, out_dir: Path) -> Path:
    fig = plt.figure(figsize=(20, 12)); gs = fig.add_gridspec(4, 3, height_ratios=[1, 2, 1, 1], hspace=0.3, wspace=0.2); ax_title = fig.add_subplot(gs[0, :]); ax_title.text(0.5, 0.5, 'PETA DETAIL LOKALISASI TAMPERING', ha='center', va='center', fontsize=20, weight='bold'); ax_title.axis('off'); ax_timeline = fig.add_subplot(gs[1, :]); total_frames = len(result.frames); frame_indices = list(range(total_frames)); ax_timeline.axhspan(0, 1, facecolor='lightgreen', alpha=0.3, label='Normal'); anomaly_types = {'anomaly_duplication': {'color': '#FF6B6B', 'height': 0.8, 'label': 'Duplikasi', 'marker': 'o'}, 'anomaly_insertion': {'color': '#4ECDC4', 'height': 0.7, 'label': 'Penyisipan', 'marker': 's'}, 'anomaly_discontinuity': {'color': '#45B7D1', 'height': 0.6, 'label': 'Diskontinuitas', 'marker': '^'}}
    for loc in result.localizations:
        event_type = loc['event']
        if event_type in anomaly_types:
            style, start_idx, end_idx = anomaly_types[event_type], loc['start_frame'], loc['end_frame']; rect = plt.Rectangle((start_idx, 0), end_idx - start_idx + 1, style['height'], facecolor=style['color'], alpha=0.6, edgecolor='black', linewidth=2); ax_timeline.add_patch(rect)
            conf_color = 'red' if loc['confidence'] == 'SANGAT TINGGI' else 'orange' if loc['confidence'] == 'TINGGI' else 'yellow'
            ax_timeline.plot((start_idx + end_idx) / 2, style['height'] + 0.05, marker='*', markersize=15, color=conf_color, markeredgecolor='black')
    ax_timeline.set_xlim(0, total_frames); ax_timeline.set_ylim(0, 1); ax_timeline.set_xlabel('Indeks Frame', fontsize=14); ax_timeline.set_title('Timeline Anomali Terdeteksi', fontsize=16, pad=20)
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=s['color'], alpha=0.6, label=s['label']) for s in anomaly_types.values()] + [plt.Line2D([0], [0], marker='*', color='red', markersize=10, label='Kepercayaan Tinggi', linestyle='None')]
    ax_timeline.legend(handles=legend_elements, loc='upper right', fontsize=12); ax_timeline.grid(True, axis='x', alpha=0.3)
    ax_stats = fig.add_subplot(gs[2, 0]); stats_text = f"STATISTIK ANOMALI\n\nTotal Frame: {total_frames}\nAnomali Terdeteksi: {sum(1 for f in result.frames if f.type.startswith('anomaly'))}\nPeristiwa Terlokalisasi: {len(result.localizations)}\n\nDistribusi Kepercayaan:\n- Sangat Tinggi: {sum(1 for l in result.localizations if l['confidence'] == 'SANGAT TINGGI')}\n- Tinggi: {sum(1 for l in result.localizations if l['confidence'] == 'TINGGI')}\n- Sedang: {sum(1 for l in result.localizations if l['confidence'] == 'SEDANG')}\n- Rendah: {sum(1 for l in result.localizations if l['confidence'] == 'RENDAH')}"
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, fontsize=11, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)); ax_stats.axis('off')
    ax_details = fig.add_subplot(gs[2, 1:]); details_text = "DETAIL PERISTIWA SIGNIFIKAN\n\n"; significant_events = sorted(result.localizations, key=lambda x: (x.get('confidence') == 'SANGAT TINGGI', x['end_frame'] - x['start_frame']), reverse=True)[:5]
    for i, event in enumerate(significant_events): details_text += f"{i+1}. {event['event'].replace('anomaly_', '').capitalize()} @ {event['start_ts']:.1f}s-{event['end_ts']:.1f}s (Durasi: {event['end_ts'] - event['start_ts']:.1f}s, Kepercayaan: {event.get('confidence', 'N/A')})\n"
    ax_details.text(0.05, 0.95, details_text, transform=ax_details.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)); ax_details.axis('off')
    ax_pie = fig.add_subplot(gs[3, 0]); confidence_counts = Counter(loc.get('confidence', 'N/A') for loc in result.localizations)
    if confidence_counts:
        colors_conf = {'SANGAT TINGGI': '#FF0000', 'TINGGI': '#FFA500', 'SEDANG': '#FFFF00', 'RENDAH': '#00FF00', 'N/A': '#808080'}; pie_colors = [colors_conf.get(c, '#808080') for c in confidence_counts.keys()]
        ax_pie.pie(confidence_counts.values(), labels=list(confidence_counts.keys()), colors=pie_colors, autopct='%1.1f%%', startangle=90); ax_pie.set_title('Distribusi Tingkat Kepercayaan', fontsize=12)
    else: ax_pie.text(0.5, 0.5, 'Tidak ada anomali', ha='center', va='center'); ax_pie.set_xlim(0, 1); ax_pie.set_ylim(0, 1)
    ax_cluster = fig.add_subplot(gs[3, 1:]); window_size = total_frames // 20 if total_frames > 20 else 1; density = np.zeros(total_frames)
    for f in result.frames:
        if f.type.startswith('anomaly'): start, end = max(0, f.index - window_size // 2), min(total_frames, f.index + window_size // 2); density[start:end] += 1
    ax_cluster.fill_between(frame_indices, density, alpha=0.5, color='red'); ax_cluster.set_xlabel('Indeks Frame', fontsize=12); ax_cluster.set_ylabel('Kepadatan Anomali', fontsize=12); ax_cluster.set_title('Analisis Kepadatan Temporal Anomali', fontsize=12); ax_cluster.grid(True, alpha=0.3)
    enhanced_map_path = out_dir / f"enhanced_localization_map_{Path(result.video_path).stem}.png"; plt.savefig(enhanced_map_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(); return enhanced_map_path

def create_integrity_breakdown_chart(integrity_details: dict, out_dir: Path, video_path: str) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 2]}); base_score, adjustments, final_score = integrity_details['base_score'], integrity_details['adjustments'], integrity_details['final_score']
    categories, values, cumulative = ['Base Score'], [base_score], base_score
    for name, value in adjustments: categories.append(name.replace(' (Non-Sistemik)', '\n(Non-Sistemik)')); values.append(value); cumulative += value
    categories.append('Final Score'); values.append(cumulative)
    bottoms = np.cumsum(values) - values; colors = ['blue'] + ['green' if v > 0 else 'red' for v in values[1:-1]] + ['gold']
    for i, (cat, val, col, bot) in enumerate(zip(categories, values, colors, bottoms)):
        if cat == 'Final Score': ax1.bar(i, val, color=col, bottom=0, alpha=0.9, width=0.6)
        else: ax1.bar(i, val, color=col, bottom=bot, alpha=0.7, width=0.6)
    for i, (cat, val, bot) in enumerate(zip(categories, values, bottoms)): y_pos, text, fw, fs = (val/2, f'{val:.0f}', 'bold', 14) if cat == 'Final Score' else (bot + val/2, f'{val:+.0f}' if i > 0 else f'{val:.0f}', 'bold', 10); ax1.text(i, y_pos, text, ha='center', va='center', fontweight=fw, fontsize=fs, color='black')
    ax1.set_xticks(range(len(categories))); ax1.set_xticklabels(categories, rotation=45, ha='right'); ax1.set_ylabel('Skor Integritas', fontsize=12); ax1.set_title('Breakdown Perhitungan Skor Integritas', fontsize=14); ax1.set_ylim(0, 100); ax1.grid(True, axis='y', alpha=0.3)
    ax1.axhline(y=80, color='darkgreen', linestyle='--', alpha=0.7, label='Target Minimal (80)'); ax1.legend()
    ax2.axis('off'); explanation_text = f"PENJELASAN PERHITUNGAN SKOR\n{'='*28}\n1. SKOR DASAR ({base_score:.0f}%)\n   Berdasarkan persentase frame anomali.\n   (0% = 95, <5% = 90, <10% = 85, etc.)\n\n2. FAKTOR PENYESUAIAN:\n"
    for name, value in adjustments: explanation_text += f"\n   {'✅' if value > 0 else '❌'} {name}: {value:+.0f}%"
    explanation_text += f"\n{'='*28}\n3. SKOR AKHIR: {final_score:.0f}%\n   Kategori: {integrity_details.get('description', 'N/A')}\n\nINTERPRETASI SKOR:\n{'-'*20}\n- 90-95%: Sangat Baik\n- 85-89%: Baik\n- 80-84%: Cukup Baik\n- 70-79%: Sedang\n- <70%:  Buruk\n"
    ax2.text(0.05, 0.95, explanation_text, transform=ax2.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    plt.suptitle('Analisis Detail Skor Integritas Video', fontsize=16, fontweight='bold'); plt.tight_layout(rect=[0, 0, 1, 0.96])
    integrity_chart_path = out_dir / f"integrity_breakdown_{Path(video_path).stem}.png"; plt.savefig(integrity_chart_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(); return integrity_chart_path

def create_anomaly_explanation_infographic(result: AnalysisResult, out_dir: Path) -> Path:
    fig = plt.figure(figsize=(16, 10)); fig.suptitle('PANDUAN MEMAHAMI ANOMALI VIDEO', fontsize=20, fontweight='bold')
    anomaly_info = {'Duplikasi': {'icon': '🔁', 'color': '#FF6B6B', 'simple': 'Frame yang sama diulang beberapa kali', 'technical': 'Deteksi melalui perbandingan hash dan SIFT', 'implication': 'Memperpanjang durasi atau menyembunyikan penghapusan', 'example': 'Memfotokopi halaman yang sama'},'Diskontinuitas': {'icon': '✂️', 'color': '#45B7D1', 'simple': 'Terjadi "lompatan" dalam aliran video', 'technical': 'Penurunan SSIM & lonjakan optical flow', 'implication': 'Indikasi pemotongan atau penyambungan kasar', 'example': 'Halaman hilang dalam buku'},'Penyisipan': {'icon': '➕', 'color': '#4ECDC4', 'simple': 'Frame baru yang tidak ada di video asli', 'technical': 'Perbandingan dengan baseline', 'implication': 'Konten tambahan mengubah narasi', 'example': 'Menambahkan halaman baru ke buku'}}
    gs = fig.add_gridspec(len(anomaly_info), 1, hspace=0.3, wspace=0.2)
    for idx, (atype, info) in enumerate(anomaly_info.items()):
        ax = fig.add_subplot(gs[idx]); ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=info['color'], alpha=0.1, zorder=0))
        ax.text(0.02, 0.85, f"{info['icon']} {atype.upper()}", transform=ax.transAxes, fontsize=18, fontweight='bold', bbox=dict(boxstyle='round', facecolor=info['color'], alpha=0.3))
        ax.text(0.02, 0.65, "Apa itu?", transform=ax.transAxes, fontsize=12, fontweight='bold'); ax.text(0.02, 0.45, info['simple'], transform=ax.transAxes, fontsize=11, wrap=True, va='top'); ax.text(0.02, 0.25, "Analogi:", transform=ax.transAxes, fontsize=12, fontweight='bold'); ax.text(0.02, 0.05, info['example'], transform=ax.transAxes, fontsize=11, fontstyle='italic', va='top'); ax.text(0.52, 0.65, "Cara Deteksi:", transform=ax.transAxes, fontsize=12, fontweight='bold'); ax.text(0.52, 0.45, info['technical'], transform=ax.transAxes, fontsize=11, va='top'); ax.text(0.52, 0.25, "Implikasi:", transform=ax.transAxes, fontsize=12, fontweight='bold'); ax.text(0.52, 0.05, info['implication'], transform=ax.transAxes, fontsize=11, va='top')
        count = sum(1 for loc in result.localizations if atype.lower() in loc.get('event', '').lower()); ax.text(0.98, 0.85, f"Ditemukan: {count}", transform=ax.transAxes, fontsize=14, ha='right', fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96]); infographic_path = out_dir / f"anomaly_explanation_{Path(result.video_path).stem}.png"; plt.savefig(infographic_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(); return infographic_path


###############################################################################
# PIPELINE 5-TAHAP
###############################################################################
def create_anomaly_summary_visualization(result: AnalysisResult, out_dir: Path):
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Ringkasan Analisis Forensik Video', fontsize=16, fontweight='bold')
        anomaly_types_summary = result.statistical_summary.get('anomaly_types', {})
        if anomaly_types_summary:
            labels = [t.replace('anomaly_', '').title() for t in anomaly_types_summary.keys()]; sizes = list(anomaly_types_summary.values()); colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90); ax1.set_title('Distribusi Jenis Anomali'); ax1.axis('equal')
        else:
            ax1.text(0.5, 0.5, 'Tidak ada anomali terdeteksi', ha='center', va='center'); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis('off')
        confidence_dist_summary = result.statistical_summary.get('confidence_distribution', {})
        if confidence_dist_summary:
            confidence_labels = list(confidence_dist_summary.keys()); confidence_values = list(confidence_dist_summary.values())
            colors_conf = {'RENDAH': 'green', 'SEDANG': 'yellow', 'TINGGI': 'orange', 'SANGAT TINGGI': 'red', 'N/A': 'gray', 'INFORMASI': 'lightblue'}
            bar_colors = [colors_conf.get(label, 'gray') for label in confidence_labels]
            ax2.bar(confidence_labels, confidence_values, color=bar_colors); ax2.set_title('Distribusi Tingkat Kepercayaan Anomali'); ax2.set_xlabel('Tingkat Kepercayaan'); ax2.set_ylabel('Jumlah Anomali')
        anomaly_times, anomaly_types_list = [], []
        for f in result.frames:
            if f.type.startswith("anomaly"):
                anomaly_times.append(f.timestamp); anomaly_types_list.append(f.type.replace('anomaly_', ''))
        if anomaly_times:
            type_colors = {'discontinuity': 'purple', 'duplication': 'orange', 'insertion': 'red'}
            for atype in set(anomaly_types_list):
                times = [t for t, at in zip(anomaly_times, anomaly_types_list) if at == atype]
                ax3.scatter(times, [1]*len(times), label=atype.title(), color=type_colors.get(atype, 'gray'), s=100, alpha=0.7)
            ax3.set_title('Timeline Anomali'); ax3.set_xlabel('Waktu (detik)'); ax3.set_ylim(0.5, 1.5); ax3.set_yticks([]); ax3.legend(); ax3.grid(True, axis='x', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Tidak ada timeline anomali', ha='center', va='center'); ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis('off')
        stats = result.statistical_summary; total_frames = stats.get('total_frames_analyzed', 1) or 1; total_anomalies = stats.get('total_anomalies', 0)
        stats_text = f"Total Frame Dianalisis: {total_frames}\nTotal Anomali Terdeteksi: {total_anomalies}\nPersentase Anomali: {total_anomalies * 100 / total_frames:.1f}%\nKluster Temporal: {stats.get('temporal_clusters', 'N/A')}\nRata-rata Anomali per Kluster: {stats.get('average_anomalies_per_cluster', 0):.1f}"
        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_xlim(0, 1); ax4.set_ylim(0, 1); ax4.axis('off'); ax4.set_title('Statistik Ringkasan')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); summary_path = out_dir / "anomaly_summary.png"; plt.savefig(summary_path, dpi=150, bbox_inches='tight'); plt.close()
        result.plots['anomaly_summary'] = str(summary_path)
    except Exception as e:
        log(f"  {Icons.ERROR} Error creating summary visualization: {e}")

def run_tahap_1_pra_pemrosesan(video_path: Path, out_dir: Path, fps: int) -> AnalysisResult | None:
    print_stage_banner(1, "Pra-pemrosesan & Ekstraksi Fitur Dasar", Icons.COLLECTION, "Mengamankan bukti, mengekstrak metadata, menormalisasi frame, dan menerapkan metode K-Means.")
    preservation_hash = calculate_sha256(video_path); log(f"  ✅ Hash SHA-256: {preservation_hash}")
    metadata_raw = ffprobe_metadata(video_path); metadata = parse_ffprobe_output(metadata_raw)
    frames_dir_root = out_dir / f"frames_{video_path.stem}"; extracted_paths = extract_frames_with_normalization(video_path, frames_dir_root, fps)
    if not extracted_paths: log(f"  {Icons.ERROR} Gagal mengekstrak frame."); return None
    log(f"  ✅ {len(extracted_paths)} set frame berhasil diekstrak.")
    frames = []
    for idx, (p_orig, p_norm, p_comp) in enumerate(tqdm(extracted_paths, desc="    pHash", leave=False)):
        try:
            with Image.open(p_norm) as img:
                frame_hash = str(imagehash.average_hash(img))
            frames.append(FrameInfo(index=idx, timestamp=idx / fps, img_path_original=p_orig, img_path=p_norm, img_path_comparison=p_comp, hash=frame_hash ))
        except Exception as e:
            log(f"  {Icons.ERROR} Gagal memproses frame set {idx}: {e}")
    histograms = []
    for f in tqdm(frames, desc="    Histogram", leave=False):
        img = cv2.imread(f.img_path)
        if img is None:
            continue
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        f.histogram_data = hist
        metrics = calculate_frame_metrics(f.img_path)
        f.edge_density = metrics.get("edge_density")
        f.blur_metric = metrics.get("blur_metric")
        histograms.append(hist.flatten())
    if histograms and float(np.var(histograms)) < 0.15: CONFIG["KMEANS_CLUSTERS"] = 5
    kmeans_artifacts = {}
    if histograms:
        actual_n_clusters = min(CONFIG["KMEANS_CLUSTERS"], len(histograms))
        if actual_n_clusters >= 2:
            kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto').fit(histograms); labels = kmeans.labels_.tolist()
            for f, label in zip(frames, labels): f.color_cluster = int(label)
            log(f"  -> Klasterisasi K-Means selesai. {len(frames)} frame dikelompokkan ke dalam {actual_n_clusters} klaster.")
            kmeans_dir = out_dir / "kmeans_artifacts"; kmeans_dir.mkdir(exist_ok=True); cluster_counts = Counter(labels)
            plt.figure(figsize=(10, 5)); plt.bar(list(cluster_counts.keys()), list(cluster_counts.values())); plt.title('Distribusi Frame per Klaster K-Means'); plt.xlabel('Nomor Klaster'); plt.ylabel('Jumlah Frame');
            dist_path = kmeans_dir / "kmeans_distribution.png"; plt.savefig(dist_path, bbox_inches="tight"); plt.close(); kmeans_artifacts['distribution_plot_path'] = str(dist_path)
            kmeans_artifacts['clusters'] = []
            for i in range(actual_n_clusters):
                cluster_indices = [idx for idx, label in enumerate(labels) if label == i]
                if not cluster_indices:
                    continue
            log(f"  -> Artefak K-Means berhasil dibuat.")

    prnu_ref = compute_prnu_reference(frames, CONFIG["PRNU_FRAME_SAMPLES"])
    if prnu_ref is not None:
        prnu_img = cv2.normalize(prnu_ref, None, 0, 255, cv2.NORM_MINMAX)
        prnu_path = out_dir / "prnu_reference.png"
        cv2.imwrite(str(prnu_path), prnu_img.astype(np.uint8))
        kmeans_artifacts["prnu_reference_path"] = str(prnu_path)

    result = AnalysisResult(video_path=str(video_path), preservation_hash=preservation_hash, metadata=metadata, frames=frames, fps=fps, prnu_reference=prnu_ref, kmeans_artifacts=kmeans_artifacts)
    log(f"  {Icons.SUCCESS} Tahap 1 Selesai."); return result

def run_tahap_2_analisis_temporal(result: AnalysisResult, baseline_result: AnalysisResult | None = None):
    print_stage_banner(2, "Analisis Anomali Temporal & Komparatif", Icons.ANALYSIS, "Menganalisis aliran optik, SSIM, dan perbandingan dengan baseline jika ada.")
    frames = result.frames
    prev_gray = None
    mv_values = extract_motion_vectors(Path(result.video_path), result.fps, len(frames))
    for f, mv in zip(frames, mv_values):
        f.motion_vector_mag = mv
    for f in tqdm(frames, desc="    Temporal Analysis", leave=False):
        current_gray = cv2.imread(f.img_path, cv2.IMREAD_GRAYSCALE)
        if current_gray is not None:
            if prev_gray is not None and prev_gray.shape == current_gray.shape:
                data_range = float(current_gray.max() - current_gray.min()); f.ssim_to_prev = float(ssim(prev_gray, current_gray, data_range=data_range)) if data_range > 0 else 1.0
                try:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0); mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1]); f.optical_flow_mag = float(np.mean(mag))
                except cv2.error: f.optical_flow_mag = 0.0
            else:
                f.ssim_to_prev = 1.0
                f.optical_flow_mag = 0.0
        if result.prnu_reference is not None and current_gray is not None:
            f.prnu_correlation = compute_prnu_correlation(current_gray, result.prnu_reference)
        prev_gray = current_gray
    log(f"  {Icons.SUCCESS} Tahap 2 Selesai.")

# --- [SOLUSI VIDEO PERPUSTAKAAN] LOGIKA UTAMA ADA DI SINI ---
def run_tahap_3_sintesis_bukti(result: AnalysisResult, out_dir: Path):
    print_stage_banner(3, "Sintesis Bukti & Investigasi Mendalam (LOGIKA DIPERBAIKI)", "🔬",
                       "Mengkorelasikan temuan, menerapkan logika kontekstual, dan melakukan analisis ELA/SIFT.")
    frames = result.frames
    n = len(frames)
    if n < 2: return

    # Tahap 3.1: Deteksi Semua Potensi Anomali
    log(f"\n  {Icons.EXAMINATION} Deteksi semua potensi anomali (Temporal & Duplikasi)...")
    hash_map = defaultdict(list)
    for f in frames:
        if f.hash: hash_map[f.hash].append(f.index)
    dup_candidates = {k: v for k, v in hash_map.items() if len(v) > 1}
    flow_mags = [f.optical_flow_mag for f in frames if f.optical_flow_mag is not None and f.optical_flow_mag > 0]
    median_flow, mad_flow = (np.median(flow_mags), stats.median_abs_deviation(flow_mags) or 1e-9) if flow_mags else (0, 1e-9)
    mv_mags = [f.motion_vector_mag for f in frames if f.motion_vector_mag is not None]
    median_mv, mad_mv = (np.median(mv_mags), stats.median_abs_deviation(mv_mags) or 1e-9) if mv_mags else (0, 1e-9)
    prnu_vals = [f.prnu_correlation for f in frames if f.prnu_correlation is not None]
    median_prnu, mad_prnu = (np.median(prnu_vals), stats.median_abs_deviation(prnu_vals) or 1e-9) if prnu_vals else (0, 1e-9)

    for i in range(1, n):
        f_curr, f_prev = frames[i], frames[i - 1]
        is_scene_change = (f_curr.color_cluster is not None and f_curr.color_cluster != f_prev.color_cluster)
        
        # Logika Deteksi Diskontinuitas
        has_flow_spike = False
        if f_curr.optical_flow_mag is not None and mad_flow > 0:
            z_score = 0.6745 * (f_curr.optical_flow_mag - median_flow) / mad_flow
            if abs(z_score) > CONFIG["OPTICAL_FLOW_Z_THRESH"]: has_flow_spike = True; f_curr.evidence_obj.metrics["optical_flow_z_score"] = round(z_score, 2)

        has_mv_spike = False
        if f_curr.motion_vector_mag is not None and mad_mv > 0:
            mv_z = 0.6745 * (f_curr.motion_vector_mag - median_mv) / mad_mv
            if abs(mv_z) > CONFIG["MOTION_VECTOR_Z_THRESH"]:
                has_mv_spike = True
                f_curr.evidence_obj.metrics["motion_vector_z_score"] = round(mv_z, 2)

        has_ssim_drop = False
        if f_curr.ssim_to_prev is not None and f_prev.ssim_to_prev is not None:
            ssim_drop = f_prev.ssim_to_prev - f_curr.ssim_to_prev
            if ssim_drop > CONFIG["SSIM_DISCONTINUITY_DROP"]: has_ssim_drop = True; f_curr.evidence_obj.metrics["ssim_drop"] = round(ssim_drop, 4)

        has_prnu_drop = False
        if f_curr.prnu_correlation is not None and mad_prnu > 0:
            prnu_z = 0.6745 * (median_prnu - f_curr.prnu_correlation) / mad_prnu
            if prnu_z > CONFIG["PRNU_Z_THRESH"]:
                has_prnu_drop = True
                f_curr.evidence_obj.metrics["prnu_z_score"] = round(prnu_z, 2)

        # Tahap 3.2: Sintesis dengan Konteks
        if is_scene_change:
            f_curr.type = "info_scene_change"
            f_curr.evidence_obj.reasons.append("Perubahan Adegan Terkonfirmasi")
            f_curr.evidence_obj.confidence = "INFORMASI"
        elif has_prnu_drop and (has_flow_spike or has_mv_spike or has_ssim_drop):
            f_curr.type = "anomaly_insertion"
            f_curr.evidence_obj.reasons.extend(["Penurunan Korelasi PRNU", "Diskontinuitas Temporal"])
            f_curr.evidence_obj.confidence = "SANGAT TINGGI"
        elif has_mv_spike and has_ssim_drop:
            f_curr.type = "anomaly_deletion"
            f_curr.evidence_obj.reasons.extend(["Lonjakan Motion Vector", "Penurunan Drastis SSIM"])
            f_curr.evidence_obj.confidence = "TINGGI"
        elif has_flow_spike and has_ssim_drop:
            f_curr.type = "anomaly_discontinuity"
            f_curr.evidence_obj.reasons.extend(["Lonjakan Aliran Optik", "Penurunan Drastis SSIM"])
            f_curr.evidence_obj.confidence = "TINGGI"
        elif has_ssim_drop:
            f_curr.type = "anomaly_discontinuity"
            f_curr.evidence_obj.reasons.append("Penurunan Drastis SSIM")
            f_curr.evidence_obj.confidence = "SEDANG"
        elif has_flow_spike:
            f_curr.type = "anomaly_discontinuity"
            f_curr.evidence_obj.reasons.append("Lonjakan Aliran Optik")
            f_curr.evidence_obj.confidence = "RENDAH"

    # Logika Deteksi Duplikasi
    if dup_candidates:
        for hash_val, indices in dup_candidates.items():
            if all(indices[j] == indices[j-1] + 1 for j in range(1, len(indices))):
                for idx in indices:
                    frames[idx].type = "info_static_scene"
            else:
                idx1 = indices[0]
                for idx2 in indices[1:]:
                    if frames[idx2].type != 'original':
                        continue
                    p1, p2 = Path(frames[idx1].img_path), Path(frames[idx2].img_path)
                    im1 = cv2.imread(str(p1), 0); im2 = cv2.imread(str(p2), 0)
                    if im1 is None or im2 is None or im1.shape != im2.shape:
                        continue
                    h1, h2 = frames[idx1].histogram_data, frames[idx2].histogram_data
                    hist_corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) if h1 is not None and h2 is not None else 1.0
                    if hist_corr < CONFIG["HIST_CORREL_THRESHOLD"]:
                        continue
                    data_range = im1.max() - im1.min()
                    ssim_val = ssim(im1, im2, data_range=data_range) if data_range > 0 else 1
                    if ssim_val > CONFIG["DUPLICATION_SSIM_CONFIRM"]:
                        sift_result = compare_sift_enhanced(p1, p2, out_dir)
                        if sift_result.get('inliers', 0) >= CONFIG["SIFT_MIN_MATCH_COUNT"]:
                            frames[idx2].type = "anomaly_duplication"
                            frames[idx2].evidence_obj.reasons.append(f"Duplikasi dari frame {idx1}")
                            frames[idx2].evidence_obj.confidence = "SANGAT TINGGI"
    
    for f in frames:
        if isinstance(f.evidence_obj.reasons, list) and f.evidence_obj.reasons:
            f.evidence_obj.reasons = ", ".join(sorted(list(set(f.evidence_obj.reasons))))

    log(f"\n  {Icons.SUCCESS} Tahap 3 Selesai.")


# --- TAHAP 4: VISUALISASI & PENILAIAN INTEGRITAS (ENHANCED VERSION) ---
def run_tahap_4_visualisasi_dan_penilaian(result: AnalysisResult, out_dir: Path):
    print_stage_banner(4, "Visualisasi & Penilaian Integritas (ENHANCED)", "📊",
                       "Membuat plot detail, melokalisasi peristiwa, menghitung skor integritas, dan menilai pipeline.")
    log(f"  {Icons.ANALYSIS} Melakukan Localization Tampering...")
    locs, event = [], None
    for f in result.frames:
        is_anomaly = f.type.startswith("anomaly")
        if is_anomaly:
            if event and event["event"] == f.type and f.index == event["end_frame"] + 1:
                event["end_frame"], event["end_ts"], event["frame_count"] = f.index, f.timestamp, event["frame_count"] + 1
            else:
                if event: locs.append(event)
                event = {"event": f.type, "start_frame": f.index, "end_frame": f.index, "start_ts": f.timestamp, "end_ts": f.timestamp, "frame_count": 1, "confidence": f.evidence_obj.confidence}
        elif event:
            locs.append(event); event = None
    if event: locs.append(event)
    result.localizations = locs;
    
    total_anom = sum(1 for f in result.frames if f.type.startswith("anomaly"))
    total_frames = len(result.frames) if result.frames else 1
    pct_anomaly = round(total_anom * 100 / total_frames, 2)
    result.summary = { "total_frames": total_frames, "total_anomaly": total_anom, "pct_anomaly": pct_anomaly, "total_events": len(locs), "all_frames": result.frames }
    
    log(f"\n  {Icons.ANALYSIS} Menghitung Skor Integritas...")
    integrity_score, integrity_desc, integrity_details = generate_integrity_score(result.summary, result.statistical_summary)
    result.integrity_analysis = {'score': integrity_score, 'description': integrity_desc, 'calculation_details': integrity_details}
    log(f"  -> Skor Integritas: {integrity_score}% ({integrity_desc})")
    
    log(f"\n  {Icons.ANALYSIS} Membuat visualisasi detail...")
    if result.localizations: result.plots['enhanced_localization_map'] = str(create_enhanced_localization_map(result, out_dir))
    if result.integrity_analysis: result.plots['integrity_breakdown'] = str(create_integrity_breakdown_chart(result.integrity_analysis['calculation_details'], out_dir, result.video_path))
    if result.localizations: result.plots['anomaly_infographic'] = str(create_anomaly_explanation_infographic(result, out_dir))
    prnu_plot = create_prnu_correlation_plot(result.frames, out_dir)
    if prnu_plot:
        result.plots['prnu_correlation'] = str(prnu_plot)
    
    log(f"  {Icons.SUCCESS} Tahap 4 Selesai.")


# --- Helper function for calculating event severity (FIXED) ---
def calculate_event_severity(event: dict) -> float:
    severity = 0.0; type_severity = {'anomaly_insertion': 0.8,'anomaly_duplication': 0.6,'anomaly_discontinuity': 0.5}; severity = type_severity.get(event.get('event', ''), 0.3); confidence_multiplier = {'SANGAT TINGGI': 1.2,'TINGGI': 1.0,'SEDANG': 0.8,'RENDAH': 0.6,'N/A': 0.5}; severity *= confidence_multiplier.get(event.get('confidence', 'N/A'), 0.5); duration = event.get('duration', 0)
    if duration > 5.0: severity *= 1.2
    elif duration > 2.0: severity *= 1.1
    frame_count = event.get('frame_count', 0)
    if frame_count > 10: severity *= 1.1
    return min(1.0, max(0.0, severity))


# --- [FIXED PDF Creation] FUNGSI LENGKAP UNTUK MEMBUAT PDF ---
def run_tahap_5_pelaporan_dan_validasi(result: AnalysisResult, out_dir: Path, baseline_result: AnalysisResult | None = None):
    print_stage_banner(5, "Penyusunan Laporan & Validasi Forensik", Icons.REPORTING,
                       "Menghasilkan laporan PDF naratif yang komprehensif.")

    pdf_path = out_dir / f"laporan_forensik_{Path(result.video_path).stem}.pdf"

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, topMargin=30, bottomMargin=50, leftMargin=30, rightMargin=30)
    styles = getSampleStyleSheet()

    # Aman mendefinisikan style custom
    if 'SubTitle' not in styles:
        styles.add(ParagraphStyle(name='SubTitle', parent=styles['h2'], fontSize=12, textColor=colors.darkslategray))
    if 'Justify' not in styles:
        styles.add(ParagraphStyle(name='Justify', parent=styles['Normal'], alignment=4)) # Justify
    if 'H3-Box' not in styles:
        styles.add(ParagraphStyle(name='H3-Box', parent=styles['h3'], backColor=colors.lightgrey, padding=4, leading=14, leftIndent=4, borderPadding=2, textColor=colors.black))

    story = []
    def header_footer(canvas, doc):
        canvas.saveState(); canvas.setFont('Helvetica', 8)
        canvas.drawString(30, 30, f"Laporan VIFA-Pro | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        canvas.drawRightString(A4[0] - 30, 30, f"Halaman {doc.page}"); canvas.restoreState()
    
    story.append(Paragraph("Laporan Analisis Forensik Video", styles['h1']))
    story.append(Paragraph("Dihasilkan oleh Sistem VIFA-Pro (v3)", styles['SubTitle']))
    story.append(Spacer(1, 24))

    # Ringkasan
    story.append(Paragraph("Ringkasan Eksekutif", styles['h2']))
    integrity_score = result.integrity_analysis['score']
    integrity_desc = result.integrity_analysis['description']
    summary_text = f"""
    Analisis komprehensif terhadap file <b>{Path(result.video_path).name}</b> telah selesai. 
    Berdasarkan <b>{len(result.localizations)} peristiwa anomali</b> yang terdeteksi dengan logika kontekstual, 
    video ini diberikan <b>Skor Integritas: {integrity_score}/100 ({integrity_desc})</b>.
    """
    story.append(Paragraph(summary_text, styles['Justify']))
    story.append(Spacer(1, 12))
    story.append(PageBreak())
    
    # Menambahkan plot utama jika ada
    if result.plots.get('integrity_breakdown') and Path(result.plots['integrity_breakdown']).exists():
        story.append(Paragraph("Analisis Skor Integritas", styles['h2']))
        story.append(PlatypusImage(result.plots['integrity_breakdown'], width=520, height=250, kind='proportional'))
        story.append(Spacer(1, 12))
    
    if result.plots.get('enhanced_localization_map') and Path(result.plots['enhanced_localization_map']).exists():
        story.append(Paragraph("Peta Lokalisasi Anomali", styles['h2']))
        story.append(PlatypusImage(result.plots['enhanced_localization_map'], width=520, height=350, kind='proportional'))

    if result.plots.get('prnu_correlation') and Path(result.plots['prnu_correlation']).exists():
        story.append(Spacer(1, 12))
        story.append(Paragraph("Korelasi PRNU", styles['h2']))
        story.append(PlatypusImage(result.plots['prnu_correlation'], width=520, height=250, kind='proportional'))
    
    # Build PDF
    try:
        doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
        result.pdf_report_path = pdf_path
        log(f"  ✅ Laporan PDF berhasil dibuat: {pdf_path.name}")
    except Exception as e:
        log(f"  {Icons.ERROR} Gagal membuat laporan PDF: {e}")
        
    log(f"  {Icons.SUCCESS} Tahap 5 Selesai.")

###############################################################################
# MAIN EXECUTION
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIFA-Pro: Sistem Forensik Video Profesional (v3 Final Fix)")
    parser.add_argument("video_path", type=str, help="Path ke video yang akan dianalisis")
    parser.add_argument("-b", "--baseline", type=str, help="Path ke video baseline (opsional)")
    parser.add_argument("-f", "--fps", type=int, default=10, help="FPS ekstraksi frame (default: 10)")
    parser.add_argument("-o", "--output", type=str, help="Direktori output (default: auto-generated)")

    args = parser.parse_args()
    video_path = Path(args.video_path)
    if not video_path.exists(): print(f"{Icons.ERROR} File video tidak ditemukan: {video_path}"); sys.exit(1)
    if args.output: out_dir = Path(args.output)
    else: timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); out_dir = Path(f"forensik_output_final_{video_path.stem}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log(f"\n{Icons.IDENTIFICATION} VIFA-Pro: Memulai Analisis Forensik (v3 Final Fix)")
    log(f"{Icons.INFO} Video: {video_path.name}")
    
    result = run_tahap_1_pra_pemrosesan(video_path, out_dir, args.fps)
    if not result: sys.exit(1)
    
    baseline_result = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            log(f"\n{Icons.ANALYSIS} Memproses video baseline...")
            baseline_result = run_tahap_1_pra_pemrosesan(baseline_path, out_dir, args.fps)

    run_tahap_2_analisis_temporal(result, baseline_result)
    run_tahap_3_sintesis_bukti(result, out_dir)
    run_tahap_4_visualisasi_dan_penilaian(result, out_dir)
    run_tahap_5_pelaporan_dan_validasi(result, out_dir, baseline_result)

    log(f"\n{Icons.SUCCESS} Analisis selesai!")
    log(f"{Icons.INFO} Hasil tersimpan di: {out_dir}")
    if result.pdf_report_path:
        log(f"{Icons.INFO} Laporan PDF: {result.pdf_report_path.resolve()}")

