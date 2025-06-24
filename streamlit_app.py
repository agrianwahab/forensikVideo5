# --- START OF FILE streamlit_app.py ---
import streamlit as st
from pathlib import Path
from typing import Optional, Union
import tempfile
import ForensikVideo as fv
import sys
import io
import traceback
from datetime import datetime
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from history_manager import HistoryManager
from export_utils import add_export_buttons

st.set_page_config(
    page_title="VIFA-Pro | Dashboard Forensik Video",
    layout="wide"
)

# ======================= REDESIGN/FIX START =======================
st.markdown("""
    <style>
    .stApp { background-color: #F0F2F6; }
    h1 { color: #0B3D91; font-weight: bold; }
    h2, h3 { color: #0056b3; }
    .stButton>button { border-radius: 8px; border: 1px solid #0c6dd6; background-color: #0c6dd6; color: white; transition: all 0.2s; }
    .stButton>button:hover { border-color: #004494; background-color: #0056b3; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }
    .stDataFrame { width: 100%; }
    
    .explanation-box {
        background-color: #FFF8DC;
        border: 2px solid #FFD700;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .technical-box {
        background-color: #F0F8FF;
        border: 1px solid #4682B4;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        font-family: monospace;
    }
    .severity-high { color: #FF0000; font-weight: bold; }
    .severity-medium { color: #FFA500; font-weight: bold; }
    .severity-low { color: #008000; }
    
    .pipeline-stage-card {
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .localization-event-card {
        background-color: #F8F9FA;
        border-left: 5px solid #0c6dd6;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .metric-explanation {
        background-color: #E8F4F8;
        border: 1px solid #B8E0E8;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    
    .ferm-card {
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
    }
    .ferm-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #333;
        margin-bottom: 15px;
        border-bottom: 2px solid #0c6dd6;
        padding-bottom: 8px;
    }
    .ferm-section {
        margin: 15px 0;
    }
    .ferm-factor {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #f8f9fa;
    }
    .ferm-factor-positif {
        border-left: 4px solid #28a745;
    }
    .ferm-factor-negatif {
        border-left: 4px solid #dc3545;
    }
    .ferm-factor-netral {
        border-left: 4px solid #ffc107;
    }
    .ferm-finding {
        background-color: #f0f7ff;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 4px solid #0c6dd6;
    }
    .ferm-reliability-high {
        color: #155724;
        background-color: #d4edda;
        border-color: #c3e6cb;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .ferm-reliability-moderate {
        color: #856404;
        background-color: #fff3cd;
        border-color: #ffeeba;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .ferm-reliability-limited {
        color: #721c24;
        background-color: #f8d7da;
        border-color: #f5c6cb;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .ferm-tab {
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #f8f9fa;
    }
    
    .history-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
        border-left: 7px solid #0056b3;
    }
    .history-card:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        transform: translateY(-3px);
    }
    .history-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    .history-date { color: #6c757d; font-size: 0.9em; }
    .history-reliability-badge {
        font-weight: bold;
        padding: 6px 12px;
        border-radius: 50px;
        color: black;
        font-size: 0.9em;
        min-width: 120px;
        text-align: center;
    }
    .history-anomaly-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 15px;
    }
    .history-anomaly-tag {
        font-size: 0.85em;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 15px;
        color: white;
    }
    .history-actions {
        display: flex;
        gap: 10px;
        margin-top: 20px;
    }
    .history-empty {
        text-align: center; padding: 50px; color: #777; background-color: #f9f9f9;
        border-radius: 10px; margin: 20px 0; border: 2px dashed #d0d0d0;
    }
    .history-toolbar {
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;
        padding: 10px; background-color: #FFFFFF; border-radius: 8px;
    }
    .history-detail-section {
        background-color: #FFFFFF; padding: 25px; border-radius: 10px; margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .history-artifact-container {
        text-align: center; margin-top: 15px; padding: 15px;
        background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef;
    }
    .history-artifact-container img { max-width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)
# ======================= REDESIGN/FIX END =======================

st.title("🔎 VIFA-Pro: Sistem Deteksi Forensik Keaslian Video")
st.markdown("Menggunakan **Metode K-Means** dan **Localization Tampering** dengan Dukungan ELA dan SIFT")

history_manager = HistoryManager()

if 'show_ferm_details' not in st.session_state:
    st.session_state.show_ferm_details = True

with st.sidebar:
    selected_tab = st.radio(
        "Menu Navigasi",
        ["Analisis Baru", "Riwayat Analisis"],
        captions=["Jalankan forensik pada video baru", "Lihat, detailkan, atau hapus analisis sebelumnya"],
        format_func=lambda x: "🔍 " + x if x == "Analisis Baru" else "📜 " + x
    )
    
    st.markdown("---")

    if selected_tab == "Analisis Baru":
        st.header("Panel Kontrol Analisis")
        uploaded_video = st.file_uploader(
            "Unggah Video Bukti", type=["mp4", "avi", "mov", "mkv"]
        )
        baseline_video = st.file_uploader(
            "Unggah Video Baseline (Opsional)", type=["mp4", "avi", "mov", "mkv"]
        )
        fps = st.number_input("Frame Extraction FPS", min_value=1, max_value=30, value=10, step=1)
        run = st.button("🚀 Jalankan Analisis Forensik", use_container_width=True, type="primary")

        st.subheader("⚙️  Pengaturan Threshold")
        auto_threshold = st.checkbox("Otomatis (Rekomendasi)", value=True)
        ssim_slider = st.slider("Ambang Batas Penurunan SSIM", 0.20, 0.50, 0.30, 0.01, disabled=auto_threshold)
        z_slider = st.slider("Z-score Aliran Optik", 3.0, 8.0, 4.0, 0.1, disabled=auto_threshold)
        bypass_debug = st.checkbox("Lewati PRNU / JPEG-DQ", value=False)
        
        st.subheader("Pengaturan Detail")
        show_technical_details = st.checkbox("Tampilkan Detail Teknis", value=True)
        show_simple_explanations = st.checkbox("Tampilkan Penjelasan Sederhana", value=True)
        show_advanced_metrics = st.checkbox("Tampilkan Metrik Lanjutan", value=False)
        
        st.markdown("---")
        st.subheader("Pengaturan FERM")
        st.session_state.show_ferm_details = st.checkbox("Tampilkan Detail FERM", value=True)
    
    else: # This is the "Riwayat Analisis" tab's sidebar
        st.header("Pengaturan Riwayat")
        
        if st.button("🗑️ Hapus Semua Riwayat", use_container_width=True):
            st.session_state['confirm_delete_all_prompt'] = True

        if st.session_state.get('confirm_delete_all_prompt', False):
            st.warning("PERINGATAN: Tindakan ini tidak dapat diurungkan.")
            confirm_text = st.text_input("Ketik 'HAPUS SEMUA' untuk konfirmasi:", key="confirm_delete_all_text")
            if confirm_text == "HAPUS SEMUA":
                if st.button("Konfirmasi Hapus Semua", type="primary"):
                    count = history_manager.delete_all_history()
                    st.success(f"Berhasil menghapus {count} riwayat analisis!")
                    del st.session_state['confirm_delete_all_prompt']
                    del st.session_state['confirm_delete_all_text']
                    time.sleep(1)
                    st.rerun()
        
        st.subheader("Filter Riwayat")
        search_query = st.text_input("Cari Nama Video...", key="history_search_query")


def load_image_as_bytes(path_str: Optional[Union[str, Path]]) -> Optional[bytes]:
    if path_str and Path(path_str).exists():
        try:
            with open(path_str, "rb") as f: return f.read()
        except Exception: return None
    return None

def _get_metric_description(metric_name: str, value: any) -> str:
    descriptions = {
        'optical_flow_z_score': 'Z-score > 4 menunjukkan pergerakan abnormal',
        'ssim_drop': 'Penurunan > 0.25 menunjukkan perubahan drastis',
        'ssim_absolute_low': 'Nilai < 0.7 menunjukkan frame sangat berbeda',
        'color_cluster_jump': 'Perubahan nomor klaster = perubahan adegan',
        'source_frame': 'Indeks frame yang diduplikasi',
        'ssim_to_source': 'Kemiripan dengan frame sumber (1 = identik)',
        'sift_inliers': 'Jumlah titik yang cocok sempurna',
        'sift_good_matches': 'Total titik kandidat',
        'sift_inlier_ratio': 'Rasio kecocokan valid',
        'ela_max_difference': 'Perbedaan kompresi maksimal',
        'ela_suspicious_regions': 'Jumlah area mencurigakan'
    }
    return descriptions.get(metric_name, 'Metrik analisis')

def format_timestamp(iso_timestamp):
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%d %b %Y, %H:%M:%S")
    except (ValueError, TypeError):
        return iso_timestamp

def get_anomaly_style(atype):
    styles = {
        "duplication": ("🔁 Duplikasi", "#dc3545"),
        "discontinuity": ("✂️ Diskontinuitas", "#007bff"),
        "insertion": ("➕ Penyisipan", "#fd7e14"),
    }
    return styles.get(atype, ("❓ Lainnya", "#6c757d"))

def get_reliability_class(reliability):
    if "Tinggi" in reliability:
        return "ferm-reliability-high", "green"
    elif "Sedang" in reliability:
        return "ferm-reliability-moderate", "orange"
    else: # Includes "Terbatas" and "Rendah"
        return "ferm-reliability-limited", "red"

def display_history_card(entry):
    ferm = entry.get("forensic_evidence_matrix", {})
    reliability = ferm.get("conclusion", {}).get("reliability_assessment", "Status tidak diketahui")
    
    reliability_short = "N/A"
    reliability_color = "#6c757d" # grey
    if "Tinggi" in reliability: 
        reliability_short = "Reliabilitas Tinggi"
        reliability_color = "#28a745" # green
    if "Sedang" in reliability: 
        reliability_short = "Reliabilitas Sedang"
        reliability_color = "#ffc107" # yellow
    if "Terbatas" in reliability or "Rendah" in reliability: 
        reliability_short = "Reliabilitas Rendah"
        reliability_color = "#dc3545" # red

    # Using st.container with a border is a simpler way to group elements than raw HTML
    with st.container(border=True):
        st.markdown(f"""
        <style>
            /* Hack to apply a colored left border to the st.container */
            div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div[style*="border: 1px solid rgba(49, 51, 63, 0.2)"]:has(h3:contains('{entry['id']}')) {{
                 border-left: 7px solid {reliability_color} !important;
            }}
        </style>
        <h3 style='display:none;'>{entry['id']}</h3> 
        """, unsafe_allow_html=True)

        header_cols = st.columns([0.8, 0.2])
        with header_cols[0]:
            st.subheader(entry.get("video_name", "Video Tanpa Nama"))
        with header_cols[1]:
            st.caption(f"📅 {format_timestamp(entry.get('timestamp'))}")

        stat_cols = st.columns(3)
        stat_cols[0].metric("Reliabilitas", reliability_short)
        stat_cols[1].metric("Total Anomali", entry.get("summary", {}).get("total_anomaly", 0))
        stat_cols[2].metric("Total Peristiwa", entry.get("localizations_count", 0))

        # Anomaly Tags
        anomaly_types = entry.get("anomaly_types", {})
        if sum(anomaly_types.values()) > 0:
            tag_html = '<div class="history-anomaly-tags">'
            for atype, count in anomaly_types.items():
                if count > 0:
                    label, color = get_anomaly_style(atype)
                    tag_html += f'<span class="history-anomaly-tag" style="background-color: {color};">{label}: {count}</span>'
            tag_html += '</div>'
            st.markdown(tag_html, unsafe_allow_html=True)
        else:
            st.markdown('<div class="history-anomaly-tags"><span class="history-anomaly-tag" style="background-color: #28a745;">✅ Tidak ada anomali</span></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Action Buttons
        action_cols = st.columns(6)
        with action_cols[0]:
            if st.button("🔍 Lihat Detail", key=f"view_{entry['id']}", use_container_width=True, type="primary"):
                st.session_state.selected_history_id = entry["id"]
                st.rerun()
        with action_cols[1]:
            if st.button("🗑️ Hapus", key=f"delete_{entry['id']}", use_container_width=True):
                history_manager.delete_analysis(entry["id"])
                st.toast(f"Analisis untuk '{entry['video_name']}' telah dihapus.")
                time.sleep(0.5)
                st.rerun()

def display_history_detail(entry_id):
    entry = history_manager.get_analysis(entry_id)
    if not entry:
        st.error("Gagal memuat riwayat analisis. Mungkin sudah dihapus.")
        if st.button("Kembali ke Riwayat"): 
            st.session_state.selected_history_id = None
            st.rerun()
        return

    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.header(f"Detail Analisis: {entry.get('video_name', 'N/A')}")
        st.caption(f"Dianalisis pada: {format_timestamp(entry.get('timestamp'))} | Hash Preservasi: {entry.get('preservation_hash', 'N/A')[:20]}...")
    with col2:
        if st.button("⬅️ Kembali", use_container_width=True):
            st.session_state.selected_history_id = None
            st.rerun()

    tabs = st.tabs(["📊 Ringkasan Eksekutif", "🔍 Investigasi Anomali", "🧠 Analisis FERM", "🖼️ Artefak Visual", "📝 Metadata & Laporan"])

    with tabs[0]:
        with st.container(border=True):
            st.subheader("Penilaian Reliabilitas Bukti Forensik (FERM)")
            ferm = entry.get("forensic_evidence_matrix", {})
            reliability = ferm.get("conclusion", {}).get("reliability_assessment", "Tidak Tersedia")
            reliability_class, _ = get_reliability_class(reliability)
            st.markdown(f'<div class="{reliability_class}">{reliability}</div>', unsafe_allow_html=True)
            if ferm.get("conclusion", {}).get("primary_findings"):
                st.write("**Temuan Utama:**")
                for i, finding in enumerate(ferm["conclusion"]["primary_findings"]):
                    st.markdown(f"- **{finding['finding']}** ({finding['confidence']})")

        with st.container(border=True):
            st.subheader("Statistik Utama")
            summary = entry.get("summary", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Frame", f"{summary.get('total_frames', 0):,}")
            c2.metric("Total Anomali", f"{summary.get('total_anomaly', 0):,}", help="Jumlah frame individual yang ditandai anomali.")
            c3.metric("Persentase Anomali", f"{summary.get('pct_anomaly', 0):.1f}%")
            c4.metric("Jumlah Peristiwa", f"{entry.get('localizations_count', 0)}", help="Jumlah kelompok anomali yang terjadi berdekatan.")
            anomaly_types = entry.get("anomaly_types", {})
            if sum(anomaly_types.values()) > 0:
                labels = [get_anomaly_style(k)[0] for k, v in anomaly_types.items() if v > 0]
                values = [v for v in anomaly_types.values() if v > 0]
                colors = [get_anomaly_style(k)[1] for k, v in anomaly_types.items() if v > 0]
                pie_fig = px.pie(values=values, names=labels, title="Distribusi Tipe Anomali", color_discrete_sequence=colors)
                pie_fig.update_layout(height=350, legend_title_text='Tipe Anomali')
                st.plotly_chart(pie_fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Rincian Setiap Peristiwa Anomali")
        localizations = entry.get("localizations", [])
        if not localizations:
            st.success("✅ Tidak ditemukan peristiwa anomali yang signifikan dalam analisis ini.")
        else:
            st.info(f"Ditemukan {len(localizations)} peristiwa anomali. Peristiwa adalah kelompok anomali yang terjadi berdekatan dan dianggap sebagai satu kejadian.")
            sorted_locs = sorted(localizations, key=lambda x: x.get('start_ts', 0))
            saved_artifacts_dict = entry.get("saved_artifacts", {})
            for i, loc in enumerate(sorted_locs):
                atype = loc.get('event', '').replace('anomaly_', '')
                desc = history_manager.get_anomaly_description(atype)

                with st.expander(f"{desc['icon']} **Peristiwa #{i+1}: {desc['title']}** (Mulai: {loc['start_ts']:.2f}s, Durasi: {loc.get('duration', 0):.2f}s)", expanded=i==0):
                    st.markdown(f'<div class="localization-event-card" style="border-left-color: {desc["color"]};">', unsafe_allow_html=True)
                    st.markdown("#### 📖 **Apa Artinya Ini?**")
                    st.markdown(f"**Analogi Sederhana:** *{desc['example']}*")
                    st.markdown(f"**Implikasi Forensik:** {desc['implication']}")
                    st.markdown("---")
                    st.markdown("#### 🔧 **Detail Teknis**")
                    st.markdown(f"**Metode Deteksi:** {desc['technical']}")
                    if loc.get("metrics"):
                        st.write("**Metrik Kunci:**")
                        for key, val in loc["metrics"].items():
                            st.text(f"  - {key.replace('_', ' ').title()}: {val}")
                    
                    artifact_key = f"anomaly_frame_{i}"
                    img_path = saved_artifacts_dict.get(artifact_key)

                    if img_path:
                        st.markdown("#### 🖼️ **Bukti Visual Pendukung**")
                        img_b64 = history_manager.get_artifact_base64(img_path)
                        if img_b64:
                            st.image(img_b64, caption=f"Frame bukti visual untuk peristiwa #{i+1}", use_container_width=True)

                    st.markdown('</div>', unsafe_allow_html=True)
                    
    with tabs[2]:
        display_ferm_tab_content(entry)

    with tabs[3]:
        st.subheader("Galeri Visualisasi Hasil Analisis")
        st.info("Ini adalah kumpulan plot dan gambar yang dihasilkan selama proses analisis, memberikan gambaran visual dari temuan.")
        saved_artifacts = entry.get("saved_artifacts", {})
        plot_artifacts = {k: v for k, v in saved_artifacts.items() if not k.startswith("anomaly_frame") and "report" not in k and "ferm_" not in k}

        if not plot_artifacts:
            st.warning("Tidak ada artefak visual yang tersimpan untuk analisis ini.")
        else:
            for key, path in plot_artifacts.items():
                img_b64 = history_manager.get_artifact_base64(path)
                if img_b64:
                    title = key.replace('_', ' ').title()
                    with st.container(border=True):
                        st.subheader(title)
                        st.markdown(f'<div class="history-artifact-container"><img src="{img_b64}"></div>', unsafe_allow_html=True)
                        
    with tabs[4]:
        with st.container(border=True):
            st.subheader("📤 Ekspor Hasil Analisis")
            st.markdown("Anda dapat mengunduh seluruh hasil analisis ini sebagai sebuah file ZIP. File ini berisi laporan data dalam format JSON dan semua artefak visual yang tersimpan.")
            
            zip_data = history_manager.export_analysis(entry_id)
            if zip_data:
                filename = f"VIFA-Pro_Analysis_{entry_id[:8]}_{entry.get('video_name', 'export').replace(' ', '_')}.zip"
                st.download_button(
                    label="💾 Unduh Laporan ZIP",
                    data=zip_data,
                    file_name=filename,
                    mime="application/zip",
                    use_container_width=True,
                    type="primary"
                )
            else:
                st.error("Gagal membuat file ekspor.")
        
        with st.container(border=True):
            st.subheader("📝 Metadata Video")
            metadata = entry.get("metadata", {})
            if not metadata:
                st.warning("Tidak ada metadata yang tersimpan.")
            else:
                for category, items in metadata.items():
                    if items:
                        st.markdown(f"##### {category.replace('_', ' ').title()}")
                        df = pd.DataFrame.from_dict(items, orient='index', columns=['Nilai'])
                        st.table(df)

def render_history_page():
    st.title("📜 Riwayat Analisis Forensik")
    st.markdown("Telusuri, detailkan, dan kelola semua analisis video yang telah dilakukan.")
    
    if st.session_state.get('selected_history_id'):
        display_history_detail(st.session_state.selected_history_id)
        return

    history = history_manager.load_history()
    if not history:
        st.markdown('<div class="history-empty"><h3>Belum Ada Riwayat Analisis</h3><p>Mulai dengan menjalankan analisis baru di menu navigasi.</p></div>', unsafe_allow_html=True)
        return
        
    st.subheader("Ringkasan Seluruh Riwayat")
    total_anomalies = sum(entry.get("summary", {}).get("total_anomaly", 0) for entry in history)
    
    c1, c2 = st.columns(2)
    c1.metric("Total Analisis Tersimpan", len(history))
    c2.metric("Total Anomali Ditemukan", f"{total_anomalies:,}")
    st.markdown("---")

    st.subheader("Daftar Riwayat Analisis")
    
    search_query = st.session_state.get('history_search_query', '').lower()
    
    filtered_history = [
        entry for entry in history
        if search_query in entry.get("video_name", "").lower()
    ]

    if not filtered_history:
        st.warning("Tidak ada riwayat yang cocok dengan kriteria pencarian Anda.")
    else:
        sorted_history = sorted(filtered_history, key=lambda x: x.get("timestamp", ""), reverse=True)
        for entry in sorted_history:
            display_history_card(entry)

def display_ferm_tab_content(entry_or_result):
    # ======================= DUPLICATION FIX START =======================
    # Memperbaiki logika untuk menghindari duplikasi tampilan FERM
    if isinstance(entry_or_result, dict):  # Ini adalah entri riwayat (history entry)
        ferm = entry_or_result.get("forensic_evidence_matrix", {})
        all_plots = entry_or_result.get("saved_artifacts", {})
        # Untuk riwayat, semua kunci adalah path, jadi tidak ada _bytes
        ferm_visualizations = {k: v for k, v in all_plots.items() if "ferm_" in k}
    else:  # Ini adalah hasil analisis baru (new result object)
        ferm = getattr(entry_or_result, 'forensic_evidence_matrix', {})
        all_plots = getattr(entry_or_result, 'plots', {})
        # Untuk hasil baru, HANYA ambil kunci yang berakhiran '_bytes' untuk menghindari duplikasi
        ferm_visualizations = {k: v for k, v in all_plots.items() if "ferm_" in k and k.endswith('_bytes')}
    # ======================= DUPLICATION FIX END =======================

    if not ferm:
        st.warning("Data Matriks Reliabilitas Bukti Forensik (FERM) tidak tersedia untuk analisis ini.")
        return

    st.header("Matriks Reliabilitas Bukti Forensik (FERM)")
    st.markdown("""
    Matriks Reliabilitas Bukti Forensik memberikan pendekatan multi-dimensi yang lebih ilmiah untuk analisis forensik video.
    Ini mengevaluasi kekuatan bukti, karakteristik anomali, dan analisis kausalitas untuk mencapai kesimpulan yang lebih dapat dipertahankan.
    """)

    ferm_tabs = st.tabs(["📊 Kesimpulan", "💪 Kekuatan Bukti", "🔍 Karakterisasi Anomali", "⚖️ Analisis Kausalitas", "🖼️ Visualisasi FERM"])

    with ferm_tabs[0]:
        conclusion = ferm.get("conclusion", {})
        reliability = conclusion.get("reliability_assessment", "Tidak tersedia")
        reliability_class, _ = get_reliability_class(reliability)
        st.markdown(f'<div class="{reliability_class}">{reliability}</div>', unsafe_allow_html=True)
        st.subheader("Faktor-faktor Reliabilitas")
        for factor in conclusion.get("reliability_factors", []):
            impact = factor.get("impact", "Netral").lower()
            factor_class = f"ferm-factor ferm-factor-{impact}"
            st.markdown(f'''<div class="{factor_class}"><div style="flex: 1;"><b>{factor.get('factor', 'T/A')}</b></div><div style="flex: 2;">{factor.get('assessment', 'T/A')}</div></div>''', unsafe_allow_html=True)
        st.subheader("Temuan Utama")
        if not conclusion.get("primary_findings"):
            st.info("Tidak ada temuan utama yang teridentifikasi.")
        else:
            for finding in conclusion.get("primary_findings", []):
                st.markdown(f'''<div class="ferm-finding"><h4>{finding.get('finding', 'T/A')} ({finding.get('confidence', 'T/A')})</h4><p><b>Bukti:</b> {finding.get('evidence', 'T/A')}</p><p><i>Interpretasi:</i> {finding.get('interpretation', 'T/A')}</p></div>''', unsafe_allow_html=True)
        st.subheader("Rekomendasi Tindakan")
        if not conclusion.get("recommended_actions"):
            st.info("Tidak ada rekomendasi tindakan.")
        else:
            for i, action in enumerate(conclusion.get("recommended_actions", [])):
                st.markdown(f"{i+1}. {action}")

    with ferm_tabs[1]:
        evidence_strength = ferm.get("evidence_strength", {})
        st.subheader("Konfirmasi Multi-Metode")
        mmc = evidence_strength.get("multi_method_confirmation", {})
        if mmc:
            c1, c2 = st.columns(2)
            c1.metric("Rata-rata Metode per Anomali", f"{mmc.get('average_methods_per_anomaly', 0):.2f}")
            c2.metric("Persentase Dikonfirmasi Beberapa Metode", f"{mmc.get('percentage_confirmed_by_multiple', 0)*100:.1f}%")
            if mmc.get("counts"):
                fig = px.bar(x=list(mmc["counts"].keys()), y=list(mmc["counts"].values()), labels={"x": "Jumlah Metode", "y": "Jumlah Anomali"}, title="Distribusi Konfirmasi Multi-Metode")
                st.plotly_chart(fig, use_container_width=True)
        st.subheader("Distribusi Tingkat Kepercayaan")
        conf_dist = evidence_strength.get("confidence_distribution", {})
        if conf_dist:
            conf_labels = ["SANGAT TINGGI", "TINGGI", "SEDANG", "RENDAH"]
            conf_values = [conf_dist.get(label, 0) for label in conf_labels]
            conf_colors = ["red", "orange", "yellow", "green"]
            fig = px.bar(x=conf_labels, y=conf_values, color=conf_labels, color_discrete_sequence=conf_colors, labels={"x": "Tingkat Kepercayaan", "y": "Jumlah Anomali"}, title="Distribusi Tingkat Kepercayaan Anomali")
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Penilaian False Positive")
        fp = evidence_strength.get("false_positive_assessment", {})
        if fp:
            st.metric("Risiko False Positive Tertimbang", f"{fp.get('weighted_risk', 0)*100:.1f}%")
            if fp.get("risk_factors"):
                st.write("**Faktor Risiko Teridentifikasi:**")
                for factor in fp.get("risk_factors", []):
                    st.markdown(f'''<div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 5px 0;"><b>{factor.get('factor', 'T/A')}</b> ({factor.get('value', 'T/A')})<br><i>Dampak:</i> {factor.get('impact', 'T/A')}</div>''', unsafe_allow_html=True)

    with ferm_tabs[2]:
        anomaly_char = ferm.get("anomaly_characterization", {})
        st.subheader("Distribusi Temporal")
        temp_dist = anomaly_char.get("temporal_distribution", {})
        if temp_dist:
            c1, c2 = st.columns(2)
            c1.metric("Total Anomali", temp_dist.get("total_anomalies", 0))
            c1.metric("Densitas Anomali", f"{temp_dist.get('anomaly_density', 0)*100:.2f}%")
            c2.metric("Jumlah Kluster", temp_dist.get("cluster_count", 0))
            c2.metric("Ukuran Rata-rata Kluster", f"{temp_dist.get('avg_cluster_size', 0):.1f}")
            st.write(f"**Pola Distribusi:** {temp_dist.get('distribution_pattern', 'T/A').title()}")
        st.subheader("Tingkat Keparahan Teknis")
        tech_sev = anomaly_char.get("technical_severity", {})
        if tech_sev:
            st.metric("Tingkat Keparahan Rata-rata", f"{tech_sev.get('overall_mean_severity', 0):.2f} (0-1)")
            st.metric("Jumlah Tingkat Keparahan Tinggi", tech_sev.get("high_severity_count", 0))
            if tech_sev.get("by_type"):
                st.write("**Tingkat Keparahan berdasarkan Tipe Anomali:**")
                severity_data = [{"Tipe": k.replace("anomaly_", "").capitalize(), "Severity Rata-rata": f"{v.get('mean', 0):.2f}", "Jumlah": v.get("count", 0)} for k, v in tech_sev.get("by_type", {}).items()]
                if severity_data: st.dataframe(severity_data, use_container_width=True)
        st.subheader("Konteks Semantik")
        sem_context = anomaly_char.get("semantic_context", {})
        if sem_context:
            st.metric("Peristiwa Signifikan", sem_context.get("significant_events", 0))
            if sem_context.get("event_types"):
                event_types = sem_context.get("event_types", {})
                fig = px.bar(x=list(event_types.keys()), y=list(event_types.values()), labels={"x": "Tipe Peristiwa", "y": "Jumlah"}, title="Distribusi Tipe Peristiwa")
                st.plotly_chart(fig, use_container_width=True)

    with ferm_tabs[3]:
        causality = ferm.get("causality_analysis", {})
        st.subheader("Penyebab Teknikal")
        tech_causes = causality.get("technical_causes", {})
        if tech_causes:
            for cause_type, cause_data in tech_causes.items():
                if isinstance(cause_data, dict):
                    st.markdown(f'''<div style="background-color: #f0f7ff; padding: 15px; border-radius: 5px; margin: 10px 0;"><h4>{cause_type.capitalize()}</h4><p><b>Penyebab:</b> {cause_data.get('cause', 'T/A')}</p><p><b>Probabilitas:</b> {cause_data.get('probability', 'T/A')}</p><ul>{' '.join(f'<li>{indicator}</li>' for indicator in cause_data.get('technical_indicators', []))}</ul></div>''', unsafe_allow_html=True)
        st.subheader("Kompresi vs Manipulasi")
        comp_vs_manip = causality.get("compression_vs_manipulation", {})
        if comp_vs_manip:
            st.write(f"**Informasi Kompresi:** {comp_vs_manip.get('compression_info', 'T/A')}")
            assessment = comp_vs_manip.get("compression_vs_manipulation_assessment", "T/A")
            assessment_color = "green" if "normal" in assessment.lower() else "red" if "manipulasi" in assessment.lower() else "orange"
            st.markdown(f'''<div style="background-color: {assessment_color}; color: white; padding: 10px; border-radius: 5px; text-align: center;"><h4 style="margin: 0;">PENILAIAN: {assessment}</h4></div>''', unsafe_allow_html=True)
        st.subheader("Penjelasan Alternatif")
        alt_expl = causality.get("alternative_explanations", {})
        if alt_expl:
            most_likely = alt_expl.get("most_likely_alternative")
            if most_likely and alt_expl.get("all_alternatives", {}).get(most_likely):
                ml_data = alt_expl["all_alternatives"][most_likely]
                st.markdown(f'''<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #0c6dd6;"><h4>Alternatif Paling Mungkin: {most_likely.replace('_', ' ').capitalize()}</h4><p><b>Penjelasan:</b> {ml_data.get('explanation', 'T/A')}</p><p><b>Kemungkinan:</b> {ml_data.get('likelihood', 'T/A')}</p></div>''', unsafe_allow_html=True)

    with ferm_tabs[4]:
        if not ferm_visualizations:
            st.warning("Tidak ada visualisasi FERM yang tersedia.")
        else:
            for key, path_or_bytes in ferm_visualizations.items():
                img_data_b64 = None
                title = key.replace('_bytes', '').replace('ferm_', '').replace('_', ' ').title()

                if isinstance(path_or_bytes, (str, Path)):
                    img_data_b64 = history_manager.get_artifact_base64(path_or_bytes)
                elif isinstance(path_or_bytes, bytes):
                    import base64
                    img_data_b64 = "data:image/png;base64," + base64.b64encode(path_or_bytes).decode('utf-8')
                
                if img_data_b64:
                    with st.container(border=True):
                        st.subheader(title)
                        st.markdown(f'<div class="history-artifact-container"><img src="{img_data_b64}" alt="{title}"></div>', unsafe_allow_html=True)


# ======================= LOGIC FIX START =======================
if selected_tab == "Analisis Baru":
    if run:
        if uploaded_video is None:
            st.error("⚠️ Mohon unggah video bukti terlebih dahulu di sidebar.")
        else:
            if auto_threshold:
                fv.CONFIG['USE_AUTO_THRESHOLDS'] = True
            else:
                fv.CONFIG['USE_AUTO_THRESHOLDS'] = False
                fv.CONFIG['SSIM_USER_THRESHOLD'] = float(ssim_slider)
                fv.CONFIG['Z_USER_THRESHOLD'] = float(z_slider)
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                sus_path = tmpdir_path / uploaded_video.name
                with open(sus_path, "wb") as f: f.write(uploaded_video.getbuffer())

                baseline_path = None
                if baseline_video is not None:
                    baseline_path = tmpdir_path / baseline_video.name
                    with open(baseline_path, "wb") as f: f.write(baseline_video.getbuffer())

                result = None; baseline_result = None
                
                all_stages_success = True
                try:
                    with st.status("Memulai analisis forensik 5 tahap...", expanded=True) as status:
                        st.write("Tahap 1: Akuisisi & Ekstraksi Fitur Dasar...")
                        result = fv.run_tahap_1_pra_pemrosesan(sus_path, tmpdir_path, int(fps))
                        if not result: raise ValueError("Gagal pada Tahap 1.")
                        
                        if baseline_path:
                            st.write("Memproses Video Baseline...")
                            baseline_result = fv.run_tahap_1_pra_pemrosesan(baseline_path, tmpdir_path, int(fps))
                            if baseline_result:
                                fv.run_tahap_2_analisis_temporal(baseline_result)

                        st.write("Tahap 2: Menganalisis Metrik Temporal...")
                        fv.run_tahap_2_analisis_temporal(result, baseline_result)

                        st.write("Tahap 3: Sintesis Bukti & Investigasi Mendalam...")
                        fv.run_tahap_3_sintesis_bukti(result, tmpdir_path)
                        
                        st.write("Tahap 4: Visualisasi & Penilaian Keandalan Bukti...")
                        fv.run_tahap_4_visualisasi_dan_penilaian(result, tmpdir_path)

                        st.write("Tahap 5: Menyusun Laporan PDF & Validasi...")
                        fv.run_tahap_5_pelaporan_dan_validasi(
                            result,
                            tmpdir_path,
                            baseline_result,
                            include_simple=show_simple_explanations,
                            include_technical=show_technical_details,
                        )
                        
                        status.update(label="✅ Analisis 5 Tahap Forensik Berhasil!", state="complete", expanded=False)
                
                except Exception as e:
                    all_stages_success = False
                    st.error(f"Terjadi kesalahan pada saat analisis: {e}")
                    st.code(traceback.format_exc())

                if all_stages_success and result:
                    st.toast("Menyimpan hasil ke riwayat...")
                    history_manager.save_analysis(
                        result,
                        uploaded_video.name,
                        {
                            "fps_awal": int(fps),
                            "has_baseline": baseline_video is not None,
                            "ssim_threshold": float(ssim_slider) if not auto_threshold else fv.CONFIG['SSIM_DISCONTINUITY_DROP'],
                            "z_threshold": float(z_slider) if not auto_threshold else fv.CONFIG['OPTICAL_FLOW_Z_THRESH'],
                            "bypass_debug": bypass_debug
                        },
                    )
                    st.toast("✅ Hasil analisis berhasil disimpan!")

                    with st.spinner("Mengemas hasil akhir untuk ditampilkan..."):
                        for key in list(result.plots.keys()):
                            path = result.plots.get(key)
                            if path:
                                result.plots[f'{key}_bytes'] = load_image_as_bytes(path)
                        
                        if result.kmeans_artifacts.get('distribution_plot_path'):
                            result.kmeans_artifacts['distribution_plot_bytes'] = load_image_as_bytes(result.kmeans_artifacts['distribution_plot_path'])
                        for c in result.kmeans_artifacts.get('clusters', []):
                            if c.get('palette_path'): c['palette_bytes'] = load_image_as_bytes(c['palette_path'])
                            if c.get('samples_montage_path'): c['samples_montage_bytes'] = load_image_as_bytes(c['samples_montage_path'])
                        
                        if result.localizations:
                            for loc in result.localizations:
                                for key in ['image', 'ela_path', 'sift_path']:
                                    if loc.get(key): loc[f'{key}_bytes'] = load_image_as_bytes(loc.get(key))
                                for v_key, v_path in loc.get('visualizations', {}).items():
                                    if v_path:
                                        img_bytes = load_image_as_bytes(v_path)
                                        if img_bytes: loc[f'{v_key}_bytes'] = img_bytes
                        
                        if result.frames and result.frames[0].img_path_comparison:
                            result.frames[0].comparison_bytes = load_image_as_bytes(result.frames[0].img_path_comparison)
                            
                        if result.pdf_report_path:
                            result.pdf_report_data = load_image_as_bytes(result.pdf_report_path)
                            
                    st.success("Analisis selesai. Hasil ditampilkan di bawah ini.")
                    
                    tab_titles = ["📄 **Tahap 1: Akuisisi & K-Means**", "📊 **Tahap 2: Analisis Temporal**", "🔬 **Tahap 3: Investigasi**", "📈 **Tahap 4: Visualisasi & Lokalisasi**", "🧠 **Analisis FERM**", "📥 **Tahap 5: Laporan**"]
                    tabs = st.tabs(tab_titles)
    
                    with tabs[0]:
                        st.header("Hasil Tahap 1: Akuisisi & Ekstraksi Fitur Dasar")
                        st.info("Tujuan: Mengamankan bukti, mengekstrak metadata, menormalisasi frame, dan menerapkan **Metode Utama K-Means** untuk mengklasifikasikan adegan video.", icon="🛡️")
                        st.subheader("1.1. Identifikasi dan Preservasi Bukti")
                        c1, c2 = st.columns(2)
                        c1.metric("Total Frame Dianalisis", result.summary.get('total_frames', 'T/A'))
                        c2.write("**Hash Integritas (SHA-256)**"); c2.code(result.preservation_hash, language="bash")
                        with st.expander("Tampilkan Metadata Video Lengkap"):
                            for category, items in result.metadata.items():
                                st.write(f"**{category}**")
                                df = pd.DataFrame.from_dict(items, orient='index', columns=['Value'])
                                st.table(df)
                        st.subheader("1.2. Ekstraksi dan Normalisasi Frame")
                        st.write("Setiap frame diekstrak dan dinormalisasi untuk konsistensi analisis.")
                        if result.frames and hasattr(result.frames[0], 'comparison_bytes') and result.frames[0].comparison_bytes:
                            st.image(result.frames[0].comparison_bytes, caption="Kiri: Original, Kanan: Normalized (Contrast-Enhanced)")
                        st.subheader("1.3. Hasil Detail Analisis K-Means")
                        st.write(f"Frame-frame dikelompokkan ke dalam **{len(result.kmeans_artifacts.get('clusters', []))} klaster** berdasarkan kemiripan warna.")
                        if result.kmeans_artifacts.get('distribution_plot_bytes'):
                            st.image(result.kmeans_artifacts['distribution_plot_bytes'], caption="Distribusi jumlah frame untuk setiap klaster warna.")
                        st.write("**Eksplorasi Setiap Klaster:**")
                        if result.kmeans_artifacts.get('clusters'):
                            cluster_tabs = st.tabs([f"Klaster {c['id']}" for c in result.kmeans_artifacts.get('clusters', [])])
                            for i, cluster_tab in enumerate(cluster_tabs):
                                with cluster_tab:
                                    cluster_data = result.kmeans_artifacts['clusters'][i]
                                    st.metric("Jumlah Frame dalam Klaster Ini", f"{cluster_data['count']}")
                                    c1, c2 = st.columns([1,2])
                                    with c1:
                                        st.write("**Palet Warna Dominan**")
                                        if cluster_data.get('palette_bytes'): st.image(cluster_data['palette_bytes'])
                                    with c2:
                                        st.write("**Contoh Frame dari Klaster Ini (Gambar Asli)**")
                                        if cluster_data.get('samples_montage_bytes'): st.image(cluster_data['samples_montage_bytes'])
                        else:
                            st.warning("Tidak ada data klaster K-Means yang dapat ditampilkan.")

                    with tabs[1]:
                        st.header("Hasil Tahap 2: Analisis Anomali Temporal")
                        st.info("Tujuan: Menganalisis hubungan antar frame berurutan untuk mendeteksi diskontinuitas.", icon="📈")
                        st.subheader("2.1. Visualisasi Klasterisasi Warna K-Means (Sepanjang Waktu)")
                        st.write("Plot ini menunjukkan bagaimana setiap frame dikelompokkan ke dalam klaster warna tertentu. Lompatan yang tajam sering mengindikasikan perubahan adegan yang mendadak.")
                        if result.plots.get('kmeans_temporal_bytes'):
                            st.image(result.plots['kmeans_temporal_bytes'], caption="Lompatan vertikal yang tajam menandakan perubahan adegan mendadak.")
                        st.subheader("2.2. Analisis Skor SSIM (Structural Similarity Index)")
                        st.write("SSIM mengukur kemiripan struktural antara dua gambar. Penurunan drastis pada skor SSIM merupakan indikator kuat adanya diskontinuitas.")
                        if result.plots.get('ssim_temporal_bytes'):
                            st.image(result.plots['ssim_temporal_bytes'], caption="Penurunan tajam mengindikasikan diskontinuitas.")
                        st.subheader("2.3. Analisis Magnitudo Aliran Optik")
                        st.write("Aliran Optik mengukur gerakan piksel antar frame. Lonjakan besar dapat mengindikasikan perubahan adegan yang tiba-tiba atau transisi paksa.")
                        if result.plots.get('optical_flow_temporal_bytes'):
                            st.image(result.plots['optical_flow_temporal_bytes'], caption="Lonjakan menunjukkan perubahan mendadak atau pergerakan tidak wajar.")
                        st.subheader("2.4. Distribusi Metrik Anomali")
                        st.write("Histogram ini menunjukkan distribusi keseluruhan skor SSIM dan pergerakan Aliran Optik. Ini membantu mengidentifikasi apakah nilai-nilai anomali benar-benar menonjol dari perilaku normal video.")
                        if result.plots.get('metrics_histograms_bytes'):
                             st.image(result.plots['metrics_histograms_bytes'], caption="Distribusi skor SSIM dan Aliran Optik di seluruh video.")
                        if baseline_result:
                            st.subheader("2.5. Analisis Komparatif (dengan Video Baseline)")
                            insertion_events_count = len([loc for loc in result.localizations if loc['event'] == 'anomaly_insertion'])
                            st.info(f"Ditemukan **{insertion_events_count} peristiwa penyisipan** yang tidak ada di video baseline.", icon="🔎")

                    with tabs[2]:
                        st.header("Hasil Tahap 3: Investigasi Detail")
                        st.info("Inti analisis: mengkorelasikan temuan dan melakukan investigasi mendalam dengan metode pendukung (ELA dan SIFT+RANSAC).", icon="🔬")
                        if result.statistical_summary:
                            st.subheader("📊 Ringkasan Statistik Investigasi")
                            col1, col2 = st.columns(2)
                            col1.metric("Total Anomali", result.statistical_summary['total_anomalies'])
                            col2.metric("Kluster Temporal", result.statistical_summary['temporal_clusters'])
                        if not result.localizations: 
                            st.success("🎉 **Tidak Ditemukan Anomali Signifikan.**")
                        else:
                            st.warning(f"🚨 Ditemukan **{len(result.localizations)} peristiwa anomali** yang signifikan:", icon="🚨")
                            for i, loc in enumerate(result.localizations):
                                event_type = loc['event'].replace('anomaly_', '').capitalize()
                                confidence = loc.get('confidence', 'T/A')
                                conf_emoji = "🟩" if confidence == "RENDAH" else "🟨" if confidence == "SEDANG" else "🟧" if confidence == "TINGGI" else "🟥"
                                with st.expander(f"{conf_emoji} **Peristiwa #{i+1}: {event_type}** @ {loc['start_ts']:.2f} - {loc['end_ts']:.2f} detik (Keyakinan: {confidence})", expanded=(i == 0)):
                                    col1, col2 = st.columns(2)
                                    col1.metric("Tipe Anomali", event_type)
                                    col2.metric("Durasi", f"{loc['end_ts'] - loc['start_ts']:.2f} detik")
                                    if show_simple_explanations and loc.get('explanations'):
                                        for exp_type, exp_data in loc['explanations'].items():
                                            if isinstance(exp_data, dict) and exp_data.get('simple_explanation'):
                                                st.markdown(f'<div class="explanation-box">{exp_data["simple_explanation"]}</div>', unsafe_allow_html=True)
                                    if show_technical_details and isinstance(loc.get('metrics'), dict) and loc['metrics']:
                                        st.markdown("### 🔧 Detail Teknis")
                                        metrics_df = pd.DataFrame([{'Metrik': k.replace('_', ' ').title(), 'Nilai': str(v)} for k, v in loc['metrics'].items()])
                                        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                                    st.markdown("### 🖼️ Bukti Visual")
                                    visual_cols = st.columns(2)
                                    if loc.get('image_bytes'): visual_cols[0].image(loc['image_bytes'], caption=f"Frame #{loc.get('start_frame')}")
                                    if loc.get('ela_path_bytes'): visual_cols[1].image(loc['ela_path_bytes'], caption="Analisis ELA")
                                    if loc.get('sift_path_bytes'): st.image(loc.get('sift_path_bytes'), caption="Bukti Pencocokan Fitur SIFT+RANSAC", use_container_width=True)

                    with tabs[3]:
                        st.header("Hasil Tahap 4: Visualisasi & Lokalisasi")
                        st.info("Tahap ini menyajikan penilaian keandalan bukti dan mengelompokkan anomali menjadi peristiwa yang mudah dipahami.", icon="📈")
                        
                        # ======================= KEY NAME FIX START =======================
                        with st.container(border=True):
                            st.subheader("🗺️ Peta Ringkas Anomali Temporal")
                            st.write("Peta ini memberikan ringkasan visual sederhana dari semua jenis anomali yang terdeteksi di sepanjang linimasa video (frame demi frame). Ini berguna untuk melihat pola mentah.")
                            # Kunci yang benar adalah 'temporal_bytes', bukan 'anomaly_map_temporal_bytes'
                            if result.plots.get('temporal_bytes'):
                                st.image(result.plots.get('temporal_bytes'), caption="Peta Anomali Temporal (Duplikasi, Penyisipan, Diskontinuitas).", use_container_width=True)
                            else:
                                st.warning("Peta ringkas anomali temporal tidak tersedia untuk ditampilkan.")

                        st.markdown("---")
                        
                        with st.container(border=True):
                            st.subheader("📊 Peta Detail Lokalisasi Tampering")
                            st.write("Visualisasi ini menggabungkan anomali ke dalam 'peristiwa', menampilkan tingkat kepercayaan, dan menyediakan statistik ringkasan dalam satu dasbor komprehensif.")
                            if result.plots.get('enhanced_localization_map_bytes'):
                                st.image(result.plots.get('enhanced_localization_map_bytes'), caption="Peta detail lokalisasi tampering dengan timeline, statistik, dan tingkat kepercayaan.", use_container_width=True)
                            else:
                                st.warning("Peta detail lokalisasi tidak tersedia untuk ditampilkan.")
                        
                        st.markdown("---")
                        
                        st.subheader("⚙️ Penilaian Kualitas Pipeline Forensik")
                        if hasattr(result, 'pipeline_assessment') and result.pipeline_assessment:
                            for stage_id, assessment in result.pipeline_assessment.items():
                                st.markdown(f"""<div class="pipeline-stage-card"><h4>{assessment['nama']}</h4><div style="display: flex; justify-content: space-between;"><span>Status: <b>{assessment['status'].upper()}</b></span><span>Quality Score: <b>{assessment['quality_score']}%</b></span></div></div>""", unsafe_allow_html=True)
                                if assessment['issues']:
                                    for issue in assessment['issues']: st.warning(f"⚠️ {issue}")
                        
                    with tabs[4]:
                        display_ferm_tab_content(result)
                        
                    with tabs[5]:
                        st.header("Hasil Tahap 5: Laporan & Validasi")
                        st.info("Unduh laporan PDF lengkap untuk dokumentasi atau sebagai lampiran bukti digital.", icon="📄")

                        # ===== GANTI BLOK INI =====
                        
                        # Cek apakah PDF berhasil dibuat
                        if hasattr(result, 'pdf_report_path') and result.pdf_report_path and Path(result.pdf_report_path).exists():
                            # PDF ada, tampilkan semua tombol unduh
                            col1, col2, col3 = st.columns(3)

                            # 1. Tombol PDF
                            with col1:
                                with open(result.pdf_report_path, "rb") as pdf_file:
                                    st.download_button(
                                        label="📄 Download PDF",
                                        data=pdf_file.read(),
                                        file_name=Path(result.pdf_report_path).name,
                                        mime="application/pdf",
                                        use_container_width=True
                                    )
                            
                            # 2. Tombol PNG
                            with col2:
                                if hasattr(result, 'png_export_paths') and result.png_export_paths:
                                    with st.popover("🖼️ Unduh Halaman PNG", use_container_width=True):
                                        for png_path_str in result.png_export_paths:
                                            png_path = Path(png_path_str)
                                            if png_path.exists():
                                                with open(png_path, "rb") as png_file:
                                                    st.download_button(
                                                        label=f"Unduh {png_path.name}",
                                                        data=png_file.read(),
                                                        file_name=png_path.name,
                                                        mime="image/png"
                                                    )
                                else:
                                    st.warning("PNG tidak dibuat. Periksa dependensi (`pdf2image` + Poppler).", icon="⚠️")

                            # 3. Tombol DOCX
                            with col3:
                                if hasattr(result, 'docx_report_path') and result.docx_report_path:
                                    docx_path = Path(result.docx_report_path)
                                    if docx_path.exists():
                                        with open(docx_path, "rb") as docx_file:
                                            st.download_button(
                                                label="📝 Download DOCX",
                                                data=docx_file.read(),
                                                file_name=docx_path.name,
                                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                                use_container_width=True
                                            )
                                else:
                                    st.warning("DOCX tidak dibuat. Periksa dependensi `python-docx`.", icon="⚠️")

                        else:
                            st.error("Gagal membuat laporan PDF utama. Periksa log di konsol untuk detail error.", icon="❌")
                        # ===== AKHIR DARI BLOK PENGGANTI =====

                        st.subheader("✅ Validasi Proses Analisis")
                        validation_data = {"File Bukti": Path(result.video_path).name, "Hash SHA-256": result.preservation_hash, "Waktu Analisis (UTC)": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        if hasattr(result, 'forensic_evidence_matrix') and result.forensic_evidence_matrix.get('conclusion'):
                            validation_data["Penilaian Reliabilitas FERM"] = result.forensic_evidence_matrix['conclusion']['reliability_assessment']
                        validation_df = pd.DataFrame.from_dict(validation_data, orient='index', columns=['Detail'])
                        st.table(validation_df)
                        st.markdown("### 🎯 Kesimpulan Analisis")
                        reliability = result.forensic_evidence_matrix.get('conclusion', {}).get('reliability_assessment', 'Tidak Dapat Ditentukan')
                        st.info(f"Berdasarkan analisis FERM, penilaian reliabilitas bukti adalah: **{reliability}**.")
# Logic to render the correct page based on the selected tab in the sidebar
else: # This implicitly means selected_tab == "Riwayat Analisis"
    render_history_page()

# ======================= LOGIC FIX END =========================