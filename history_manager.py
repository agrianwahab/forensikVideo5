# --- START OF FILE history_manager.py ---

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
import shutil
import uuid
import base64
import zipfile
import io

class HistoryManager:
    """
    Kelas untuk mengelola riwayat analisis forensik video secara komprehensif.
    Menyimpan data, artefak, dan menyediakan fungsi untuk antarmuka pengguna yang detail.
    """
    
    def __init__(self, history_file="analysis_history.json", history_folder="analysis_artifacts"):
        """
        Inisialisasi History Manager.
        
        Args:
            history_file (str): Nama file JSON untuk menyimpan data riwayat.
            history_folder (str): Folder untuk menyimpan semua artefak visual (plot, gambar).
        """
        self.history_file = Path(history_file)
        self.history_folder = Path(history_folder)
        self.history_folder.mkdir(exist_ok=True)

        # ====== [NEW] False-Positive Fix June-2025 ======
        self.db_path = Path("analysis_settings.db")
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS settings (id TEXT PRIMARY KEY, video_name TEXT, timestamp TEXT, fps_awal REAL, fps_baru REAL, ssim_thresh REAL, z_thresh REAL)"
        )
        conn.commit()
        conn.close()
        # ====== [END NEW] ======
        
        # Buat file riwayat jika belum ada dengan struktur list kosong.
        if not self.history_file.exists():
            with open(self.history_file, 'w') as f:
                json.dump([], f)
    
    def save_analysis(self, result, video_name, additional_info=None):
        """
        Menyimpan hasil analisis lengkap ke dalam file riwayat.
        
        Args:
            result: Objek AnalysisResult dari ForensikVideo.
            video_name (str): Nama file video yang dianalisis.
            additional_info (dict): Informasi tambahan opsional seperti FPS.
            
        Returns:
            str: ID unik dari entri riwayat yang baru saja disimpan.
        """
        analysis_id = str(uuid.uuid4())
        
        # Buat sub-folder spesifik untuk artefak analisis ini.
        artifact_folder = self.history_folder / analysis_id
        artifact_folder.mkdir(exist_ok=True)
        
        # Salin artefak visual penting ke folder riwayat.
        saved_artifacts = self._save_artifacts(result, artifact_folder)
        
        # ======================= FIX START =======================
        # Struktur data riwayat yang akan disimpan ke JSON.
        # Menggunakan 'forensic_evidence_matrix' yang baru, bukan 'integrity_analysis' yang lama.
        history_entry = {
            "id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "video_name": video_name,
            "artifacts_folder": str(artifact_folder),
            "preservation_hash": result.preservation_hash,
            "summary": result.summary,
            "metadata": result.metadata,
            "forensic_evidence_matrix": result.forensic_evidence_matrix if hasattr(result, 'forensic_evidence_matrix') else None,
            "localization_details": result.localization_details if hasattr(result, 'localization_details') else None,
            "pipeline_assessment": result.pipeline_assessment if hasattr(result, 'pipeline_assessment') else None,
            "localizations": result.localizations,
            "localizations_count": len(result.localizations),
            "anomaly_types": self._count_anomaly_types(result),
            "saved_artifacts": saved_artifacts,
            "additional_info": additional_info if additional_info else {},
            "report_paths": {
                "pdf": str(result.pdf_report_path) if result.pdf_report_path else None,
                "html": str(getattr(result, 'html_report_path', '')) or None,
                "json": str(getattr(result, 'json_report_path', '')) or None,
            }
        }
        # ======================= FIX END =======================
        
        history = self.load_history()
        history.append(history_entry)
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=4)

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO settings(id, video_name, timestamp, fps_awal, fps_baru, ssim_thresh, z_thresh) VALUES (?,?,?,?,?,?,?)",
                (
                    analysis_id,
                    video_name,
                    history_entry["timestamp"],
                    additional_info.get("fps_awal") if additional_info else None,
                    additional_info.get("fps_baru") if additional_info else None,
                    additional_info.get("ssim_threshold") if additional_info else None,
                    additional_info.get("z_threshold") if additional_info else None,
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

        return analysis_id
    
    def load_history(self):
        """
        Memuat seluruh riwayat analisis dari file JSON.
        
        Returns:
            list: Daftar semua entri riwayat analisis.
        """
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            with open(self.history_file, 'w') as f:
                json.dump([], f)
            return []
    
    def get_analysis(self, analysis_id):
        """
        Mengambil satu entri riwayat berdasarkan ID uniknya.
        
        Args:
            analysis_id (str): ID analisis yang dicari.
            
        Returns:
            dict: Entri riwayat yang ditemukan, atau None jika tidak ada.
        """
        history = self.load_history()
        return next((entry for entry in history if entry["id"] == analysis_id), None)
    
    def delete_analysis(self, analysis_id):
        """
        Menghapus satu entri riwayat dan semua artefak terkait.
        
        Args:
            analysis_id (str): ID analisis yang akan dihapus.
            
        Returns:
            bool: True jika berhasil dihapus, False jika tidak.
        """
        history = self.load_history()
        
        entry_to_delete = self.get_analysis(analysis_id)
        if not entry_to_delete:
            return False

        artifact_folder = Path(entry_to_delete.get("artifacts_folder", ""))
        if artifact_folder.exists() and artifact_folder.is_dir():
            shutil.rmtree(artifact_folder)
        
        updated_history = [entry for entry in history if entry["id"] != analysis_id]
        
        with open(self.history_file, 'w') as f:
            json.dump(updated_history, f, indent=4)
                
        return True
    
    def delete_all_history(self):
        """
        Menghapus SEMUA riwayat analisis dan semua artefaknya. Operasi ini tidak dapat diurungkan.
        
        Returns:
            int: Jumlah entri yang berhasil dihapus.
        """
        history = self.load_history()
        count = len(history)
        
        if self.history_folder.exists():
            shutil.rmtree(self.history_folder)
        self.history_folder.mkdir(exist_ok=True)
        
        with open(self.history_file, 'w') as f:
            json.dump([], f)

        return count

    def _generate_html_report(self, entry):
        """
        Membangun laporan HTML komprehensif yang merangkum seluruh tahap DFRWS beserta visualisasi-visualisasi kunci.
        
        Args:
            entry (dict): Data entri riwayat yang akan dikonversi menjadi laporan HTML
            
        Returns:
            str: Konten HTML laporan
        """
        # Fungsi helper untuk membuat judul seksi HTML
        def section_header(title, level=2):
            return f"<h{level} style='color:#0B3D91;border-bottom:1px solid #ddd;padding-bottom:8px;margin-top:25px;'>{title}</h{level}>"
        
        # Fungsi helper untuk menyiapkan path gambar relatif dari artefak
        def get_artifact_relative_path(path_key):
            if path_key in entry.get("saved_artifacts", {}):
                return f"artifacts/{Path(entry['saved_artifacts'][path_key]).name}"
            return None
            
        # Metadata dasar
        ferm = entry.get("forensic_evidence_matrix", {})
        reliability = ferm.get("conclusion", {}).get("reliability_assessment", "N/A")
        video_name = entry.get("video_name", "Unnamed Video")
        timestamp = entry.get("timestamp", "Unknown Date")
        
        # Warna dan style berdasarkan penilaian reliabilitas
        reliability_color = "#28a745" if "Tinggi" in reliability else "#ffc107" if "Sedang" in reliability else "#dc3545"
        
        # Mulai membangun HTML
        html = [
            "<!DOCTYPE html>",
            "<html lang='id'>",
            "<head>",
            "    <meta charset='utf-8'>",
            "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"    <title>Laporan Forensik Video - {video_name}</title>",
            "    <style>",
            "        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }",
            "        h1, h2, h3, h4 { color: #0B3D91; }",
            "        h1 { text-align: center; border-bottom: 2px solid #0B3D91; padding-bottom: 10px; }",
            "        img { max-width: 100%; height: auto; border-radius: 5px; margin: 15px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }",
            "        .metadata { background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #0B3D91; }",
            "        .reliability-badge { display: inline-block; padding: 8px 15px; color: white; font-weight: bold; border-radius: 4px; margin: 10px 0; }",
            "        .two-column { display: flex; flex-wrap: wrap; gap: 20px; }",
            "        .two-column > div { flex: 1; min-width: 300px; }",
            "        .explanation-box { background-color: #f0f7ff; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #0c6dd6; }",
            "        .disclaimer { background-color: #ffe6cc; padding: 15px; border-radius: 8px; margin: 20px 0; font-style: italic; }",
            "        .method-card { background-color: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; }",
            "        .method-card h3 { margin-top: 0; border-bottom: 1px solid #eee; padding-bottom: 8px; }",
            "        .anomaly-type { margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #eee; }",
            "        .findings-section { background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }",
            "        .metric { display: inline-block; background-color: #e9ecef; padding: 5px 10px; border-radius: 4px; margin: 3px; font-size: 0.9em; }",
            "        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 0.9em; color: #666; }",
            "    </style>",
            "</head>",
            "<body>",
            
            # Header dan metadata
            f"<h1>Laporan Analisis Forensik Video</h1>",
            f"<div class='metadata'>",
            f"    <p><strong>File:</strong> {video_name}</p>",
            f"    <p><strong>Waktu Analisis:</strong> {self._format_timestamp(timestamp)}</p>",
            f"    <p><strong>Hash SHA-256:</strong> {entry.get('preservation_hash', 'N/A')[:20]}...</p>",
            f"    <p><strong>Penilaian Reliabilitas:</strong> <span class='reliability-badge' style='background-color:{reliability_color}'>{reliability}</span></p>",
            f"</div>",
            
            # Ringkasan eksekutif
            section_header("Ringkasan Eksekutif"),
            "<p>Analisis forensik video ini dilakukan menggunakan metodologi standar Digital Forensics Research Workshop (DFRWS) yang ",
            "terdiri dari enam tahap: Identifikasi, Preservasi, Pengumpulan, Pemeriksaan, Analisis, dan Pelaporan. ",
            "Sistem menggunakan metode utama <strong>K-Means</strong> dan <strong>Localization Tampering</strong> dengan ",
            "dukungan <strong>Error Level Analysis (ELA)</strong> dan <strong>Scale-Invariant Feature Transform (SIFT)</strong>.</p>",
        ]
        
        # Tambahkan disclaimer forensik
        html.extend([
            "<div class='disclaimer'>",
            "    <p><strong>CATATAN PENTING:</strong> Hasil analisis yang disajikan dalam laporan ini adalah produk dari sistem otomatis ",
            "    forensik video. Meskipun dirancang menggunakan metodologi dan algoritma ilmiah, semua temuan harus divalidasi dan ",
            "    diinterpretasikan lebih lanjut oleh ahli forensik video yang berkualifikasi. Sistem hanya dapat mengidentifikasi anomali ",
            "    berdasarkan pola statistik dan visual; interpretasi akhir tentang implikasi forensik dan konteks faktual dari anomali ",
            "    tersebut memerlukan penilaian manusia.</p>",
            "</div>",
        ])
        
        # Tambahkan temuan utama jika ada
        primary_findings = ferm.get("conclusion", {}).get("primary_findings", [])
        if primary_findings:
            html.append("<div class='findings-section'>")
            html.append("<h3>Temuan Utama:</h3>")
            html.append("<ul>")
            for finding in primary_findings:
                html.append(f"<li><strong>{finding.get('finding', '')}</strong> ({finding.get('confidence', 'N/A')})")
                html.append(f"<p><em>Interpretasi:</em> {finding.get('interpretation', '')}</p></li>")
            html.append("</ul>")
            html.append("</div>")
        
        # Bagian metodologi DFRWS
        html.append(section_header("Metodologi DFRWS", 2))
        html.append("<p>Analisis forensik video ini mengikuti kerangka kerja DFRWS yang terstruktur dalam enam tahap berikut:</p>")
        
        dfrws_phases = [
            {"name": "Identifikasi", "icon": "🔍", "desc": "Mengidentifikasi bukti potensial (video) dan metadata-nya."},
            {"name": "Preservasi", "icon": "🛡️", "desc": "Menjaga integritas bukti dengan hash SHA-256 dan penyimpanan frame asli."},
            {"name": "Pengumpulan", "icon": "📥", "desc": "Ekstraksi frame, normalisasi warna, dan penghitungan pHash."},
            {"name": "Pemeriksaan", "icon": "🔬", "desc": "Analisis temporal, K-Means, Error Level Analysis, dan SIFT+RANSAC."},
            {"name": "Analisis", "icon": "📈", "desc": "Localization Tampering dan Forensic Evidence Reliability Matrix (FERM)."},
            {"name": "Pelaporan", "icon": "📄", "desc": "Dokumentasi sistematis temuan dengan visualisasi dan penjelasan."}
        ]
        
        html.append("<div class='two-column'>")
        for phase in dfrws_phases:
            html.append(f"<div class='method-card'>")
            html.append(f"<h3>{phase['icon']} {phase['name']}</h3>")
            html.append(f"<p>{phase['desc']}</p>")
            html.append(f"</div>")
        html.append("</div>")
        
        # Bagian analisis visual
        html.append(section_header("Visualisasi Analisis Utama", 2))
        html.append("<p>Berikut adalah visualisasi-visualisasi kunci dari hasil analisis forensik:</p>")
        
        # Tambahkan visualisasi K-Means, SSIM, dan Optical Flow jika tersedia
        timeline_visuals = [
            {"key": "kmeans_temporal", "title": "Klasterisasi Warna K-Means Sepanjang Waktu", 
            "desc": "Menunjukkan perpindahan antar klaster warna yang dapat mengindikasikan perubahan adegan atau diskontinuitas."},
            {"key": "ssim_temporal", "title": "Analisis Structural Similarity Index (SSIM)", 
            "desc": "Menampilkan perubahan kemiripan struktural antar frame berurutan. Penurunan tajam mengindikasikan diskontinuitas."},
            {"key": "optical_flow_temporal", "title": "Analisis Aliran Optik (Optical Flow)", 
            "desc": "Mengukur pergerakan piksel antar frame. Lonjakan besar menandakan gerakan abnormal yang mungkin akibat manipulasi."}
        ]
        
        # Tambahkan visualisasi-visualisasi yang ada
        for visual in timeline_visuals:
            img_path = get_artifact_relative_path(visual["key"])
            if img_path:
                html.extend([
                    f"<h3>{visual['title']}</h3>",
                    f"<p>{visual['desc']}</p>",
                    f"<img src='{img_path}' alt='{visual['title']}'>",
                    "<hr>"
                ])
        
        # Tambahkan Peta Lokalisasi dan Infografis Anomali jika tersedia
        enhanced_maps = [
            {"key": "enhanced_localization_map", "title": "Peta Lokalisasi Tampering", 
            "desc": "Visualisasi komprehensif yang menunjukkan lokasi, durasi, dan tingkat kepercayaan peristiwa anomali dalam video."},
            {"key": "anomaly_infographic", "title": "Infografis Penjelasan Anomali", 
            "desc": "Penjelasan visual tentang berbagai jenis anomali, metode deteksinya, dan implikasi forensiknya."}
        ]
        
        for visual in enhanced_maps:
            img_path = get_artifact_relative_path(visual["key"])
            if img_path:
                html.extend([
                    f"<h3>{visual['title']}</h3>",
                    f"<p>{visual['desc']}</p>",
                    f"<img src='{img_path}' alt='{visual['title']}'>",
                    "<hr>"
                ])
        
        # Tambahkan visualisasi FERM jika tersedia
        ferm_visuals = [
            {"key": "ferm_evidence_strength", "title": "Kekuatan Bukti FERM", 
            "desc": "Menunjukkan efektivitas relatif dari berbagai metode deteksi untuk setiap jenis anomali."},
            {"key": "ferm_reliability", "title": "Faktor-faktor Reliabilitas FERM", 
            "desc": "Menampilkan faktor-faktor yang berkontribusi positif atau negatif terhadap penilaian keandalan bukti."}
        ]
        
        html.append("<h3>Analisis Matriks Keandalan Bukti Forensik (FERM)</h3>")
        html.append("<p>FERM adalah pendekatan multi-dimensi untuk menilai keandalan bukti forensik, mempertimbangkan faktor kekuatan bukti, karakteristik anomali, dan analisis kausalitas.</p>")
        
        for visual in ferm_visuals:
            img_path = get_artifact_relative_path(visual["key"])
            if img_path:
                html.extend([
                    f"<h4>{visual['title']}</h4>",
                    f"<p>{visual['desc']}</p>",
                    f"<img src='{img_path}' alt='{visual['title']}'>",
                ])
        
        # Bagian detail anomali jika ada
        localizations = entry.get("localizations", [])
        if localizations:
            html.append(section_header("Detail Peristiwa Anomali", 2))
            html.append(f"<p>Analisis menemukan <strong>{len(localizations)} peristiwa anomali</strong> dalam video ini:</p>")
            
            for i, loc in enumerate(localizations):
                event_type = loc.get('event', '').replace('anomaly_', '')
                anomaly_desc = self.get_anomaly_description(event_type)
                
                html.append(f"<div class='anomaly-type'>")
                html.append(f"<h3>{anomaly_desc['icon']} Peristiwa #{i+1}: {anomaly_desc['title']}</h3>")
                html.append(f"<p><strong>Lokasi:</strong> {loc.get('start_ts', 0):.2f}s - {loc.get('end_ts', 0):.2f}s (Durasi: {loc.get('duration', 0):.2f}s)</p>")
                html.append(f"<p><strong>Tingkat Kepercayaan:</strong> {loc.get('confidence', 'N/A')}</p>")
                
                html.append("<div class='explanation-box'>")
                html.append(f"<p><strong>Penjelasan:</strong> {anomaly_desc['simple']}</p>")
                html.append(f"<p><strong>Implikasi Forensik:</strong> {anomaly_desc['implication']}</p>")
                html.append("</div>")
                
                # Tambahkan bukti visual jika tersedia
                for artifact_key in ['anomaly_frame_0', 'anomaly_frame_1', 'anomaly_frame_2']:
                    img_path = get_artifact_relative_path(artifact_key)
                    if img_path:
                        html.append(f"<p><strong>Bukti Visual:</strong></p>")
                        html.append(f"<img src='{img_path}' alt='Bukti visual anomali'>")
                        break
                
                # Tambahkan metrik teknis
                if loc.get("metrics"):
                    html.append("<p><strong>Metrik Teknis:</strong></p>")
                    html.append("<div>")
                    for key, value in loc["metrics"].items():
                        html.append(f"<span class='metric'>{key.replace('_', ' ').title()}: {value}</span>")
                    html.append("</div>")
                
                html.append("</div>")
        else:
            html.append(section_header("Tidak Ditemukan Anomali Signifikan", 2))
            html.append("<p>Analisis forensik tidak menemukan bukti anomali yang signifikan dalam video ini.</p>")
        
        # Kesimpulan
        html.append(section_header("Kesimpulan", 2))
        if len(localizations) > 0:
            html.append(f"<p>Berdasarkan analisis forensik 5 tahap yang telah dilakukan, video \"{video_name}\" memiliki penilaian reliabilitas \"{reliability}\". Sistem telah mendeteksi {len(localizations)} peristiwa anomali yang memerlukan perhatian.</p>")
        else:
            html.append(f"<p>Berdasarkan analisis forensik 5 tahap yang telah dilakukan, video \"{video_name}\" memiliki penilaian reliabilitas \"{reliability}\". Sistem tidak mendeteksi adanya peristiwa anomali yang signifikan dalam video ini.</p>")
        
        html.append("<p>PENTING: Hasil analisis ini adalah produk dari sistem otomatis, dan meskipun menggunakan metodologi DFRWS yang diakui secara profesional, penting untuk dipahami bahwa penilaian akhir dan interpretasi temuan memerlukan validasi dan analisis lebih lanjut oleh ahli forensik video berkualifikasi. Sistem hanya menganalisis temuan yang terdeteksi melalui algoritma; interpretasi kontekstual dan legal dari temuan tersebut berada di luar kemampuan sistem dan memerlukan penilaian manusia.</p>")
        
        # Footer
        html.extend([
            "<div class='footer'>",
            f"    <p>Laporan ini dihasilkan oleh Sistem VIFA-Pro pada {self._format_timestamp(timestamp)}</p>",
            "    <p>© VIFA-Pro - Sistem Deteksi Forensik Keaslian Video</p>",
            "</div>",
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html)
        
    def _format_timestamp(self, iso_timestamp):
        """
        Memformat timestamp ISO menjadi format yang lebih mudah dibaca.
        """
        try:
            dt = datetime.fromisoformat(iso_timestamp)
            return dt.strftime("%d %B %Y, %H:%M:%S")
        except (ValueError, TypeError):
            return iso_timestamp    
    def export_analysis(self, analysis_id):
        """
        Mengekspor data analisis lengkap (metadata + artefak) sebagai file ZIP.
        
        Args:
            analysis_id (str): ID analisis yang akan diekspor.
            
        Returns:
            bytes: Data file ZIP dalam bentuk bytes, atau None jika gagal.
        """
        entry = self.get_analysis(analysis_id)
        if not entry:
            return None
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            report_data = json.dumps(entry, indent=4)
            zip_file.writestr('analysis_report.json', report_data)

            html_report = self._generate_html_report(entry)
            zip_file.writestr('analysis_report.html', html_report)

            artifact_folder = Path(entry.get("artifacts_folder", ""))
            if artifact_folder.exists():
                for artifact in artifact_folder.glob('**/*'):
                    if artifact.is_file():
                        arcname = artifact.relative_to(artifact_folder.parent)
                        zip_file.write(artifact, arcname=f'artifacts/{artifact.name}')
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    def _count_anomaly_types(self, result):
        """Helper untuk menghitung jumlah setiap jenis anomali."""
        counts = {"duplication": 0, "insertion": 0, "discontinuity": 0}
        for loc in getattr(result, 'localizations', []):
            event_type = loc.get('event', '').replace('anomaly_', '')
            if event_type in counts:
                counts[event_type] += 1
        return counts
    
    def _save_artifacts(self, result, folder):
        """Helper untuk menyalin artefak visual penting ke folder riwayat."""
        saved = {}
        
        for plot_name, plot_path in result.plots.items():
            if isinstance(plot_path, (str, Path)) and os.path.exists(plot_path):
                target_path = folder / Path(plot_path).name
                shutil.copy(plot_path, target_path)
                saved[plot_name] = str(target_path)
        
        localizations = getattr(result, 'localizations', [])
        for i, loc in enumerate(localizations[:3]): # Simpan hanya 3 contoh anomali pertama
            if loc.get('image') and os.path.exists(loc['image']):
                target_path = folder / f"sample_anomaly_frame_{i}.jpg"
                shutil.copy(loc['image'], target_path)
                saved[f"anomaly_frame_{i}"] = str(target_path) # Kunci menjadi 'anomaly_frame_0', 'anomaly_frame_1', dll

        if getattr(result, 'pdf_report_path', None) and os.path.exists(result.pdf_report_path):
            target_path = folder / Path(result.pdf_report_path).name
            shutil.copy(result.pdf_report_path, target_path)
            saved['pdf_report'] = str(target_path)
        if getattr(result, 'html_report_path', None) and os.path.exists(result.html_report_path):
            target_path = folder / Path(result.html_report_path).name
            shutil.copy(result.html_report_path, target_path)
            saved['html_report'] = str(target_path)
        if getattr(result, 'json_report_path', None) and os.path.exists(result.json_report_path):
            target_path = folder / Path(result.json_report_path).name
            shutil.copy(result.json_report_path, target_path)
            saved['json_report'] = str(target_path)
        
        return saved
    
    def get_artifact_base64(self, artifact_path):
        """Mengonversi file gambar artefak menjadi string base64 untuk ditampilkan di web."""
        path = Path(artifact_path)
        if not path.is_file():
            return None
            
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            mime_type = "image/png" if path.suffix.lower() == '.png' else "image/jpeg"
            return f"data:{mime_type};base64,{data}"
        except Exception:
            return None

    def get_anomaly_description(self, anomaly_type):
        """Menyediakan deskripsi lengkap untuk setiap jenis anomali."""
        descriptions = {
            "duplication": {
                "title": "Duplikasi Frame", "icon": "🔁", "color": "#FF6B6B",
                "simple": "Frame yang sama diulang beberapa kali dalam video.",
                "technical": "Dideteksi melalui perbandingan pHash dan dikonfirmasi dengan SIFT+RANSAC yang menemukan kecocokan fitur yang sangat tinggi antar frame.",
                "implication": "Ini bisa menjadi indikasi untuk memperpanjang durasi secara artifisial atau untuk menyembunyikan/menutupi konten yang telah dihapus di antara frame yang diduplikasi.",
                "example": "Seperti Anda menyalin sebuah halaman dari buku dan menempelkannya lagi di tempat lain untuk membuat buku terlihat lebih tebal."
            },
            "discontinuity": {
                "title": "Diskontinuitas Video", "icon": "✂️", "color": "#45B7D1",
                "simple": "Terjadi 'lompatan' atau patahan mendadak dalam aliran visual atau gerakan video.",
                "technical": "Dideteksi melalui penurunan drastis pada skor SSIM (kemiripan struktural) atau lonjakan tajam pada magnitudo Optical Flow (aliran gerakan).",
                "implication": "Seringkali ini adalah tanda kuat dari pemotongan (cut) dan penyambungan (paste) video. Aliran alami video terganggu.",
                "example": "Bayangkan sebuah kalimat di mana beberapa kata di tengahnya hilang, membuat kalimatnya terasa aneh dan melompat."
            },
            "insertion": {
                "title": "Penyisipan Konten", "icon": "➕", "color": "#4ECDC4",
                "simple": "Adanya frame atau segmen baru yang tidak ada di video asli/baseline.",
                "technical": "Dideteksi secara definitif dengan membandingkan hash setiap frame dari video bukti dengan video baseline. Frame yang ada di bukti tapi tidak di baseline dianggap sebagai sisipan.",
                "implication": "Ini adalah bukti kuat dari penambahan konten yang bisa mengubah konteks atau narasi video secara signifikan.",
                "example": "Seperti menambahkan sebuah paragraf karangan Anda sendiri ke tengah-tengah novel karya orang lain."
            }
        }
        return descriptions.get(anomaly_type, {
            "title": "Anomali Lain", "icon": "❓", "color": "#808080", "simple": "Jenis anomali tidak dikenali.",
            "technical": "-", "implication": "-", "example": "-"
        })

    # ======================= FIX START =======================
    # Menghapus fungsi get_integrity_explanation yang sudah usang
    # def get_integrity_explanation(self, score): ...
    # ======================= FIX END =======================