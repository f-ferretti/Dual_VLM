import os
import re
import cv2
import base64
import logging
import urllib.request
import json
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import io

logger = logging.getLogger(__name__)


def extract_youtube_video_id(url: str) -> Optional[str]:
    if not url:
        return None
    match = re.search(r"(?:v=|\/embed\/|\.be\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None


def clean_youtube_url(url: str) -> str:
    if not url:
        return ""
    stripped = url.strip()
    vid_id = extract_youtube_video_id(stripped)
    if vid_id:
        return f"https://www.youtube.com/watch?v={vid_id}"
    return stripped


def _get_youtube_video_info(url: str) -> Dict[str, Any]:
    clean_url = clean_youtube_url(url)
    info = {
        "duration_seconds": 0.0,
        "fps": 30.0,
        "frame_count": 0,
        "width": 1920,
        "height": 1080,
        "file_size_mb": 0.0,
        "title": "Video YouTube (Streaming Remoto)"
    }
    try:
        req = urllib.request.Request(
            clean_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        title_m = re.search(r"<title>(.*?)</title>", html)
        if title_m:
            raw_title = title_m.group(1).replace(" - YouTube", "").strip()
            if raw_title:
                info["title"] = raw_title

        m_len = re.search(r'"lengthSeconds":"(\d+)"', html)
        if m_len:
            sec = float(m_len.group(1))
            info["duration_seconds"] = sec
            info["frame_count"] = int(sec * 30)
        else:
            m_dur = re.search(r'approxDurationMs\\":\\"(\d+)\\"', html)
            if m_dur:
                sec = round(float(m_dur.group(1)) / 1000.0, 2)
                info["duration_seconds"] = sec
                info["frame_count"] = int(sec * 30)
    except Exception as e:
        logger.warning(f"Impossibile estrarre metadati YouTube per {url}: {e}")

    return info


def get_video_info(video_path: str) -> Dict[str, Any]:
    if "youtube.com" in video_path.lower() or "youtu.be" in video_path.lower():
        return _get_youtube_video_info(video_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"File video non trovato: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossibile aprire il video con OpenCV: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    duration = 0.0
    if fps > 0 and frame_count > 0:
        duration = frame_count / fps

    cap.release()

    return {
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2)
    }


def format_seconds_to_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 10)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}.{millis}"
    return f"{mins:02d}:{secs:02d}.{millis}"


def format_seconds_to_mm_ss(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def extract_base64_frames(video_path: str, max_frames: int = 32, max_dimension: int = 768) -> List[Tuple[float, str]]:
    """
    Estrae una sequenza uniforme di frame dal video con i rispettivi timestamp,
    ridimensionati e codificati in base64.
    Restituisce una lista di tuple: (timestamp_secondi, base64_jpeg).
    """
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0 or fps <= 0:
        cap.release()
        return []

    actual_frames_to_extract = min(max_frames, total_frames)
    frame_indices = [int(i * (total_frames - 1) / max(1, actual_frames_to_extract - 1)) for i in range(actual_frames_to_extract)]

    extracted = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        timestamp_sec = round(idx / fps, 2)
        
        h, w = frame.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        extracted.append((timestamp_sec, b64_str))

    cap.release()
    return extracted


def cut_video_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_dir: str = "results/clips",
    bug_id: str = "BUG",
    margin_sec: float = 1.5
) -> Optional[str]:
    """
    Estrae uno spezzone video temporizzato intorno all'anomalia rilevata.
    Aggiunge un margine di contesto prima e dopo il timestamp del bug per visualizzare l'azione completa.
    """
    import subprocess
    import re

    if not os.path.exists(video_path):
        return None

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        video_info = get_video_info(video_path)
        total_dur = video_info.get("duration_seconds", 0.0)
    except Exception:
        total_dur = 0.0

    actual_start = max(0.0, start_sec - margin_sec)
    if total_dur > 0:
        actual_end = min(total_dur, max(actual_start + 1.0, end_sec + margin_sec))
    else:
        actual_end = max(actual_start + 1.0, end_sec + margin_sec)
        
    clip_duration = max(0.5, actual_end - actual_start)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    safe_base = re.sub(r'[^a-zA-Z0-9_.-]', '_', base_name)
    safe_bug_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', bug_id)
    output_filename = f"{safe_base}_{safe_bug_id}_{actual_start:.1f}s-{actual_end:.1f}s.mp4"
    output_path = os.path.join(output_dir, output_filename)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{actual_start:.2f}",
        "-i", video_path,
        "-t", f"{clip_duration:.2f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-avoid_negative_ts", "make_zero",
        output_path
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except Exception:
        pass

    return None


def compress_video_for_upload(
    video_path: str,
    max_size_mb: float = 12.0,
    output_dir: Optional[str] = None
) -> str:
    """
    Verifica ricorsivamente che il file video rientri sotto max_size_mb.
    Se supera la soglia, applica profili di compressione progressivi (da 720p fino a 360p con target bitrate vincolato)
    ricontrollando ad ogni iterazione la dimensione effettiva fino a garantire che risulti <= max_size_mb.
    """
    import subprocess
    import tempfile

    if not os.path.exists(video_path):
        return video_path

    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if file_size_mb <= max_size_mb:
        return video_path

    if not output_dir:
        output_dir = os.path.join(tempfile.gettempdir(), "dual_llm_compressed")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Calcolo durata per eventuale target bitrate vincolante
    try:
        v_info = get_video_info(video_path)
        duration_sec = max(1.0, v_info.get("duration_seconds", 30.0))
    except Exception:
        duration_sec = 30.0

    # Profili di compressione progressivi ordinati per qualita decrescente
    profiles = [
        {"height": 720, "crf": 26, "audio_k": "128k"},
        {"height": 720, "crf": 30, "audio_k": "96k"},
        {"height": 480, "crf": 28, "audio_k": "64k"},
        {"height": 480, "crf": 32, "audio_k": "64k"},
        {"height": 360, "crf": 34, "audio_k": "48k"}
    ]

    for idx, prof in enumerate(profiles, start=1):
        compressed_path = os.path.join(output_dir, f"{base_name}_step{idx}_{prof['height']}p.mp4")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"scale=-2:{prof['height']}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", str(prof["crf"]),
            "-c:a", "aac",
            "-b:a", prof["audio_k"],
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            compressed_path
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if os.path.exists(compressed_path):
                curr_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                if curr_size_mb <= max_size_mb:
                    return compressed_path
        except Exception:
            continue

    # Fallback con bitrate vincolato per rispetto limite dimensionale
    target_kbit = max(100, int((max_size_mb * 0.80 * 8192) / duration_sec))
    fallback_path = os.path.join(output_dir, f"{base_name}_hard_limit.mp4")
    cmd_fallback = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", "scale=-2:360",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-b:v", f"{target_kbit}k",
        "-maxrate", f"{int(target_kbit * 1.2)}k",
        "-bufsize", f"{target_kbit * 2}k",
        "-c:a", "aac",
        "-b:a", "48k",
        "-pix_fmt", "yuv420p",
        fallback_path
    ]
    try:
        subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if os.path.exists(fallback_path) and os.path.getsize(fallback_path) > 0:
            return fallback_path
    except Exception:
        pass

    return video_path