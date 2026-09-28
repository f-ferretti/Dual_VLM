import os
import sys
import time
import glob
import json
import re
import argparse
import csv
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from pipeline import DualLLMPipeline
from video_utils import get_video_info

load_dotenv()

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def find_video_files(video_dir: str, recursive: bool = False) -> list:
    video_files = []
    p = Path(video_dir)
    if not p.exists() or not p.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    for item in p.glob(pattern):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            video_files.append(str(item.resolve()))
    return sorted(video_files)


def run_batch(
    video_dir: str,
    output_dir: str = "results",
    det_provider: str = "google",
    det_model: str = "gemini-3.8-flash",
    val_provider: str = "google",
    val_model: str = "gemini-3.8-flash",
    recursive: bool = False,
    limit: int = None,
    resume: bool = False,
    enable_agentic_video: bool = True
):
    video_files = find_video_files(video_dir, recursive=recursive)
    if not video_files:
        print(f"Nessun file video supportato trovato in: {video_dir}")
        return

    if limit and limit > 0:
        video_files = video_files[:limit]

    total_videos = len(video_files)
    print(f"Avvio elaborazione batch per {total_videos} video in: {video_dir}")
    print(f"Detector: {det_provider} ({det_model})")
    print(f"Validator: {val_provider} ({val_model})")
    print(f"Cartella Output: {output_dir}")
    print("-" * 70)

    os.makedirs(output_dir, exist_ok=True)

    pipeline = DualLLMPipeline(
        detector_provider=det_provider,
        detector_model=det_model,
        validator_provider=val_provider,
        validator_model=val_model,
        results_dir=output_dir,
        enable_agentic_video=enable_agentic_video
    )

    batch_start_time = time.time()
    batch_records = []
    success_count = 0
    error_count = 0
    skipped_count = 0

    for idx, video_path in enumerate(video_files, start=1):
        filename = os.path.basename(video_path)
        raw_name = os.path.splitext(filename)[0]
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_name)

        if resume:
            existing_reports = glob.glob(os.path.join(output_dir, f"report_{sanitized_name}_*.json"))
            if existing_reports:
                print(f"[{idx:02d}/{total_videos:02d}] {filename} - GIA ELABORATO (Skip per flag --resume)")
                skipped_count += 1
                continue

        print(f"[{idx:02d}/{total_videos:02d}] Elaborazione: {filename}")
        t0 = time.time()

        def progress_cb(msg: str, p: float):
            print(f"   [{int(p * 100):02d}%] {msg}")

        try:
            result = pipeline.run(video_path, progress_callback=progress_cb)
            elapsed = round(time.time() - t0, 2)
            d_rep = result.detector_report
            v_rep = result.validator_report

            confirmed_types = [
                (b.corrected_bug_type.value if b.corrected_bug_type else "unknown")
                for b in v_rep.validated_bugs
                if str(b.verdict).upper().endswith("CONFIRMED")
            ]

            record = {
                "index": idx,
                "video_name": filename,
                "video_path": video_path,
                "duration_seconds": result.video_duration_seconds,
                "status": "SUCCESS",
                "execution_time_seconds": elapsed,
                "detected_bugs_count": len(d_rep.bugs),
                "confirmed_count": v_rep.confirmed_count,
                "rejected_count": v_rep.rejected_count,
                "uncertain_count": v_rep.uncertain_count,
                "confirmed_bug_types": ", ".join(confirmed_types) if confirmed_types else "none",
                "json_report_path": result.saved_json_path or "",
                "error_message": ""
            }
            batch_records.append(record)
            success_count += 1
            print(f"   -> Completato in {elapsed}s | Rilevati: {len(d_rep.bugs)} | Confermati: {v_rep.confirmed_count} | Scartati: {v_rep.rejected_count}")

        except Exception as ex:
            elapsed = round(time.time() - t0, 2)
            error_count += 1
            print(f"   -> ERRORE durante l'elaborazione di {filename}: {ex}")
            record = {
                "index": idx,
                "video_name": filename,
                "video_path": video_path,
                "duration_seconds": 0.0,
                "status": "ERROR",
                "execution_time_seconds": elapsed,
                "detected_bugs_count": 0,
                "confirmed_count": 0,
                "rejected_count": 0,
                "uncertain_count": 0,
                "confirmed_bug_types": "",
                "json_report_path": "",
                "error_message": str(ex)
            }
            batch_records.append(record)

        print("-" * 70)

    total_batch_time = round(time.time() - batch_start_time, 2)
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_json_path = os.path.join(output_dir, f"batch_summary_{timestamp_slug}.json")
    summary_csv_path = os.path.join(output_dir, f"batch_summary_{timestamp_slug}.csv")

    summary_payload = {
        "batch_execution_timestamp": datetime.now().isoformat(),
        "total_videos_found": total_videos,
        "processed_count": len(batch_records),
        "success_count": success_count,
        "error_count": error_count,
        "skipped_count": skipped_count,
        "total_execution_time_seconds": total_batch_time,
        "configuration": {
            "detector_provider": det_provider,
            "detector_model": det_model,
            "validator_provider": val_provider,
            "validator_model": val_model,
            "video_dir": video_dir
        },
        "records": batch_records
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    if batch_records:
        keys = list(batch_records[0].keys())
        with open(summary_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, delimiter=";")
            writer.writeheader()
            writer.writerows(batch_records)

    print("=" * 70)
    print("ESECUZIONE BATCH COMPLETATA")
    print(f"Video Totali: {total_videos} (Successi: {success_count}, Errori: {error_count}, Saltati: {skipped_count})")
    print(f"Tempo Totale Batch: {total_batch_time}s")
    print(f"Report Riepilogativo JSON: {summary_json_path}")
    print(f"Report Riepilogativo CSV:  {summary_csv_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Batch Runner per Dual LLM Bug Detector.")
    parser.add_argument("--video-dir", type=str, required=True, help="Directory contenente i video da analizzare")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory di output per i report")
    parser.add_argument("--det-provider", type=str, default="google", choices=["google", "openrouter"], help="Provider per il rilevamento")
    parser.add_argument("--det-model", type=str, default=None, help="Nome modello Detector (se omesso e openrouter, legge OPENROUTER_MODEL da .env)")
    parser.add_argument("--val-provider", type=str, default="google", choices=["google", "openrouter"], help="Provider per la validazione")
    parser.add_argument("--val-model", type=str, default=None, help="Nome modello Validator (se omesso e openrouter, legge OPENROUTER_MODEL da .env)")
    parser.add_argument("--recursive", action="store_true", help="Cerca video ricorsivamente nelle sottocartelle")
    parser.add_argument("--limit", type=int, default=None, help="Numero massimo di video da elaborare")
    parser.add_argument("--resume", action="store_true", help="Salta i video già elaborati precedentemente")
    parser.add_argument("--disable-agentic-video", action="store_true", help="Disabilita la frammentazione agentica automatica")

    args = parser.parse_args()

    if args.det_model:
        det_model = args.det_model
    elif args.det_provider == "openrouter":
        det_model = os.getenv("OPENROUTER_MODEL")
    else:
        det_model = None

    if args.val_model:
        val_model = args.val_model
    elif args.val_provider == "openrouter":
        val_model = os.getenv("OPENROUTER_MODEL")
    else:
        val_model = None

    if not det_model:
        print(
            f"Errore: Nessun modello specificato per il Detector (provider '{args.det_provider}'). "
            f"Specificare --det-model oppure impostare OPENROUTER_MODEL nel file .env.",
            file=sys.stderr
        )
        sys.exit(1)

    if not val_model:
        print(
            f"Errore: Nessun modello specificato per il Validator (provider '{args.val_provider}'). "
            f"Specificare --val-model oppure impostare OPENROUTER_MODEL nel file .env.",
            file=sys.stderr
        )
        sys.exit(1)

    run_batch(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        det_provider=args.det_provider,
        det_model=det_model,
        val_provider=args.val_provider,
        val_model=val_model,
        recursive=args.recursive,
        limit=args.limit,
        resume=args.resume,
        enable_agentic_video=not args.disable_agentic_video
    )


if __name__ == "__main__":
    main()
