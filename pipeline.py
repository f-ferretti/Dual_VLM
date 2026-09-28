import os
import re
import json
import time
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any

from schemas import (
    DualLLMPipelineResult,
    DetectorReport,
    ValidatorReport,
    ValidationVerdictEnum
)
from detector_agent import DetectorAgent
from validator_agent import ValidatorAgent
from video_utils import (
    get_video_info,
    format_seconds_to_timestamp,
    cut_video_clip,
    clean_youtube_url
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class DualLLMPipeline:
    def __init__(
        self,
        detector_provider: str = "google",
        detector_model: str = "gemini-3.8-flash",
        detector_api_key: Optional[str] = None,
        validator_provider: str = "google",
        validator_model: str = "gemini-3.8-flash",
        validator_api_key: Optional[str] = None,
        results_dir: str = "results",
        enable_agentic_video: bool = True
    ):
        self.detector_provider = detector_provider
        self.detector_model = detector_model
        self.detector_api_key = detector_api_key

        self.validator_provider = validator_provider
        self.validator_model = validator_model
        self.validator_api_key = validator_api_key

        self.results_dir = results_dir
        self.enable_agentic_video = enable_agentic_video
        os.makedirs(self.results_dir, exist_ok=True)

        self.detector = DetectorAgent(
            provider_type=self.detector_provider,
            model_name=self.detector_model,
            api_key=self.detector_api_key,
            enable_agentic_video=self.enable_agentic_video
        )

        self.validator = ValidatorAgent(
            provider_type=self.validator_provider,
            model_name=self.validator_model,
            api_key=self.validator_api_key,
            enable_agentic_video=self.enable_agentic_video
        )

    def run(self, video_path: str, progress_callback: Optional[Callable[[str, float], None]] = None) -> DualLLMPipelineResult:
        if "youtube.com" in video_path.lower() or "youtu.be" in video_path.lower():
            video_path = clean_youtube_url(video_path)

        start_time = time.time()
        video_info = get_video_info(video_path)
        video_filename = os.path.basename(video_path)

        if progress_callback:
            progress_callback(f"Avvio rilevamento anomalie nel video ({self.detector_model})...", 0.15)

        logger.info(f"--- Rilevamento Anomalie ({self.detector_model}) ---")
        detector_report = self.detector.analyze_video(video_path)

        is_youtube = ("youtube.com" in video_path.lower() or "youtu.be" in video_path.lower())

        if progress_callback:
            progress_callback(f"Rilevate {len(detector_report.bugs)} anomalie. Avvio validazione ({self.validator_model})...", 0.55)

        if is_youtube:
            youtube_cooldown_seconds = 8
            logger.info(f"Attesa di sicurezza di {youtube_cooldown_seconds}s per rilascio stream YouTube prima della validazione...")
            time.sleep(youtube_cooldown_seconds)

        logger.info(f"--- Validazione Anomalie ({self.validator_model}) ---")
        validator_report = self.validator.validate_detections(video_path, detector_report)

        # Estrazione spezzoni video per file locali
        clips_dir = os.path.join(self.results_dir, "clips")
        val_map = {item.bug_id: item for item in validator_report.validated_bugs}

        if not is_youtube and os.path.exists(video_path):
            for bug in detector_report.bugs:
                # Usa i timestamp corretti dal validatore se disponibili
                v_item = val_map.get(bug.bug_id)
                s_time = v_item.corrected_timestamp_start if (v_item and v_item.corrected_timestamp_start is not None) else bug.timestamp_start
                e_time = v_item.corrected_timestamp_end if (v_item and v_item.corrected_timestamp_end is not None) else bug.timestamp_end

                clip_file = cut_video_clip(
                    video_path=video_path,
                    start_sec=s_time,
                    end_sec=e_time,
                    output_dir=clips_dir,
                    bug_id=bug.bug_id
                )
                bug.clip_path = clip_file
                if v_item:
                    v_item.clip_path = clip_file

        if progress_callback:
            progress_callback("Validazione completata. Generazione report finale...", 0.90)

        total_duration = round(time.time() - start_time, 2)
        utc_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        result = DualLLMPipelineResult(
            timestamp_utc=utc_timestamp,
            video_path=video_path if is_youtube else os.path.abspath(video_path),
            video_duration_seconds=video_info["duration_seconds"],
            detector_provider=self.detector_provider,
            detector_model=self.detector_model,
            validator_provider=self.validator_provider,
            validator_model=self.validator_model,
            detector_report=detector_report,
            validator_report=validator_report,
            execution_time_seconds=total_duration
        )

        self._save_results(result, video_filename)

        if progress_callback:
            progress_callback("Elaborazione completata.", 1.0)

        return result

    def _save_results(self, result: DualLLMPipelineResult, base_name: str) -> None:
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_name = os.path.splitext(base_name)[0]
        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_name)
        
        json_path = os.path.join(self.results_dir, f"report_{sanitized_name}_{timestamp_slug}.json")
        md_path = os.path.join(self.results_dir, f"report_{sanitized_name}_{timestamp_slug}.md")
        result.saved_json_path = os.path.abspath(json_path)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

        md_content = self.generate_markdown_report(result)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Report salvati in:\n- JSON: {json_path}\n- Markdown: {md_path}")

    @staticmethod
    def generate_markdown_report(result: DualLLMPipelineResult) -> str:
        d_rep = result.detector_report
        v_rep = result.validator_report

        val_map = {item.bug_id: item for item in v_rep.validated_bugs}

        dur_str = f"{format_seconds_to_timestamp(result.video_duration_seconds)} ({result.video_duration_seconds}s)" if result.video_duration_seconds > 0 else "N/D"
        display_name = d_rep.video_filename or os.path.basename(result.video_path)

        lines = [
            f"# Report Analisi Anomalie Gameplay",
            f"",
            f"- **Data esecuzione**: {result.timestamp_utc}",
            f"- **Sorgente video**: `{display_name}` ({dur_str})",
            f"- **Modulo Rilevamento**: `{result.detector_provider} / {result.detector_model}`",
            f"- **Modulo Validazione**: `{result.validator_provider} / {result.validator_model}`",
            f"- **Tempo totale di elaborazione**: {result.execution_time_seconds}s",
            f"",
            f"---",
            f"",
            f"## Sintesi Metriche",
            f"",
            f"| Metrica | Valore |",
            f"| :--- | :--- |",
            f"| Anomalie Segnalate | **{len(d_rep.bugs)}** |",
            f"| Anomalie Confermate | **{v_rep.confirmed_count}** |",
            f"| Falsi Positivi Scartati | **{v_rep.rejected_count}** |",
            f"| Casi Incerti / Da approfondire | **{v_rep.uncertain_count}** |",
            f"",
            f"### Sintesi del Gameplay",
            f"> {d_rep.gameplay_summary}",
            f"",
            f"### Valutazione Globale",
            f"> {v_rep.global_critique_summary}",
            f"",
            f"---",
            f"",
            f"## Dettaglio Anomalie e Validazione",
            f""
        ]

        if not d_rep.bugs:
            lines.append("Nessuna anomalia riscontrata nel video.")
        else:
            for b in d_rep.bugs:
                v_item = val_map.get(b.bug_id)
                verdict_str = v_item.verdict.value if v_item else "NON VALUTATO"
                
                t_start = format_seconds_to_timestamp(b.timestamp_start)
                t_end = format_seconds_to_timestamp(b.timestamp_end)

                b_type_str = b.bug_type.value if hasattr(b.bug_type, "value") else str(b.bug_type)
                b_sev_str = b.severity.value if hasattr(b.severity, "value") else str(b.severity)

                lines.append(f"### [{b.bug_id}] {b.title}")
                lines.append(f"- **Timestamp**: `{t_start} - {t_end}` ({b.timestamp_start}s - {b.timestamp_end}s)")
                lines.append(f"- **Tipologia**: `{b_type_str}` | **Severita**: `{b_sev_str}` | **Confidenza**: `{b.confidence:.2f}`")
                lines.append(f"- **Descrizione**: {b.description}")
                lines.append(f"- **Comportamento Atteso**: {b.expected_behavior}")
                lines.append(f"- **Comportamento Effettivo**: {b.actual_behavior}")
                lines.append(f"")
                
                if v_item:
                    lines.append(f"**Esito Validazione**: `{verdict_str}` (Confidenza: `{v_item.confidence:.2f}`)")
                    lines.append(f"**Motivazione Critica**: {v_item.critique_explanation}")
                    if v_item.is_intended_mechanic_or_style:
                        lines.append(f"- *Nota*: Riconosciuto come meccanica di gioco voluta o stile artistico.")
                    if v_item.corrected_bug_type:
                        corr_t_str = v_item.corrected_bug_type.value if hasattr(v_item.corrected_bug_type, "value") else str(v_item.corrected_bug_type)
                        lines.append(f"- *Tipologia corretta*: `{corr_t_str}`")
                    if v_item.corrected_severity:
                        corr_s_str = v_item.corrected_severity.value if hasattr(v_item.corrected_severity, "value") else str(v_item.corrected_severity)
                        lines.append(f"- *Severita corretta*: `{corr_s_str}`")
                
                if b.clip_path and os.path.exists(b.clip_path):
                    lines.append(f"- **Clip Video Ritagliata**: `{b.clip_path}`")
                lines.append("")
                lines.append("---")
                lines.append("")

        return "\n".join(lines)
