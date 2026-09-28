from enum import Enum
import re
from typing import List, Optional, Any
from pydantic import BaseModel, Field


class BugTypeEnum(str, Enum):
    PHYSICS_COLLISION = "physics_collision"
    RENDERING_GRAPHICS = "rendering_graphics"
    ANIMATION_GLITCH = "animation_glitch"
    UI_HUD = "ui_hud"
    LOGIC_AI = "logic_ai"
    AUDIO_GLITCH = "audio_glitch"
    PERFORMANCE_ISSUE = "performance_issue"
    CRASH_STABILITY = "crash_stability"
    OTHER = "other"


class SeverityEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationVerdictEnum(str, Enum):
    CONFIRMED = "CONFIRMED"
    REJECTED_FALSE_POSITIVE = "REJECTED_FALSE_POSITIVE"
    UNCERTAIN = "UNCERTAIN"


def normalize_bug_type(raw_val: Any) -> BugTypeEnum:
    if isinstance(raw_val, BugTypeEnum):
        return raw_val
    s = str(raw_val or "").lower().strip()
    if "crash" in s or "fatal" in s or "deadlock" in s or "ctd" in s or "oom" in s or "stabilit" in s:
        return BugTypeEnum.CRASH_STABILITY
    elif "physic" in s or "collis" in s or "clip" in s:
        return BugTypeEnum.PHYSICS_COLLISION
    elif "render" in s or "graph" in s or "textur" in s or "flicker" in s:
        return BugTypeEnum.RENDERING_GRAPHICS
    elif "anim" in s or "pose" in s or "ik" in s:
        return BugTypeEnum.ANIMATION_GLITCH
    elif "ui" in s or "hud" in s or "menu" in s:
        return BugTypeEnum.UI_HUD
    elif "ai" in s or "logic" in s or "path" in s or "enemy" in s:
        return BugTypeEnum.LOGIC_AI
    elif "audio" in s or "sound" in s:
        return BugTypeEnum.AUDIO_GLITCH
    elif "perf" in s or "stutter" in s or "drop" in s or "lag" in s or "fps" in s:
        return BugTypeEnum.PERFORMANCE_ISSUE
    return BugTypeEnum.OTHER


def normalize_severity(raw_val: Any) -> SeverityEnum:
    if isinstance(raw_val, SeverityEnum):
        return raw_val
    s = str(raw_val or "").lower().strip()
    if "crit" in s:
        return SeverityEnum.CRITICAL
    elif "high" in s or "alt" in s:
        return SeverityEnum.HIGH
    elif "low" in s or "bass" in s:
        return SeverityEnum.LOW
    return SeverityEnum.MEDIUM


def normalize_confidence(raw_val: Any) -> float:
    try:
        if isinstance(raw_val, (int, float)):
            val = float(raw_val)
        else:
            cleaned = re.sub(r"[^\d.]", "", str(raw_val or "0.8"))
            val = float(cleaned) if cleaned else 0.8
        if val > 1.0:
            val = val / 100.0
        return max(0.0, min(1.0, round(val, 2)))
    except Exception:
        return 0.8


def normalize_timestamp(raw_val: Any) -> float:
    if raw_val is None:
        return 0.0
    if isinstance(raw_val, (int, float)):
        return max(0.0, round(float(raw_val), 2))
    s = str(raw_val).strip()
    if not s:
        return 0.0
    parts = s.split(":")
    if len(parts) == 3:
        try:
            h = float(parts[0])
            m = float(parts[1])
            sec = float(parts[2])
            return max(0.0, round(h * 3600.0 + m * 60.0 + sec, 2))
        except ValueError:
            pass
    elif len(parts) == 2:
        try:
            m = float(parts[0])
            sec = float(parts[1])
            return max(0.0, round(m * 60.0 + sec, 2))
        except ValueError:
            pass
    cleaned = re.sub(r"[^\d.]", "", s)
    try:
        return max(0.0, round(float(cleaned), 2))
    except ValueError:
        return 0.0


class DetectedBug(BaseModel):
    bug_id: str = Field(description="Identificativo univoco del bug, es. BUG-01")
    timestamp_start: float = Field(description="Timestamp di inizio del bug in secondi all'interno del video")
    timestamp_end: float = Field(description="Timestamp di fine del bug in secondi all'interno del video")
    bug_type: BugTypeEnum = Field(description="Categoria dell'anomalia rilevata")
    severity: SeverityEnum = Field(description="Livello di gravita stimato dal detector")
    confidence: float = Field(ge=0.0, le=1.0, description="Punteggio di confidenza del detector da 0.0 a 1.0")
    title: str = Field(description="Titolo sintetico del bug")
    description: str = Field(description="Descrizione dettagliata dell'anomalia osservata")
    actual_behavior: str = Field(description="Cosa accade effettivamente nel video")
    expected_behavior: str = Field(description="Cosa ci si aspetterebbe normalmente nel gioco")
    clip_path: Optional[str] = Field(default=None, description="Percorso dello spezzone video ritagliato relativo all'anomalia")
    user_validation: Optional[dict] = Field(default=None, description="Annotazione e verdetto umano Ground Truth")


class DetectorReport(BaseModel):
    video_filename: str = Field(default="", description="Nome del file video analizzato")
    video_duration_seconds: float = Field(default=0.0, description="Durata totale del video in secondi")
    gameplay_summary: str = Field(description="Breve descrizione del gameplay e del contesto osservato")
    bugs: List[DetectedBug] = Field(default_factory=list, description="Lista dei bug rilevati")
    raw_response: Optional[str] = Field(default=None, description="Risposta testuale grezza del modello")


class ValidatedBugItem(BaseModel):
    bug_id: str = Field(description="ID del bug corrispondente")
    verdict: ValidationVerdictEnum = Field(description="Verdetto della validazione: CONFIRMED, REJECTED_FALSE_POSITIVE o UNCERTAIN")
    confidence: float = Field(ge=0.0, le=1.0, description="Punteggio di confidenza da 0.0 a 1.0")
    critique_explanation: str = Field(description="Motivazione tecnica e critica a supporto del verdetto")
    is_intended_mechanic_or_style: bool = Field(description="True se il comportamento sembra una meccanica di gioco voluta, effetto stilistico o animazione prevista")
    corrected_bug_type: Optional[BugTypeEnum] = Field(default=None, description="Categoria corretta dal validatore, se diversa")
    corrected_severity: Optional[SeverityEnum] = Field(default=None, description="Livello di gravita rettificato")
    corrected_timestamp_start: Optional[float] = Field(default=None, description="Timestamp di inizio rettificato, se necessario")
    corrected_timestamp_end: Optional[float] = Field(default=None, description="Timestamp di fine rettificato, se necessario")
    clip_path: Optional[str] = Field(default=None, description="Percorso dello spezzone video ritagliato")
    user_validation: Optional[dict] = Field(default=None, description="Annotazione e verdetto umano Ground Truth")


class ValidatorReport(BaseModel):
    total_evaluated: int = Field(default=0, description="Numero totale di bug esaminati")
    confirmed_count: int = Field(default=0, description="Numero di bug confermati")
    rejected_count: int = Field(default=0, description="Numero di falsi positivi scartati")
    uncertain_count: int = Field(default=0, description="Numero di casi incerti")
    validated_bugs: List[ValidatedBugItem] = Field(default_factory=list, description="Lista delle valutazioni dettagliate")
    global_critique_summary: str = Field(description="Sintesi globale della qualita delle segnalazioni e considerazioni finali")
    raw_response: Optional[str] = Field(default=None, description="Risposta testuale grezza del validatore")


class DualLLMPipelineResult(BaseModel):
    timestamp_utc: str = Field(description="Data e ora di esecuzione dell'analisi")
    video_path: str = Field(description="Percorso del video analizzato")
    video_duration_seconds: float = Field(default=0.0)
    detector_provider: str = Field(description="Provider usato per il Detector (google/openrouter)")
    detector_model: str = Field(description="Nome del modello Detector")
    validator_provider: str = Field(description="Provider usato per il Validator (google/openrouter)")
    validator_model: str = Field(description="Nome del modello Validator")
    detector_report: DetectorReport
    validator_report: ValidatorReport
    execution_time_seconds: float = Field(default=0.0, description="Tempo totale impiegato per l'intera pipeline")
    saved_json_path: Optional[str] = Field(default=None, description="Percorso del report JSON salvato su disco")
    user_validation: Optional[dict] = Field(default=None, description="Verdetto umano complessivo sul video")