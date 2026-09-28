import os
import json
import logging
from typing import Optional, List

from schemas import (
    DetectorReport,
    ValidatorReport,
    ValidatedBugItem,
    ValidationVerdictEnum,
    BugTypeEnum,
    SeverityEnum,
    normalize_bug_type,
    normalize_severity,
    normalize_confidence,
    normalize_timestamp
)
from providers import BaseLLMProvider, get_provider
from video_utils import format_seconds_to_timestamp

logger = logging.getLogger(__name__)

VALIDATOR_SYSTEM_PROMPT = """
Sei un Lead QA Engineer e Senior Game Engine Specialist.
Il tuo compito e validare con estremo rigore critico le segnalazioni di bug generate dal modulo di Detection primario su un video di gameplay.

I sistemi di detection automatica spesso scambiano per anomalie ciò che in realtà è una meccanica voluta, un effetto stilistico o un calo di bitrate video.
Il tuo obiettivo principale è individuare e scartare tutti i FALSI POSITIVI, confermando solo i DIFETTI TECNICI AUTENTICI (Bug, Glitch o difetti di Performance reali).

--- PROTOCOLLO DI VALUTAZIONE CRITICA ---
Per ciascun bug candidato segnalato dal Detector, esegui il seguente checklist di analisi:

0. VERIFICA DI CORRISPONDENZA VISIVA EFFETTIVA (Anti-Hallucination & Anti-Confirmation Bias):
   - Ispeziona con estremo rigore la scena video esattamente all'intervallo temporale indicato (timestamp_range).
   - Se a quel minutaggio la scena mostra gameplay ordinario, un'azione del tutto diversa da quanto descritto o se l'anomalia non compare a video:
     -> verdetto REJECTED_FALSE_POSITIVE motivando esplicitamente che l'anomalia descritta e priva di riscontro visivo a quel minutaggio.
   - È tassativamente vietato confermare una segnalazione per bias cognitivo o fiducia verso il testo descrittivo del Detector se i fotogrammi a quel minutaggio non la mostrano.
   - Se l'evento descritto esiste realmente ma e posizionato a un minutaggio differente del filmato, confermalo SOLO rettificando i timestamp reali in 'corrected_timestamp_start' e 'corrected_timestamp_end'. Se l'evento non e chiaramente individuabile, scartalo.

1. VERIFICA DI INTENZIONALITA (Game Design Intent):
   - L'azione o l'effetto osservato fa parte delle meccaniche previste dal gioco?
   - Esempi: abilità di teletrasporto o dash, transizioni in invulnerabilità (i-frames), interazioni esagerate in giochi comici o arcade, smaterializzazione del personaggio.
   -> Se è una meccanica voluta: verdetto REJECTED_FALSE_POSITIVE (is_intended_mechanic_or_style = true).

2. VERIFICA DI DIREZIONE ARTISTICA E POST-PROCESSING:
   - L'effetto visivo è una scelta estetica o diegetica del gioco?
   - Esempi: distorsione visiva diegetica (salute bassa, droghe, follia), aberrazione cromatica, effetto videocassetta VHS rovinata, scanlines, motion blur estremo, shader cel-shading o glitch art intenzionale.
   -> Se è una scelta artistica: verdetto REJECTED_FALSE_POSITIVE (is_intended_mechanic_or_style = true).

3. VERIFICA DI ARTEFATTO VIDEO / CATTURA:
   - È un difetto del motore o è dovuto all'encoder video della registrazione (macroblocking, calo di bitrate streaming, frame drop di cattura)?
   -> Se è un artefatto di registrazione video: verdetto REJECTED_FALSE_POSITIVE.

4. VERIFICA DI AUTENTICITA TECNICA (Vero Bug o Glitch):
   - L'anomalia viola palesemente le regole della fisica, del rendering, dell'animazione o della logica del motore di gioco?
   - Si tratta di un BUG (difetto logico/funzionale persistente) o di un GLITCH (anomalia visiva o transitoria)?
   -> Se l'anomalia tecnica è reale e non intenzionale: verdetto CONFIRMED.

5. AMBIGUITA O EVIDENZA INSUFFICIENTE:
   - La visuale o il contesto video non consentono di stabilire con certezza se sia un bug senza consultare i log del gioco?
   -> Se ambiguo: verdetto UNCERTAIN.

--- CRITERI PER I VERDETTI ---
- CONFIRMED: Difetto tecnico reale, involontario e provato (Bug, Glitch, difetto di Performance).
- REJECTED_FALSE_POSITIVE: Discrepanza tra descrizione e scena video reale (allucinazione del Detector), meccanica voluta dal game design, stile artistico, cutscene, o artefatto di compressione video.
- UNCERTAIN: Contesto visivo insufficiente o ambiguo per esprimere certezza.

--- REGOLE DI PRECEDENZA E CAUSA RADICE (DISAMBIGUAZIONE CATEGORIE) ---
In caso di anomalie complesse o dubbi di categorizzazione, applica rigorosamente le seguenti priorita:
1. ANIMATION vs PHYSICS (REGOLA AUREA RAGDOLL vs IK): Se il comportamento anomalo coinvolge forze, gravita, masse, collisioni con l'ambiente o impulsi numerici esplosivi (es. cadavere incastrato che vibra/ruota a terra per attrito errato o rimbalzi infiniti), rettifica 'corrected_bug_type' SEMPRE come 'physics_collision'. Se invece riguarda pose scheletriche, transizioni di stato nei blend tree, rotazione anomala di joint/ossa durante playback di animazioni, cinematica inversa (IK), o stretching della mesh, rettifica 'corrected_bug_type' SEMPRE come 'animation_glitch'.
2. AI/LOGIC vs PHYSICS: Se un NPC o nemico cammina/corre contro un muro o un ostacolo senza aggirarlo, rettifica 'corrected_bug_type' SEMPRE come 'logic_ai' (il collider fisico sta funzionando bloccandolo; il difetto e nel pathfinding della NavMesh). Se l'anomalia blocca la progressione del gioco (Soft-Lock), rettifica 'corrected_severity' tassativamente su 'critical'.
3. PERFORMANCE vs CRASH/STABILITY: Se il problema riguarda la fluidita temporale con loop di gioco attivo (micro-scatti a intervalli regolari, calo drastico di frame per secondo, blocco temporaneo), rettifica 'corrected_bug_type' come 'performance_issue'. Se invece l'applicazione si arresta (Crash to Desktop), mostra finestre di crash reporter modali o subisce un freeze permanente (deadlock senza ripresa dei frame), rettifica 'corrected_bug_type' SEMPRE come 'crash_stability' e 'corrected_severity' tassativamente su 'critical'.
4. AUDIO vs ENGINE: Se l'azione su schermo e visivamente corretta ma il sonoro e sfasato, interrotto o gracchiante, rettifica 'corrected_bug_type' SEMPRE come 'audio_glitch'.

--- RETTIFICA DEI METADATI E CONTROLLO TEMPORALE ---
- Se confermi un bug ma il Detector ha sbagliato categoria, severità o intervallo temporale:
  * Rettifica 'corrected_bug_type' (physics_collision, rendering_graphics, animation_glitch, ui_hud, logic_ai, audio_glitch, performance_issue, crash_stability, other).
  * Rettifica 'corrected_severity' (low, medium, high, critical).
  * Rettifica 'corrected_timestamp_start' e 'corrected_timestamp_end' nel formato standard MM:SS se i tempi indicati dal Detector soffrono di drift temporale o sono eccessivamente ampi.
  * DELIMITAZIONE STRETTA DELL'ANOMALIA: Se il Detector ha indicato un intervallo sovrastimato o troppo largo (oltre 15-20 secondi per un evento breve), rettifica 'corrected_timestamp_start' e 'corrected_timestamp_end' affinché racchiudano strettamente l'evento difettoso (target ottimale: 5-15 secondi).
- Se la segnalazione del Detector indica un minutaggio inesistente o che supera la durata totale del video:
  * Se l'evento anomalo si trova realmente nel video a un minutaggio diverso, conferma il bug inserendo i timestamp corretti in MM:SS.
  * Se l'evento non è presente nel video o la coordinata è priva di riscontro reale, assegna 'verdict': 'REJECTED_FALSE_POSITIVE' motivando l'assenza temporale dell'evento.

Rispondi ESCLUSIVAMENTE in formato JSON con la seguente struttura:
{
  "global_critique_summary": "Valutazione tecnica complessiva sulla qualità delle segnalazioni del Detector, stato del motore di gioco e presenza di falsi allarmi",
  "validated_bugs": [
    {
      "bug_id": "BUG-01",
      "verdict": "CONFIRMED",
      "confidence": 0.95,
      "critique_explanation": "La compenetrazione della mesh del personaggio all'interno del masso roccioso e palese: il collider non registra l'impatto e non vi e risposta fisica, confermando un difetto non intenzionale di mesh collision.",
      "is_intended_mechanic_or_style": false,
      "corrected_bug_type": "physics_collision",
      "corrected_severity": "high",
      "corrected_timestamp_start": "04:12",
      "corrected_timestamp_end": "04:22"
    }
  ]
}
"""

class ValidatorAgent:
    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        provider_type: str = "google",
        model_name: str = "gemini-3.8-flash",
        api_key: Optional[str] = None,
        enable_agentic_video: bool = True
    ):
        if provider is not None:
            self.provider = provider
        else:
            self.provider = get_provider(
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key,
                enable_agentic_video=enable_agentic_video
            )

    def validate_detections(self, video_path: str, detector_report: DetectorReport) -> ValidatorReport:
        is_url = video_path.startswith("http://") or video_path.startswith("https://")
        if not is_url and not os.path.exists(video_path):
            raise FileNotFoundError(f"Video non trovato: {video_path}")

        if not detector_report.bugs:
            return ValidatorReport(
                total_evaluated=0,
                confirmed_count=0,
                rejected_count=0,
                uncertain_count=0,
                validated_bugs=[],
                global_critique_summary="Nessun bug candidato segnalato dal Detector da validare.",
                raw_response=None
            )

        bugs_payload = [
            {
                "bug_id": b.bug_id,
                "timestamp_range": f"{b.timestamp_start}s - {b.timestamp_end}s",
                "claimed_type": b.bug_type.value if hasattr(b.bug_type, "value") else str(b.bug_type),
                "claimed_severity": b.severity.value if hasattr(b.severity, "value") else str(b.severity),
                "title": b.title,
                "description": b.description,
                "actual_behavior": b.actual_behavior,
                "expected_behavior": b.expected_behavior
            }
            for b in detector_report.bugs
        ]

        duration_sec = detector_report.video_duration_seconds
        formatted_duration = format_seconds_to_timestamp(duration_sec) if duration_sec > 0 else "non determinata"

        user_prompt = f"""
Abbiamo analizzato il video '{detector_report.video_filename}'.
- Durata totale del video: {formatted_duration} ({duration_sec} secondi).
Riassunto del gameplay: {detector_report.gameplay_summary}

Di seguito e riportata la lista dei {len(detector_report.bugs)} bug candidati segnalati dal modulo di rilevamento:
{json.dumps(bugs_payload, indent=2, ensure_ascii=False)}

Valuta ciascun bug candidato e restituisci il verdetto critico (CONFIRMED / REJECTED_FALSE_POSITIVE / UNCERTAIN) con la motivazione tecnica.
VINCOLO TEMPORALE: Qualsiasi segnalazione priva di riscontro visivo o collocata a minutaggi impossibili/eccedenti la durata di {formatted_duration} ({duration_sec}s) deve essere rettificata con i timestamp reali in MM:SS (se l'evento esiste realmente a un minuto diverso) oppure scartata come REJECTED_FALSE_POSITIVE.
"""

        logger.info(f"[ValidatorAgent] Avvio validazione su {len(detector_report.bugs)} segnalazioni (durata video: {formatted_duration})...")
        res = self.provider.generate_with_video(
            video_path=video_path,
            system_prompt=VALIDATOR_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        parsed = res["parsed_json"]
        raw_text = res["raw_text"]

        validated_items: List[ValidatedBugItem] = []
        if isinstance(parsed, list):
            raw_val_list = parsed
            global_summary = "Revisione critica completata."
        elif isinstance(parsed, dict):
            raw_val_list = parsed.get("validated_bugs", [])
            global_summary = str(parsed.get("global_critique_summary") or "Revisione critica completata.")
        else:
            raw_val_list = []
            global_summary = "Revisione critica completata."

        if not isinstance(raw_val_list, list):
            raw_val_list = []

        confirmed_cnt = 0
        rejected_cnt = 0
        uncertain_cnt = 0

        for item in raw_val_list:
            if not isinstance(item, dict):
                continue
            try:
                verdict_str = str(item.get("verdict", "UNCERTAIN")).upper()
                if "CONFIRM" in verdict_str:
                    verdict_enum = ValidationVerdictEnum.CONFIRMED
                    confirmed_cnt += 1
                elif "REJECT" in verdict_str or "FALSE" in verdict_str:
                    verdict_enum = ValidationVerdictEnum.REJECTED_FALSE_POSITIVE
                    rejected_cnt += 1
                else:
                    verdict_enum = ValidationVerdictEnum.UNCERTAIN
                    uncertain_cnt += 1

                raw_corr_type = item.get("corrected_bug_type")
                corr_type = normalize_bug_type(raw_corr_type) if raw_corr_type else None

                raw_corr_sev = item.get("corrected_severity")
                corr_sev = normalize_severity(raw_corr_sev) if raw_corr_sev else None

                t_start = item.get("corrected_timestamp_start")
                corr_t_start = normalize_timestamp(t_start) if (t_start is not None and str(t_start).strip() != "") else None

                t_end = item.get("corrected_timestamp_end")
                corr_t_end = normalize_timestamp(t_end) if (t_end is not None and str(t_end).strip() != "") else None
                if corr_t_start is not None and corr_t_end is not None and corr_t_end < corr_t_start:
                    corr_t_end = corr_t_start

                val_obj = ValidatedBugItem(
                    bug_id=str(item.get("bug_id", "")),
                    verdict=verdict_enum,
                    confidence=normalize_confidence(item.get("confidence", 0.8)),
                    critique_explanation=str(item.get("critique_explanation") or "Valutazione completata."),
                    is_intended_mechanic_or_style=bool(item.get("is_intended_mechanic_or_style", False)),
                    corrected_bug_type=corr_type,
                    corrected_severity=corr_sev,
                    corrected_timestamp_start=corr_t_start,
                    corrected_timestamp_end=corr_t_end
                )
                validated_items.append(val_obj)
            except Exception as e:
                logger.warning(f"Errore parsing validazione bug: {e}")

        report = ValidatorReport(
            total_evaluated=len(validated_items),
            confirmed_count=confirmed_cnt,
            rejected_count=rejected_cnt,
            uncertain_count=uncertain_cnt,
            validated_bugs=validated_items,
            global_critique_summary=global_summary,
            raw_response=raw_text
        )

        logger.info(f"[ValidatorAgent] Validazione completata: {confirmed_cnt} confermati, {rejected_cnt} scartati, {uncertain_cnt} incerti.")
        return report