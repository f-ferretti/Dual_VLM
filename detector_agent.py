import os
import json
import logging
from typing import Optional

from schemas import (
    DetectorReport,
    DetectedBug,
    normalize_bug_type,
    normalize_severity,
    normalize_confidence,
    normalize_timestamp
)
from providers import BaseLLMProvider, get_provider
from video_utils import get_video_info, format_seconds_to_timestamp

logger = logging.getLogger(__name__)

DETECTOR_SYSTEM_PROMPT = """
Sei un analista esperto in Quality Assurance (QA), Game Engine Architecture e Game Testing per videogiochi.
Il tuo compito e esaminare attentamente l'intero video di gameplay fornito ed individuare tutte le anomalie tecniche effettive.

--- DEFINIZIONI FORMALI DI DOMINIO (GAME QA) ---
Per operare un'analisi accurata e scientifica, adotta rigorosamente le seguenti definizioni:

1. BUG (Software / Logic Defect):
   - Difetto riproducibile o persistente nel codice, nella logica di gioco, nelle collisioni o nei trigger.
   - Esempi: assenza totale di un collider (il personaggio cammina nel vuoto o attraversa muri), nemici la cui IA si disattiva completamente rimanendo inerti, trigger di missione che non scatta, interfaccia che non si aggiorna o mostra valori corrotti, porte o leve non interagibili.

2. GLITCH (Transient Malfunction):
   - Malfunzionamento o anomalia transitoria/temporanea che si verifica in condizioni specifiche senza necessariamente bloccare la continuità di gioco.
   - Esempi: micro-compenetrazione della mesh (clipping) che si risolve dopo pochi decimi di secondo, Z-fighting (superfici poligonali sovrapposte che sfarfallano), animazioni con scatti o snap improvvisi dello scheletro/ragdoll, popping brusco di asset geometrici (LOD failure).

3. PERFORMANCE ISSUE (Stuttering & Framedrop):
   - Problema legato all'efficienza computazionale del motore (rendering o garbage collection), visibile come vistoso calo di framerate, micro-stuttering o blocco temporaneo dei frame (hitching).

4. VISUAL / AUDIO ARTIFACT:
   - Difetto visivo o sonoro legato alla corruzione degli asset o della pipeline: texture mancanti (viola/scacchi neri), shader corrotti, ombre deformate, desincronizzazione labiale (lip-sync) o audio popping/crackling.

5. EXPLOIT (Gameplay Abusing):
   - Combinazione anomala o abuso di meccaniche lecite da parte del giocatore che rompe le regole previste (es. saltare aree di mappa tramite animation canceling o salti non previsti). Segnalalo solo se evidente ed esplicito.

6. FATAL CRASH & STABILITY (Showstopper / Process Failure):
   - Arresto irreversibile del processo o deadlock del motore.
   - Segnali visivi nel video: Crash to Desktop (CTD, chiusura immediata della viewport e ritorno a desktop/launcher), comparsa di finestre modali di crash (Unreal Engine Crash Reporter, Unity Crash Handler, dialoghi di errore di sistema), o deadlock permanente della finestra (il video rimane congelato perennemente senza ripresa dei frame).

--- COSA NON E UN BUG (NON SEGNALARE) ---
Attenzione: non segnalare i seguenti comportamenti come anomalie:
- MECCANICHE VOLUTE (Intended Mechanics): abilità speciali come scatti rapidi (dash), teletrasporto, smaterializzazione, invulnerabilità temporanea, o fisica deliberatamente esagerata/parodistica.
- SCELTE ARTISTICHE E POST-PROCESSING: filtri retro (es. effetto videocassetta VHS rovinata, aberrazione cromatica, rumore diegetico, cel-shading, motion blur direzionale).
- ARTEFATTI DI COMPRESSIONE VIDEO: squadrettamento (macroblocking) o cali di qualità dovuti all'acquisizione video streaming e non al motore di gioco.

--- TASSONOMIA PER IL CAMPO 'bug_type' ---
Usa esclusivamente uno dei seguenti valori:
- physics_collision: compenetrazione mesh, collisioni mancanti, caduta nel vuoto, gravita anomala, forze cinematiche sproporzionate o esplosioni fisiche.
- rendering_graphics: texture mancanti, sfarfallio (flickering), Z-fighting, ombre corrotte, LOD popping, geometrie statiche fluttuanti.
- animation_glitch: T-pose/A-pose, freeze dell'animazione, transizioni rotte dello scheletro, inverse kinematics (IK) disallineate, stretching di ossa/mesh, socket di aggancio prop disallineati.
- ui_hud: elementi grafici o testuali sovrapposti, barre della vita o statistiche congelate/errate, menu illeggibili.
- logic_ai: difetti logici suddivisi in:
  * Progression Blocker (Soft-Lock): porte chiuse con chiave posseduta, trigger invisibili non attivati, cutscene che non restituisce l'input (imposta SEMPRE severity: 'critical').
  * Behavioral Glitches: loop di pathfinding su NavMesh, nemici incastrati o inerti, perdita di aggro (imposta severity: 'minor' o 'major').
- audio_glitch: audio desincronizzato o assente, popping sonoro, loop audio corrotto.
- performance_issue: vistoso framedrop, micro-stuttering marcato o hitching temporaneo recuperabile con gioco ancora in esecuzione.
- crash_stability: Crash to Desktop (CTD), finestre modali di Crash Reporter, blocco totale irreversibile (deadlock permanente della finestra) (imposta SEMPRE severity: 'critical').
- other: qualsiasi altra anomalia reale non riconducibile alle precedenti.

--- REGOLE DI PRECEDENZA E CAUSA RADICE (DISAMBIGUAZIONE CATEGORIE) ---
In caso di anomalie complesse o composte, applica rigorosamente le seguenti priorita:
1. ANIMATION vs PHYSICS (REGOLA AUREA RAGDOLL vs IK): Se il comportamento anomalo coinvolge forze, gravita, masse, collisioni con l'ambiente o impulsi numerici esplosivi (es. cadavere incastrato che vibra/ruota a terra per attrito errato o rimbalzi infiniti), classifica SEMPRE come 'physics_collision'. Se invece riguarda pose scheletriche, transizioni di stato nei blend tree, rotazione anomala di joint/ossa durante playback di animazioni, cinematica inversa (IK), o stretching della mesh, classifica SEMPRE come 'animation_glitch'.
2. AI/LOGIC vs PHYSICS: Se un NPC o nemico cammina/corre contro un muro o un ostacolo senza aggirarlo, classifica SEMPRE come 'logic_ai' (il collider fisico sta funzionando correttamente bloccandolo; il difetto e nel pathfinding della NavMesh).
3. PERFORMANCE vs CRASH/STABILITY: Se il problema riguarda la fluidita temporale con loop di gioco attivo (micro-scatti a intervalli regolari, calo drastico di frame per secondo, blocco temporaneo recuperato), classifica come 'performance_issue'. Se l'applicazione collassa (Crash to Desktop), mostra finestre di crash reporter o subisce un freeze permanente (deadlock senza ripresa), classifica SEMPRE come 'crash_stability' con severita 'critical'.
4. AUDIO vs ENGINE: Se l'azione su schermo e visivamente corretta ma il sonoro e sfasato, interrotto o gracchiante, classifica SEMPRE come 'audio_glitch'.

--- LINEE GUIDA OPERATIVE PER I TIMESTAMP ---
1. Sii estremamente accurato e circoscritto nei timestamp:
   - Esprimi i timestamp nel formato temporale MM:SS (oppure HH:MM:SS se oltre l'ora), ad esempio "04:12" per 4 minuti e 12 secondi, oppure "18:38". Puoi aggiungere decimali se necessario, es. "04:12.5".
   - ATTENZIONE: Tutti i timestamp (timestamp_start e timestamp_end) DEVONO essere compresi rigorosamente entro la durata totale del video indicata nel prompt. È categoricamente vietato e invalido inserire timestamp che superano la fine del video.
   - DELIMITAZIONE STRETTA DELL'ANOMALIA (FINESTRA TEMPORALE FOCALIZZATA):
     Isola l'intervallo temporale più breve e preciso possibile attorno all'anomalia. La durata della clip (timestamp_end - timestamp_start) deve coprire unicamente l'innesco, la manifestazione e l'immediata risoluzione del difetto (target tipico: tra 5 e 15 secondi). È tassativamente vietato includere decine di secondi di gameplay ordinario precedente o successivo al difetto.
2. Distingui e descrivi chiaramente:
   - cosa accade visivamente/logicamente (actual_behavior).
   - cosa il motore o il game design dovrebbero fare correttamente (expected_behavior).
3. Assegna una severita motivata (low, medium, high, critical) e un grado di confidenza (da 0.0 a 1.0).
4. Se non rilevi anomalie genuine, restituisci una lista vuota in 'bugs'.

Rispondi ESCLUSIVAMENTE in formato JSON con la seguente struttura:
{
  "gameplay_summary": "Breve sintesi del gioco, genere, ambientazione e azione osservata nel video",
  "bugs": [
    {
      "bug_id": "BUG-01",
      "timestamp_start": "04:12",
      "timestamp_end": "04:22",
      "bug_type": "physics_collision",
      "severity": "high",
      "confidence": 0.90,
      "title": "Compenetrazione della mesh del personaggio attraverso la roccia",
      "description": "Il modello poligonale del personaggio attraversa la roccia sulla destra senza attivazione di collisione fisica.",
      "actual_behavior": "La mesh attraversa liberamente la geometria statica della roccia senza reazione elastica o blocco.",
      "expected_behavior": "Il collider del personaggio dovrebbe intersecare il collider della roccia e arrestarne il movimento."
    }
  ]
}
"""

class DetectorAgent:
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

    def analyze_video(self, video_path: str) -> DetectorReport:
        is_url = video_path.startswith("http://") or video_path.startswith("https://")
        if not is_url and not os.path.exists(video_path):
            raise FileNotFoundError(f"Video non trovato: {video_path}")

        video_info = get_video_info(video_path)
        video_filename = video_info.get("title") or os.path.basename(video_path)
        duration_sec = video_info["duration_seconds"]
        formatted_duration = format_seconds_to_timestamp(duration_sec) if duration_sec > 0 else "non determinata"

        user_prompt = f"""
Analizza questo video di gameplay intitolato '{video_filename}'.
- Durata totale del video: {formatted_duration} ({duration_sec} secondi, FPS: {video_info['fps']}).
Ispeziona l'intera sequenza e restituisci l'elenco di tutte le anomalie e bug individuati con timestamp accurati in formato MM:SS.

VINCOLO TEMPORALE TASSATIVO:
Tutti i timestamp (timestamp_start e timestamp_end) DEVONO essere compresi nell'intervallo 00:00 - {formatted_duration}. Nessun timestamp può eccedere la durata effettiva del video ({formatted_duration} / {duration_sec}s).
"""

        logger.info(f"[DetectorAgent] Avvio analisi per il video '{video_filename}' ({formatted_duration})...")
        res = self.provider.generate_with_video(
            video_path=video_path,
            system_prompt=DETECTOR_SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        parsed = res["parsed_json"]
        raw_text = res["raw_text"]

        bugs_list = []
        if isinstance(parsed, list):
            raw_bugs = parsed
            summary_text = "Analisi completata."
        elif isinstance(parsed, dict):
            raw_bugs = parsed.get("bugs", [])
            summary_text = str(parsed.get("gameplay_summary") or "Analisi completata.")
        else:
            raw_bugs = []
            summary_text = "Analisi completata."

        if not isinstance(raw_bugs, list):
            raw_bugs = []

        for i, b in enumerate(raw_bugs):
            if not isinstance(b, dict):
                continue
            try:
                bug_id = str(b.get("bug_id") or f"BUG-{i+1:02d}")
                t_start = normalize_timestamp(b.get("timestamp_start", 0.0))
                t_end = normalize_timestamp(b.get("timestamp_end", t_start))

                if t_end < t_start:
                    t_end = t_start

                # Scarta segnalazioni oltre la durata video con tolleranza di 3 secondi
                if duration_sec > 0 and t_start > duration_sec + 3.0:
                    logger.warning(f"Scartato bug {bug_id} con timestamp ({t_start}s) eccedente la durata del video ({duration_sec}s): {b.get('title')}")
                    continue

                if duration_sec > 0:
                    t_start = min(t_start, duration_sec)
                    t_end = min(max(t_start, t_end), duration_sec)
                
                bug_item = DetectedBug(
                    bug_id=bug_id,
                    timestamp_start=round(t_start, 2),
                    timestamp_end=round(t_end, 2),
                    bug_type=normalize_bug_type(b.get("bug_type")),
                    severity=normalize_severity(b.get("severity")),
                    confidence=normalize_confidence(b.get("confidence")),
                    title=str(b.get("title") or f"Anomalia {bug_id}"),
                    description=str(b.get("description") or ""),
                    actual_behavior=str(b.get("actual_behavior") or ""),
                    expected_behavior=str(b.get("expected_behavior") or "")
                )
                bugs_list.append(bug_item)
            except Exception as e:
                logger.warning(f"Errore parsing bug #{i}: {e}")

        report = DetectorReport(
            video_filename=video_filename,
            video_duration_seconds=duration_sec,
            gameplay_summary=summary_text,
            bugs=bugs_list,
            raw_response=raw_text
        )

        logger.info(f"[DetectorAgent] Analisi completata. Rilevati {len(report.bugs)} bug candidati.")
        return report