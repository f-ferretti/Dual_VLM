import os
import sys
import tempfile
import json
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

# Gestione silenziosa warning e auto-avvio con Streamlit CLI se eseguito come `python app.py`
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging_level = os.environ.get("STREAMLIT_LOG_LEVEL", "error")

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        import subprocess
        cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
        subprocess.run(cmd + sys.argv[1:])
        sys.exit(0)
except Exception:
    pass

import streamlit as st
from dotenv import load_dotenv

from pipeline import DualLLMPipeline
from video_utils import get_video_info, format_seconds_to_timestamp, clean_youtube_url
from schemas import ValidationVerdictEnum

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path, override=True)

st.set_page_config(
    page_title="Gameplay Bug Detector - Dual LLM Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container {
        background: #fdfdfd;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #212529;
    }
    .metric-label {
        font-size: 13px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-confirmed {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .badge-rejected {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .badge-uncertain {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }
    .status-box {
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Gameplay Bug Detector - Dual LLM Pipeline")
st.caption("Sistema per il rilevamento e la validazione di anomalie in registrazioni di gameplay.")


def extract_youtube_id(url: str) -> Optional[str]:
    if not url:
        return None
    match = re.search(r"(?:v=|\/embed\/|\.be\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None


def render_youtube_clip_player(video_url: str, start_sec: float, end_sec: Optional[float] = None) -> None:
    vid_id = extract_youtube_id(video_url)
    start_int = max(0, int(start_sec))
    if end_sec is not None and float(end_sec) > start_int:
        end_int = int(end_sec)
    else:
        end_int = start_int + 15

    s_fmt = format_seconds_to_timestamp(start_int)
    e_fmt = format_seconds_to_timestamp(end_int)
    direct_link = f"https://www.youtube.com/watch?v={vid_id}&t={start_int}s" if vid_id else f"{video_url}&t={start_int}s"

    st.markdown(f"**Spezzone Video YouTube (Intervallo: `{s_fmt} - {e_fmt}`):** [Apri su YouTube]({direct_link})")

    if vid_id:
        embed_url = f"https://www.youtube-nocookie.com/embed/{vid_id}?start={start_int}&end={end_int}&autoplay=0&rel=0"
        st.iframe(src=embed_url, height=355)
    else:
        st.video(video_url, start_time=start_int)


def persist_result_to_disk(res):
    if res.saved_json_path and os.path.exists(res.saved_json_path):
        try:
            with open(res.saved_json_path, "w", encoding="utf-8") as f:
                json.dump(res.model_dump(), f, indent=2, ensure_ascii=False)
        except Exception as ex:
            st.error(f"Errore durante il salvataggio dell'annotazione: {ex}")

def update_bug_user_validation(bug_id: str, user_says_bug_present: bool):
    if "last_result" not in st.session_state:
        return
    res = st.session_state["last_result"]
    val_map = {item.bug_id: item for item in res.validator_report.validated_bugs}
    v_item = val_map.get(bug_id)
    
    model_says_bug = (v_item.verdict == ValidationVerdictEnum.CONFIRMED) if v_item else True
    
    if user_says_bug_present and model_says_bug:
        verdict = "TRUE_POSITIVE"
    elif not user_says_bug_present and model_says_bug:
        verdict = "FALSE_POSITIVE"
    elif user_says_bug_present and not model_says_bug:
        verdict = "FALSE_NEGATIVE"
    else:
        verdict = "TRUE_NEGATIVE"
        
    validation_info = {
        "user_says_bug_present": user_says_bug_present,
        "model_says_bug_present": model_says_bug,
        "verdict": verdict,
        "is_true_positive": (verdict == "TRUE_POSITIVE"),
        "annotated_at": datetime.now().isoformat()
    }
    
    for b in res.detector_report.bugs:
        if b.bug_id == bug_id:
            b.user_validation = validation_info
    if v_item:
        v_item.user_validation = validation_info
        
    persist_result_to_disk(res)

def update_global_user_validation(user_confirms_clean: bool):
    if "last_result" not in st.session_state:
        return
    res = st.session_state["last_result"]
    
    if user_confirms_clean:
        verdict = "TRUE_NEGATIVE"
    else:
        verdict = "FALSE_NEGATIVE"
        
    validation_info = {
        "user_confirms_clean": user_confirms_clean,
        "model_detected_bugs": False,
        "verdict": verdict,
        "is_true_negative": (verdict == "TRUE_NEGATIVE"),
        "annotated_at": datetime.now().isoformat()
    }
    
    res.user_validation = validation_info
    persist_result_to_disk(res)

with st.sidebar:
    st.header("Configurazione Modelli & API")
    
    st.subheader("Modulo Rilevamento")
    detector_provider = st.selectbox(
        "Provider Rilevamento",
        ["Google AI Studio", "OpenRouter"],
        index=0,
        key="det_provider"
    )
    
    if "Google" in detector_provider:
        detector_model = st.text_input(
            "Modello Rilevamento",
            value="gemini-3.8-flash",
            key="det_model_gemini",
            autocomplete="off"
        )
        det_prov_code = "google"
    else:
        detector_model = st.text_input(
            "Modello OpenRouter Rilevamento",
            value="qwen/qwen3.8-flash",
            key="det_model_or",
            autocomplete="off"
        )
        det_prov_code = "openrouter"

    st.subheader("Modulo Validazione")
    validator_provider = st.selectbox(
        "Provider Validazione",
        ["Google AI Studio", "OpenRouter"],
        index=0,
        key="val_provider"
    )
    
    if "Google" in validator_provider:
        validator_model = st.text_input(
            "Modello Validazione",
            value="gemini-3.8-flash",
            key="val_model_gemini",
            autocomplete="off"
        )
        val_prov_code = "google"
    else:
        validator_model = st.text_input(
            "Modello OpenRouter Validazione",
            value="qwen/qwen3.8-flash",
            key="val_model_or",
            autocomplete="off"
        )
        val_prov_code = "openrouter"

    st.subheader("Chiavi API")
    env_gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    env_or_key = os.getenv("OPENROUTER_API_KEY") or ""

    user_gemini_key = st.text_input(
        "Google Gemini API Key",
        value="",
        type="password",
        autocomplete="off",
        placeholder="Configurata da file .env" if env_gemini_key else "Inserisci chiave API...",
        help="Necessaria se usi Google AI Studio come provider."
    )
    if env_gemini_key:
        st.caption("Chiave Gemini rilevata nel file .env (lascia vuoto il campo per utilizzarla).")

    user_or_key = st.text_input(
        "OpenRouter API Key",
        value="",
        type="password",
        autocomplete="off",
        placeholder="Configurata da file .env" if env_or_key else "Inserisci chiave API...",
        help="Necessaria se usi OpenRouter come provider."
    )
    if env_or_key:
        st.caption("Chiave OpenRouter rilevata nel file .env (lascia vuoto il campo per utilizzarla).")

    st.subheader("Opzioni Avanzate")
    enable_agentic_video = st.checkbox(
        "Agentic Video Understanding (Gemini)",
        value=True,
        help="Attiva il loop dinamico Think-Act-Observe per esplorazione video adattiva, minor consumo di token e massima precisione visiva."
    )

def render_batch_review_dual():
    st.subheader("Esplorazione e Convalida Report Batch")
    
    col_dir, col_refresh = st.columns([4, 1])
    with col_dir:
        batch_dir = st.text_input("Cartella Report JSON", value="results", key="dual_batch_dir_input", autocomplete="off")
    with col_refresh:
        st.write("")
        st.write("")
        if st.button("Aggiorna Elenco", key="btn_refresh_dual_batch", width='stretch'):
            st.rerun()

    if not os.path.exists(batch_dir):
        st.warning(f"La cartella specificata non esiste: {batch_dir}")
        return

    all_json_files = [
        f for f in sorted(os.listdir(batch_dir))
        if f.endswith(".json") and not f.startswith("batch_summary_")
    ]

    if not all_json_files:
        st.info(f"Nessun file report JSON trovato nella cartella '{batch_dir}'. Esegui prima un'elaborazione con 'batch_runner.py'.")
        return

    reports_meta = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_unrev = 0

    for fname in all_json_files:
        fpath = os.path.join(batch_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                rdata = json.load(f)
        except Exception:
            continue

        d_bugs = rdata.get("detector_report", {}).get("bugs", [])
        g_val = rdata.get("user_validation")

        file_tp = 0
        file_fp = 0
        file_fn = 0
        file_tn = 0
        file_unrev = 0

        if d_bugs:
            for b in d_bugs:
                u_v = b.get("user_validation")
                verdict = u_v.get("verdict") if isinstance(u_v, dict) else u_v
                if verdict in ("TRUE_POSITIVE", "tp"):
                    file_tp += 1
                elif verdict in ("FALSE_POSITIVE", "fp"):
                    file_fp += 1
                elif verdict in ("FALSE_NEGATIVE", "fn"):
                    file_fn += 1
                elif verdict in ("TRUE_NEGATIVE", "tn"):
                    file_tn += 1
                else:
                    file_unrev += 1
        else:
            verdict = g_val.get("verdict") if isinstance(g_val, dict) else g_val
            if verdict in ("TRUE_NEGATIVE", "tn"):
                file_tn += 1
            elif verdict in ("FALSE_NEGATIVE", "fn"):
                file_fn += 1
            else:
                file_unrev += 1

        is_completed = (file_unrev == 0)

        total_tp += file_tp
        total_fp += file_fp
        total_fn += file_fn
        total_tn += file_tn
        total_unrev += file_unrev

        reports_meta.append({
            "filename": fname,
            "filepath": fpath,
            "video_path": rdata.get("video_path", ""),
            "bugs_count": len(d_bugs),
            "is_completed": is_completed,
            "tp": file_tp,
            "fp": file_fp,
            "fn": file_fn,
            "tn": file_tn,
            "unreviewed": file_unrev,
            "status_label": "REVISIONATO" if is_completed else "DA REVISIONARE"
        })

    st.markdown("#### Cruscotto Metriche Complessive Batch")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #28a745;">{total_tp}</div><div class="metric-label">Veri Positivi (TP)</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #dc3545;">{total_fp}</div><div class="metric-label">Falsi Positivi (FP)</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #fd7e14;">{total_fn}</div><div class="metric-label">Falsi Negativi (FN)</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #007bff;">{total_tn}</div><div class="metric-label">Veri Negativi (TN)</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #6c757d;">{total_unrev}</div><div class="metric-label">Voci da Revisionare</div></div>', unsafe_allow_html=True)

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
    with kpi_c1:
        st.metric("Precision (Accuratezza Rilevamenti)", f"{prec:.1%}")
    with kpi_c2:
        st.metric("Recall (Copertura Difetti Reali)", f"{rec:.1%}")
    with kpi_c3:
        st.metric("F1-Score", f"{f1:.3f}")

    st.markdown("---")

    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        filter_opt = st.radio(
            "Filtro Elenco Report",
            ["Tutti", "Solo da revisionare", "Gia revisionati"],
            index=0,
            horizontal=True,
            key="dual_batch_filter"
        )

    if filter_opt == "Solo da revisionare":
        filtered_reports = [r for r in reports_meta if not r["is_completed"]]
    elif filter_opt == "Gia revisionati":
        filtered_reports = [r for r in reports_meta if r["is_completed"]]
    else:
        filtered_reports = reports_meta

    if not filtered_reports:
        st.info("Nessun report corrisponde al criterio di filtro selezionato.")
        return

    report_options = [
        f"[{r['status_label']}] {r['filename']} ({r['bugs_count']} anomalie)"
        for r in filtered_reports
    ]

    if "dual_batch_curr_idx" not in st.session_state:
        st.session_state.dual_batch_curr_idx = 0

    if st.session_state.dual_batch_curr_idx >= len(filtered_reports):
        st.session_state.dual_batch_curr_idx = 0

    with f_col2:
        selected_option = st.selectbox(
            f"Seleziona Report ({len(filtered_reports)} disponibili)",
            options=report_options,
            index=st.session_state.dual_batch_curr_idx,
            key="dual_batch_select"
        )
        st.session_state.dual_batch_curr_idx = report_options.index(selected_option)

    curr_report = filtered_reports[st.session_state.dual_batch_curr_idx]

    btn_p_col, btn_n_col, info_col = st.columns([1, 1, 4])
    with btn_p_col:
        if st.button("Precedente", key="btn_prev_dual_batch", disabled=(st.session_state.dual_batch_curr_idx == 0), width='stretch'):
            st.session_state.dual_batch_curr_idx -= 1
            st.rerun()
    with btn_n_col:
        if st.button("Successivo", key="btn_next_dual_batch", disabled=(st.session_state.dual_batch_curr_idx >= len(filtered_reports) - 1), width='stretch'):
            st.session_state.dual_batch_curr_idx += 1
            st.rerun()
    with info_col:
        st.caption(f"Visualizzazione {st.session_state.dual_batch_curr_idx + 1} di {len(filtered_reports)}: `{curr_report['filename']}`")

    with open(curr_report["filepath"], "r", encoding="utf-8") as f:
        data = json.load(f)

    st.markdown("### Ispezione Video e Contesto")
    vid_col, meta_col = st.columns([3, 2])

    video_source = data.get("video_path", "")
    with vid_col:
        if video_source and os.path.exists(video_source):
            st.video(video_source)
        elif video_source and ("youtube.com" in video_source.lower() or "youtu.be" in video_source.lower()):
            st.video(video_source)
        else:
            st.warning(f"File video non trovato al percorso originario: `{video_source}`.")
            manual_vid = st.text_input("Specifica percorso alternativo del video (opzionale):", key=f"man_vid_{curr_report['filename']}", autocomplete="off")
            if manual_vid and os.path.exists(manual_vid):
                st.video(manual_vid)

    with meta_col:
        st.subheader("Metadati Esecuzione")
        st.write(f"- **File Report**: `{curr_report['filename']}`")
        st.write(f"- **Video Path**: `{video_source}`")
        det_rep = data.get("detector_report") or {}
        val_rep = data.get("validator_report") or {}
        st.write(f"- **Provider Rilevamento**: `{det_rep.get('model_name', 'N/D')}`")
        st.write(f"- **Provider Validazione**: `{val_rep.get('model_name', 'N/D')}`")
        st.write(f"- **Tempo Elaborazione**: `{data.get('execution_time_seconds', 'N/D')}s`")
        st.write(f"- **Anomalie Segnalate**: `{len(det_rep.get('bugs', []))}`")

    if det_rep.get("gameplay_summary") or val_rep.get("global_critique_summary"):
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Sintesi Gameplay**")
            st.info(det_rep.get("gameplay_summary") or "Nessuna sintesi disponibile.")
        with s2:
            st.markdown("**Valutazione Globale Validator**")
            st.info(val_rep.get("global_critique_summary") or "Nessuna sintesi disponibile.")

    st.markdown("### Segnalazioni e Validazione Umana")
    bugs = det_rep.get("bugs", [])
    val_map = {item.get("bug_id"): item for item in val_rep.get("validated_bugs", [])}

    if not bugs:
        st.write("I modelli non hanno rilevato alcuna anomalia in questo video (classificato come pulito).")
        g_val = data.get("user_validation")
        u_verdict = g_val.get("verdict") if isinstance(g_val, dict) else g_val

        if not u_verdict or u_verdict in ("UNREVIEWED", "unreviewed"):
            status_text = "Stato: Non ancora revisionato dall'utente (I modelli non hanno rilevato anomalie)"
            status_class = "background-color: #334155; color: #f8fafc;"
        elif u_verdict in ("TRUE_NEGATIVE", "tn"):
            status_text = "Stato: Validato dall'utente: VERO NEGATIVO (Confermato: Nessun bug presente nel video)"
            status_class = "background-color: #1e3a8a; color: #93c5fd;"
        elif u_verdict in ("FALSE_NEGATIVE", "fn"):
            status_text = "Stato: Validato dall'utente: FALSO NEGATIVO (Bug presente nel video, mancato dai modelli)"
            status_class = "background-color: #7c2d12; color: #fdba74;"
        else:
            status_text = f"Stato: Validato ({u_verdict})"
            status_class = "background-color: #334155; color: #f8fafc;"

        st.markdown(f'<div class="status-box" style="{status_class}">{status_text}</div>', unsafe_allow_html=True)

        cg1, cg2 = st.columns(2)
        with cg1:
            if st.button("Confermo: Video Pulito (True Negative)", key=f"btn_b_clean_{curr_report['filename']}", width='stretch'):
                data["user_validation"] = {
                    "user_confirms_clean": True,
                    "model_detected_bugs": False,
                    "verdict": "TRUE_NEGATIVE",
                    "annotated_at": datetime.now().isoformat()
                }
                with open(curr_report["filepath"], "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                st.rerun()
        with cg2:
            if st.button("Errato: C'era un Bug Mancato (False Negative)", key=f"btn_b_miss_{curr_report['filename']}", width='stretch'):
                data["user_validation"] = {
                    "user_confirms_clean": False,
                    "model_detected_bugs": False,
                    "verdict": "FALSE_NEGATIVE",
                    "annotated_at": datetime.now().isoformat()
                }
                with open(curr_report["filepath"], "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                st.rerun()
    else:
        for idx_b, bug in enumerate(bugs):
            b_id = bug.get("bug_id", f"BUG-{idx_b+1}")
            v_item = val_map.get(b_id, {})
            st.markdown(f"#### [{b_id}] {bug.get('title', 'Anomalia')}")

            c_b1, c_b2 = st.columns(2)
            with c_b1:
                st.markdown("**Segnalazione Rilevata dal Detector**")
                st.write(f"- **Timestamp**: `{format_seconds_to_timestamp(bug.get('timestamp_start', 0.0))} - {format_seconds_to_timestamp(bug.get('timestamp_end', 0.0))}` ({bug.get('timestamp_start')}s - {bug.get('timestamp_end')}s)")
                st.write(f"- **Categoria**: `{bug.get('bug_type')}`")
                st.write(f"- **Severita**: `{bug.get('severity')}` | **Confidenza**: `{bug.get('confidence', 0.0):.2f}`")
                st.write(f"- **Descrizione**: {bug.get('description')}")
                st.write(f"- **Atteso**: {bug.get('expected_behavior', 'N/D')}")
                st.write(f"- **Effettivo**: {bug.get('actual_behavior', 'N/D')}")
                if bug.get("visual_evidence"):
                    st.write(f"- **Evidenze Visive**: {bug.get('visual_evidence')}")
                if bug.get("detector_rationale"):
                    st.write(f"- **Motivazione Detector**: {bug.get('detector_rationale')}")

            with c_b2:
                st.markdown("**Verifica del Validator**")
                v_verdict = v_item.get("verdict", "N/D")
                st.write(f"- **Verdetto Validatore**: `{v_verdict}`")
                st.write(f"- **Confidenza**: `{v_item.get('confidence', 0.0):.2f}`")
                st.write(f"- **Critica Validatore**: {v_item.get('critique_explanation', 'Nessuna critica registrata.')}")
                if v_item.get("is_intended_mechanic_or_style"):
                    st.write("- **Nota**: Considerato meccanica voluta o scelta artistica.")

            clip_path = bug.get("clip_path") or v_item.get("clip_path")
            is_yt_report = ("youtube.com" in video_source.lower() or "youtu.be" in video_source.lower()) if video_source else False

            if is_yt_report:
                s_sec = v_item.get("corrected_timestamp_start") if (v_item.get("corrected_timestamp_start") is not None) else bug.get("timestamp_start", 0.0)
                e_sec = v_item.get("corrected_timestamp_end") if (v_item.get("corrected_timestamp_end") is not None) else bug.get("timestamp_end", 0.0)
                render_youtube_clip_player(video_source, s_sec, e_sec)
            elif clip_path and os.path.exists(clip_path):
                st.markdown("**Spezzone Ritagliato:**")
                st.video(clip_path)

            val_info = bug.get("user_validation") or v_item.get("user_validation")
            u_verdict = val_info.get("verdict") if isinstance(val_info, dict) else val_info

            if not u_verdict or u_verdict in ("UNREVIEWED", "unreviewed"):
                status_text = "Stato: Non ancora revisionato dall'utente"
                status_class = "background-color: #334155; color: #f8fafc;"
            elif u_verdict in ("TRUE_POSITIVE", "tp"):
                status_text = "Stato: VERO POSITIVO (Bug Reale Confermato)"
                status_class = "background-color: #14532d; color: #86efac;"
            elif u_verdict in ("FALSE_POSITIVE", "fp"):
                status_text = "Stato: FALSO POSITIVO (Segnalazione Errata del Modello)"
                status_class = "background-color: #7f1d1d; color: #fca5a5;"
            else:
                status_text = f"Stato: Validato ({u_verdict})"
                status_class = "background-color: #334155; color: #f8fafc;"

            st.markdown(f'<div class="status-box" style="{status_class}">{status_text}</div>', unsafe_allow_html=True)

            bv1, bv2 = st.columns(2)
            with bv1:
                if st.button("Si (Bug Reale Presente - TP)", key=f"btn_b_tp_{curr_report['filename']}_{b_id}", width='stretch'):
                    v_record = {
                        "user_says_bug_present": True,
                        "verdict": "TRUE_POSITIVE",
                        "annotated_at": datetime.now().isoformat()
                    }
                    bug["user_validation"] = v_record
                    if b_id in val_map:
                        val_map[b_id]["user_validation"] = v_record
                    all_u = [(b.get("user_validation") or {}).get("verdict") for b in bugs]
                    has_tp = any(v in ("TRUE_POSITIVE", "tp") for v in all_u)
                    data["user_validation"] = "tp" if has_tp else "fp"
                    with open(curr_report["filepath"], "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    st.rerun()

            with bv2:
                if st.button("No (Falso Positivo - FP)", key=f"btn_b_fp_{curr_report['filename']}_{b_id}", width='stretch'):
                    v_record = {
                        "user_says_bug_present": False,
                        "verdict": "FALSE_POSITIVE",
                        "annotated_at": datetime.now().isoformat()
                    }
                    bug["user_validation"] = v_record
                    if b_id in val_map:
                        val_map[b_id]["user_validation"] = v_record
                    all_u = [(b.get("user_validation") or {}).get("verdict") for b in bugs]
                    has_tp = any(v in ("TRUE_POSITIVE", "tp") for v in all_u)
                    all_reviewed = all(v in ("TRUE_POSITIVE", "FALSE_POSITIVE", "tp", "fp") for v in all_u)
                    if all_reviewed:
                        data["user_validation"] = "tp" if has_tp else "fp"
                    with open(curr_report["filepath"], "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    st.rerun()

            st.divider()

    with st.expander("Ispezione Raw JSON del Report"):
        st.json(data)


def render_benchmark_dashboard_dual():
    st.subheader("Benchmark Analitico e Grafici Prestazionali")

    col_dir, col_out, col_gen = st.columns([3, 2, 2])
    with col_dir:
        res_dir = st.text_input("Cartella Risultati Batch", value="results", key="bench_results_dir", autocomplete="off")
    with col_out:
        out_dir = st.text_input("Cartella Output Grafici", value="plots", key="bench_output_dir", autocomplete="off")
    with col_gen:
        st.write("")
        st.write("")
        if st.button("Genera / Aggiorna Grafici", key="btn_gen_bench_charts", width='stretch', type="primary"):
            with st.spinner("Elaborazione dati e generazione grafici in corso..."):
                from evaluate_results import generate_benchmark_charts
                generate_benchmark_charts(output_dir=out_dir, results_dir=res_dir)
            st.success("Grafici e report aggiornati con successo.")
            st.rerun()

    from evaluate_results import parse_all_results
    df_runs, df_attempts, df_summary = parse_all_results(res_dir)

    if df_summary.empty:
        st.info(f"Nessun dato trovato nella cartella '{res_dir}'. Esegui prima un batch o verifica il percorso.")
        return

    total_runs_count = len(df_runs)
    total_detected = int(df_runs["detected_bugs_count"].sum()) if "detected_bugs_count" in df_runs.columns else 0
    total_confirmed = int(df_runs["confirmed_bugs"].sum()) if "confirmed_bugs" in df_runs.columns else 0
    total_fp_count = int(df_runs["fp_count"].sum()) if "fp_count" in df_runs.columns else 0
    avg_exec_time = df_runs["execution_time_seconds"].mean() if "execution_time_seconds" in df_runs.columns else 0.0

    st.markdown("#### Riepilogo Esecuzioni e Filtraggio Dual-LLM")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_runs_count}</div><div class="metric-label">Video Analizzati</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #2b5c8f;">{total_detected}</div><div class="metric-label">Segnalati dal Detector</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #2e7d32;">{total_confirmed}</div><div class="metric-label">Confermati dal Validatore</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #2e7d32;">{total_fp_count}</div><div class="metric-label">FP Abbattuti dal Validatore</div></div>', unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_exec_time:.2f}s</div><div class="metric-label">Tempo Medio / Video</div></div>', unsafe_allow_html=True)

    tot_tp = int(df_runs["TP"].sum()) if "TP" in df_runs.columns else 0
    tot_fp = int(df_runs["FP"].sum()) if "FP" in df_runs.columns else 0
    tot_fn = int(df_runs["FN"].sum()) if "FN" in df_runs.columns else 0
    tot_tn = int(df_runs["TN"].sum()) if "TN" in df_runs.columns else 0

    pipe_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    pipe_rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    pipe_f1 = 2 * (pipe_prec * pipe_rec) / (pipe_prec + pipe_rec) if (pipe_prec + pipe_rec) > 0 else 0.0

    f2_den = (4.0 * pipe_prec) + pipe_rec
    f2_val = (5.0 * pipe_prec * pipe_rec) / f2_den if f2_den > 0 else 0.0

    f05_den = (0.25 * pipe_prec) + pipe_rec
    f05_val = (1.25 * pipe_prec * pipe_rec) / f05_den if f05_den > 0 else 0.0

    st.markdown("#### Metriche di Accuratezza Game QA (Ground Truth)")
    q1, q2, q3, q4, q5 = st.columns(5)
    with q1:
        st.metric("Precision Pipeline", f"{pipe_prec:.1%}")
    with q2:
        st.metric("Recall Pipeline", f"{pipe_rec:.1%}")
    with q3:
        st.metric("F1-Score Pipeline", f"{pipe_f1:.3f}")
    with q4:
        st.metric("F2-Score (Recall Focus)", f"{f2_val:.3f}")
    with q5:
        st.metric("F0.5-Score (Precision Focus)", f"{f05_val:.3f}")

    st.markdown("---")
    st.markdown("### Grafici Comparativi Scientifici")

    master_path = os.path.join(out_dir, "06_master_benchmark_dashboard.png")
    if os.path.exists(master_path):
        st.markdown("#### Quadro Comparativo Benchmark (Riepilogo Generale)")
        st.image(master_path, width='stretch')

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        p1 = os.path.join(out_dir, "01_model_performance_comparison.png")
        if os.path.exists(p1):
            st.image(p1, caption="Confronto Metriche di Valutazione - Detector e Pipeline Dual-LLM", width='stretch')

        p3 = os.path.join(out_dir, "03_execution_time_comparison.png")
        if os.path.exists(p3):
            st.image(p3, caption="Tempo Medio di Elaborazione per Video (s)", width='stretch')

        p5 = os.path.join(out_dir, "05_ablation_stage1_vs_pipeline.png")
        if os.path.exists(p5):
            st.image(p5, caption="Studio di Ablazione: Detector vs Pipeline per Configurazione Sperimentale", width='stretch')

    with g_col2:
        p2 = os.path.join(out_dir, "02_confusion_matrix_breakdown.png")
        if os.path.exists(p2):
            st.image(p2, caption="Distribuzione Esiti di Validazione per Configurazione", width='stretch')

        p4 = os.path.join(out_dir, "04_bug_types_by_model.png")
        if os.path.exists(p4):
            st.image(p4, caption="Distribuzione delle Tipologie di Difetto per Modello", width='stretch')

    st.markdown("---")
    st.markdown("### Tabella Riepilogativa Benchmark per Modello")
    st.dataframe(df_summary, width='stretch')

    st.markdown("---")
    st.markdown("### Esportazione Report Benchmark")
    rep_csv = os.path.join(out_dir, "benchmark_summary.csv")
    rep_md = os.path.join(out_dir, "benchmark_report.md")

    exp1, exp2 = st.columns(2)
    with exp1:
        if os.path.exists(rep_csv):
            with open(rep_csv, "rb") as f:
                st.download_button("Scarica Tabella Metriche (.csv)", f.read(), file_name="benchmark_summary.csv", mime="text/csv", key="dl_bench_csv")
    with exp2:
        if os.path.exists(rep_md):
            with open(rep_md, "rb") as f:
                st.download_button("Scarica Report Markdown (.md)", f.read(), file_name="benchmark_report.md", mime="text/markdown", key="dl_bench_md")


view_mode = st.radio(
    "Modalita Operativa",
    ["Nuova Analisi Video", "Revisione Report Batch", "Benchmark & Grafici"],
    horizontal=True,
    key="main_view_mode"
)

if view_mode == "Revisione Report Batch":
    render_batch_review_dual()
    st.stop()
elif view_mode == "Benchmark & Grafici":
    render_benchmark_dashboard_dual()
    st.stop()

st.header("Sorgente Video di Gameplay")
tab_file, tab_yt = st.tabs(["File Locale", "Link YouTube"])

target_video_input = None

with tab_file:
    uploaded_file = st.file_uploader(
        "Seleziona un video di gameplay (.mp4, .mov, .avi, .webm)",
        type=["mp4", "mov", "avi", "webm"],
        key="local_file_uploader"
    )
    if uploaded_file is not None:
        if st.session_state.get("last_uploaded_source") != uploaded_file.name:
            st.session_state["last_uploaded_source"] = uploaded_file.name
            st.session_state.pop("last_result", None)

        temp_dir = os.path.join(tempfile.gettempdir(), "dual_llm_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        sanitized_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', uploaded_file.name)
        if not sanitized_filename:
            sanitized_filename = "gameplay_video.mp4"
            
        temp_video_path = os.path.join(temp_dir, sanitized_filename)

        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        target_video_input = temp_video_path

        col_vid, col_meta = st.columns([2, 1])
        with col_vid:
            st.video(temp_video_path)
        with col_meta:
            try:
                info = get_video_info(temp_video_path)
                st.subheader("Dettagli File")
                st.write(f"- **Nome file**: `{uploaded_file.name}`")
                st.write(f"- **Durata**: `{info['duration_seconds']}s` ({format_seconds_to_timestamp(info['duration_seconds'])})")
                st.write(f"- **Risoluzione**: `{info['width']}x{info['height']}`")
                st.write(f"- **Frame Rate**: `{info['fps']} FPS`")
                st.write(f"- **Dimensione**: `{info['file_size_mb']} MB`")
            except Exception as e:
                st.warning(f"Impossibile estrarre i metadati del video: {e}")

with tab_yt:
    youtube_url_input = st.text_input(
        "Incolla l'URL del video YouTube",
        placeholder="https://www.youtube.com/watch?v=... oppure https://youtu.be/...",
        key="yt_url_input",
        autocomplete="off"
    )
    if youtube_url_input and ("youtube.com" in youtube_url_input.lower() or "youtu.be" in youtube_url_input.lower()):
        yt_clean = clean_youtube_url(youtube_url_input)
        if st.session_state.get("last_uploaded_source") != yt_clean:
            st.session_state["last_uploaded_source"] = yt_clean
            st.session_state.pop("last_result", None)

        target_video_input = yt_clean

        col_yt_vid, col_yt_meta = st.columns([2, 1])
        with col_yt_vid:
            st.video(target_video_input)
        with col_yt_meta:
            st.subheader("Dettagli YouTube")
            try:
                yt_info = get_video_info(target_video_input)
                st.write(f"- **Titolo**: `{yt_info.get('title', 'Video YouTube')}`")
                st.write(f"- **Durata**: `{yt_info['duration_seconds']}s` ({format_seconds_to_timestamp(yt_info['duration_seconds'])})")
                st.write(f"- **Risoluzione**: `{yt_info['width']}x{yt_info['height']}`")
            except Exception:
                st.write("- **Sorgente**: `YouTube URL`")
            st.info("Con Google Gemini l'elaborazione avviene in streaming diretto da YouTube.")

st.divider()

if st.button("Avvia Analisi", type="primary", disabled=(target_video_input is None)):
    if not target_video_input:
        st.error("Seleziona prima un file video locale o incolla un link YouTube valido.")
    else:
        st.session_state.pop("last_result", None)

        effective_gemini_key = user_gemini_key.strip() if user_gemini_key.strip() else env_gemini_key
        effective_or_key = user_or_key.strip() if user_or_key.strip() else env_or_key

        det_key = effective_gemini_key if det_prov_code == "google" else effective_or_key
        val_key = effective_gemini_key if val_prov_code == "google" else effective_or_key

        if det_prov_code == "google" and not det_key:
            st.error("Inserisci la chiave API per Google Gemini.")
            st.stop()
        if det_prov_code == "openrouter" and not det_key:
            st.error("Inserisci la chiave API per OpenRouter.")
            st.stop()
        if val_prov_code == "google" and not val_key:
            st.error("Inserisci la chiave API per Google Gemini per il validatore.")
            st.stop()
        if val_prov_code == "openrouter" and not val_key:
            st.error("Inserisci la chiave API per OpenRouter per il validatore.")
            st.stop()

        progress_bar = st.progress(0.0)
        status_box = st.empty()

        def update_progress(message: str, progress_val: float):
            status_box.info(message)
            progress_bar.progress(progress_val)

        try:
            pipeline = DualLLMPipeline(
                detector_provider=det_prov_code,
                detector_model=detector_model,
                detector_api_key=det_key,
                validator_provider=val_prov_code,
                validator_model=validator_model,
                validator_api_key=val_key,
                results_dir="results",
                enable_agentic_video=enable_agentic_video
            )

            result = pipeline.run(target_video_input, progress_callback=update_progress)
            st.session_state["last_result"] = result
            st.rerun()

        except Exception as ex:
            status_box.error(f"Errore durante l'esecuzione dell'analisi: {ex}")
            st.exception(ex)

if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    d_rep = res.detector_report
    v_rep = res.validator_report

    st.header("Risultati dell'Analisi")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(d_rep.bugs)}</div>
            <div class="metric-label">Anomalie Rilevate</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #28a745;">{v_rep.confirmed_count}</div>
            <div class="metric-label">Confermate</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #dc3545;">{v_rep.rejected_count}</div>
            <div class="metric-label">Falsi Positivi Scartati</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #ffc107;">{v_rep.uncertain_count}</div>
            <div class="metric-label">Incerte</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{res.execution_time_seconds}s</div>
            <div class="metric-label">Tempo di Elaborazione</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    if d_rep.bugs:
        tp_cnt = sum(1 for b in d_rep.bugs if (b.user_validation or {}).get("verdict") == "TRUE_POSITIVE")
        fp_cnt = sum(1 for b in d_rep.bugs if (b.user_validation or {}).get("verdict") == "FALSE_POSITIVE")
        fn_cnt = sum(1 for b in d_rep.bugs if (b.user_validation or {}).get("verdict") == "FALSE_NEGATIVE")
        tn_cnt = sum(1 for b in d_rep.bugs if (b.user_validation or {}).get("verdict") == "TRUE_NEGATIVE")
        unrev_cnt = sum(1 for b in d_rep.bugs if not b.user_validation)
    else:
        u_v = (res.user_validation or {}).get("verdict")
        tp_cnt = 0
        fp_cnt = 0
        fn_cnt = 1 if u_v == "FALSE_NEGATIVE" else 0
        tn_cnt = 1 if u_v == "TRUE_NEGATIVE" else 0
        unrev_cnt = 1 if not u_v else 0

    st.subheader("Verifica Umana Ground Truth")
    gt1, gt2, gt3, gt4, gt5 = st.columns(5)
    with gt1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #28a745;">{tp_cnt}</div>
            <div class="metric-label">Veri Positivi (TP)</div>
        </div>
        """, unsafe_allow_html=True)
    with gt2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #dc3545;">{fp_cnt}</div>
            <div class="metric-label">Falsi Positivi (FP)</div>
        </div>
        """, unsafe_allow_html=True)
    with gt3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #fd7e14;">{fn_cnt}</div>
            <div class="metric-label">Falsi Negativi (FN)</div>
        </div>
        """, unsafe_allow_html=True)
    with gt4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #007bff;">{tn_cnt}</div>
            <div class="metric-label">Veri Negativi (TN)</div>
        </div>
        """, unsafe_allow_html=True)
    with gt5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #6c757d;">{unrev_cnt}</div>
            <div class="metric-label">Da Revisionare</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    col_sum1, col_sum2 = st.columns(2)
    with col_sum1:
        st.subheader("Sintesi Contesto Gameplay")
        st.info(d_rep.gameplay_summary)
    with col_sum2:
        st.subheader("Valutazione Globale delle Segnalazioni")
        st.info(v_rep.global_critique_summary)

    st.subheader("Dettaglio Rilevamento e Validazione")
    
    if not d_rep.bugs:
        st.write("Nessuna anomalia riscontrata nel video dai modelli.")
        st.markdown("---")
        st.markdown("#### Validazione Umana del Video (Ground Truth)")
        val_info = res.user_validation
        if not val_info:
            status_text = "Stato: Non ancora revisionato dall'utente (I modelli non hanno rilevato anomalie)"
            status_class = "background-color: #334155; color: #f8fafc;"
        else:
            verdict = val_info.get("verdict")
            if verdict == "TRUE_NEGATIVE":
                status_text = "Stato: Validato dall'utente: VERO NEGATIVO (Confermato: Nessun bug presente nel video)"
                status_class = "background-color: #1e3a8a; color: #93c5fd;"
            elif verdict == "FALSE_NEGATIVE":
                status_text = "Stato: Validato dall'utente: FALSO NEGATIVO (Bug presente nel video, mancato dai modelli)"
                status_class = "background-color: #7c2d12; color: #fdba74;"
            else:
                status_text = f"Stato: Validato dall'utente ({verdict})"
                status_class = "background-color: #334155; color: #f8fafc;"

        st.markdown(f'<div class="status-box" style="{status_class}">{status_text}</div>', unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if st.button("Confermo: Video Pulito (True Negative)", key="btn_global_clean", width='stretch'):
                update_global_user_validation(True)
                st.rerun()
        with col_g2:
            if st.button("Errato: C'era un Bug Mancato (False Negative)", key="btn_global_missed", width='stretch'):
                update_global_user_validation(False)
                st.rerun()
    else:
        val_map = {item.bug_id: item for item in v_rep.validated_bugs}

        for bug in d_rep.bugs:
            v_item = val_map.get(bug.bug_id)
            
            with st.container():
                st.markdown(f"#### [{bug.bug_id}] {bug.title}")
                c_det, c_val = st.columns([1, 1])

                with c_det:
                    b_t_str = bug.bug_type.value if hasattr(bug.bug_type, "value") else str(bug.bug_type)
                    b_s_str = bug.severity.value if hasattr(bug.severity, "value") else str(bug.severity)

                    st.markdown("**Segnalazione Rilevata**")
                    st.write(f"- **Timestamp**: `{format_seconds_to_timestamp(bug.timestamp_start)} - {format_seconds_to_timestamp(bug.timestamp_end)}` ({bug.timestamp_start}s - {bug.timestamp_end}s)")
                    st.write(f"- **Categoria**: `{b_t_str}`")
                    st.write(f"- **Severita**: `{b_s_str}` | **Confidenza**: `{bug.confidence:.2f}`")
                    st.write(f"- **Descrizione**: {bug.description}")
                    st.write(f"- **Atteso**: {bug.expected_behavior}")
                    st.write(f"- **Effettivo**: {bug.actual_behavior}")

                with c_val:
                    st.markdown("**Esito Validazione**")
                    if v_item:
                        if v_item.verdict == ValidationVerdictEnum.CONFIRMED:
                            badge_html = '<span class="badge-confirmed">VERDETTO: CONFERMATO</span>'
                        elif v_item.verdict == ValidationVerdictEnum.REJECTED_FALSE_POSITIVE:
                            badge_html = '<span class="badge-rejected">VERDETTO: FALSO POSITIVO SCARTATO</span>'
                        else:
                            badge_html = '<span class="badge-uncertain">VERDETTO: INCERTO</span>'

                        st.markdown(badge_html, unsafe_allow_html=True)
                        st.write(f"- **Confidenza Validatore**: `{v_item.confidence:.2f}`")
                        st.write(f"- **Motivazione Critica**: {v_item.critique_explanation}")
                        if v_item.is_intended_mechanic_or_style:
                            st.write("- **Nota**: Identificato come meccanica di gioco voluta o stile artistico.")
                        if v_item.corrected_bug_type:
                            c_t_str = v_item.corrected_bug_type.value if hasattr(v_item.corrected_bug_type, "value") else str(v_item.corrected_bug_type)
                            st.write(f"- **Categoria Rettificata**: `{c_t_str}`")
                        if v_item.corrected_severity:
                            c_s_str = v_item.corrected_severity.value if hasattr(v_item.corrected_severity, "value") else str(v_item.corrected_severity)
                            st.write(f"- **Severita Rettificata**: `{c_s_str}`")
                        if v_item.corrected_timestamp_start is not None:
                            c_s_time = format_seconds_to_timestamp(v_item.corrected_timestamp_start)
                            c_e_time = format_seconds_to_timestamp(v_item.corrected_timestamp_end or v_item.corrected_timestamp_start)
                            st.write(f"- **Timestamp Rettificato**: `{c_s_time} - {c_e_time}`")
                    else:
                        st.write("Nessuna valutazione disponibile.")

                is_yt = ("youtube.com" in res.video_path.lower() or "youtu.be" in res.video_path.lower())

                if is_yt:
                    s_sec = v_item.corrected_timestamp_start if (v_item and v_item.corrected_timestamp_start is not None) else bug.timestamp_start
                    e_sec = v_item.corrected_timestamp_end if (v_item and v_item.corrected_timestamp_end is not None) else bug.timestamp_end
                    render_youtube_clip_player(res.video_path, s_sec, e_sec)
                else:
                    clip_path_to_show = bug.clip_path or (v_item.clip_path if v_item else None)
                    if clip_path_to_show and os.path.exists(clip_path_to_show):
                        st.markdown(f"**Spezzone Video Ritagliato (`{os.path.basename(clip_path_to_show)}`):**")
                        col_clip_vid, col_clip_btn = st.columns([3, 1])
                        with col_clip_vid:
                            st.video(clip_path_to_show)
                        with col_clip_btn:
                            try:
                                with open(clip_path_to_show, "rb") as cf:
                                    clip_bytes = cf.read()
                                st.download_button(
                                    label="Scarica Spezzone (.mp4)",
                                    data=clip_bytes,
                                    file_name=os.path.basename(clip_path_to_show),
                                    mime="video/mp4",
                                    key=f"dl_clip_{bug.bug_id}"
                                )
                            except Exception as e:
                                st.caption(f"Impossibile preparare download: {e}")

                st.markdown("##### Modulo di Validazione Umana (Ground Truth)")
                val_info = bug.user_validation or (v_item.user_validation if v_item else None)
                model_says_bug = (v_item.verdict == ValidationVerdictEnum.CONFIRMED) if v_item else True

                if not val_info:
                    status_text = f"Stato: Non ancora revisionato dall'utente (Esito Modello: {'CONFERMATO' if model_says_bug else 'SCARTATO'})"
                    status_class = "background-color: #334155; color: #f8fafc;"
                else:
                    u_verdict = val_info.get("verdict")
                    if u_verdict == "TRUE_POSITIVE":
                        status_text = "Stato: Validato dall'utente: VERO POSITIVO (Bug Reale confermato)"
                        status_class = "background-color: #14532d; color: #86efac;"
                    elif u_verdict == "FALSE_POSITIVE":
                        status_text = "Stato: Validato dall'utente: FALSO POSITIVO (Rilevamento Errato del modello)"
                        status_class = "background-color: #7f1d1d; color: #fca5a5;"
                    elif u_verdict == "FALSE_NEGATIVE":
                        status_text = "Stato: Validato dall'utente: FALSO NEGATIVO (Bug presente, scartato dal validatore)"
                        status_class = "background-color: #7c2d12; color: #fdba74;"
                    elif u_verdict == "TRUE_NEGATIVE":
                        status_text = "Stato: Validato dall'utente: VERO NEGATIVO (Confermato: Non e un bug)"
                        status_class = "background-color: #1e3a8a; color: #93c5fd;"
                    else:
                        status_text = f"Stato: Validato dall'utente ({u_verdict})"
                        status_class = "background-color: #334155; color: #f8fafc;"

                st.markdown(f'<div class="status-box" style="{status_class}">{status_text}</div>', unsafe_allow_html=True)

                col_btn_yes, col_btn_no = st.columns(2)
                with col_btn_yes:
                    if st.button("Si (Bug Reale Presente)", key=f"btn_val_yes_{bug.bug_id}", width='stretch'):
                        update_bug_user_validation(bug.bug_id, True)
                        st.rerun()
                with col_btn_no:
                    if st.button("No (Falso Positivo / Non e un Bug)", key=f"btn_val_no_{bug.bug_id}", width='stretch'):
                        update_bug_user_validation(bug.bug_id, False)
                        st.rerun()

                st.divider()

    st.subheader("Esportazione Report")
    raw_name = os.path.splitext(os.path.basename(res.video_path))[0]
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_name)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        json_data = json.dumps(res.model_dump(), indent=2, ensure_ascii=False)
        st.download_button(
            label="Scarica Report JSON",
            data=json_data,
            file_name=f"report_{clean_name}.json",
            mime="application/json"
        )
    with col_d2:
        md_data = DualLLMPipeline.generate_markdown_report(res)
        st.download_button(
            label="Scarica Report Markdown",
            data=md_data,
            file_name=f"report_{clean_name}.md",
            mime="text/markdown"
        )
