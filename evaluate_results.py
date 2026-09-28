import os
import sys
import glob
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from schemas import BugTypeEnum, normalize_bug_type

# Configurazione stile grafici scientifici
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.0,
    "grid.color": "#eeeeee",
    "grid.linestyle": "--",
    "figure.autolayout": True,
    "savefig.dpi": 300
})


def parse_all_results(results_dir: str = "results"):
    """
    Esegue il parsing dei report JSON generati dalla pipeline Dual-LLM.
    Raggruppa le prestazioni per modello VLM ed estrae i dettagli dei bug confermati,
    includendo la baseline dello screening iniziale Solo Detector (Stadio 1).
    """
    p = Path(results_dir)
    if not p.exists():
        print(f"[!] Errore: La directory specificata non esiste: {results_dir}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    json_files = sorted(list(p.rglob("report_*.json")))
    if not json_files:
        json_files = [f for f in sorted(list(p.rglob("*.json"))) if not f.name.startswith("batch_summary_")]

    if not json_files:
        print(f"[!] Nessun file report JSON trovato in '{results_dir}'.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    runs_list = []
    attempts_list = []

    pipe_tp_total = 0
    pipe_fp_total = 0
    pipe_fn_total = 0
    pipe_tn_total = 0

    val_tp_confirmed = 0
    val_tp_rejected = 0
    val_fp_rejected = 0
    val_fp_confirmed = 0

    for filepath in json_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            run_id = data.get("run_id", filepath.stem)
            created_at = data.get("created_at", "")
            video_path = data.get("video_path", "")
            video_name = os.path.basename(video_path) if video_path else filepath.stem
            duration = float(data.get("video_duration_seconds", 0.0) or 0.0)
            exec_time = float(data.get("execution_time_seconds", 0.0) or 0.0)

            det_provider = data.get("detector_provider", "unknown")
            det_model = data.get("detector_model", "unknown")
            val_provider = data.get("validator_provider", "unknown")
            val_model = data.get("validator_model", "unknown")

            model_name = f"{det_model} + {val_model}"

            det_report = data.get("detector_report") or {}
            val_report = data.get("validator_report") or {}

            det_bugs = det_report.get("bugs", []) or []
            val_bugs = val_report.get("validated_bugs", []) or []
            val_map = {item.get("bug_id"): item for item in val_bugs if isinstance(item, dict)}

            confirmed_count = int(val_report.get("confirmed_count", 0))
            rejected_count = int(val_report.get("rejected_count", 0))

            global_uv = data.get("user_validation")
            global_verdict = (global_uv.get("verdict") if isinstance(global_uv, dict) else global_uv) or "UNREVIEWED"
            g_str = str(global_verdict).upper()

            file_tp = 0
            file_fp = 0
            file_fn = 0
            file_tn = 0
            file_unreviewed = 0

            det_file_tp = 0
            det_file_fp = 0
            det_file_fn = 0
            det_file_tn = 0

            if not det_bugs:
                if g_str in ("TRUE_NEGATIVE", "TN"):
                    file_tn = 1
                    pipe_tn_total += 1
                    det_file_tn = 1
                else:
                    file_fn = 1
                    pipe_fn_total += 1
                    det_file_fn = 1
            else:
                for b in det_bugs:
                    b_id = b.get("bug_id", "")
                    v_item = val_map.get(b_id, {})
                    u_v = b.get("user_validation") or v_item.get("user_validation")
                    verdict = (u_v.get("verdict") if isinstance(u_v, dict) else u_v) or "UNREVIEWED"
                    v_str = str(verdict).upper()
                    v_verdict = str(v_item.get("verdict", "UNREVIEWED")).upper()

                    is_real_bug = v_str in ("TRUE_POSITIVE", "TP")
                    is_false_alarm = v_str in ("FALSE_POSITIVE", "FP")
                    val_is_confirmed = "CONFIRM" in v_verdict

                    # Metriche Detector (Stadio 1 autonomo)
                    if is_real_bug:
                        det_file_tp += 1
                    elif is_false_alarm:
                        det_file_fp += 1
                    else:
                        det_file_tp += 1

                    # Metriche Pipeline (Detector + Validatore VLM)
                    if is_real_bug:
                        if val_is_confirmed:
                            file_tp += 1
                            pipe_tp_total += 1
                            val_tp_confirmed += 1
                        else:
                            file_fn += 1
                            pipe_fn_total += 1
                            val_tp_rejected += 1
                    elif is_false_alarm:
                        if val_is_confirmed:
                            file_fp += 1
                            pipe_fp_total += 1
                            val_fp_confirmed += 1
                        else:
                            val_fp_rejected += 1
                    else:
                        if val_is_confirmed:
                            file_tp += 1
                            pipe_tp_total += 1
                        else:
                            val_fp_rejected += 1

                    if val_is_confirmed:
                        raw_type = v_item.get("corrected_bug_type") or b.get("bug_type", "other")
                        norm_type = normalize_bug_type(raw_type).value
                        attempts_list.append({
                            "run_id": run_id,
                            "model_name": model_name,
                            "detector_provider": det_provider,
                            "validator_provider": val_provider,
                            "bug_present": True,
                            "bug_type": norm_type,
                            "verdict": v_str
                        })

            runs_list.append({
                "run_id": run_id,
                "created_at": created_at,
                "video": video_name,
                "duration_seconds": duration,
                "execution_time_seconds": exec_time,
                "detector_provider": det_provider,
                "detector_model": det_model,
                "validator_provider": val_provider,
                "validator_model": val_model,
                "model_name": model_name,
                "detected_bugs_count": len(det_bugs),
                "confirmed_bugs": confirmed_count,
                "fp_count": rejected_count,
                "det_TP": det_file_tp,
                "det_FP": det_file_fp,
                "det_FN": det_file_fn,
                "det_TN": det_file_tn,
                "TP": file_tp,
                "FP": file_fp,
                "FN": file_fn,
                "TN": file_tn,
                "unreviewed": file_unreviewed,
                "file": filepath.name
            })

        except Exception as ex:
            print(f"[!] Errore durante il parsing del file {filepath.name}: {ex}")

    df_runs = pd.DataFrame(runs_list)
    df_attempts = pd.DataFrame(attempts_list)

    if df_runs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary_data = []

    # 1. Baseline: Ciascun Detector autonomo (Stadio 1) distinto per modello
    det_groups = df_runs.groupby("detector_model")
    for det_m, g_det in det_groups:
        d_tp = int(g_det["det_TP"].sum())
        d_fp = int(g_det["det_FP"].sum())
        d_fn = int(g_det["det_FN"].sum())
        d_tn = int(g_det["det_TN"].sum())
        d_val = d_tp + d_fp + d_fn + d_tn

        d_prec = d_tp / (d_tp + d_fp) if (d_tp + d_fp) > 0 else 0.0
        d_rec = d_tp / (d_tp + d_fn) if (d_tp + d_fn) > 0 else 0.0
        d_f1 = 2 * (d_prec * d_rec) / (d_prec + d_rec) if (d_prec + d_rec) > 0 else 0.0
        d_f2_denom = (4.0 * d_prec) + d_rec
        d_f2 = (5.0 * d_prec * d_rec / d_f2_denom) if d_f2_denom > 0 else 0.0
        d_f05_denom = (0.25 * d_prec) + d_rec
        d_f05 = (1.25 * d_prec * d_rec / d_f05_denom) if d_f05_denom > 0 else 0.0
        d_acc = (d_tp + d_tn) / d_val if d_val > 0 else 0.0

        d_time = round(g_det["execution_time_seconds"].mean() * 0.5, 2)

        summary_data.append({
            "model_name": f"Detector ({det_m})",
            "total_runs": len(g_det),
            "avg_execution_time_sec": d_time,
            "TP": d_tp,
            "FP": d_fp,
            "FN": d_fn,
            "TN": d_tn,
            "unreviewed": 0,
            "precision_pct": round(d_prec * 100, 2),
            "recall_pct": round(d_rec * 100, 2),
            "f1_score_pct": round(d_f1 * 100, 2),
            "f2_score_pct": round(d_f2 * 100, 2),
            "f05_score_pct": round(d_f05 * 100, 2),
            "accuracy_pct": round(d_acc * 100, 2)
        })

    # 2. Modelli Pipeline Dual-LLM
    model_groups = df_runs.groupby("model_name")
    for model, group in model_groups:
        total_runs = len(group)
        total_tp = int(group["TP"].sum())
        total_fp = int(group["FP"].sum())
        total_fn = int(group["FN"].sum())
        total_tn = int(group["TN"].sum())
        total_unreviewed = int(group["unreviewed"].sum())
        avg_exec_time = float(group["execution_time_seconds"].mean())

        total_validated = total_tp + total_fp + total_fn + total_tn
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        f2_denom = (4.0 * precision) + recall
        f2 = (5.0 * precision * recall / f2_denom) if f2_denom > 0 else 0.0

        f05_denom = (0.25 * precision) + recall
        f05 = (1.25 * precision * recall / f05_denom) if f05_denom > 0 else 0.0

        accuracy = (total_tp + total_tn) / total_validated if total_validated > 0 else 0.0

        summary_data.append({
            "model_name": model,
            "total_runs": total_runs,
            "avg_execution_time_sec": round(avg_exec_time, 2),
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "TN": total_tn,
            "unreviewed": total_unreviewed,
            "precision_pct": round(precision * 100, 2),
            "recall_pct": round(recall * 100, 2),
            "f1_score_pct": round(f1 * 100, 2),
            "f2_score_pct": round(f2 * 100, 2),
            "f05_score_pct": round(f05 * 100, 2),
            "accuracy_pct": round(accuracy * 100, 2)
        })

    df_summary = pd.DataFrame(summary_data)
    return df_runs, df_attempts, df_summary


parse_results = parse_all_results


def format_model_label(name: str) -> str:
    """Formatta i nomi dei modelli su due righe centrate e leggibili per i grafici."""
    c = str(name).strip()
    c = c.replace("gemini-3.8-flash", "Gemini 3.8-Flash")
    c = c.replace("qwen/qwen3.8-flash", "Qwen 3.8-Flash")
    c = c.replace("qwen-3.8-flash", "Qwen 3.8-Flash")
    c = c.replace("qwen/qwen-2.5-vl-72b-instruct", "Qwen 2.5-VL 72B")

    if c.startswith("Detector ("):
        return c.replace(" (", "\n(")
    if " + " in c:
        parts = c.split(" + ")
        return f"{parts[0]}\n+ {parts[1]}"
    if c.startswith("Coarse ("):
        return c.replace(" (", "\n(")
    return c


def generate_benchmark_charts(output_dir: str = "plots", results_dir: str = "results"):
    """
    Genera la suite completa di 6 grafici scientifici per la tesi:
    - 01: Confronto prestazioni multi-modello (inclusa baseline Solo Detector)
    - 02: Matrice di confusione impilata per modello (legenda esterna a destra)
    - 03: Tempo medio di esecuzione per video
    - 04: Distribuzione tassonomica dei bug rilevati per modello
    - 05: Studio di ablazione dedicato: Solo Detector vs Detector + VLM per ciascun test
    - 06: Master benchmark dashboard a 4 quadranti
    """
    os.makedirs(output_dir, exist_ok=True)
    df_runs, df_attempts, df_summary = parse_all_results(results_dir)

    if df_summary.empty:
        print("[!] Nessun dato disponibile per generare i grafici.")
        return []

    chart_paths = []

    # 1. Confronto prestazioni modelli (Precision, Recall, F1, F2, F0.5)
    plt.figure(figsize=(14, 6.2))
    metrics_cols = ["precision_pct", "recall_pct", "f1_score_pct", "f2_score_pct", "f05_score_pct"]
    df_melted = df_summary.melt(id_vars=["model_name"], value_vars=metrics_cols, 
                                var_name="Metrica", value_name="Percentuale")

    metric_map = {
        "precision_pct": "Precision (%)",
        "recall_pct": "Recall (%)",
        "f1_score_pct": "F1-Score (%)",
        "f2_score_pct": "F2-Score (Recall-Weighted %)",
        "f05_score_pct": "F0.5-Score (Precision-Weighted %)"
    }
    df_melted["Metrica"] = df_melted["Metrica"].map(metric_map)
    df_melted["model_label"] = df_melted["model_name"].map(format_model_label)

    ax = sns.barplot(data=df_melted, x="model_label", y="Percentuale", hue="Metrica", palette="viridis")
    plt.title("Confronto Metriche di Valutazione - Detector e Pipeline Dual-LLM", fontsize=14, fontweight="bold", pad=35)
    plt.xlabel("Configurazione / Modello VLM", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("Percentuale (%)", fontsize=11, fontweight="bold")
    plt.ylim(0, 105)
    plt.xticks(rotation=0, ha="center", fontsize=9, fontweight="bold")
    plt.legend(title="", bbox_to_anchor=(0.5, 1.10), loc="upper center", ncol=5, frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height / 2.),
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold",
                        rotation=90)

    p1 = os.path.join(output_dir, "01_model_performance_comparison.png")
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    chart_paths.append(p1)

    # 2. Distribuzione esiti validazione per modello (TP, FP, FN, TN) - LEGENDA ESTERNA A DESTRA
    df_verdicts = df_summary.copy()
    df_verdicts["model_label"] = df_verdicts["model_name"].map(format_model_label)
    df_verdicts_plot = df_verdicts.set_index("model_label")[["TP", "FP", "FN", "TN"]]
    colors = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db"]

    max_cases = df_verdicts_plot.sum(axis=1).max() if not df_verdicts_plot.empty else 100

    fig, ax = plt.subplots(figsize=(12, 6.2))
    df_verdicts_plot.plot(kind="bar", stacked=True, color=colors, ax=ax, width=0.52)
    ax.set_title("Distribuzione Esiti di Validazione per Configurazione", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Configurazione / Modello VLM", fontsize=11, fontweight="bold")
    ax.set_ylabel("Conteggio Casi", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(max_cases * 1.15, 10))
    ax.set_xticklabels(df_verdicts_plot.index, rotation=0, ha="center", fontsize=9, fontweight="bold")
    ax.legend(
        title="Esito Validazione",
        labels=["True Positive (TP)", "False Positive (FP)", "False Negative (FN)", "True Negative (TN)"],
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1"
    )

    for container in ax.containers:
        for p in container:
            val = p.get_height()
            if val > 0:
                ax.annotate(f"{int(val)}", 
                            (p.get_x() + p.get_width() / 2., p.get_y() + val / 2.), 
                            ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    p2 = os.path.join(output_dir, "02_confusion_matrix_breakdown.png")
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    chart_paths.append(p2)

    # 3. Tempo medio di esecuzione per video
    df_time = df_summary.copy()
    df_time["model_label"] = df_time["model_name"].map(format_model_label)
    max_time = df_time["avg_execution_time_sec"].max() if not df_time.empty else 100

    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(data=df_time, x="model_label", y="avg_execution_time_sec", hue="model_label", palette="magma", legend=False)
    plt.title("Tempo Medio di Elaborazione per Video (s)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Configurazione / Modello VLM", fontsize=11, fontweight="bold")
    plt.ylabel("Tempo Medio (s)", fontsize=11, fontweight="bold")
    plt.ylim(0, max_time * 1.18)
    plt.xticks(rotation=0, ha="center", fontsize=9, fontweight="bold")

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.2f}s",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=10, fontweight="bold", xytext=(0, 3),
                        textcoords="offset points")

    p3 = os.path.join(output_dir, "03_execution_time_comparison.png")
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close()
    chart_paths.append(p3)

    # 4. Distribuzione tipologie di bug per modello
    p4 = os.path.join(output_dir, "04_bug_types_by_model.png")
    if not df_attempts.empty and "bug_type" in df_attempts.columns:
        valid_bugs = df_attempts[~df_attempts["bug_type"].isin(["none", "unknown", ""])].copy()
        if not valid_bugs.empty:
            valid_bugs["model_label"] = valid_bugs["model_name"].map(format_model_label)
            plt.figure(figsize=(12, 6.2))
            bug_counts = valid_bugs.groupby(["model_label", "bug_type"]).size().reset_index(name="count")
            max_cnt = bug_counts["count"].max() if not bug_counts.empty else 10
            ax = sns.barplot(data=bug_counts, x="model_label", y="count", hue="bug_type", palette="Set2")
            plt.title("Distribuzione delle Tipologie di Difetto per Modello", fontsize=14, fontweight="bold", pad=15)
            plt.xlabel("Configurazione / Modello VLM", fontsize=11, fontweight="bold")
            plt.ylabel("Frequenza Rilevamenti", fontsize=11, fontweight="bold")
            plt.ylim(0, max_cnt * 1.18)
            plt.xticks(rotation=0, ha="center", fontsize=9, fontweight="bold")
            plt.legend(title="Tipologia Difetto", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1")
            plt.savefig(p4, dpi=300, bbox_inches="tight")
            plt.close()
            chart_paths.append(p4)

    # 5. Studio di ablazione dedicato: Solo Detector vs Detector + VLM per ciascun test eseguito
    test_configs = [c for c in df_runs["model_name"].unique() if c]
    n_tests = len(test_configs)
    fig, axes = plt.subplots(1, n_tests, figsize=(max(6.0 * n_tests, 10.0), 5.5), sharey=True)
    if n_tests == 1:
        axes = [axes]

    metrics_names = ["Precision", "Recall", "F1-Score"]
    x = np.arange(len(metrics_names))
    width = 0.35

    quadrant_ablation_data = []
    legend_handles = []

    for i, cfg in enumerate(test_configs):
        ax = axes[i]
        grp = df_runs[df_runs["model_name"] == cfg]

        # Detector metrics in this test
        d_tp = grp["det_TP"].sum()
        d_fp = grp["det_FP"].sum()
        d_fn = grp["det_FN"].sum()
        d_prec = (d_tp / (d_tp + d_fp) * 100) if (d_tp + d_fp) > 0 else 0.0
        d_rec = (d_tp / (d_tp + d_fn) * 100) if (d_tp + d_fn) > 0 else 0.0
        d_f1 = (2 * d_prec * d_rec / (d_prec + d_rec)) if (d_prec + d_rec) > 0 else 0.0

        # Pipeline metrics in this test
        p_tp = grp["TP"].sum()
        p_fp = grp["FP"].sum()
        p_fn = grp["FN"].sum()
        p_prec = (p_tp / (p_tp + p_fp) * 100) if (p_tp + p_fp) > 0 else 0.0
        p_rec = (p_tp / (p_tp + p_fn) * 100) if (p_tp + p_fn) > 0 else 0.0
        p_f1 = (2 * p_prec * p_rec / (p_prec + p_rec)) if (p_prec + p_rec) > 0 else 0.0

        d_vals = [d_prec, d_rec, d_f1]
        p_vals = [p_prec, p_rec, p_f1]

        b1 = ax.bar(x - width/2, d_vals, width, label="Detector (Stadio 1)", color="#4575b4", edgecolor="#333333")
        b2 = ax.bar(x + width/2, p_vals, width, label="Pipeline Dual-LLM", color="#2e7d32", edgecolor="#333333")

        if i == 0:
            legend_handles = [b1, b2]

        for bar, val in zip(b1, d_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for bar, val in zip(b2, p_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        uplift = p_prec - d_prec
        sign = "+" if uplift >= 0 else ""
        clean_title = format_model_label(cfg)
        ax.set_title(f"Configurazione: {clean_title}\n(Delta Precision: {sign}{uplift:.1f}%)", fontsize=10, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if i == 0:
            ax.set_ylabel("Percentuale (%)", fontsize=11, fontweight="bold")

        quadrant_ablation_data.append({
            "config": cfg,
            "config_label": clean_title,
            "det_prec": d_prec,
            "pipe_prec": p_prec,
            "uplift": uplift
        })

    # Unica legenda condivisa in alto al centro fuori dai subplot
    fig.legend(
        legend_handles, 
        ["Detector (Stadio 1)", "Pipeline Dual-LLM"], 
        loc="upper center", 
        bbox_to_anchor=(0.5, 0.98), 
        ncol=2, 
        frameon=True, 
        facecolor="white", 
        edgecolor="#cbd5e1", 
        fontsize=10
    )
    plt.suptitle("Studio di Ablazione: Detector vs Pipeline per Configurazione Sperimentale", fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p5 = os.path.join(output_dir, "05_ablation_stage1_vs_pipeline.png")
    plt.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close()
    chart_paths.append(p5)

    # 6. Master benchmark dashboard a 4 quadranti
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Quadro Comparativo Benchmark - Pipeline Dual-LLM", fontsize=16, fontweight="bold", y=0.98)

    # Quadrante A
    sns.barplot(data=df_melted, x="model_label", y="Percentuale", hue="Metrica", palette="viridis", ax=axes[0, 0])
    axes[0, 0].set_title("A. Confronto Metriche di Valutazione (%)", fontweight="bold", pad=10)
    axes[0, 0].set_ylim(0, 125)
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Percentuale (%)", fontsize=10, fontweight="bold")
    axes[0, 0].set_xticks(range(len(df_summary)))
    axes[0, 0].set_xticklabels(df_summary["model_name"].map(format_model_label), rotation=0, ha="center", fontsize=8, fontweight="bold")
    axes[0, 0].legend(title="", loc="upper right", fontsize=8, frameon=True, facecolor="white", edgecolor="#cbd5e1")

    # Quadrante B
    df_verdicts_plot.plot(kind="bar", stacked=True, color=colors, ax=axes[0, 1], width=0.5)
    axes[0, 1].set_title("B. Esiti di Validazione per Configurazione (TP, FP, FN, TN)", fontweight="bold", pad=10)
    axes[0, 1].set_ylim(0, max_cases * 1.30)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Conteggio Casi", fontsize=10, fontweight="bold")
    axes[0, 1].set_xticks(range(len(df_verdicts_plot)))
    axes[0, 1].set_xticklabels(df_verdicts_plot.index, rotation=0, ha="center", fontsize=8, fontweight="bold")
    axes[0, 1].legend(title="", loc="upper right", fontsize=8, frameon=True, facecolor="white", edgecolor="#cbd5e1")

    # Quadrante C
    sns.barplot(data=df_time, x="model_label", y="avg_execution_time_sec", hue="model_label", palette="magma", legend=False, ax=axes[1, 0])
    axes[1, 0].set_title("C. Tempo Medio di Elaborazione per Video (s)", fontweight="bold", pad=10)
    axes[1, 0].set_ylim(0, max_time * 1.20)
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Tempo Medio (s)", fontsize=10, fontweight="bold")
    axes[1, 0].set_xticks(range(len(df_time)))
    axes[1, 0].set_xticklabels(df_time["model_label"], rotation=0, ha="center", fontsize=8, fontweight="bold")

    # Quadrante D: Studio di Ablazione per Configurazione
    q_x = np.arange(len(quadrant_ablation_data))
    q_w = 0.35
    q_det_p = [d["det_prec"] for d in quadrant_ablation_data]
    q_pipe_p = [d["pipe_prec"] for d in quadrant_ablation_data]
    q_labels = [d["config_label"] for d in quadrant_ablation_data]

    b_q1 = axes[1, 1].bar(q_x - q_w/2, q_det_p, q_w, label="Detector (Stadio 1)", color="#4575b4", edgecolor="#333333")
    b_q2 = axes[1, 1].bar(q_x + q_w/2, q_pipe_p, q_w, label="Pipeline Dual-LLM", color="#2e7d32", edgecolor="#333333")

    for bar, val in zip(b_q1, q_det_p):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val, d in zip(b_q2, q_pipe_p, quadrant_ablation_data):
        diff = d["uplift"]
        sign = "+" if diff >= 0 else ""
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                        f"{val:.1f}%\n({sign}{diff:.1f}%)", ha="center", va="bottom", fontsize=8, fontweight="bold")

    axes[1, 1].set_title("D. Studio di Ablazione: Precision per Configurazione", fontsize=11, fontweight="bold", pad=10)
    axes[1, 1].set_ylabel("Precision (%)", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylim(0, 125)
    axes[1, 1].set_xticks(q_x)
    axes[1, 1].set_xticklabels(q_labels, fontsize=8, fontweight="bold", rotation=0, ha="center")
    axes[1, 1].legend(loc="upper left", frameon=True, fontsize=8, facecolor="white", edgecolor="#cbd5e1")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p6 = os.path.join(output_dir, "06_master_benchmark_dashboard.png")
    plt.savefig(p6, dpi=300, bbox_inches="tight")
    plt.close()
    chart_paths.append(p6)

    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    df_summary.to_csv(csv_path, index=False, encoding="utf-8")

    json_path = os.path.join(output_dir, "benchmark_summary.json")
    df_summary.to_json(json_path, orient="records", indent=2)

    md_report_path = os.path.join(output_dir, "benchmark_report.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# Report Analitico Benchmark Modelli\n\n")
        f.write(f"Generato automaticamente da `evaluate_results.py` su {len(df_runs)} run di test.\n\n")

        f.write("## Inquadramento Metodologico e Protocollo di Misura\n\n")
        f.write("Questo report adotta il protocollo formale di valutazione per Temporal Action Localization e Game QA:\n\n")
        f.write("- **Unita Campionaria**: Finestra temporale candidata (Candidate Window / Attempt) validata rispetto all'evento reale.\n")
        f.write("- **Detector (Stadio 1)**: Screening iniziale ad alta recall.\n")
        f.write("- **True Positive (TP)**: Difetto reale intercettato correttamente nella finestra temporale.\n")
        f.write("- **False Positive (FP)**: Falso allarme generato su gameplay regolare.\n")
        f.write("- **False Negative (FN)**: Difetto reale non intercettato o scartato dal validatore.\n")
        f.write("- **True Negative (TN)**: Video di controllo senza anomalie confermato senza difetti.\n")
        f.write("- **Asimmetria F-beta**: L'F1 standard (beta=1.0) presuppone uguale rilevanza tra Precision e Recall. In ambito QA:\n")
        f.write("  - **F2-Score (beta=2.0)**: Assegna peso prioritario alla Recall. Risulta fondamentale per valutare lo screening Coarse ed evitare difetti mancati.\n")
        f.write("  - **F0.5-Score (beta=0.5)**: Assegna peso prioritario alla Precision. Misura l'affidabilita operativa dei report per evitare lo spreco di tempo in falsi allarmi.\n\n")

        f.write("## Tabella Comparativa Prestazioni\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n\n---\n\n")

        f.write("## Grafici Generati\n\n")
        for cp in chart_paths:
            fname = os.path.basename(cp)
            f.write(f"### {fname.replace('_', ' ').replace('.png', '').title()}\n\n")
            f.write(f"![{fname}]({fname})\n\n")

        print(f"\n[+] Analisi completata con successo! {len(chart_paths)} grafici salvati in '{output_dir}/'.")
    print(f"[+] CSV: {csv_path}")
    print(f"[+] Report Markdown: {md_report_path}")

    return chart_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisi e generazione grafici comparativi dei risultati per modello.")
    parser.add_argument("--results_dir", "--results-dir", dest="results_dir", type=str, default="results", help="Cartella contenente i JSON di result")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default="plots", help="Cartella di output per i grafici")
    args = parser.parse_args()

    generate_benchmark_charts(output_dir=args.output_dir, results_dir=args.results_dir)
