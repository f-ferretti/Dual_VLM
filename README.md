# Dual VLM: Automated Gameplay Bug Detection Pipeline

A two-stage multimodal agentic architecture designed for automated video game bug and anomaly detection directly from native gameplay video footage using Vision-Language Models (VLMs).

## Architectural Overview

The system addresses the game test oracle problem by decoupling anomaly localization from formal verification through a sequential, coarse-to-fine multi-agent workflow:

1. Stage 1 - Detector Agent (Screening)
The Detector Agent scans the entire continuous gameplay video (or YouTube stream) to identify candidate temporal intervals (timestamp_start, timestamp_end) where anomalous behaviors, physics failures, visual glitches, or unexpected events are suspected.

2. Stage 2 - Validator Agent (Fine Confirmation)
For each candidate segment detected in Stage 1, a targeted video clip is extracted and submitted to the Validator Agent. The Validator independently inspects the localized footage, rules out false positives (such as intended game mechanics, stylized rendering, or video compression artifacts), assesses root-cause dynamics, and categorizes confirmed bugs according to a formal QA taxonomy.

The final output is a structured JSON and Markdown report validated against strict Pydantic schemas.

## QA Bug Taxonomy

The pipeline classifies confirmed anomalies into formal software engineering and game QA categories:

- physics_collision: Collision detection failures, mesh clipping, falling through terrain, unnatural gravity, or excessive ragdoll impulse forces.
- rendering_graphics: Missing textures, flickering, Z-fighting, corrupted shadows, LOD popping, or floating static geometry.
- animation_glitch: T-poses, frozen skeletal meshes, broken inverse kinematics (IK) transitions, mesh stretching, or socket misalignments.
- ui_hud: Overlapping interface elements, frozen status bars, broken menus, or misplaced HUD indicators.
- logic_ai: Progression blockers (soft-locks, broken mission triggers) and NPC behavioral anomalies (pathfinding navigation loops, unresponsive enemy AI).
- audio_glitch: Sound desynchronization, corrupted loops, popping, or missing sound cues.
- performance_issue: Severe frame drops, micro-stuttering, or asset streaming hitching during active gameplay.
- crash_stability: Process termination (Crash to Desktop), engine crash reporter dialogs (Unreal Engine / Unity), permanent game window deadlocks, or out-of-memory errors.
- other: Real defects not covered by the categories above.

## Repository Structure

- app.py: Interactive Streamlit web interface for single-video analysis, real-time stage execution inspection, clip playback, and report export.
- batch_runner.py: Automated CLI runner for batch execution across video directories or benchmark datasets.
- evaluate_results.py: Statistical analysis and benchmark module that aggregates execution reports, computes classification metrics (Precision, Recall, F1, F2, Accuracy), and produces publication-quality comparison charts.
- detector_agent.py: Implementation of the Stage 1 Detector Agent for coarse temporal screening.
- validator_agent.py: Implementation of the Stage 2 Validator Agent for fine clip confirmation and taxonomy assignment.
- pipeline.py: Core Dual-VLM orchestrator coordinating Stage 1, video clipping, Stage 2, and report compilation.
- providers.py: Model provider abstractions supporting Google AI Studio (Gemini) and OpenRouter (Alibaba DashScope / Qwen).
- schemas.py: Formal Pydantic schemas for structured outputs, confidence scoring, bug classification, and batch metrics.
- video_utils.py: Utilities for video metadata extraction, temporal clipping via FFmpeg, and automated video compression for API payload limits.
- requirements.txt: Python dependency specifications.
- .env.example: Template for API credentials and runtime configurations.

## System Prerequisites

1. Python 3.10 or higher.
2. FFmpeg: FFmpeg must be installed and accessible in the system PATH for video clipping and compression routines.
   - On Windows: Install via winget (winget install Gyan.FFmpeg) or Chocolatey (choco install ffmpeg), and ensure ffmpeg.exe is in your PATH.
   - On Linux / macOS: Install via package manager (sudo apt install ffmpeg or brew install ffmpeg).

## Installation

1. Clone or open the repository:
   cd Dual_VLM

2. Create and activate a Python virtual environment:
   python -m venv .venv
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate

3. Install required dependencies:
   pip install -r requirements.txt

## Environment Configuration

Copy the example environment configuration file to create your local .env:

cp .env.example .env

Edit .env to supply your API credentials:

# Google AI Studio API Key (for Gemini models)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenRouter API Key (for Qwen and third-party models)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Default OpenRouter model name
OPENROUTER_MODEL=qwen/qwen-2.5-vl-72b-instruct:free

Note: The system strictly prohibits automatic fallback models. A valid model must be explicitly declared either in the .env configuration or supplied via command-line arguments.

## Usage

### 1. Interactive Web Application (Streamlit)

Launch the web interface:

streamlit run app.py

Through the application, you can:
- Select the VLM provider and model for both Detector and Validator.
- Upload a local video file (MP4, MOV, WEBM, MKV, AVI) or enter a YouTube streaming link (supported for Google Gemini).
- Execute the pipeline and monitor coarse detection intervals.
- Inspect the generated clips, validation rationale, and confirmed bug details.
- Download the generated report in JSON or Markdown format.
- Access the Benchmark and Charts tab to visualize aggregate performance metrics directly within the UI.

### 2. Batch Processing via Command Line

Run the automated batch runner over a directory of video files:

# Example using Google Gemini models:
python batch_runner.py --video-dir "D:/Datasets/Videos" --output-dir "results" --det-provider google --det-model gemini-3.8-flash --val-provider google --val-model gemini-3.8-flash

# Example using OpenRouter models:
python batch_runner.py --video-dir "D:/Datasets/Videos" --output-dir "results" --det-provider openrouter --det-model "qwen/qwen-2.5-vl-72b-instruct:free" --val-provider openrouter --val-model "qwen/qwen-2.5-vl-72b-instruct:free"

Available options for batch_runner.py:
- --video-dir: Path to directory containing video files (required).
- --output-dir: Output directory for JSON and Markdown reports (default: results).
- --det-provider: Provider for Stage 1 detector (google or openrouter, default: google).
- --det-model: Model name for Stage 1 detector.
- --val-provider: Provider for Stage 2 validator (google or openrouter, default: google).
- --val-model: Model name for Stage 2 validator.
- --recursive: Recursively search for video files in subdirectories.
- --limit: Maximum number of videos to process in this run.
- --resume: Skip videos that already have an existing report in the output directory.
- --disable-agentic-video: Disable automatic video fragmentation for long inputs.

### 3. Statistical Evaluation and Benchmark Generation

Process test reports and generate high-resolution comparison charts, confusion matrices, and tabular summaries:

python evaluate_results.py --results-dir results --output-dir plots

This script parses all JSON reports in the results directory, computes formal Game QA metrics (Precision, Recall, F1-Score, F2-Score, Accuracy), and outputs:
- Confusion matrix breakdown (TP, FP, FN, TN).
- Model performance comparisons across single models and the Dual-VLM pipeline.
- Execution time distributions.
- Bug taxonomy distribution charts.
- Summary tables in CSV format and an analytical Markdown report (benchmark_report.md).

## Output Structure

When tests run, reports and plots are written to the designated output directories:
- results/report__[video_name]__[timestamp].json: Structured machine-readable report containing video metadata, detector findings, and validator verdicts.
- results/report__[video_name]__[timestamp].md: Human-readable QA summary report.
- results/batch_summary__[timestamp].json: Aggregated metrics across all processed videos in a batch session.
- results/batch_summary__[timestamp].csv: Tabular summary of all processed videos, timings, and detected anomalies.
- plots/: High-resolution scientific visualization charts and benchmark summaries produced by evaluate_results.py.
