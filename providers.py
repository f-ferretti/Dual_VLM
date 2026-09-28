import os
import json
import re
import time
import logging
import base64
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import requests
from dotenv import load_dotenv

from video_utils import extract_base64_frames, compress_video_for_upload

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path, override=True)
logger = logging.getLogger(__name__)


def clean_json_response(raw_text: Any) -> Any:
    """
    Rimuove blocchi markdown e sanitizza la stringa per ottenere un dizionario o lista JSON valida.
    """
    if raw_text is None:
        raise ValueError("Impossibile effettuare il parsing del JSON: il testo della risposta e None.")
    text = str(raw_text).strip()
    if not text:
        raise ValueError("Impossibile effettuare il parsing del JSON: risposta vuota dal modello.")
    
    # Rimuove blocchi di codice ```json ... ``` o ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Ricerca delimitatori JSON esterni
    first_brace = text.find("{")
    first_bracket = text.find("[")
    
    start_idx = -1
    end_char = None
    if first_brace != -1 and first_bracket != -1:
        if first_brace < first_bracket:
            start_idx = first_brace
            end_char = "}"
        else:
            start_idx = first_bracket
            end_char = "]"
    elif first_brace != -1:
        start_idx = first_brace
        end_char = "}"
    elif first_bracket != -1:
        start_idx = first_bracket
        end_char = "]"

    if start_idx != -1 and end_char is not None:
        end_idx = text.rfind(end_char)
        if end_idx != -1 and end_idx > start_idx:
            sub = text[start_idx:end_idx + 1]
            try:
                return json.loads(sub)
            except json.JSONDecodeError:
                # Rimuove eventuali trailing commas come `",\n}"` o `",\n]"`
                sub_cleaned = re.sub(r',\s*([}\]])', r'\1', sub)
                try:
                    return json.loads(sub_cleaned)
                except json.JSONDecodeError:
                    pass

    raise ValueError(f"Impossibile effettuare il parsing del JSON dalla risposta del modello: {raw_text[:200]}...")


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_with_video(self, video_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        pass


class GoogleGeminiProvider(BaseLLMProvider):
    def __init__(
        self,
        model_name: str = "gemini-3.8-flash",
        api_key: Optional[str] = None,
        enable_agentic_video: bool = True
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.enable_agentic_video = enable_agentic_video

    def generate_with_video(self, video_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY non trovata. Inseriscila nel file .env o nell'interfaccia.")

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        is_youtube = ("youtube.com" in video_path.lower() or "youtu.be" in video_path.lower())

        if is_youtube:
            logger.info(f"[Google Gemini] Analisi diretta YouTube URL: {video_path}...")
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.2
            )

            video_part = types.Part.from_uri(file_uri=video_path, mime_type="video/youtube")

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=[video_part, user_prompt],
                        config=config
                    )
                    raw_text = response.text
                    parsed_json = clean_json_response(raw_text)
                    return {
                        "parsed_json": parsed_json,
                        "raw_text": raw_text
                    }
                except Exception as e:
                    err_str = str(e)
                    transient_keywords = ["429", "RESOURCE_EXHAUSTED", "Quota", "503", "UNAVAILABLE", "500", "502", "504", "overloaded", "Service Unavailable"]
                    is_parsing_error = isinstance(e, ValueError) and ("parsing del JSON" in err_str or "risposta vuota" in err_str)
                    if (any(k in err_str for k in transient_keywords) or is_parsing_error) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 8 if not is_parsing_error else 3
                        logger.warning(f"[Google Gemini] Errore temporaneo o output non valido ({err_str[:70]}...). Nuovo tentativo tra {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    raise e

        import tempfile
        import shutil

        # Assicura che il path e il nome del file passati all'SDK contengano solo caratteri ASCII sicuri
        safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', os.path.basename(video_path))
        if not safe_name:
            safe_name = "gameplay_video.mp4"

        temp_ascii_copy = None
        upload_path = video_path

        try:
            video_path.encode("ascii")
        except UnicodeEncodeError:
            # Copia temporanea del file per percorsi non-ASCII
            ext = os.path.splitext(video_path)[1] or ".mp4"
            temp_ascii_dir = os.path.join(tempfile.gettempdir(), "ascii_video_safe")
            os.makedirs(temp_ascii_dir, exist_ok=True)
            temp_ascii_copy = os.path.join(temp_ascii_dir, f"safe_video_{int(time.time())}_{safe_name}")
            shutil.copy2(video_path, temp_ascii_copy)
            upload_path = temp_ascii_copy

        logger.info(f"[Google Gemini] Caricamento video: {upload_path}...")
        
        try:
            upload_config = types.UploadFileConfig(display_name=safe_name)
            uploaded_file = client.files.upload(file=upload_path, config=upload_config)
        except Exception:
            uploaded_file = client.files.upload(file=upload_path)
        finally:
            if temp_ascii_copy and os.path.exists(temp_ascii_copy):
                try:
                    os.remove(temp_ascii_copy)
                except Exception:
                    pass

        max_wait_seconds = 180
        start_time = time.time()
        while uploaded_file.state.name == "PROCESSING":
            if time.time() - start_time > max_wait_seconds:
                raise TimeoutError("Timeout durante l'elaborazione del video su Google Gemini Files API.")
            time.sleep(3)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise RuntimeError("L'elaborazione del video su Google Gemini e fallita.")

        logger.info(f"[Google Gemini] Invio richiesta al modello {self.model_name}...")
        
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=0.2
        )

        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=[uploaded_file, user_prompt],
                        config=config
                    )
                    try:
                        raw_text = response.text if (response and hasattr(response, "text") and response.text is not None) else ""
                    except Exception:
                        raw_text = ""

                    if not raw_text:
                        candidates = getattr(response, "candidates", None)
                        if candidates and len(candidates) > 0:
                            cand = candidates[0]
                            content_obj = getattr(cand, "content", None)
                            parts = getattr(content_obj, "parts", None) if content_obj else None
                            if parts:
                                for p in parts:
                                    part_text = getattr(p, "text", None)
                                    if part_text:
                                        raw_text += part_text

                    if not raw_text:
                        raise RuntimeError("Risposta priva di testo restituita da Google Gemini.")

                    parsed_json = clean_json_response(raw_text)
                    return {
                        "parsed_json": parsed_json,
                        "raw_text": raw_text
                    }
                except Exception as e:
                    err_str = str(e)
                    transient_keywords = ["429", "RESOURCE_EXHAUSTED", "Quota", "503", "UNAVAILABLE", "500", "502", "504", "overloaded", "Service Unavailable"]
                    is_parsing_error = isinstance(e, ValueError) and ("parsing del JSON" in err_str or "risposta vuota" in err_str)
                    if (any(k in err_str for k in transient_keywords) or is_parsing_error) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 8 if not is_parsing_error else 3
                        logger.warning(f"[Google Gemini] Errore temporaneo o output non valido ({err_str[:70]}...). Nuovo tentativo tra {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    raise e
        finally:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as e:
                logger.warning(f"Impossibile eliminare il file temporaneo su Google Gemini: {e}")


class OpenRouterLLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "qwen/qwen3.7-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_with_video(self, video_path: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY non trovata. Inseriscila nel file .env o nell'interfaccia.")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video non trovato: {video_path}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://tesi-gameplay-bug-detector.local",
            "X-Title": "Gameplay Bug Detector"
        }

        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        logger.info(f"[OpenRouter] Codifica Base64 del video ({size_mb:.2f} MB) per invio diretto a /chat/completions...")

        with open(video_path, "rb") as vf:
            video_bytes = vf.read()
            video_b64 = base64.b64encode(video_bytes).decode("utf-8")

        ext = os.path.splitext(video_path)[1].lower().replace(".", "")
        mime_type = f"video/{ext}" if ext in ["mp4", "webm", "mov", "avi", "mkv"] else "video/mp4"
        if ext == "mov":
            mime_type = "video/quicktime"

        content_elements = [
            {"type": "text", "text": user_prompt},
            {
                "type": "video_url",
                "video_url": {
                    "url": f"data:{mime_type};base64,{video_b64}"
                }
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_elements}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"[OpenRouter] Invio richiesta di analisi al modello {self.model_name} (tentativo {attempt + 1}/{max_retries})...")
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=180)

                # Riprova senza response_format se non supportato dal modello
                if response.status_code == 400 and "response_format" in payload and "data_inspection_failed" not in response.text:
                    logger.warning(f"[OpenRouter] Errore 400 con response_format, riprovo senza response_format...")
                    payload_no_rf = payload.copy()
                    del payload_no_rf["response_format"]
                    response = requests.post(self.base_url, headers=headers, json=payload_no_rf, timeout=180)

                # Gestione rate limit e saturazione upstream
                is_rate_limited = (
                    response.status_code in [429, 502, 503, 504, 529]
                    or "rate-limited" in response.text.lower()
                    or "temporarily rate-limited" in response.text.lower()
                )
                if is_rate_limited:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 12
                        logger.warning(f"[OpenRouter] Rate limit o server sovraccarico ({response.status_code}). Nuovo tentativo tra {wait_time}s...")
                        time.sleep(wait_time)
                        continue

                # Gestione rifiuto moderazione contenuti
                if response.status_code == 400 and "data_inspection_failed" in response.text:
                    if attempt < max_retries - 1:
                        wait_time = 5
                        logger.warning(f"[OpenRouter] Ispezione video fallita (data_inspection_failed). Tentativo di riprova tra {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RuntimeError(f"OpenRouter Content Filter Rejected (400 data_inspection_failed): {response.text}")

                if response.status_code != 200:
                    raise RuntimeError(f"Errore chiamata OpenRouter ({response.status_code}): {response.text}")

                res_json = response.json()
                if "error" in res_json:
                    err_msg = res_json["error"].get("message") if isinstance(res_json["error"], dict) else str(res_json["error"])
                    if any(k in err_msg.lower() for k in ["rate", "limit", "temporarily", "busy"]) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 12
                        logger.warning(f"[OpenRouter] Segnalazione rate limit da provider: {err_msg}. Nuovo tentativo tra {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    raise RuntimeError(f"Errore restituito da OpenRouter: {err_msg}")

                if "choices" not in res_json or not res_json["choices"]:
                    raise RuntimeError(f"Risposta inattesa da OpenRouter: {json.dumps(res_json, ensure_ascii=False)}")

                choice = res_json["choices"][0]
                message = choice.get("message", {})
                raw_text = message.get("content")

                if not raw_text:
                    if message.get("reasoning"):
                        raw_text = message.get("reasoning")
                    elif message.get("reasoning_content"):
                        raw_text = message.get("reasoning_content")
                    elif message.get("refusal"):
                        raise RuntimeError(f"Il modello ha rifiutato la richiesta (refusal): {message.get('refusal')}")
                    else:
                        finish_reason = choice.get("finish_reason")
                        raise RuntimeError(
                            f"Risposta priva di contenuto dal modello OpenRouter. finish_reason: {finish_reason}. Payload: {json.dumps(res_json, ensure_ascii=False)}"
                        )

                parsed_json = clean_json_response(raw_text)

                return {
                    "parsed_json": parsed_json,
                    "raw_text": raw_text
                }

            except ValueError as ve:
                err_str = str(ve)
                if ("parsing del JSON" in err_str or "risposta vuota" in err_str) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    logger.warning(f"[OpenRouter] Errore parsing JSON ({err_str[:70]}...). Nuovo tentativo tra {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise ve
            except requests.exceptions.RequestException as re_err:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 8
                    logger.warning(f"[OpenRouter] Errore di rete ({re_err}). Nuovo tentativo tra {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise re_err


def get_provider(
    provider_type: str,
    model_name: str,
    api_key: Optional[str] = None,
    enable_agentic_video: bool = True
) -> BaseLLMProvider:
    provider_type = provider_type.lower().strip()
    if provider_type in ["google", "gemini", "google_ai_studio"]:
        return GoogleGeminiProvider(
            model_name=model_name,
            api_key=api_key,
            enable_agentic_video=enable_agentic_video
        )
    elif provider_type in ["openrouter", "open_router"]:
        return OpenRouterLLMProvider(model_name=model_name, api_key=api_key)
    else:
        raise ValueError(f"Provider sconosciuto: {provider_type}. Scegli tra 'google' o 'openrouter'.")
