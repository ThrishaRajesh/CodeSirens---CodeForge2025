from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
from PIL import Image
import io
import json
from datetime import datetime
import os
import tempfile
import math
import re
import unicodedata
import shutil
import threading

app = FastAPI()

# Global variables for models
fire_model = None
disaster_model = None
general_model = None
asr_model = None  # Whisper / Distil-Whisper

# Storage for disaster reports (in production, use a proper database)
disaster_reports = []


# ------------------------------
# Model loading
# ------------------------------
def load_models():
    """Load all models (runs in background on startup)."""
    global fire_model, disaster_model, general_model, asr_model

    print("Loading models...")

    try:
        print("Loading fire detection model...")
        fire_model = pipeline(model="prithivMLmods/Forest-Fire-Detection", task="image-classification")
        print("✓ Fire detection model loaded")
    except Exception as e:
        print(f"Warning: Could not load fire model: {e}")

    try:
        print("Loading general vision model (ViT)...")
        disaster_model = pipeline(model="google/vit-base-patch16-224", task="image-classification")
        print("✓ ViT model loaded")
    except Exception as e:
        print(f"Warning: Could not load disaster model: {e}")

    try:
        print("Loading general classification model (ResNet-50)...")
        general_model = pipeline(model="microsoft/resnet-50", task="image-classification")
        print("✓ ResNet-50 model loaded")
    except Exception as e:
        print(f"Warning: Could not load general model: {e}")

    # Whisper (or Distil-Whisper) for ASR
    try:
        print("Loading Whisper ASR...")
        # You can swap to a lighter model if needed:
        # asr_model = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")
        asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-small")
        print("✓ ASR loaded")
        if not shutil.which("ffmpeg"):
            print("WARNING: ffmpeg not found on PATH. Audio decoding will fail. Install FFmpeg and restart.")
    except Exception as e:
        print(f"Warning: Could not load ASR model: {e}")

    print("Model loading finished.")


@app.on_event("startup")
def startup_event():
    # Load models in a background thread so the server can start immediately
    threading.Thread(target=load_models, daemon=True).start()


# ------------------------------
# Utilities
# ------------------------------
def save_reports_to_file():
    """Save disaster reports to JSON file"""
    try:
        with open("disaster_reports.json", "w") as f:
            json.dump(disaster_reports, f, indent=2)
        print(f"Saved {len(disaster_reports)} disaster reports")
    except Exception as e:
        print(f"Error saving reports: {e}")


def _normalize_text(t: str) -> str:
    # lower, strip accents, collapse spaces
    t = t.lower()
    t = ''.join(c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c))
    t = re.sub(r'\s+', ' ', t)
    return t


def classify_from_transcript(text: str):
    """
    Multilingual-ish, fuzzy keyword scorer for disaster labels.
    Returns dict: {"label": "...", "score": 0..1, "tags": [..]} or None
    """
    if not text:
        return None

    t = _normalize_text(text)

    # English + Hinglish/Hindi keyword buckets (extend as needed)
    buckets = {
        "fire": [
            "fire", "flame", "burn", "burning", "wildfire", "smoulder", "smoke",
            "aag", "aag lag", "ag lag", "jal", "jal rha", "jal raha", "jal rahi",
            "dhua", "dhuaan", "dhuan"
        ],
        "flood": [
            "flood", "flooding", "water rising", "river overflow", "inundat",
            "pani", "paani", "pani bhar", "paani bhar", "baadh", "baarh", "barish", "nadi", "talab", "dub", "doob", "dooba"
        ],
        "earthquake/building collapse": [
            "earthquake", "tremor", "aftershock", "collapsed", "collapse", "rubble", "debris", "crack",
            "bhukamp", "bukamp", "kamp", "kampan", "deewar gir", "diwar gir", "ghar gir", "tuta", "tuti", "gir gaya"
        ],
        "landslide": [
            "landslide", "mudslide", "mud flow", "slope failure", "rockfall",
            "bhuskhlaan", "bhuskhulan", "pahad khisak", "mitti khisak", "kachra pahad se"
        ],
        "cyclone/storm": [
            "cyclone", "hurricane", "typhoon", "storm", "tornado", "gale", "strong wind",
            "toofan", "andhi", "aandhi", "tez hawa", "hawa tez", "baarish ke saath tez hawa"
        ],
        "tsunami": [
            "tsunami", "tidal wave", "big wave", "samundar ki lehr", "samundar se pani"
        ],
        "explosion": [
            "explosion", "blast", "bomb", "blast wave",
            "visphot", "dhamaka", "phatna", "phat gaya"
        ],
    }

    def score_for_keywords(keywords):
        score = 0.0
        hits = []
        for kw in keywords:
            if kw in t:
                freq = t.count(kw)
                score += 0.45 + 0.15 * math.log2(1 + freq)  # base + log boost
                hits.append(kw)
        return score, hits

    best = None
    best_hits = []

    for label, kws in buckets.items():
        s, hits = score_for_keywords(kws)
        if s > 0 and (best is None or s > best[1]):
            best = (label, s)
            best_hits = hits

    if not best:
        return None

    label, raw = best
    norm = min(0.2 + raw, 0.95)
    return {"label": label, "score": norm, "tags": sorted(set(best_hits))}


def analyze_disaster_type(predictions, voice_vote=None):
    """Analyze image predictions + optional voice to determine disaster type"""
    disaster_summary = {
        "primary_disaster": "unknown",
        "confidence": 0.0,
        "detected_features": [],
        "risk_level": "low"
    }

    high_confidence_threshold = 0.6

    def image_inference():
        earthquake_indicators = ["rubble", "debris", "wreck", "destroyed", "collapsed", "ruin", "demolish", "damage", "broken", "shatter", "crash"]
        building_indicators = ["building", "house", "structure", "wall", "concrete", "brick", "construction"]
        water_indicators = ["water", "flood", "river", "lake", "ocean", "sea", "wet", "liquid"]

        earthquake_score = 0.0
        water_score = 0.0
        building_damage_detected = False

        # Fire (priority)
        if "fire_detection" in predictions and isinstance(predictions["fire_detection"], list) and not any("error" in p for p in predictions["fire_detection"]):
            for pred in predictions["fire_detection"]:
                if pred.get("score", 0) > high_confidence_threshold:
                    if "fire" in pred["label"].lower():
                        return {
                            "primary_disaster": "fire",
                            "confidence": pred["score"],
                            "detected_features": [f"Fire detected ({pred['score']:.1%})"],
                            "risk_level": "high"
                        }
                    elif "smoke" in pred["label"].lower():
                        return {
                            "primary_disaster": "smoke/potential fire",
                            "confidence": pred["score"],
                            "detected_features": [f"Smoke detected ({pred['score']:.1%})"],
                            "risk_level": "medium"
                        }

        # Vision model indicators
        if "vision_detection" in predictions and isinstance(predictions["vision_detection"], list) and not any("error" in p for p in predictions["vision_detection"]):
            for pred in predictions["vision_detection"]:
                label_lower = pred["label"].lower()
                score = pred["score"]
                if any(ind in label_lower for ind in earthquake_indicators):
                    earthquake_score += score * 2
                    disaster_summary["detected_features"].append(f"Destruction detected: {pred['label']} ({score:.1%})")
                if any(ind in label_lower for ind in building_indicators):
                    building_damage_detected = True
                    disaster_summary["detected_features"].append(f"Building-related: {pred['label']} ({score:.1%})")
                if any(ind in label_lower for ind in water_indicators):
                    water_score += score

        # General classifier
        if "general_classification" in predictions and isinstance(predictions["general_classification"], list) and not any("error" in p for p in predictions["general_classification"]):
            for pred in predictions["general_classification"]:
                label_lower = pred["label"].lower()
                score = pred["score"]
                if any(ind in label_lower for ind in ["rubble", "debris", "collapsed", "wreck"]):
                    earthquake_score += score
                    disaster_summary["detected_features"].append(f"Damage indicator: {pred['label']} ({score:.1%})")

        if earthquake_score > 0.4 or (earthquake_score > 0.2 and building_damage_detected):
            return {
                "primary_disaster": "earthquake/building collapse",
                "confidence": min(earthquake_score, 0.95),
                "detected_features": disaster_summary["detected_features"],
                "risk_level": "high" if earthquake_score > 0.7 else "medium",
            }
        elif water_score > 0.5 and earthquake_score < 0.3:
            disaster_summary["detected_features"].append(f"Water/flooding detected ({water_score:.1%})")
            return {
                "primary_disaster": "flood",
                "confidence": water_score,
                "detected_features": disaster_summary["detected_features"],
                "risk_level": "high" if water_score > 0.7 else "medium",
            }
        elif building_damage_detected:
            return {
                "primary_disaster": "structural damage",
                "confidence": 0.6,
                "detected_features": disaster_summary["detected_features"],
                "risk_level": "medium",
            }

        return None

    img_result = image_inference()

    # Voice-only
    if voice_vote and not img_result:
        disaster_summary.update({
            "primary_disaster": voice_vote["label"],
            "confidence": voice_vote["score"],
            "risk_level": "high" if voice_vote["score"] > 0.75 else ("medium" if voice_vote["score"] > 0.45 else "low"),
        })
        disaster_summary["detected_features"].append(f"Voice indicates {voice_vote['label']} ({voice_vote['score']:.1%})")
        disaster_summary["voice_vote"] = voice_vote
        return disaster_summary

    # Image-only
    if img_result and not voice_vote:
        img_result["voice_vote"] = None
        return img_result

    # Fuse image + voice
    if img_result and voice_vote:
        aligned = voice_vote["label"] == img_result["primary_disaster"]
        if aligned:
            fused_conf = min(0.98, max(img_result["confidence"], voice_vote["score"]) + 0.1)
            img_result["confidence"] = fused_conf
            img_result["risk_level"] = "high" if fused_conf > 0.7 else ("medium" if fused_conf > 0.45 else "low")
            img_result.setdefault("detected_features", []).append(f"Voice corroborates {voice_vote['label']} ({voice_vote['score']:.1%})")
            img_result["voice_vote"] = voice_vote
            return img_result
        else:
            stronger = img_result if img_result["confidence"] >= voice_vote["score"] else {
                "primary_disaster": voice_vote["label"],
                "confidence": voice_vote["score"],
                "detected_features": img_result.get("detected_features", []),
                "risk_level": "high" if voice_vote["score"] > 0.7 else "medium",
            }
            stronger.setdefault("detected_features", []).append(
                f"Voice disagrees with image ({voice_vote['label']} vs {img_result.get('primary_disaster')})"
            )
            stronger["voice_vote"] = voice_vote
            return stronger

    # Neither produced anything strong
    return {
        "primary_disaster": "no clear disaster detected",
        "confidence": 0.1,
        "detected_features": disaster_summary["detected_features"],
        "risk_level": "low",
        "voice_vote": voice_vote
    }


# ------------------------------
# API endpoints
# ------------------------------
@app.post("/predict/")
async def predict(
    file: UploadFile = File(None),              # optional image
    audio: UploadFile = File(None),             # optional voice
    voice_transcript: str = Form(None),         # optional manual transcript
    latitude: float = Form(None),
    longitude: float = Form(None),
    address: str = Form(None)
):
    try:
        # Early guard: if user sent media but models not ready yet
        if (file is not None or audio is not None) and not any([fire_model, disaster_model, general_model, asr_model]):
            return JSONResponse(content={"error": "Models are still loading. Please try again in a moment."})

        img = None
        if file:
            img_bytes = await file.read()
            img = Image.open(io.BytesIO(img_bytes))
            print(f"Processing image - Mode: {img.mode}, Size: {img.size}")
            if img.mode != "RGB":
                img = img.convert("RGB")

        if latitude and longitude:
            print(f"Location: {latitude}, {longitude}")
            if address:
                print(f"Address: {address}")

        all_predictions = {}

        # ---------- Image models ----------
        if img is not None:
            if fire_model:
                try:
                    fire_results = fire_model(img)
                    fire_results = sorted(fire_results, key=lambda x: x['score'], reverse=True)
                    all_predictions["fire_detection"] = [{"label": r["label"], "score": float(r["score"])} for r in fire_results]
                    print("Fire detection completed")
                except Exception as e:
                    print(f"Fire detection error: {e}")
                    all_predictions["fire_detection"] = [{"error": str(e)}]

            if disaster_model:
                try:
                    disaster_results = disaster_model(img, top_k=15)
                    disaster_results = sorted(disaster_results, key=lambda x: x['score'], reverse=True)
                    all_predictions["vision_detection"] = [{"label": r["label"], "score": float(r["score"])} for r in disaster_results]
                    print("Vision-based detection completed")
                except Exception as e:
                    print(f"Vision detection error: {e}")
                    all_predictions["vision_detection"] = [{"error": str(e)}]

            if general_model:
                try:
                    general_results = general_model(img, top_k=10)
                    all_predictions["general_classification"] = [{"label": r["label"], "score": float(r["score"])} for r in general_results]
                    print("General classification completed")
                except Exception as e:
                    print(f"General classification error: {e}")
                    all_predictions["general_classification"] = [{"error": str(e)}]

        # ---------- Voice / transcript ----------
        transcript_text = (voice_transcript or "").strip()
        voice_vote = None

        if audio:
            # Ensure ffmpeg + ASR are available
            if not shutil.which("ffmpeg"):
                return JSONResponse(content={"error": "Audio received but FFmpeg is not installed/visible in PATH on this machine."})
            if not asr_model:
                return JSONResponse(content={"error": "Audio received but ASR model is not loaded yet. Check /health or server logs."})

            # Persist to temp file for Whisper
            suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await audio.read())
                tmp_path = tmp.name
            print(f"Saved temp audio at {tmp_path}")

            try:
                result = asr_model(tmp_path, return_timestamps=False, chunk_length_s=30)
                if isinstance(result, dict) and "text" in result:
                    if result["text"].strip():
                        transcript_text = (transcript_text + " " + result["text"]).strip() if transcript_text else result["text"]
                        print("ASR transcript:", transcript_text[:120], "...")
                    else:
                        return JSONResponse(content={"error": "ASR ran but produced an empty transcript. Try speaking louder/closer or upload a WAV/MP4."})
                else:
                    return JSONResponse(content={"error": "ASR did not return a transcript. Check server logs."})
            except Exception as e:
                print(f"ASR error: {e}")
                return JSONResponse(content={"error": f"ASR error: {e}"})
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        if transcript_text:
            voice_vote = classify_from_transcript(transcript_text)

        # Fuse to final summary
        disaster_summary = analyze_disaster_type(all_predictions, voice_vote)

        # Store report
        disaster_report = {
            "timestamp": datetime.now().isoformat(),
            "disaster_type": disaster_summary["primary_disaster"],
            "confidence": disaster_summary["confidence"],
            "risk_level": disaster_summary["risk_level"],
            "detected_features": disaster_summary.get("detected_features", []),
            "voice_vote": voice_vote,
            "voice_transcript": transcript_text,
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "address": address
            },
            "filename": file.filename if file else None,
        }
        disaster_reports.append(disaster_report)
        save_reports_to_file()

        return JSONResponse(content={
            "disaster_analysis": disaster_summary,
            "detailed_predictions": all_predictions,
            "location_data": {
                "latitude": latitude,
                "longitude": longitude,
                "address": address,
                "timestamp": disaster_report["timestamp"]
            },
            "voice": {
                "transcript": transcript_text,
                "vote": voice_vote,
                "debug_tags": (voice_vote or {}).get("tags") if voice_vote else None
            }
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return JSONResponse(content={"error": str(e)})


@app.get("/reports/")
async def get_reports():
    """Get all disaster reports with locations"""
    return JSONResponse(content={"reports": disaster_reports})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = {
        "fire_model": fire_model is not None,
        "disaster_model": disaster_model is not None,
        "general_model": general_model is not None,
        "asr_model": asr_model is not None
    }
    return {
        "status": "healthy",
        "models_loaded": model_status,
        "total_reports": len(disaster_reports)
    }


# Load existing reports on startup
try:
    if os.path.exists("disaster_reports.json"):
        with open("disaster_reports.json", "r") as f:
            disaster_reports = json.load(f)
        print(f"Loaded {len(disaster_reports)} existing disaster reports")
except Exception as e:
    print(f"Could not load existing reports: {e}")

# Static files
app.mount("/", StaticFiles(directory=".", html=True), name="static")
