"""
pycaps-api v3: Microserviço FastAPI para editar e legendar vídeos.
Endpoints:
  POST /edit      - Edita vídeo (corta silêncio + zoom com IA)
  POST /caption   - Adiciona legendas estilizadas
  GET  /health    - Health check
  GET  /cleanup   - Limpa arquivos temporários
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

app = FastAPI(title="pycaps-api", version="3.1.0")

WORK_DIR = Path("/tmp/pycaps-work")
WORK_DIR.mkdir(parents=True, exist_ok=True)

CAPTION_LOCKED_LAYOUT = {
    "position": "bottom",
    "position_offset": 0.24,
    "max_width": 0.76,
    "max_lines": 2,
}

CAPTION_STYLE_PRESETS = {
    "capcut_clean": {
        "label": "CapCut Clean",
        "template": "minimalist",
        "layout": CAPTION_LOCKED_LAYOUT,
        "css": {
            "font_size": 56,
            "font_color": "white",
            "font_family": "Arial Black",
            "font_weight": 900,
            "highlight_color": "white",
            "highlight_bg": "",
            "text_transform": "none",
            "stroke_color": "black",
            "stroke_width": "3px",
        },
    },
    "minimalist": {
        "label": "Minimalist",
        "template": "minimalist",
        "layout": CAPTION_LOCKED_LAYOUT,
        "css": {
            "font_size": 52,
            "font_color": "white",
            "font_family": "Inter",
            "font_weight": 850,
            "highlight_color": "white",
            "highlight_bg": "",
            "text_transform": "none",
            "stroke_color": "black",
            "stroke_width": "3px",
        },
    },
    "karaoke_yellow": {
        "label": "Karaoke Yellow",
        "template": "minimalist",
        "layout": CAPTION_LOCKED_LAYOUT,
        "css": {
            "font_size": 56,
            "font_color": "white",
            "font_family": "Arial Black",
            "font_weight": 900,
            "highlight_color": "#facc15",
            "highlight_bg": "",
            "text_transform": "none",
            "stroke_color": "black",
            "stroke_width": "3px",
        },
    },
    "bold_quote": {
        "label": "Bold Quote",
        "template": "minimalist",
        "layout": CAPTION_LOCKED_LAYOUT,
        "css": {
            "font_size": 54,
            "font_color": "white",
            "font_family": "Montserrat",
            "font_weight": 900,
            "highlight_color": "white",
            "highlight_bg": "",
            "text_transform": "none",
            "stroke_color": "black",
            "stroke_width": "3px",
        },
    },
    "hook_yellow": {
        "label": "Hook Yellow",
        "template": "minimalist",
        "layout": CAPTION_LOCKED_LAYOUT,
        "css": {
            "font_size": 58,
            "font_color": "#facc15",
            "font_family": "Arial Black",
            "font_weight": 900,
            "highlight_color": "#facc15",
            "highlight_bg": "",
            "text_transform": "none",
            "stroke_color": "black",
            "stroke_width": "3px",
        },
    },
}


# ═══════════════════════════════════════════════════════════════════
# STARTUP & UTILS
# ═══════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_cleanup():
    _cleanup_old_jobs()


def _cleanup_old_jobs(max_age_minutes: int = 30):
    now = time.time()
    for item in WORK_DIR.iterdir():
        try:
            if (now - item.stat().st_mtime) / 60 > max_age_minutes:
                if item.is_dir():
                    shutil.rmtree(str(item), ignore_errors=True)
                else:
                    item.unlink()
        except Exception:
            pass


@app.get("/health")
async def health():
    ffmpeg_ok = os.system("ffmpeg -version > /dev/null 2>&1") == 0
    pycaps_ok = False
    try:
        import pycaps
        pycaps_ok = True
    except ImportError:
        pass
    return {
        "status": "ok" if (ffmpeg_ok and pycaps_ok) else "degraded",
        "ffmpeg": ffmpeg_ok,
        "pycaps": pycaps_ok,
        "pycaps_cli": shutil.which("pycaps") or "not found",
    }


@app.get("/cleanup")
async def manual_cleanup():
    count = 0
    for item in WORK_DIR.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(str(item), ignore_errors=True)
            else:
                item.unlink()
            count += 1
        except Exception:
            pass
    return {"cleaned": count}


@app.get("/templates")
async def list_templates():
    builtin = ["minimalist", "default"]
    custom_dir = Path("/app/templates")
    custom = [d.name for d in custom_dir.iterdir() if d.is_dir()] if custom_dir.exists() else []
    caption_styles = {
        name: {
            "label": preset["label"],
            "template": preset["template"],
            "layout": preset["layout"],
        }
        for name, preset in CAPTION_STYLE_PRESETS.items()
    }
    return {"builtin": builtin, "custom": custom, "caption_styles": caption_styles}


# ═══════════════════════════════════════════════════════════════════
# ENDPOINT /edit
# ═══════════════════════════════════════════════════════════════════

@app.post("/edit")
async def edit_video(
    video: UploadFile = File(default=None, description="Vídeo MP4"),
    video_url: str = Form(default="", description="URL do vídeo (alternativa ao upload)"),
    openai_api_key: str = Form(default="", description="API Key da OpenAI"),
    openai_model: str = Form(default="gpt-4o", description="Modelo OpenAI"),
    remove_silence: bool = Form(default=True, description="Remover silêncios"),
    silence_threshold: float = Form(default=0.4, description="Duração mín de silêncio pra cortar (s)"),
    silence_db: str = Form(default="-30dB", description="Limiar de volume pra silêncio"),
    add_zooms: bool = Form(default=True, description="Adicionar zooms dinâmicos"),
    zoom_intensity: float = Form(default=1.15, description="Intensidade do zoom"),
    custom_prompt: str = Form(default="", description="Instrução extra pra IA editora"),
    speed_factor: float = Form(default=1.0, description="Velocidade do vídeo. 1.0 = normal"),
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / f"edit_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.mp4"
    output_path = job_dir / "edited.mp4"
    start_time = time.time()

    try:
        await _save_input(video, video_url, input_path, f"EDIT-{job_id}")
        openai_api_key = _resolve_openai_api_key(openai_api_key)

        speed_factor = max(0.5, min(2.0, speed_factor))
        if speed_factor != 1.0:
            sped_path = job_dir / "sped.mp4"
            _apply_speed(str(input_path), str(sped_path), speed_factor, job_id)
            if sped_path.exists():
                input_path = sped_path
                print(f"[EDIT-{job_id}] Velocidade {speed_factor}x aplicada")
            else:
                print(f"[EDIT-{job_id}] Velocidade falhou, usando original")

        video_info = _get_video_info(str(input_path))
        duration = video_info["duration"]
        print(f"[EDIT-{job_id}] Vídeo: {duration:.1f}s, {video_info['width']}x{video_info['height']}")

        silences = []
        if remove_silence:
            silences = _detect_silences(str(input_path), silence_db, silence_threshold)
            print(f"[EDIT-{job_id}] {len(silences)} silêncios detectados")

        transcript = _whisper_transcribe(str(input_path), job_id)
        print(f"[EDIT-{job_id}] Transcrição: {len(transcript.get('segments', []))} segmentos")

        edit_plan = _ai_edit_plan(
            transcript=transcript, silences=silences, duration=duration,
            zoom_intensity=zoom_intensity, openai_key=openai_api_key,
            openai_model=openai_model, add_zooms=add_zooms,
            custom_prompt=custom_prompt, job_id=job_id,
        )
        (job_dir / "edit_plan.json").write_text(json.dumps(edit_plan, indent=2, ensure_ascii=False))
        cuts = edit_plan.get("cuts", [])
        zooms = edit_plan.get("zooms", [])
        print(f"[EDIT-{job_id}] Plano: {len(cuts)} cortes, {len(zooms)} zooms")

        _apply_edits(str(input_path), str(output_path), cuts, zooms, video_info, job_id)

        if not output_path.exists():
            raise HTTPException(status_code=500, detail="FFmpeg não gerou vídeo editado")

        elapsed = round(time.time() - start_time, 2)
        print(f"[EDIT-{job_id}] Concluído em {elapsed}s → {output_path.stat().st_size/1024/1024:.1f} MB")

        return FileResponse(
            path=str(output_path), media_type="video/mp4",
            filename=f"editado_{video.filename if video and video.filename else 'video.mp4'}",
            headers={
                "X-Edit-Duration": str(elapsed),
                "X-Edit-Cuts": str(len(cuts)),
                "X-Edit-Zooms": str(len(zooms)),
                "X-Edit-Speed": str(speed_factor),
            },
            background=_cleanup_bg(job_dir),
        )
    except HTTPException:
        _cleanup_dir(job_dir); raise
    except Exception as e:
        _cleanup_dir(job_dir); traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════
# ENDPOINT /caption
# ═══════════════════════════════════════════════════════════════════

@app.post("/caption")
async def caption_video(
    video: UploadFile = File(default=None, description="Vídeo MP4"),
    video_url: str = Form(default="", description="URL do vídeo"),
    template: str = Form(default="minimalist"),
    caption_style: str = Form(default="capcut_clean"),
    language: str = Form(default="pt"),
    whisper_model: str = Form(default="small"),
    position: str = Form(default="center"),
    position_offset: float = Form(default=0.2),
    max_width: float = Form(default=0.85),
    max_lines: int = Form(default=2),
    font_size: int = Form(default=24),
    font_color: str = Form(default="white"),
    font_family: str = Form(default="system-ui"),
    font_weight: int = Form(default=800),
    highlight_color: str = Form(default="white"),
    highlight_bg: str = Form(default="#22c55e"),
    text_transform: str = Form(default="uppercase"),
    stroke_color: str = Form(default="black"),
    stroke_width: str = Form(default="2px"),
    custom_css: str = Form(default=""),
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / f"cap_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.mp4"
    output_path = job_dir / "output.mp4"
    css_path = job_dir / "style.css"
    config_path = job_dir / "pycaps.template.json"
    start_time = time.time()

    try:
        await _save_input(video, video_url, input_path, f"CAP-{job_id}")

        preset = _resolve_caption_style(caption_style, template)
        if preset:
            template = preset["template"]
            layout_preset = preset["layout"]
            position = layout_preset.get("position", position)
            position_offset = layout_preset.get("position_offset", position_offset)
            max_width = layout_preset.get("max_width", max_width)
            max_lines = layout_preset.get("max_lines", max_lines)
            css_preset = preset["css"]
            font_size = css_preset.get("font_size", font_size)
            font_color = css_preset.get("font_color", font_color)
            font_family = css_preset.get("font_family", font_family)
            font_weight = css_preset.get("font_weight", font_weight)
            highlight_color = css_preset.get("highlight_color", highlight_color)
            highlight_bg = css_preset.get("highlight_bg", highlight_bg)
            text_transform = css_preset.get("text_transform", text_transform)
            stroke_color = css_preset.get("stroke_color", stroke_color)
            stroke_width = css_preset.get("stroke_width", stroke_width)

        position = CAPTION_LOCKED_LAYOUT["position"]
        position_offset = CAPTION_LOCKED_LAYOUT["position_offset"]
        max_width = CAPTION_LOCKED_LAYOUT["max_width"]
        max_lines = CAPTION_LOCKED_LAYOUT["max_lines"]

        css_content = custom_css if custom_css.strip() else _build_css(
            font_size=font_size, font_color=font_color, font_family=font_family,
            font_weight=font_weight, highlight_color=highlight_color,
            highlight_bg=highlight_bg, text_transform=text_transform,
            stroke_color=stroke_color, stroke_width=stroke_width,
        )
        css_path.write_text(css_content)

        config = {
            "css": "style.css",
            "whisper": {"model": whisper_model, "language": language},
            "layout": {
                "max_width_ratio": max_width, "max_number_of_lines": max_lines,
                "vertical_align": {"align": position, "offset": position_offset},
            },
        }
        config_path.write_text(json.dumps(config, indent=2))
        print(f"[CAP-{job_id}] Config: style={caption_style}, template={template}, pos={position}({position_offset}), font={font_size}px")

        result = _run_pycaps(str(input_path), str(output_path), str(job_dir), template, job_id)
        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"pycaps: {result.get('error')}")

        final = _find_output(output_path, job_dir)
        if not final:
            raise HTTPException(status_code=500, detail="Vídeo legendado não gerado")

        elapsed = round(time.time() - start_time, 2)
        print(f"[CAP-{job_id}] Concluído em {elapsed}s → {final.stat().st_size/1024/1024:.1f} MB")

        return FileResponse(
            path=str(final), media_type="video/mp4",
            filename=f"legendado_{video.filename if video and video.filename else 'video.mp4'}",
            headers={"X-Caption-Duration": str(elapsed)},
            background=_cleanup_bg(job_dir),
        )
    except HTTPException:
        _cleanup_dir(job_dir); raise
    except Exception as e:
        _cleanup_dir(job_dir); traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════
# HELPERS — Input
# ═══════════════════════════════════════════════════════════════════

@app.post("/clips")
async def create_clips(
    video: UploadFile = File(default=None, description="VÃ­deo MP4"),
    video_url: str = Form(default="", description="URL do vÃ­deo"),
    openai_api_key: str = Form(default="", description="API Key da OpenAI"),
    openai_model: str = Form(default="gpt-4o", description="Modelo OpenAI"),
    num_clips: int = Form(default=5, description="Quantidade de clipes"),
    min_duration: float = Form(default=18.0, description="DuraÃ§Ã£o mÃ­nima por clipe"),
    max_duration: float = Form(default=60.0, description="DuraÃ§Ã£o mÃ¡xima por clipe"),
    add_captions: bool = Form(default=True, description="Adicionar legenda aos clipes"),
    caption_style: str = Form(default="capcut_clean", description="Estilo de legenda"),
    language: str = Form(default="pt"),
    whisper_model: str = Form(default="small"),
    custom_prompt: str = Form(default="", description="CritÃ©rios extras para escolher clipes"),
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / f"clips_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.mp4"
    zip_path = job_dir / "clips.zip"
    start_time = time.time()

    try:
        await _save_input(video, video_url, input_path, f"CLIPS-{job_id}")
        openai_api_key = _resolve_openai_api_key(openai_api_key)

        num_clips = max(1, min(12, num_clips))
        min_duration = max(8.0, min(90.0, min_duration))
        max_duration = max(min_duration, min(180.0, max_duration))

        video_info = _get_video_info(str(input_path))
        transcript = _whisper_transcribe(str(input_path), job_id)
        clip_plan = _ai_clip_plan(
            transcript=transcript,
            duration=video_info["duration"],
            openai_key=openai_api_key,
            openai_model=openai_model,
            num_clips=num_clips,
            min_duration=min_duration,
            max_duration=max_duration,
            custom_prompt=custom_prompt,
            job_id=job_id,
        )
        clips = _sanitize_clip_plan(clip_plan.get("clips", []), video_info["duration"], min_duration, max_duration, num_clips)
        if not clips:
            raise HTTPException(status_code=500, detail="Nenhum clipe encontrado")

        rendered = []
        for idx, clip in enumerate(clips, 1):
            raw_path = job_dir / f"clip_{idx:02d}_raw.mp4"
            final_path = job_dir / f"clip_{idx:02d}.mp4"
            _cut_clip(str(input_path), str(raw_path), clip["start"], clip["end"], job_id, idx)

            if add_captions:
                cap_dir = job_dir / f"cap_{idx:02d}"
                cap_dir.mkdir(parents=True, exist_ok=True)
                template = _write_caption_assets(cap_dir, caption_style, "minimalist", whisper_model, language)
                result = _run_pycaps(str(raw_path), str(final_path), str(cap_dir), template, f"{job_id}-{idx:02d}")
                if not result["success"] or not final_path.exists():
                    print(f"[CLIPS-{job_id}] Legenda falhou no clipe {idx}, usando clipe sem legenda")
                    shutil.copy2(raw_path, final_path)
            else:
                shutil.copy2(raw_path, final_path)

            clip["file"] = final_path.name
            clip["duration"] = round(clip["end"] - clip["start"], 2)
            rendered.append({"path": final_path, "meta": clip})

        metadata = {
            "job_id": job_id,
            "source_duration": round(video_info["duration"], 2),
            "elapsed": round(time.time() - start_time, 2),
            "clips": [item["meta"] for item in rendered],
        }
        (job_dir / "clips.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(job_dir / "clips.json", "clips.json")
            for item in rendered:
                zf.write(item["path"], item["path"].name)

        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=f"clips_{job_id}.zip",
            headers={"X-Clips-Count": str(len(rendered))},
            background=_cleanup_bg(job_dir),
        )
    except HTTPException:
        _cleanup_dir(job_dir); raise
    except Exception as e:
        _cleanup_dir(job_dir); traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def _save_input(video, video_url, input_path, tag):
    if video and video.filename:
        content = await video.read()
        input_path.write_bytes(content)
        print(f"[{tag}] Upload: {video.filename} ({len(content)/1024/1024:.1f} MB)")
    elif video_url and video_url.strip():
        print(f"[{tag}] Baixando: {video_url[:80]}...")
        _download_file(video_url.strip(), str(input_path))
        print(f"[{tag}] Baixado: {input_path.stat().st_size/1024/1024:.1f} MB")
    else:
        raise HTTPException(status_code=400, detail="Envie 'video' ou 'video_url'")
    if input_path.stat().st_size > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Max 500MB")


def _download_file(url, output_path):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "pycaps-api/3.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(output_path, "wb") as f:
            while chunk := resp.read(8192):
                f.write(chunk)


# ═══════════════════════════════════════════════════════════════════
# HELPERS — Speed
# ═══════════════════════════════════════════════════════════════════

def _apply_speed(input_path, output_path, speed_factor, job_id):
    atempo_filters = []
    remaining = speed_factor
    while remaining > 2.0:
        atempo_filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        atempo_filters.append("atempo=0.5")
        remaining /= 0.5
    atempo_filters.append(f"atempo={remaining:.4f}")

    audio_filter = ",".join(atempo_filters)
    video_filter = f"setpts={1.0/speed_factor:.4f}*PTS"

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", video_filter,
        "-af", audio_filter,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]

    print(f"[EDIT-{job_id}] Speed FFmpeg: vf={video_filter} | af={audio_filter}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(f"[EDIT-{job_id}] Speed falhou: {proc.stderr[-300:]}")


# ═══════════════════════════════════════════════════════════════════
# HELPERS — Video Info & Silence
# ═══════════════════════════════════════════════════════════════════

def _get_video_info(path):
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True, timeout=30,
    )
    data = json.loads(proc.stdout)
    vs = next(s for s in data["streams"] if s["codec_type"] == "video")
    return {
        "duration": float(data["format"]["duration"]),
        "width": int(vs["width"]), "height": int(vs["height"]),
        "fps": eval(vs.get("r_frame_rate", "30/1")),
    }


def _detect_silences(path, silence_db, min_dur):
    proc = subprocess.run(
        ["ffmpeg", "-i", path, "-af", f"silencedetect=noise={silence_db}:d={min_dur}", "-f", "null", "-"],
        capture_output=True, text=True, timeout=120,
    )
    silences, start = [], None
    for line in proc.stderr.split("\n"):
        if "silence_start:" in line:
            try: start = float(line.split("silence_start:")[1].strip().split()[0])
            except: start = None
        elif "silence_end:" in line and start is not None:
            try:
                end = float(line.split("silence_end:")[1].strip().split()[0])
                silences.append({"start": round(start, 3), "end": round(end, 3)})
            except: pass
            start = None
    return silences


# ═══════════════════════════════════════════════════════════════════
# HELPERS — Whisper
# ═══════════════════════════════════════════════════════════════════

def _whisper_transcribe(video_path, job_id):
    script = f'''
import json, whisper
model = whisper.load_model("small")
result = model.transcribe("{video_path}", language="pt", word_timestamps=True)
output = {{"text": result["text"], "segments": []}}
for seg in result["segments"]:
    segment = {{"text": seg["text"].strip(), "start": seg["start"], "end": seg["end"], "words": []}}
    for w in seg.get("words", []):
        segment["words"].append({{"word": w["word"].strip(), "start": w["start"], "end": w["end"]}})
    output["segments"].append(segment)
print(json.dumps(output, ensure_ascii=False))
'''
    sp = WORK_DIR / f"whisper_{job_id}.py"
    sp.write_text(script)
    try:
        proc = subprocess.run([sys.executable, str(sp)], capture_output=True, text=True, timeout=300)
        sp.unlink(missing_ok=True)
        if proc.returncode != 0:
            print(f"[{job_id}] Whisper stderr: {proc.stderr[-300:]}")
            return {"text": "", "segments": []}
        for line in reversed(proc.stdout.strip().split("\n")):
            try: return json.loads(line)
            except: continue
        return {"text": "", "segments": []}
    except Exception as e:
        print(f"[{job_id}] Whisper falhou: {e}")
        return {"text": "", "segments": []}


# ═══════════════════════════════════════════════════════════════════
# HELPERS — AI Edit Plan
# ═══════════════════════════════════════════════════════════════════

def _ai_clip_plan(transcript, duration, openai_key, openai_model, num_clips, min_duration, max_duration, custom_prompt, job_id):
    import urllib.request

    if not openai_key:
        print(f"[CLIPS-{job_id}] OpenAI key ausente, usando fallback")
        return _fallback_clip_plan(transcript, duration, num_clips, min_duration, max_duration)

    timeline = _transcript_timeline(transcript, max_chars=24000)
    prompt = f"""VocÃª Ã© um editor especialista em criar cortes virais para Reels, TikTok e Shorts, no estilo Opus Clip.

OBJETIVO:
Escolha os {num_clips} melhores clipes independentes deste vÃ­deo.

VÃDEO:
DuraÃ§Ã£o total: {duration:.1f}s
DuraÃ§Ã£o por clipe: entre {min_duration:.0f}s e {max_duration:.0f}s

TRANSCRIÃ‡ÃƒO COM TIMESTAMPS:
{timeline}

CRITÃ‰RIOS:
1. Comece em um gancho forte, sem depender de contexto anterior.
2. Termine em uma conclusÃ£o natural, frase de impacto ou virada.
3. Priorize valor claro, emoÃ§Ã£o, opiniÃ£o forte, histÃ³ria ou promessa.
4. Evite comeÃ§ar no meio de uma palavra/frase.
5. DÃª score de 0 a 100 considerando hook, clareza, retenÃ§Ã£o e potencial de compartilhamento.
{f"6. CritÃ©rio extra do usuÃ¡rio: {custom_prompt}" if custom_prompt else ""}

Responda APENAS JSON vÃ¡lido, sem markdown:
{{"clips":[{{"start":12.3,"end":45.6,"title":"tÃ­tulo curto","score":91,"reason":"por que esse trecho funciona"}}]}}"""

    body = json.dumps({
        "model": openai_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.25,
        "max_tokens": 2500,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"},
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"): content = "\n".join(content.split("\n")[1:])
        if content.endswith("```"): content = "\n".join(content.split("\n")[:-1])
        plan = json.loads(content)
        print(f"[CLIPS-{job_id}] IA: {len(plan.get('clips', []))} clipes sugeridos")
        return plan
    except Exception as e:
        print(f"[CLIPS-{job_id}] IA falhou ({e}), usando fallback")
        return _fallback_clip_plan(transcript, duration, num_clips, min_duration, max_duration)


def _fallback_clip_plan(transcript, duration, num_clips, min_duration, max_duration):
    clips, current = [], None
    for seg in transcript.get("segments", []):
        if current is None:
            current = {"start": float(seg.get("start", 0)), "end": float(seg.get("end", 0)), "text": seg.get("text", "").strip()}
        else:
            current["end"] = float(seg.get("end", current["end"]))
            current["text"] = f"{current['text']} {seg.get('text', '').strip()}".strip()

        clip_duration = current["end"] - current["start"]
        sentence_end = current["text"].endswith((".", "!", "?"))
        if clip_duration >= min_duration and (sentence_end or clip_duration >= max_duration):
            clips.append({
                "start": current["start"],
                "end": min(current["end"], current["start"] + max_duration),
                "title": current["text"][:70] or "Clip",
                "score": max(50, 82 - len(clips) * 4),
                "reason": "Fallback por bloco natural da transcriÃ§Ã£o",
            })
            current = None
            if len(clips) >= num_clips:
                break

    if not clips:
        end = min(duration, max_duration)
        clips.append({"start": 0.0, "end": end, "title": "Clip principal", "score": 60, "reason": "Fallback sem transcriÃ§Ã£o suficiente"})
    return {"clips": clips}


def _sanitize_clip_plan(clips, duration, min_duration, max_duration, num_clips):
    clean = []
    for clip in clips:
        try:
            start = max(0.0, float(clip.get("start", 0)))
            end = min(duration, float(clip.get("end", start + min_duration)))
        except Exception:
            continue

        if end <= start:
            continue
        if end - start < min_duration:
            end = min(duration, start + min_duration)
        if end - start > max_duration:
            end = start + max_duration
        if end - start < 3:
            continue

        try:
            score = int(float(clip.get("score", 60)))
        except Exception:
            score = 60

        clean.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "title": str(clip.get("title") or "Clip")[:90],
            "score": max(0, min(100, score)),
            "reason": str(clip.get("reason") or "")[:240],
        })

    clean.sort(key=lambda c: c["score"], reverse=True)
    deduped = []
    for clip in clean:
        overlaps = any(not (clip["end"] <= kept["start"] or clip["start"] >= kept["end"]) for kept in deduped)
        if not overlaps:
            deduped.append(clip)
        if len(deduped) >= num_clips:
            break
    return sorted(deduped, key=lambda c: c["start"])


def _transcript_timeline(transcript, max_chars=24000):
    lines = []
    for seg in transcript.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{seg.get('start', 0):.1f}s - {seg.get('end', 0):.1f}s] {text}")
    timeline = "\n".join(lines)
    return timeline[:max_chars]


def _cut_clip(input_path, output_path, start, end, job_id, index):
    duration = max(0.1, end - start)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", input_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    print(f"[CLIPS-{job_id}] Cortando clip {index}: {start:.2f}s-{end:.2f}s")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not Path(output_path).exists():
        raise RuntimeError(f"FFmpeg falhou no clipe {index}: {proc.stderr[-400:]}")


def _ai_edit_plan(transcript, silences, duration, zoom_intensity, openai_key, openai_model, add_zooms, custom_prompt, job_id):
    import urllib.request

    if not openai_key:
        print(f"[EDIT-{job_id}] OpenAI key ausente, usando fallback")
        return _fallback_plan(silences, duration, zoom_intensity, add_zooms)

    txt = ""
    for seg in transcript.get("segments", []):
        txt += f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}\n"

    prompt = f"""Você é um editor de vídeo profissional para Reels/TikTok/Shorts.

VÍDEO: duração {duration:.1f}s

TRANSCRIÇÃO:
{txt}

SILÊNCIOS DETECTADOS:
{json.dumps(silences, ensure_ascii=False)}

REGRAS:
1. CORTES: Confirme quais silêncios cortar. Mantenha pausas naturais (<0.3s).
2. {"ZOOMS: zoom in nos momentos de ênfase, zoom out entre frases. Max: " + str(zoom_intensity) + "x" if add_zooms else "NÃO adicione zooms."}
3. Não coloque zoom em trechos cortados.
{f"4. EXTRA: {custom_prompt}" if custom_prompt else ""}

Responda APENAS JSON válido, sem markdown:
{{"cuts": [{{"start": 0.0, "end": 0.0}}], "zooms": [{{"start": 0.0, "end": 0.0, "scale": 1.15, "type": "in"}}]}}"""

    body = json.dumps({"model": openai_model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 2000}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"})

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"): content = "\n".join(content.split("\n")[1:])
        if content.endswith("```"): content = "\n".join(content.split("\n")[:-1])
        plan = json.loads(content)
        print(f"[EDIT-{job_id}] IA: {len(plan.get('cuts',[]))} cortes, {len(plan.get('zooms',[]))} zooms")
        return plan
    except Exception as e:
        print(f"[EDIT-{job_id}] IA falhou ({e}), usando fallback")
        return _fallback_plan(silences, duration, zoom_intensity, add_zooms)


def _fallback_plan(silences, duration, zoom_intensity, add_zooms):
    cuts = [s for s in silences if (s["end"] - s["start"]) > 0.5]
    zooms = []
    if add_zooms:
        t, zi = 0.5, True
        while t < duration - 1:
            zooms.append({"start": t, "end": t + (1.0 if zi else 0.5), "scale": zoom_intensity if zi else 1.0, "type": "in" if zi else "out"})
            t += 3.0; zi = not zi
    return {"cuts": cuts, "zooms": zooms}


# ═══════════════════════════════════════════════════════════════════
# HELPERS — FFmpeg Edits
# ═══════════════════════════════════════════════════════════════════

def _apply_edits(input_path, output_path, cuts, zooms, video_info, job_id):
    duration, w, h = video_info["duration"], video_info["width"], video_info["height"]

    if not cuts and not zooms:
        shutil.copy2(input_path, output_path); return

    cut_path = input_path
    if cuts:
        speech = _speech_segments(cuts, duration)
        if not speech:
            shutil.copy2(input_path, output_path); return

        cut_path = output_path.replace(".mp4", "_cut.mp4") if zooms else output_path
        sel = "+".join(f"between(t,{s['start']},{s['end']})" for s in speech)

        cmd = ["ffmpeg", "-y", "-i", input_path,
               "-vf", f"select='{sel}',setpts=N/FRAME_RATE/TB",
               "-af", f"aselect='{sel}',asetpts=N/SR/TB",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23",
               "-c:a", "aac", "-b:a", "128k", cut_path]

        print(f"[EDIT-{job_id}] FFmpeg: {len(cuts)} cortes, {len(speech)} segmentos mantidos")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"[EDIT-{job_id}] Corte falhou: {proc.stderr[-300:]}")
            cut_path = input_path

    if zooms:
        source = cut_path
        zoom_expr = "1"
        for z in zooms:
            s, e, sc = z["start"], z["end"], z.get("scale", 1.15)
            if z.get("type") == "out":
                zoom_expr = f"if(between(t\\,{s}\\,{e})\\,{sc}-({sc}-1)*(t-{s})/({e}-{s})\\,{zoom_expr})"
            else:
                zoom_expr = f"if(between(t\\,{s}\\,{e})\\,1+({sc}-1)*(t-{s})/({e}-{s})\\,{zoom_expr})"

        cmd = ["ffmpeg", "-y", "-i", source,
               "-vf", f"scale=iw*({zoom_expr}):ih*({zoom_expr}),crop={w}:{h}",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23",
               "-c:a", "copy", output_path]

        print(f"[EDIT-{job_id}] FFmpeg: {len(zooms)} zooms")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"[EDIT-{job_id}] Zoom falhou: {proc.stderr[-300:]}")
            if source != output_path: shutil.copy2(source, output_path)

        temp = output_path.replace(".mp4", "_cut.mp4")
        if Path(temp).exists() and temp != output_path:
            Path(temp).unlink(missing_ok=True)
    elif not cuts:
        shutil.copy2(input_path, output_path)


def _speech_segments(cuts, duration):
    sc = sorted(cuts, key=lambda c: c["start"])
    segs, cur = [], 0.0
    for c in sc:
        if c["start"] > cur + 0.05: segs.append({"start": round(cur, 3), "end": round(c["start"], 3)})
        cur = c["end"]
    if cur < duration - 0.05: segs.append({"start": round(cur, 3), "end": round(duration, 3)})
    return segs


# ═══════════════════════════════════════════════════════════════════
# HELPERS — Pycaps (FIX: layout position)
# ═══════════════════════════════════════════════════════════════════

def _run_pycaps(input_path, output_path, job_dir, template, job_id):
    css_path = Path(job_dir) / "style.css"
    config_path = Path(job_dir) / "pycaps.template.json"
    script_path = Path(job_dir) / "run_pycaps.py"

    script = f'''
import sys, json
from pathlib import Path
try:
    from pycaps import CapsPipelineBuilder, TemplateLoader

    # Carrega template e input
    builder = TemplateLoader("{template}").with_input_video("{input_path}").load(False)

    # Aplica CSS customizado
    css = Path("{css_path}")
    if css.exists():
        builder.add_css_content(css.read_text())

    # Carrega config
    cfg = Path("{config_path}")
    layout = {{}}
    if cfg.exists():
        c = json.loads(cfg.read_text())
        layout = c.get("layout", {{}})

    # DEBUG: lista todos os métodos do builder
    builder_methods = [m for m in dir(builder) if not m.startswith("_")]
    print(f"BUILDER_METHODS: {{builder_methods}}")

    # Tenta aplicar layout de várias formas
    applied = False

    # Forma 1: with_layout
    if hasattr(builder, "with_layout"):
        try:
            builder.with_layout(layout)
            print(f"Layout aplicado via with_layout: {{layout}}")
            applied = True
        except Exception as e:
            print(f"with_layout falhou: {{e}}")

    # Forma 2: layout direto
    if not applied and hasattr(builder, "layout"):
        try:
            builder.layout = layout
            print(f"Layout aplicado via builder.layout = ...")
            applied = True
        except Exception as e:
            print(f"builder.layout falhou: {{e}}")

    # Forma 3: with_max_width_ratio e with_vertical_align separados
    if hasattr(builder, "with_max_width_ratio"):
        try:
            builder.with_max_width_ratio(layout.get("max_width_ratio", 0.85))
            print(f"max_width_ratio aplicado")
        except Exception as e:
            print(f"with_max_width_ratio falhou: {{e}}")

    if hasattr(builder, "with_max_number_of_lines"):
        try:
            builder.with_max_number_of_lines(layout.get("max_number_of_lines", 2))
            print(f"max_number_of_lines aplicado")
        except Exception as e:
            print(f"with_max_number_of_lines falhou: {{e}}")

    if hasattr(builder, "with_vertical_align"):
        try:
            va = layout.get("vertical_align", {{}})
            builder.with_vertical_align(va.get("align", "center"), va.get("offset", 0.2))
            print(f"vertical_align aplicado: {{va}}")
            applied = True
        except Exception as e:
            print(f"with_vertical_align falhou: {{e}}")

    # Build pipeline
    pipeline = builder.build()

    # DEBUG: lista métodos do pipeline
    pipeline_methods = [m for m in dir(pipeline) if not m.startswith("_")]
    print(f"PIPELINE_METHODS: {{pipeline_methods}}")

    # Tenta setar layout no pipeline também
    if hasattr(pipeline, "layout"):
        try:
            pipeline.layout = layout
            print(f"Layout aplicado via pipeline.layout")
        except Exception as e:
            print(f"pipeline.layout falhou: {{e}}")

    if hasattr(pipeline, "config"):
        try:
            if hasattr(pipeline.config, "layout"):
                pipeline.config.layout = layout
                print(f"Layout aplicado via pipeline.config.layout")
        except Exception as e:
            print(f"pipeline.config.layout falhou: {{e}}")

    if not applied:
        print(f"AVISO: Nenhum metodo de layout funcionou. Layout: {{layout}}")

    # Executa
    pipeline.run()
    print("SUCCESS")

except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    import traceback; traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''
    script_path.write_text(script)
    print(f"[CAP-{job_id}] Executando pycaps...")

    try:
        proc = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, timeout=600, cwd=job_dir)
        if proc.stdout: print(f"[CAP-{job_id}] stdout: {proc.stdout}")
        if proc.stderr: print(f"[CAP-{job_id}] stderr: {proc.stderr[-500:]}")
        out = _find_output(Path(output_path), Path(job_dir))
        if proc.returncode == 0 and out:
            if str(out) != output_path: shutil.move(str(out), output_path)
            return {"success": True}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        print(f"[CAP-{job_id}] Script falhou: {e}")

    # Fallback CLI
    pycaps_bin = shutil.which("pycaps")
    if not pycaps_bin:
        return {"success": False, "error": "pycaps não encontrado"}
    cmd = [pycaps_bin, "render", "--input", input_path, "--output", output_path, "--template", template]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode == 0 and os.path.exists(output_path):
        return {"success": True}
    return {"success": False, "error": f"CLI exit {proc.returncode}"}


def _resolve_caption_style(caption_style, template):
    if caption_style in CAPTION_STYLE_PRESETS:
        return CAPTION_STYLE_PRESETS[caption_style]
    if template in CAPTION_STYLE_PRESETS:
        return CAPTION_STYLE_PRESETS[template]
    return CAPTION_STYLE_PRESETS["capcut_clean"]


def _resolve_openai_api_key(form_value: str = ""):
    return (form_value or os.getenv("OPENAI_API_KEY") or "").strip()


def _write_caption_assets(job_dir, caption_style, template, whisper_model, language):
    preset = _resolve_caption_style(caption_style, template)
    template = preset["template"]
    css_preset = preset["css"]
    css_content = _build_css(
        font_size=css_preset.get("font_size", 56),
        font_color=css_preset.get("font_color", "white"),
        font_family=css_preset.get("font_family", "Arial Black"),
        font_weight=css_preset.get("font_weight", 900),
        highlight_color=css_preset.get("highlight_color", "white"),
        highlight_bg=css_preset.get("highlight_bg", ""),
        text_transform=css_preset.get("text_transform", "none"),
        stroke_color=css_preset.get("stroke_color", "black"),
        stroke_width=css_preset.get("stroke_width", "3px"),
    )
    config = {
        "css": "style.css",
        "whisper": {"model": whisper_model, "language": language},
        "layout": {
            "max_width_ratio": CAPTION_LOCKED_LAYOUT["max_width"],
            "max_number_of_lines": CAPTION_LOCKED_LAYOUT["max_lines"],
            "vertical_align": {
                "align": CAPTION_LOCKED_LAYOUT["position"],
                "offset": CAPTION_LOCKED_LAYOUT["position_offset"],
            },
        },
    }
    Path(job_dir, "style.css").write_text(css_content)
    Path(job_dir, "pycaps.template.json").write_text(json.dumps(config, indent=2))
    return template


def _build_css(font_size, font_color, font_family, font_weight, highlight_color, highlight_bg, text_transform, stroke_color, stroke_width):
    sw = int(stroke_width.replace("px", "").strip() or "1")
    shadows = [f"{dx}px {dy}px 0 {stroke_color}" for dx in range(-sw, sw+1) for dy in range(-sw, sw+1) if dx or dy]
    shadows.append("0 4px 10px rgba(0,0,0,0.65)")
    hl = f"\n    background-color: {highlight_bg};\n    border-radius: 4px;\n    padding: 2px 6px;" if highlight_bg.strip() else ""
    return f""".word {{
    font-family: {font_family}, sans-serif;
    font-weight: {font_weight};
    font-size: {font_size}px;
    line-height: 1.08;
    letter-spacing: 0;
    color: {font_color};
    text-transform: {text_transform};
    padding: 0 3px;
    text-shadow: {', '.join(shadows)};
    text-align: center;
}}
.word-being-narrated {{
    color: {highlight_color};{hl}
}}
"""


# ═══════════════════════════════════════════════════════════════════
# HELPERS — Cleanup & Output
# ═══════════════════════════════════════════════════════════════════

def _find_output(expected, job_dir):
    if expected.exists(): return expected
    for f in job_dir.glob("*.mp4"):
        if "input" not in f.name: return f
    return None


def _cleanup_dir(d):
    try:
        if d.exists(): shutil.rmtree(str(d), ignore_errors=True)
    except: pass


class _cleanup_bg:
    def __init__(self, d): self.d = d
    async def __call__(self):
        import asyncio; await asyncio.sleep(10); _cleanup_dir(self.d)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
