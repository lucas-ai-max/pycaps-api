"""
pycaps-api: Microserviço FastAPI para legendar vídeos com pycaps.
Deploy como App separado no Easypanel, mesmo projeto que o n8n.

Endpoints:
  POST /caption     - Recebe vídeo, retorna vídeo legendado
  GET  /health      - Health check
  GET  /templates   - Lista templates disponíveis
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

app = FastAPI(
    title="pycaps-api",
    description="API para adicionar legendas estilizadas a vídeos",
    version="2.0.0",
)

WORK_DIR = Path("/tmp/pycaps-work")
WORK_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup_cleanup():
    """Limpar jobs antigos que sobraram de crashes anteriores."""
    _cleanup_old_jobs()


def _cleanup_old_jobs(max_age_minutes: int = 30):
    """Remove jobs com mais de X minutos."""
    now = time.time()
    for item in WORK_DIR.iterdir():
        try:
            age_minutes = (now - item.stat().st_mtime) / 60
            if age_minutes > max_age_minutes:
                if item.is_dir():
                    shutil.rmtree(str(item), ignore_errors=True)
                else:
                    item.unlink()
        except Exception:
            pass


@app.get("/cleanup")
async def manual_cleanup():
    """Limpar todos os arquivos temporários manualmente."""
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


# ─── Health & Info ────────────────────────────────────────────────

@app.get("/health")
async def health():
    ffmpeg_ok = os.system("ffmpeg -version > /dev/null 2>&1") == 0
    pycaps_ok = False
    pycaps_bin = shutil.which("pycaps")
    try:
        import pycaps
        pycaps_ok = True
    except ImportError:
        pass

    return {
        "status": "ok" if (ffmpeg_ok and pycaps_ok) else "degraded",
        "ffmpeg": ffmpeg_ok,
        "pycaps": pycaps_ok,
        "pycaps_cli": pycaps_bin or "not found",
    }


@app.get("/templates")
async def list_templates():
    builtin = ["minimalist", "default"]
    custom_dir = Path("/app/templates")
    custom = []
    if custom_dir.exists():
        custom = [d.name for d in custom_dir.iterdir() if d.is_dir()]
    return {"builtin": builtin, "custom": custom}


# ─── Edit endpoint ────────────────────────────────────────────────

@app.post("/edit")
async def edit_video(
    video: UploadFile = File(..., description="Arquivo de vídeo (MP4)"),
    openai_api_key: str = Form(..., description="OpenAI API key"),
    openai_model: str = Form(default="gpt-4o", description="Modelo OpenAI"),
    remove_silence: bool = Form(default=True, description="Remover silêncios/suspiros"),
    silence_threshold: float = Form(default=0.4, description="Duração mínima do silêncio pra cortar (segundos)"),
    add_zooms: bool = Form(default=True, description="Adicionar zoom in/out dinâmico"),
    zoom_intensity: float = Form(default=1.15, description="Intensidade do zoom (1.1=sutil, 1.3=forte)"),
    custom_prompt: str = Form(default="", description="Prompt extra pra IA (ex: 'zoom em momentos emocionais')"),
):
    """
    Edita vídeo automaticamente com IA: remove silêncios e aplica zooms dinâmicos.
    Retorna o vídeo editado (sem legendas — use /caption depois).
    """
    job_id = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / f"edit_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.mp4"
    output_path = job_dir / "edited.mp4"
    start_time = time.time()

    try:
        # 1. Salvar vídeo
        content = await video.read()
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > 500:
            raise HTTPException(status_code=413, detail="Arquivo muito grande (max 500MB)")
        input_path.write_bytes(content)
        print(f"[EDIT-{job_id}] Recebido: {video.filename} ({file_size_mb:.1f} MB)")

        # 2. Extrair áudio e transcrever com Whisper
        transcript = _transcribe_video(str(input_path), job_id)
        if not transcript:
            raise HTTPException(status_code=500, detail="Falha na transcrição - vídeo sem áudio?")
        print(f"[EDIT-{job_id}] Transcrito: {len(transcript)} segmentos")

        # 3. Obter duração do vídeo
        duration = _get_video_duration(str(input_path))
        print(f"[EDIT-{job_id}] Duração: {duration:.1f}s")

        # 4. IA analisa e gera decisões de edição
        edit_plan = _ai_generate_edit_plan(
            transcript=transcript,
            video_duration=duration,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            remove_silence=remove_silence,
            silence_threshold=silence_threshold,
            add_zooms=add_zooms,
            zoom_intensity=zoom_intensity,
            custom_prompt=custom_prompt,
            job_id=job_id,
        )

        # Salvar plano pra debug
        plan_path = job_dir / "edit_plan.json"
        plan_path.write_text(json.dumps(edit_plan, indent=2, ensure_ascii=False))
        print(f"[EDIT-{job_id}] Plano: {len(edit_plan.get('cuts', []))} cortes, {len(edit_plan.get('zooms', []))} zooms")

        # 5. Executar edições com FFmpeg
        _execute_edits(
            input_path=str(input_path),
            output_path=str(output_path),
            edit_plan=edit_plan,
            job_id=job_id,
        )

        if not output_path.exists():
            raise HTTPException(status_code=500, detail="Vídeo editado não foi gerado")

        elapsed = round(time.time() - start_time, 2)
        output_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[EDIT-{job_id}] Concluído em {elapsed}s → {output_mb:.1f} MB")

        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=f"editado_{video.filename or 'video.mp4'}",
            headers={
                "X-Edit-Duration": str(elapsed),
                "X-Edit-Cuts": str(len(edit_plan.get("cuts", []))),
                "X-Edit-Zooms": str(len(edit_plan.get("zooms", []))),
                "X-Edit-Job-Id": job_id,
            },
            background=_cleanup_task(job_dir),
        )

    except HTTPException:
        _cleanup_dir(job_dir)
        raise
    except Exception as e:
        _cleanup_dir(job_dir)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Transcription ────────────────────────────────────────────────

def _transcribe_video(input_path: str, job_id: str) -> list:
    """Transcreve vídeo com Whisper via subprocess."""
    script = f'''
import json, sys
try:
    import whisper
    model = whisper.load_model("small")
    result = model.transcribe("{input_path}", language="pt", word_timestamps=True)
    segments = []
    for seg in result.get("segments", []):
        words = []
        for w in seg.get("words", []):
            words.append({{"word": w["word"].strip(), "start": round(w["start"], 3), "end": round(w["end"], 3)}})
        segments.append({{
            "text": seg["text"].strip(),
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "words": words
        }})
    print(json.dumps(segments, ensure_ascii=False))
except Exception as e:
    print(json.dumps({{"error": str(e)}}), file=sys.stderr)
    sys.exit(1)
'''
    script_path = WORK_DIR / f"transcribe_{job_id}.py"
    script_path.write_text(script)

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=300,
        )
        script_path.unlink(missing_ok=True)

        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip())
        else:
            print(f"[EDIT-{job_id}] Whisper erro: {proc.stderr[-300:]}")
            return []
    except Exception as e:
        print(f"[EDIT-{job_id}] Transcribe falhou: {e}")
        return []


def _get_video_duration(input_path: str) -> float:
    """Obtém duração do vídeo via ffprobe."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", input_path],
            capture_output=True, text=True, timeout=30,
        )
        info = json.loads(proc.stdout)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0


# ─── AI Edit Planning ─────────────────────────────────────────────

def _ai_generate_edit_plan(
    transcript: list,
    video_duration: float,
    openai_api_key: str,
    openai_model: str,
    remove_silence: bool,
    silence_threshold: float,
    add_zooms: bool,
    zoom_intensity: float,
    custom_prompt: str,
    job_id: str,
) -> dict:
    """Chama GPT-4o para analisar transcript e gerar plano de edição."""
    import urllib.request

    # Montar transcript resumido pra IA
    transcript_text = ""
    for seg in transcript:
        transcript_text += f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}\n"

    system_prompt = f"""Você é um editor de vídeo profissional para conteúdo de redes sociais (Reels/TikTok/Shorts).

Analise o transcript abaixo e gere um plano de edição em JSON.

REGRAS:
- Duração total do vídeo: {video_duration:.1f} segundos
- Zoom intensity máxima: {zoom_intensity}
- Threshold mínimo de silêncio pra cortar: {silence_threshold}s
{"- REMOVER silêncios, suspiros, hesitações (éééé, hmmm, pausas longas)" if remove_silence else "- NÃO remover silêncios"}
{"- ADICIONAR zooms dinâmicos nos momentos de ênfase, palavras fortes, mudanças de energia" if add_zooms else "- NÃO adicionar zooms"}
{f"- INSTRUÇÃO EXTRA: {custom_prompt}" if custom_prompt else ""}

RETORNE APENAS JSON (sem markdown, sem explicação):
{{
  "cuts": [
    {{"start": 2.1, "end": 3.4, "reason": "silêncio entre frases"}},
    ...
  ],
  "zooms": [
    {{"start": 1.0, "end": 2.5, "scale": 1.15, "center_x": 0.5, "center_y": 0.4}},
    ...
  ]
}}

- cuts: trechos pra REMOVER do vídeo (silêncios, suspiros)
- zooms: trechos pra aplicar zoom (scale 1.0=normal, 1.15=sutil, 1.3=forte)
- center_x/center_y: ponto focal do zoom (0.5,0.4 = rosto centralizado)
- NÃO sobreponha zooms (um zoom por vez)
- NÃO corte no meio de palavras
- Mantenha pelo menos 0.15s de respiro entre cortes"""

    payload = json.dumps({
        "model": openai_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"TRANSCRIPT:\n{transcript_text}"},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        },
    )

    try:
        print(f"[EDIT-{job_id}] Chamando {openai_model}...")
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        content = result["choices"][0]["message"]["content"]
        # Limpar markdown se vier
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]

        edit_plan = json.loads(content)
        print(f"[EDIT-{job_id}] IA retornou plano com {len(edit_plan.get('cuts', []))} cortes e {len(edit_plan.get('zooms', []))} zooms")
        return edit_plan

    except Exception as e:
        print(f"[EDIT-{job_id}] IA falhou: {e}")
        # Fallback: detectar silêncios automaticamente sem IA
        return _fallback_detect_silences(transcript, video_duration, silence_threshold)


def _fallback_detect_silences(transcript: list, video_duration: float, threshold: float) -> dict:
    """Detecta silêncios baseado nos gaps entre segmentos (sem IA)."""
    cuts = []
    for i in range(len(transcript) - 1):
        gap_start = transcript[i]["end"]
        gap_end = transcript[i + 1]["start"]
        gap = gap_end - gap_start
        if gap >= threshold:
            # Manter 0.1s de respiro nas bordas
            cuts.append({
                "start": round(gap_start + 0.1, 3),
                "end": round(gap_end - 0.1, 3),
                "reason": f"silêncio ({gap:.1f}s)",
            })
    return {"cuts": cuts, "zooms": []}


# ─── FFmpeg Execution ─────────────────────────────────────────────

def _execute_edits(input_path: str, output_path: str, edit_plan: dict, job_id: str):
    """Executa cortes e zooms via FFmpeg."""

    cuts = edit_plan.get("cuts", [])
    zooms = edit_plan.get("zooms", [])

    # Se não tem edições, copiar o arquivo
    if not cuts and not zooms:
        print(f"[EDIT-{job_id}] Nenhuma edição — copiando original")
        shutil.copy2(input_path, output_path)
        return

    # Obter duração total
    duration = _get_video_duration(input_path)

    # ─── Montar filter_complex ───
    filter_parts = []

    # 1. Zoom filters
    if zooms:
        zoom_filter = _build_zoom_filter(zooms, duration)
        if zoom_filter:
            filter_parts.append(zoom_filter)

    # 2. Cortes (trim + concat)
    if cuts:
        _execute_cuts_and_zooms(input_path, output_path, cuts, zooms, duration, job_id)
        return

    # Só zooms, sem cortes
    if zooms and not cuts:
        _execute_zooms_only(input_path, output_path, zooms, duration, job_id)
        return

    shutil.copy2(input_path, output_path)


def _execute_cuts_and_zooms(
    input_path: str, output_path: str, cuts: list, zooms: list, duration: float, job_id: str
):
    """Remove silêncios e aplica zooms."""

    # Calcular os segmentos a MANTER (inverso dos cortes)
    cuts_sorted = sorted(cuts, key=lambda c: c["start"])
    keep_segments = []
    current = 0.0

    for cut in cuts_sorted:
        cut_start = max(0, cut["start"])
        cut_end = min(duration, cut["end"])
        if cut_start > current:
            keep_segments.append({"start": current, "end": cut_start})
        current = cut_end

    if current < duration:
        keep_segments.append({"start": current, "end": duration})

    if not keep_segments:
        print(f"[EDIT-{job_id}] Nenhum segmento restante após cortes!")
        shutil.copy2(input_path, output_path)
        return

    print(f"[EDIT-{job_id}] Mantendo {len(keep_segments)} segmentos, removendo {len(cuts_sorted)} cortes")

    # Montar filter_complex com trim + concat
    n = len(keep_segments)
    filters = []
    streams_v = []
    streams_a = []

    for i, seg in enumerate(keep_segments):
        # Video trim
        vf = f"[0:v]trim=start={seg['start']:.3f}:end={seg['end']:.3f},setpts=PTS-STARTPTS"

        # Aplicar zoom se houver sobreposição
        zoom = _find_zoom_for_segment(seg, zooms)
        if zoom:
            scale = zoom.get("scale", 1.15)
            cx = zoom.get("center_x", 0.5)
            cy = zoom.get("center_y", 0.4)
            # Zoom com crop centralizado
            vf += f",scale=iw*{scale}:ih*{scale}"
            offset_x = f"(iw-ow)*{cx}"
            offset_y = f"(ih-oh)*{cy}"
            vf += f",crop=iw/{scale}:ih/{scale}:{offset_x}:{offset_y}"

        vf += f"[v{i}]"
        filters.append(vf)
        streams_v.append(f"[v{i}]")

        # Audio trim
        af = f"[0:a]atrim=start={seg['start']:.3f}:end={seg['end']:.3f},asetpts=PTS-STARTPTS[a{i}]"
        filters.append(af)
        streams_a.append(f"[a{i}]")

    # Concat
    concat_in = "".join(streams_v) + "".join(streams_a)
    filters.append(f"{concat_in}concat=n={n}:v=1:a=1[outv][outa]")

    filter_complex = ";\n".join(filters)

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    print(f"[EDIT-{job_id}] Executando FFmpeg com {n} segmentos...")

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if proc.returncode != 0:
        print(f"[EDIT-{job_id}] FFmpeg erro: {proc.stderr[-500:]}")
        # Fallback: tentar sem zoom
        _execute_cuts_only(input_path, output_path, keep_segments, job_id)


def _execute_cuts_only(input_path: str, output_path: str, keep_segments: list, job_id: str):
    """Fallback: só cortes, sem zoom."""
    n = len(keep_segments)
    filters = []
    streams_v = []
    streams_a = []

    for i, seg in enumerate(keep_segments):
        filters.append(f"[0:v]trim=start={seg['start']:.3f}:end={seg['end']:.3f},setpts=PTS-STARTPTS[v{i}]")
        filters.append(f"[0:a]atrim=start={seg['start']:.3f}:end={seg['end']:.3f},asetpts=PTS-STARTPTS[a{i}]")
        streams_v.append(f"[v{i}]")
        streams_a.append(f"[a{i}]")

    concat_in = "".join(streams_v) + "".join(streams_a)
    filters.append(f"{concat_in}concat=n={n}:v=1:a=1[outv][outa]")

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter_complex", ";\n".join(filters),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]

    print(f"[EDIT-{job_id}] FFmpeg fallback (sem zoom)...")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(f"[EDIT-{job_id}] FFmpeg fallback falhou: {proc.stderr[-300:]}")


def _execute_zooms_only(input_path: str, output_path: str, zooms: list, duration: float, job_id: str):
    """Só zooms, sem cortes."""
    # Construir filtro zoompan dinâmico
    zoom_filters = []
    for z in zooms:
        scale = z.get("scale", 1.15)
        start = z.get("start", 0)
        end = z.get("end", start + 1)
        cx = z.get("center_x", 0.5)
        cy = z.get("center_y", 0.4)

        # Usar crop pra simular zoom
        zoom_filters.append(
            f"between(t,{start:.3f},{end:.3f})"
        )

    if not zoom_filters:
        shutil.copy2(input_path, output_path)
        return

    # Simplificação: aplicar zooms via crop com enable
    filter_parts = []
    base = "[0:v]split={n}".format(n=len(zooms) + 1)

    # Abordagem simples: zoom mais forte no segmento inteiro
    z = zooms[0]  # Primeiro zoom como teste
    scale = z.get("scale", 1.15)

    # Um filtro de zoom suave com zoompan
    vf = (
        f"zoompan=z='if(between(on,0,1),{scale},{scale})':"
        f"d=1:s=1080x1920:fps=30"
    )

    # Usar abordagem mais simples: scale + crop por segmento
    enables = []
    for z in zooms:
        enables.append(f"between(t\\,{z['start']:.2f}\\,{z['end']:.2f})")

    enable_expr = "+".join(enables)
    scale_val = zooms[0].get("scale", 1.15)

    vf = (
        f"[0:v]scale=iw*{scale_val}:ih*{scale_val},"
        f"crop=iw/{scale_val}:ih/{scale_val}:(iw-ow)/2:(ih-oh)/2:"
        f"enable='{enable_expr}'[outv]"
    )

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter_complex", vf,
        "-map", "[outv]", "-map", "0:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        output_path,
    ]

    print(f"[EDIT-{job_id}] FFmpeg zooms only...")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(f"[EDIT-{job_id}] Zoom falhou: {proc.stderr[-300:]}")
        shutil.copy2(input_path, output_path)


def _find_zoom_for_segment(segment: dict, zooms: list) -> dict | None:
    """Encontra zoom que sobrepõe com o segmento."""
    for z in zooms:
        # Se o zoom sobrepõe com o segmento
        if z["start"] < segment["end"] and z["end"] > segment["start"]:
            return z
    return None


def _build_zoom_filter(zooms: list, duration: float) -> str:
    """Constrói filtro de zoom (placeholder)."""
    return ""


# ─── Main endpoint ────────────────────────────────────────────────

@app.post("/caption")
async def caption_video(
    video: UploadFile = File(..., description="Arquivo de vídeo (MP4, MOV, etc.)"),
    # Template base
    template: str = Form(default="minimalist", description="Template pycaps base"),
    language: str = Form(default="pt", description="Idioma Whisper (pt, en, es...)"),
    whisper_model: str = Form(default="small", description="Modelo Whisper: tiny, base, small, medium, large"),
    # Posição
    position: str = Form(default="bottom", description="Posição vertical: top, center, bottom"),
    position_offset: float = Form(default=0.0, description="Ajuste fino da posição: -1.0 (topo) a 1.0 (base)"),
    max_width: float = Form(default=0.8, description="Largura máxima da legenda (0.0 a 1.0 do vídeo)"),
    max_lines: int = Form(default=2, description="Máximo de linhas por segmento"),
    # Estilo
    font_size: int = Form(default=18, description="Tamanho da fonte (px)"),
    font_color: str = Form(default="white", description="Cor do texto (CSS: white, #FF0000, rgb(...))"),
    font_family: str = Form(default="system-ui", description="Família da fonte CSS"),
    font_weight: int = Form(default=700, description="Peso da fonte (400=normal, 700=bold, 800=extra-bold)"),
    highlight_color: str = Form(default="#ffc107", description="Cor da palavra sendo falada"),
    highlight_bg: str = Form(default="", description="Background da palavra sendo falada (vazio=sem bg)"),
    text_transform: str = Form(default="uppercase", description="Transformação: uppercase, lowercase, none"),
    stroke_color: str = Form(default="black", description="Cor do contorno do texto"),
    stroke_width: str = Form(default="1px", description="Espessura do contorno (ex: 1px, 2px)"),
    # CSS customizado (override total)
    custom_css: str = Form(default="", description="CSS customizado completo (ignora params de estilo acima)"),
):
    """
    Recebe um vídeo e retorna com legendas estilizadas.
    Controle total de posição e estilo via parâmetros.
    """
    job_id = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / f"job_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.mp4"
    output_path = job_dir / "output.mp4"
    css_path = job_dir / "style.css"
    config_path = job_dir / "pycaps.template.json"
    start_time = time.time()

    try:
        # 1. Salvar vídeo
        content = await video.read()
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > 500:
            raise HTTPException(status_code=413, detail="Arquivo muito grande (max 500MB)")
        input_path.write_bytes(content)
        print(f"[{job_id}] Recebido: {video.filename} ({file_size_mb:.1f} MB)")

        # 2. Gerar CSS
        if custom_css.strip():
            css_content = custom_css
            print(f"[{job_id}] Usando CSS customizado")
        else:
            css_content = _build_css(
                font_size=font_size,
                font_color=font_color,
                font_family=font_family,
                font_weight=font_weight,
                highlight_color=highlight_color,
                highlight_bg=highlight_bg,
                text_transform=text_transform,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
            )
            print(f"[{job_id}] CSS gerado: font={font_size}px {font_color}, highlight={highlight_color}")

        css_path.write_text(css_content)

        # 3. Gerar config JSON
        config = _build_config(
            css_path="style.css",
            position=position,
            position_offset=position_offset,
            max_width=max_width,
            max_lines=max_lines,
            whisper_model=whisper_model,
            language=language,
        )
        config_path.write_text(json.dumps(config, indent=2))
        print(f"[{job_id}] Config: pos={position}({position_offset}), width={max_width}, lines={max_lines}")

        # 4. Processar
        result = _process_with_pycaps(
            input_path=str(input_path),
            output_path=str(output_path),
            job_dir=str(job_dir),
            template=template,
            job_id=job_id,
        )

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Erro no processamento: {result.get('error', 'desconhecido')}",
            )

        # 5. Encontrar output
        final_output = _find_output(output_path, job_dir)
        if not final_output:
            raise HTTPException(status_code=500, detail="Vídeo de saída não foi gerado")

        duration = round(time.time() - start_time, 2)
        output_size_mb = final_output.stat().st_size / (1024 * 1024)
        print(f"[{job_id}] Concluído em {duration}s → {output_size_mb:.1f} MB")

        return FileResponse(
            path=str(final_output),
            media_type="video/mp4",
            filename=f"legendado_{video.filename or 'video.mp4'}",
            headers={
                "X-Pycaps-Duration": str(duration),
                "X-Pycaps-Template": template,
                "X-Pycaps-Position": position,
                "X-Pycaps-Job-Id": job_id,
            },
            background=_cleanup_task(job_dir),
        )

    except HTTPException:
        _cleanup_dir(job_dir)
        raise
    except Exception as e:
        _cleanup_dir(job_dir)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── CSS Builder ──────────────────────────────────────────────────

def _build_css(
    font_size: int,
    font_color: str,
    font_family: str,
    font_weight: int,
    highlight_color: str,
    highlight_bg: str,
    text_transform: str,
    stroke_color: str,
    stroke_width: str,
) -> str:
    """Gera CSS para o pycaps baseado nos parâmetros."""

    # Text shadow para simular stroke/contorno
    sw = stroke_width.replace("px", "").strip()
    try:
        sw_val = int(sw)
    except ValueError:
        sw_val = 1

    shadow_parts = []
    for dx in range(-sw_val, sw_val + 1):
        for dy in range(-sw_val, sw_val + 1):
            if dx != 0 or dy != 0:
                shadow_parts.append(f"{dx}px {dy}px 0 {stroke_color}")
    # Adicionar sombra suave para legibilidade
    shadow_parts.append(f"2px 2px 4px rgba(0, 0, 0, 0.8)")
    text_shadow = ", ".join(shadow_parts)

    # Highlight background
    highlight_bg_rule = ""
    if highlight_bg.strip():
        highlight_bg_rule = f"""
    background-color: {highlight_bg};
    border-radius: 4px;
    padding: 2px 6px;"""

    css = f""".word {{
    font-family: {font_family}, sans-serif;
    font-weight: {font_weight};
    font-size: {font_size}px;
    color: {font_color};
    text-transform: {text_transform};
    padding: 2px 4px;
    text-shadow: {text_shadow};
}}

.word-being-narrated {{
    color: {highlight_color};{highlight_bg_rule}
}}
"""
    return css


# ─── Config Builder ───────────────────────────────────────────────

def _build_config(
    css_path: str,
    position: str,
    position_offset: float,
    max_width: float,
    max_lines: int,
    whisper_model: str,
    language: str,
) -> dict:
    """Gera pycaps.template.json com layout e whisper config."""

    config = {
        "css": css_path,
        "whisper": {
            "model": whisper_model,
        },
        "layout": {
            "max_width_ratio": max_width,
            "max_number_of_lines": max_lines,
            "vertical_align": {
                "align": position,
                "offset": position_offset,
            },
        },
    }

    # Adicionar language se especificado
    if language:
        config["whisper"]["language"] = language

    return config


# ─── Processing ───────────────────────────────────────────────────

def _process_with_pycaps(
    input_path: str,
    output_path: str,
    job_dir: str,
    template: str,
    job_id: str,
) -> dict:
    """Executa pycaps via script Python em subprocess (evita conflito Playwright+asyncio)."""

    css_path = Path(job_dir) / "style.css"
    config_path = Path(job_dir) / "pycaps.template.json"
    script_path = Path(job_dir) / "run_pycaps.py"

    # Gerar script Python que roda fora do asyncio
    script = f'''
import sys
import json
from pathlib import Path

try:
    from pycaps import CapsPipelineBuilder, TemplateLoader

    # Carregar template minimalist como base
    builder = TemplateLoader("{template}").with_input_video("{input_path}").load(False)

    # Sobrescrever CSS se existir customizado
    css_path = Path("{css_path}")
    if css_path.exists():
        css_content = css_path.read_text()
        builder.add_css_content(css_content)

    # Aplicar layout do config
    config_path = Path("{config_path}")
    if config_path.exists():
        config = json.loads(config_path.read_text())
        layout = config.get("layout", {{}})

        # Aplicar configurações de layout disponíveis
        if hasattr(builder, "with_layout"):
            builder.with_layout(layout)

    # Build e executar
    pipeline = builder.build()
    pipeline.run()
    print("SUCCESS")

except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''

    script_path.write_text(script)
    print(f"[{job_id}] Executando pycaps via script Python isolado...")

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=job_dir,
        )

        if proc.stdout:
            print(f"[{job_id}] stdout: {proc.stdout[-500:]}")
        if proc.stderr:
            print(f"[{job_id}] stderr: {proc.stderr[-500:]}")

        # Encontrar output gerado
        output = _find_output(Path(output_path), Path(job_dir))

        if proc.returncode == 0 and output:
            # Mover output pro path esperado se necessário
            if str(output) != output_path:
                shutil.move(str(output), output_path)
            return {"success": True}

        # Se o script falhou, tentar CLI puro com template padrão
        print(f"[{job_id}] Script falhou, tentando CLI com template '{template}'...")

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout: processamento excedeu 10 minutos"}
    except Exception as e:
        print(f"[{job_id}] Script falhou: {e}")

    # --- Fallback: CLI puro com template padrão ---
    try:
        pycaps_bin = shutil.which("pycaps")
        if not pycaps_bin:
            return {"success": False, "error": "pycaps CLI não encontrado"}

        cmd = [
            pycaps_bin, "render",
            "--input", input_path,
            "--output", output_path,
            "--template", template,
        ]

        print(f"[{job_id}] CLI fallback: {' '.join(cmd)}")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if proc.stdout:
            print(f"[{job_id}] stdout: {proc.stdout[-500:]}")
        if proc.stderr:
            print(f"[{job_id}] stderr: {proc.stderr[-500:]}")

        if proc.returncode == 0 and os.path.exists(output_path):
            return {"success": True}
        else:
            return {
                "success": False,
                "error": f"CLI exit {proc.returncode}: {proc.stderr[-300:] if proc.stderr else ''}",
            }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout CLI"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─── Helpers ──────────────────────────────────────────────────────

def _find_output(expected_path: Path, job_dir: Path) -> Path | None:
    """Encontra o arquivo de output (pycaps pode gerar com nome diferente)."""
    if expected_path.exists():
        return expected_path

    # Procurar qualquer .mp4 gerado (exceto input)
    for f in job_dir.glob("*.mp4"):
        if "input" not in f.name:
            return f

    # Procurar no diretório de trabalho geral
    for f in WORK_DIR.glob(f"*output*"):
        if f.is_file() and f.suffix == ".mp4":
            return f

    return None


def _cleanup_dir(job_dir: Path):
    """Remove diretório do job."""
    try:
        if job_dir.exists():
            shutil.rmtree(str(job_dir), ignore_errors=True)
    except Exception:
        pass


def _cleanup_files(*paths):
    for p in paths:
        try:
            if isinstance(p, (str, Path)) and Path(p).exists():
                Path(p).unlink()
        except Exception:
            pass


class _cleanup_task:
    """Background task para limpar job após enviar resposta."""

    def __init__(self, job_dir: Path):
        self.job_dir = job_dir

    async def __call__(self):
        import asyncio
        await asyncio.sleep(10)
        _cleanup_dir(self.job_dir)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ═══════════════════════════════════════════════════════════════════
# ENDPOINT /edit — Pipeline completo: cortar silêncio + zoom AI + legendar
# ═══════════════════════════════════════════════════════════════════

@app.post("/edit")
async def edit_video(
    video: UploadFile = File(..., description="Vídeo MP4"),
    openai_key: str = Form(..., description="API Key da OpenAI"),
    # Edição
    silence_threshold: float = Form(default=0.4, description="Duração mín de silêncio pra cortar (segundos)"),
    silence_db: str = Form(default="-30dB", description="Limiar de volume pra considerar silêncio"),
    zoom_intensity: float = Form(default=1.15, description="Intensidade do zoom (1.0=sem, 1.15=suave, 1.3=forte)"),
    # Legendas
    position: str = Form(default="center"),
    position_offset: float = Form(default=0.0),
    font_size: int = Form(default=24),
    font_weight: int = Form(default=800),
    font_color: str = Form(default="white"),
    highlight_color: str = Form(default="white"),
    highlight_bg: str = Form(default="#22c55e"),
    text_transform: str = Form(default="uppercase"),
    stroke_color: str = Form(default="black"),
    stroke_width: str = Form(default="2px"),
):
    """
    Pipeline completo de edição:
    1. Detecta silêncios/suspiros via FFmpeg
    2. Transcreve áudio via Whisper
    3. GPT-4o analisa e gera decisões de zoom
    4. FFmpeg corta silêncios + aplica zooms
    5. Pycaps adiciona legendas estilizadas
    """
    job_id = str(uuid.uuid4())[:8]
    job_dir = WORK_DIR / f"edit_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.mp4"
    edited_path = job_dir / "edited.mp4"
    final_path = job_dir / "final.mp4"
    start_time = time.time()

    try:
        # Salvar vídeo
        content = await video.read()
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > 500:
            raise HTTPException(status_code=413, detail="Max 500MB")
        input_path.write_bytes(content)
        print(f"[{job_id}] EDIT recebido: {file_size_mb:.1f} MB")

        # Obter info do vídeo
        video_info = _get_video_info(str(input_path))
        print(f"[{job_id}] Vídeo: {video_info['duration']:.1f}s, {video_info['width']}x{video_info['height']}")

        # 1. Detectar silêncios
        print(f"[{job_id}] Detectando silêncios...")
        silences = _detect_silences(str(input_path), silence_db, silence_threshold)
        print(f"[{job_id}] {len(silences)} silêncios detectados")

        # 2. Transcrever com Whisper
        print(f"[{job_id}] Transcrevendo com Whisper...")
        transcript = _transcribe_video(str(input_path), str(job_dir))
        print(f"[{job_id}] Transcrição: {len(transcript['segments'])} segmentos")

        # 3. IA decide zooms
        print(f"[{job_id}] Consultando GPT-4o para edição...")
        edit_plan = _get_ai_edit_plan(
            transcript=transcript,
            silences=silences,
            duration=video_info["duration"],
            zoom_intensity=zoom_intensity,
            openai_key=openai_key,
        )
        print(f"[{job_id}] Plano: {len(edit_plan.get('cuts', []))} cortes, {len(edit_plan.get('zooms', []))} zooms")

        # 4. FFmpeg aplica edições
        print(f"[{job_id}] Aplicando edições com FFmpeg...")
        _apply_edits(
            input_path=str(input_path),
            output_path=str(edited_path),
            cuts=edit_plan.get("cuts", []),
            zooms=edit_plan.get("zooms", []),
            video_info=video_info,
        )

        if not edited_path.exists():
            raise HTTPException(status_code=500, detail="FFmpeg não gerou vídeo editado")
        print(f"[{job_id}] Vídeo editado: {edited_path.stat().st_size / 1024 / 1024:.1f} MB")

        # 5. Legendar com pycaps
        print(f"[{job_id}] Legendando com pycaps...")
        css_content = _build_css(
            font_size=font_size, font_color=font_color, font_family="system-ui",
            font_weight=font_weight, highlight_color=highlight_color,
            highlight_bg=highlight_bg, text_transform=text_transform,
            stroke_color=stroke_color, stroke_width=stroke_width,
        )
        css_path = job_dir / "style.css"
        css_path.write_text(css_content)

        config = _build_config(
            css_path="style.css", position=position, position_offset=position_offset,
            max_width=0.8, max_lines=2, whisper_model="small", language="pt",
        )
        config_json_path = job_dir / "pycaps.template.json"
        config_json_path.write_text(json.dumps(config, indent=2))

        caption_result = _process_with_pycaps(
            input_path=str(edited_path),
            output_path=str(final_path),
            job_dir=str(job_dir),
            template="minimalist",
            job_id=job_id,
        )

        # Encontrar output final
        output_file = _find_output(final_path, job_dir)
        if not output_file or not caption_result.get("success"):
            # Se legenda falhou, retorna pelo menos o vídeo editado
            output_file = edited_path
            print(f"[{job_id}] Legenda falhou, retornando vídeo editado sem legenda")

        duration = round(time.time() - start_time, 2)
        output_size = output_file.stat().st_size / 1024 / 1024
        print(f"[{job_id}] CONCLUÍDO em {duration}s → {output_size:.1f} MB")

        return FileResponse(
            path=str(output_file),
            media_type="video/mp4",
            filename=f"editado_{video.filename or 'video.mp4'}",
            headers={
                "X-Edit-Duration": str(duration),
                "X-Edit-Cuts": str(len(edit_plan.get("cuts", []))),
                "X-Edit-Zooms": str(len(edit_plan.get("zooms", []))),
                "X-Edit-Job-Id": job_id,
            },
            background=_cleanup_task(job_dir),
        )

    except HTTPException:
        _cleanup_dir(job_dir)
        raise
    except Exception as e:
        _cleanup_dir(job_dir)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Video Info ───────────────────────────────────────────────────

def _get_video_info(video_path: str) -> dict:
    """Obtém duração, resolução e fps do vídeo."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    video_stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    return {
        "duration": float(data["format"]["duration"]),
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": eval(video_stream.get("r_frame_rate", "30/1")),
    }


# ─── Silence Detection ───────────────────────────────────────────

def _detect_silences(video_path: str, silence_db: str, min_duration: float) -> list:
    """Detecta trechos de silêncio no vídeo via FFmpeg."""
    cmd = [
        "ffmpeg", "-i", video_path, "-af",
        f"silencedetect=noise={silence_db}:d={min_duration}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    silences = []
    lines = stderr.split("\n")
    start = None
    for line in lines:
        if "silence_start:" in line:
            try:
                start = float(line.split("silence_start:")[1].strip().split()[0])
            except (ValueError, IndexError):
                start = None
        elif "silence_end:" in line and start is not None:
            try:
                parts = line.split("silence_end:")[1].strip().split()
                end = float(parts[0])
                silences.append({"start": round(start, 3), "end": round(end, 3)})
            except (ValueError, IndexError):
                pass
            start = None

    return silences


# ─── Whisper Transcription ────────────────────────────────────────

def _transcribe_video(video_path: str, job_dir: str) -> dict:
    """Transcreve vídeo com Whisper via subprocess."""
    script = f'''
import json
import whisper

model = whisper.load_model("small")
result = model.transcribe("{video_path}", language="pt", word_timestamps=True)

output = {{
    "text": result["text"],
    "segments": []
}}

for seg in result["segments"]:
    segment = {{
        "text": seg["text"].strip(),
        "start": seg["start"],
        "end": seg["end"],
        "words": []
    }}
    for w in seg.get("words", []):
        segment["words"].append({{
            "word": w["word"].strip(),
            "start": w["start"],
            "end": w["end"]
        }})
    output["segments"].append(segment)

print(json.dumps(output, ensure_ascii=False))
'''

    script_path = Path(job_dir) / "transcribe.py"
    script_path.write_text(script)

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, timeout=300,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Whisper falhou: {proc.stderr[-300:]}")

    # Pegar última linha com JSON válido
    for line in reversed(proc.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    raise RuntimeError("Whisper não retornou transcript válido")


# ─── AI Edit Planning ────────────────────────────────────────────

def _get_ai_edit_plan(
    transcript: dict,
    silences: list,
    duration: float,
    zoom_intensity: float,
    openai_key: str,
) -> dict:
    """Pede ao GPT-4o para gerar plano de edição baseado no transcript."""
    import urllib.request

    prompt = f"""Você é um editor de vídeo profissional para Reels/TikTok/Shorts.

VÍDEO: duração {duration:.1f}s

TRANSCRIÇÃO COM TIMESTAMPS:
{json.dumps(transcript['segments'], ensure_ascii=False, indent=2)}

SILÊNCIOS DETECTADOS:
{json.dumps(silences, ensure_ascii=False)}

REGRAS:
1. CORTES: Confirme quais silêncios devem ser cortados. Mantenha pausas curtas naturais (<0.5s). Corte suspiros, hesitações longas e silêncios mortos.
2. ZOOMS: Adicione zoom in nos momentos de ênfase, palavras fortes, ou mudanças de energia. Use zoom out suave entre frases pra dar ritmo. Intensidade máxima: {zoom_intensity}x
3. Cada zoom deve ter: start, end (segundos), scale (1.0 a {zoom_intensity}), type ("in" ou "out")
4. Não coloque zoom em trechos que serão cortados.
5. Pense em ritmo — alterne zoom in/out pra manter dinâmico.

Responda APENAS com JSON válido, sem markdown, neste formato:
{{
  "cuts": [{{"start": 0.0, "end": 0.0}}],
  "zooms": [{{"start": 0.0, "end": 0.0, "scale": 1.15, "type": "in"}}]
}}"""

    body = json.dumps({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2000,
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"OpenAI falhou: {e}, usando plano padrão")
        return _default_edit_plan(silences, duration, zoom_intensity)

    content = data["choices"][0]["message"]["content"]

    # Limpar markdown se tiver
    content = content.strip()
    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:])
    if content.endswith("```"):
        content = "\n".join(content.split("\n")[:-1])

    try:
        plan = json.loads(content)
        return plan
    except json.JSONDecodeError:
        print(f"GPT retornou JSON inválido, usando plano padrão")
        return _default_edit_plan(silences, duration, zoom_intensity)


def _default_edit_plan(silences: list, duration: float, zoom_intensity: float) -> dict:
    """Plano de edição padrão se a IA falhar."""
    cuts = [s for s in silences if (s["end"] - s["start"]) > 0.5]
    zooms = []
    # Zoom in a cada 3 segundos, alternando
    t = 0.5
    zoom_in = True
    while t < duration - 1:
        if zoom_in:
            zooms.append({"start": t, "end": t + 1.0, "scale": zoom_intensity, "type": "in"})
        else:
            zooms.append({"start": t, "end": t + 0.5, "scale": 1.0, "type": "out"})
        t += 3.0
        zoom_in = not zoom_in
    return {"cuts": cuts, "zooms": zooms}


# ─── FFmpeg Edit Execution ────────────────────────────────────────

def _apply_edits(
    input_path: str,
    output_path: str,
    cuts: list,
    zooms: list,
    video_info: dict,
) -> None:
    """Aplica cortes de silêncio e zooms via FFmpeg."""

    duration = video_info["duration"]
    w = video_info["width"]
    h = video_info["height"]

    # --- Passo 1: Cortar silêncios ---
    if cuts:
        # Calcular segmentos de fala (inverso dos cortes)
        speech_segments = _get_speech_segments(cuts, duration)
        cut_path = output_path.replace(".mp4", "_cut.mp4")

        # Gerar filtro select pra manter apenas os trechos de fala
        select_parts = []
        aselect_parts = []
        for seg in speech_segments:
            select_parts.append(f"between(t,{seg['start']},{seg['end']})")
            aselect_parts.append(f"between(t,{seg['start']},{seg['end']})")

        v_select = "+".join(select_parts)
        a_select = "+".join(aselect_parts)

        cmd_cut = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"select='{v_select}',setpts=N/FRAME_RATE/TB",
            "-af", f"aselect='{a_select}',asetpts=N/SR/TB",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            cut_path,
        ]

        proc = subprocess.run(cmd_cut, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"Corte falhou: {proc.stderr[-300:]}")
            cut_path = input_path  # Fallback: usar original
    else:
        cut_path = input_path

    # --- Passo 2: Aplicar zooms ---
    source_for_zoom = cut_path if cut_path != input_path else input_path

    if zooms:
        # Gerar expressão de zoom dinâmico
        # Scale up e crop centrado para simular zoom
        zoom_expr_parts = []
        for z in zooms:
            s, e, scale = z["start"], z["end"], z["scale"]
            if z["type"] == "in":
                # Zoom in: de 1.0 até scale
                zoom_expr_parts.append(
                    f"if(between(t,{s},{e}),1+({scale}-1)*(t-{s})/({e}-{s})"
                )
            else:
                # Zoom out: de scale até 1.0
                zoom_expr_parts.append(
                    f"if(between(t,{s},{e}),{scale}-({scale}-1)*(t-{s})/({e}-{s})"
                )

        # Construir expressão completa de zoom
        # Cada zoom fecha com ",1)" — fallback pra 1.0
        zoom_expr = "1"
        for z in zooms:
            s, e, scale = z["start"], z["end"], z["scale"]
            if z["type"] == "in":
                zoom_expr = f"if(between(t\\,{s}\\,{e})\\,1+({scale}-1)*(t-{s})/({e}-{s})\\,{zoom_expr})"
            else:
                zoom_expr = f"if(between(t\\,{s}\\,{e})\\,{scale}-({scale}-1)*(t-{s})/({e}-{s})\\,{zoom_expr})"

        # Aplicar zoom via scale + crop centralizado
        vf = (
            f"scale=iw*({zoom_expr}):ih*({zoom_expr}),"
            f"crop={w}:{h}"
        )

        cmd_zoom = [
            "ffmpeg", "-y", "-i", source_for_zoom,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            output_path,
        ]

        proc = subprocess.run(cmd_zoom, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"Zoom falhou: {proc.stderr[-300:]}")
            # Fallback: copiar sem zoom
            if source_for_zoom != output_path:
                shutil.copy2(source_for_zoom, output_path)
    else:
        # Sem zooms, copiar do passo anterior
        if source_for_zoom != output_path:
            shutil.copy2(source_for_zoom, output_path)

    # Limpar intermediário
    cut_temp = output_path.replace(".mp4", "_cut.mp4")
    if Path(cut_temp).exists() and cut_temp != output_path:
        Path(cut_temp).unlink()


def _get_speech_segments(cuts: list, duration: float) -> list:
    """Calcula segmentos de fala (inverso dos cortes de silêncio)."""
    if not cuts:
        return [{"start": 0, "end": duration}]

    sorted_cuts = sorted(cuts, key=lambda c: c["start"])
    segments = []
    current = 0.0

    for cut in sorted_cuts:
        if cut["start"] > current + 0.05:
            segments.append({"start": round(current, 3), "end": round(cut["start"], 3)})
        current = cut["end"]

    if current < duration - 0.05:
        segments.append({"start": round(current, 3), "end": round(duration, 3)})

    return segments


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
