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
        "splitters": [
            {
                "type": "limit_by_chars",
                "min_limit": 10,
                "max_limit": 20,
            }
        ],
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
    """Tenta processar via API Python primeiro, fallback para CLI."""

    # --- Tentativa 1: API Python (suporta config dinâmico) ---
    try:
        print(f"[{job_id}] Processando via API Python...")
        from pycaps import CapsPipelineBuilder

        config_path = Path(job_dir) / "pycaps.template.json"
        css_path = Path(job_dir) / "style.css"

        builder = CapsPipelineBuilder()
        builder.with_input_video(input_path)

        # Carregar CSS gerado
        if css_path.exists():
            css_content = css_path.read_text()
            builder.add_css_content(css_content)

        # Carregar config para layout
        if config_path.exists():
            config = json.loads(config_path.read_text())

            # Aplicar layout se o builder suportar
            layout = config.get("layout", {})
            if hasattr(builder, "with_layout"):
                builder.with_layout(layout)

            # Aplicar splitters
            splitters = config.get("splitters", [])
            for sp in splitters:
                if sp.get("type") == "limit_by_chars":
                    try:
                        from pycaps import LimitByCharsSplitter
                        builder.add_segment_splitter(
                            LimitByCharsSplitter(
                                min_limit=sp.get("min_limit", 10),
                                max_limit=sp.get("max_limit", 20),
                            )
                        )
                    except ImportError:
                        pass

        pipeline = builder.build()
        pipeline.run()

        if os.path.exists(output_path):
            return {"success": True}

        # pycaps pode gerar o output com nome diferente
        return {"success": True, "note": "checking output location"}

    except Exception as e:
        error_api = f"{type(e).__name__}: {str(e)}"
        print(f"[{job_id}] API Python falhou: {error_api}")
        traceback.print_exc()

    # --- Tentativa 2: CLI com template local ---
    try:
        pycaps_bin = shutil.which("pycaps")
        if not pycaps_bin:
            return {"success": False, "error": f"CLI não encontrado e API falhou: {error_api}"}

        config_path = Path(job_dir) / "pycaps.template.json"

        # Se temos config local, usar como template
        if config_path.exists():
            cmd = [
                pycaps_bin, "render",
                "--input", input_path,
                "--output", output_path,
                "--template", job_dir,  # diretório com pycaps.template.json + style.css
            ]
        else:
            cmd = [
                pycaps_bin, "render",
                "--input", input_path,
                "--output", output_path,
                "--template", template,
            ]

        print(f"[{job_id}] Executando CLI: {' '.join(cmd)}")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=job_dir,
        )

        if proc.stdout:
            print(f"[{job_id}] stdout: {proc.stdout[-500:]}")
        if proc.stderr:
            print(f"[{job_id}] stderr: {proc.stderr[-500:]}")

        if proc.returncode == 0:
            return {"success": True}
        else:
            return {
                "success": False,
                "error": f"CLI exit {proc.returncode}: {proc.stderr[-300:] if proc.stderr else 'sem output'}",
            }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout: processamento excedeu 10 minutos"}
    except Exception as e:
        return {"success": False, "error": f"CLI: {str(e)}"}


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
    import subprocess
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
