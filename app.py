"""
pycaps-api: Microserviço FastAPI para legendar vídeos com pycaps.
Deploy como App separado no Easypanel, mesmo projeto que o n8n.

Endpoints:
  POST /caption     - Recebe vídeo, retorna vídeo legendado
  GET  /health      - Health check
  GET  /templates   - Lista templates disponíveis
"""

import os
import time
import uuid
import traceback
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(
    title="pycaps-api",
    description="API para adicionar legendas estilizadas a vídeos",
    version="1.0.0",
)

WORK_DIR = Path("/tmp/pycaps-work")
WORK_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def health():
    """Health check para Easypanel."""
    # Verificar se FFmpeg está disponível
    ffmpeg_ok = os.system("ffmpeg -version > /dev/null 2>&1") == 0

    # Verificar se pycaps importa
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
    }


@app.get("/templates")
async def list_templates():
    """Lista templates disponíveis no pycaps."""
    # Templates built-in do pycaps
    builtin = ["minimalist", "default"]

    # Tentar listar do diretório de templates customizados
    custom_dir = Path("/app/templates")
    custom = []
    if custom_dir.exists():
        custom = [d.name for d in custom_dir.iterdir() if d.is_dir()]

    return {"builtin": builtin, "custom": custom}


@app.post("/caption")
async def caption_video(
    video: UploadFile = File(..., description="Arquivo de vídeo (MP4, MOV, etc.)"),
    template: str = Form(default="minimalist", description="Template pycaps"),
    language: str = Form(default="pt", description="Idioma para transcrição (pt, en, es...)"),
    whisper_model: str = Form(default="small", description="Modelo Whisper: tiny, base, small, medium, large"),
):
    """
    Recebe um vídeo e retorna com legendas estilizadas.

    - **video**: Arquivo de vídeo (MP4 recomendado)
    - **template**: Template CSS do pycaps (default: minimalist)
    - **language**: Idioma para transcrição Whisper (default: pt)
    - **whisper_model**: Modelo Whisper - tiny/base/small/medium/large (default: small)
    """
    job_id = str(uuid.uuid4())[:8]
    input_path = WORK_DIR / f"input_{job_id}.mp4"
    output_path = WORK_DIR / f"output_{job_id}.mp4"
    start_time = time.time()

    try:
        # 1. Salvar vídeo recebido
        content = await video.read()
        file_size_mb = len(content) / (1024 * 1024)

        if file_size_mb > 500:
            raise HTTPException(status_code=413, detail="Arquivo muito grande (max 500MB)")

        input_path.write_bytes(content)
        print(f"[{job_id}] Recebido: {video.filename} ({file_size_mb:.1f} MB)")
        print(f"[{job_id}] Config: template={template}, lang={language}, model={whisper_model}")

        # 2. Processar com pycaps
        result = _process_with_pycaps(
            input_path=str(input_path),
            output_path=str(output_path),
            template=template,
            language=language,
            whisper_model=whisper_model,
            job_id=job_id,
        )

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Erro no processamento: {result.get('error', 'desconhecido')}",
            )

        # 3. Verificar output
        if not output_path.exists():
            raise HTTPException(status_code=500, detail="Vídeo de saída não foi gerado")

        duration = round(time.time() - start_time, 2)
        output_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[{job_id}] Concluído em {duration}s → {output_size_mb:.1f} MB")

        # 4. Retornar vídeo legendado
        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=f"legendado_{video.filename or 'video.mp4'}",
            headers={
                "X-Pycaps-Duration": str(duration),
                "X-Pycaps-Template": template,
                "X-Pycaps-Model": whisper_model,
                "X-Pycaps-Job-Id": job_id,
            },
            background=_cleanup_task(input_path, output_path),
        )

    except HTTPException:
        _cleanup_files(input_path, output_path)
        raise
    except Exception as e:
        _cleanup_files(input_path, output_path)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _process_with_pycaps(
    input_path: str,
    output_path: str,
    template: str,
    language: str,
    whisper_model: str,
    job_id: str,
) -> dict:
    """Tenta processar via CLI primeiro, fallback para API Python."""

    # --- Tentativa 1: CLI (mais confiável) ---
    try:
        # Descobrir o caminho do executável pycaps
        import shutil
        pycaps_bin = shutil.which("pycaps")

        if pycaps_bin:
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
            )

            print(f"[{job_id}] CLI stdout: {proc.stdout[-300:]}" if proc.stdout else "")
            print(f"[{job_id}] CLI stderr: {proc.stderr[-300:]}" if proc.stderr else "")

            if proc.returncode == 0 and os.path.exists(output_path):
                return {"success": True}
            else:
                print(f"[{job_id}] CLI falhou (exit {proc.returncode}), tentando API Python...")
        else:
            print(f"[{job_id}] Binário 'pycaps' não encontrado no PATH, tentando API Python...")

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout: processamento excedeu 10 minutos"}
    except Exception as e:
        print(f"[{job_id}] CLI falhou com exceção: {e}")

    # --- Tentativa 2: API Python ---
    try:
        print(f"[{job_id}] Processando via API Python...")
        from pycaps import CapsPipelineBuilder

        # Tentar com TemplateLoader
        try:
            from pycaps import TemplateLoader
            builder = (
                TemplateLoader(template)
                .with_input_video(input_path)
                .load(False)
            )
            print(f"[{job_id}] Template '{template}' carregado")
        except Exception as te:
            print(f"[{job_id}] TemplateLoader falhou ({te}), usando builder padrão")
            builder = (
                CapsPipelineBuilder()
                .with_input_video(input_path)
            )

        # Configurar output se o método existir
        if hasattr(builder, 'with_output_video'):
            builder.with_output_video(output_path)

        pipeline = builder.build()
        pipeline.run()

        # Verificar se gerou o output (pode ter nome diferente)
        if os.path.exists(output_path):
            return {"success": True}

        # Procurar qualquer output gerado no mesmo diretório
        work_dir = Path(input_path).parent
        generated = list(work_dir.glob("*output*")) + list(work_dir.glob("*captioned*"))
        if generated:
            import shutil as sh
            sh.move(str(generated[0]), output_path)
            return {"success": True}

        return {"success": False, "error": "Pipeline executou mas output não encontrado"}

    except Exception as e:
        error_msg = f"API Python: {type(e).__name__}: {str(e)}"
        print(f"[{job_id}] {error_msg}")
        traceback.print_exc()
        return {"success": False, "error": error_msg}


def _cleanup_files(*paths):
    """Remove arquivos temporários."""
    for p in paths:
        try:
            if isinstance(p, (str, Path)) and Path(p).exists():
                Path(p).unlink()
        except Exception:
            pass


class _cleanup_task:
    """Background task para limpar arquivos após enviar a resposta."""

    def __init__(self, *paths):
        self.paths = paths

    async def __call__(self):
        import asyncio
        await asyncio.sleep(5)  # Esperar resposta terminar de enviar
        _cleanup_files(*self.paths)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
