"""
Zero-Shot Singing Voice Conversion - REST API Server

Provides HTTP endpoints for programmatic voice conversion.

Endpoints:
    POST /api/convert          - Convert single audio file
    POST /api/batch-convert    - Batch convert multiple files
    GET  /api/similarity       - Compute voice similarity
    POST /api/profiles         - Save speaker profile
    GET  /api/profiles         - List saved profiles
    GET  /api/health           - Health check
    GET  /api/status           - System status (models loaded, device, etc.)

Launch:
    python api_server.py                    # Default: http://0.0.0.0:8000
    python api_server.py --port 9000        # Custom port
    python api_server.py --host 127.0.0.1   # Local only
"""
import os
import sys
import json
import time
import uuid
import tempfile
import argparse
from typing import Optional, Dict, Any, List
from functools import lru_cache

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn


# ── FastAPI App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="Zero-Shot SVC API",
    description="REST API for Zero-Shot Singing Voice Conversion",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global State ──────────────────────────────────────────────────────────

class AppState:
    """Global application state."""
    
    def __init__(self):
        self.svc_instances = {}
        self.output_cache = {}
        self.start_time = time.time()
        
    def get_svc(self, use_neural=False, device=None):
        """Get or create cached SVC instance."""
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        key = (use_neural, device)
        if key not in self.svc_instances:
            from pipeline.voice_converter import ZeroShotSVC
            self.svc_instances[key] = ZeroShotSVC(
                device=device,
                use_neural=use_neural
            )
        return self.svc_instances[key]


state = AppState()


# ── Data Models ───────────────────────────────────────────────────────────

class ConvertResponse(BaseModel):
    success: bool
    output_id: str
    output_path: str
    message: str
    duration_s: float = 0.0
    mode: str = "signal"
    device: str = "cpu"
    metrics: Dict[str, Any] = {}


class SimilarityRequest(BaseModel):
    audio1_path: str
    audio2_path: str


class SimilarityResponse(BaseModel):
    overall_similarity: float
    mfcc_cosine_similarity: float
    spectral_centroid_correlation: float
    rms_correlation: float


class ProfileResponse(BaseModel):
    name: str
    created_at: str
    duration: float
    f0_mean: Optional[float] = None
    spectral_centroid_mean: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    uptime_s: float
    version: str
    device: str
    models_loaded: Dict[str, bool]


class StatusResponse(BaseModel):
    status: str
    uptime_s: float
    active_conversions: int
    cached_outputs: int
    svc_instances: int
    system_info: Dict[str, Any]


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import torch
    
    return HealthResponse(
        status="healthy",
        uptime_s=time.time() - state.start_time,
        version="2.0.0",
        device="cuda" if torch.cuda.is_available() else "cpu",
        models_loaded={
            "signal": True,
            "neural": any(k[0] for k in state.svc_instances.keys() if k[0])
        }
    )


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get detailed system status."""
    import torch
    import psutil
    
    process = psutil.Process(os.getpid())
    
    return StatusResponse(
        status="running",
        uptime_s=time.time() - state.start_time,
        active_conversions=0,
        cached_outputs=len(state.output_cache),
        svc_instances=len(state.svc_instances),
        system_info={
            "cpu_percent": psutil.cpu_percent(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "gpu_available": torch.cuda.is_available(),
            "python_version": sys.version.split()[0],
        }
    )


@app.post("/api/convert", response_model=ConvertResponse)
async def convert_audio(
    source: UploadFile = File(..., description="Source audio (singing to convert)"),
    reference: UploadFile = File(..., description="Reference audio (target voice)"),
    f0_transpose: int = Form(default=0, ge=-24, le=24),
    f0_curve_factor: float = Form(default=1.0, ge=0.5, le=2.0),
    noise_scale: float = Form(default=0.4, ge=0.05, le=1.0),
    use_neural: bool = Form(default=False),
    formant_shift: int = Form(default=0, ge=-6, le=6),
    noise_reduction: float = Form(default=0.0, ge=0.0, le=1.0),
    breathiness: float = Form(default=0.0, ge=0.0, le=1.0),
    protect_consonants: bool = Form(default=True),
):
    """
    Convert singing voice from source to target speaker.
    
    Returns output_id for downloading the converted audio.
    """
    import torch
    from utils.audio import load_audio
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_id = str(uuid.uuid4())[:8]
    
    tmp_dir = tempfile.mkdtemp(prefix=f"zsvc_api_{output_id}_")
    
    try:
        source_path = os.path.join(tmp_dir, f"source_{source.filename}")
        reference_path = os.path.join(tmp_dir, f"reference_{reference.filename}")
        
        with open(source_path, 'wb') as f:
            f.write(await source.read())
        with open(reference_path, 'wb') as f:
            f.write(await reference.read())
        
        if os.path.getsize(source_path) == 0:
            raise HTTPException(status_code=400, detail="Source file is empty")
        if os.path.getsize(reference_path) == 0:
            raise HTTPException(status_code=400, detail="Reference file is empty")
        
        svc = state.get_svc(use_neural=use_neural, device=device)
        
        output_path = os.path.join(tmp_dir, f"converted_{output_id}.wav")
        
        t0 = time.time()
        svc.convert(
            source_path=source_path,
            reference_path=reference_path,
            output_path=output_path,
            f0_transpose=f0_transpose,
            f0_curve_factor=f0_curve_factor,
            noise_scale=noise_scale,
            formant_shift=formant_shift,
            noise_reduction=noise_reduction,
            breathiness=breathiness,
            protect_consonants=protect_consonants,
        )
        elapsed = time.time() - t0
        
        import soundfile as sf
        out_audio, out_sr = sf.read(output_path)
        duration_s = len(out_audio) / out_sr
        
        state.output_cache[output_id] = {
            "path": output_path,
            "created_at": time.time(),
            "tmp_dir": tmp_dir,
        }
        
        return ConvertResponse(
            success=True,
            output_id=output_id,
            output_path=output_path,
            message=f"Conversion completed in {elapsed:.2f}s",
            duration_s=duration_s,
            mode="neural" if use_neural else "signal",
            device=device,
            metrics={
                "duration_s": round(duration_s, 2),
                "sample_rate": out_sr,
                "processing_time_s": round(elapsed, 2),
            }
        )
        
    except Exception as e:
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch-convert")
async def batch_convert_audio(
    sources: List[UploadFile] = File(..., description="Multiple source audio files"),
    reference: UploadFile = File(..., description="Reference audio (target voice)"),
    f0_transpose: int = Form(default=0, ge=-24, le=24),
    noise_scale: float = Form(default=0.4, ge=0.05, le=1.0),
    use_neural: bool = Form(default=False),
):
    """Batch convert multiple source files using the same reference."""
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_id = str(uuid.uuid4())[:8]
    tmp_dir = tempfile.mkdtemp(prefix=f"zsvc_batch_{batch_id}_")
    
    results = []
    
    try:
        ref_path = os.path.join(tmp_dir, "reference.wav")
        with open(ref_path, 'wb') as f:
            f.write(await reference.read())
        
        for idx, source in enumerate(sources):
            src_path = os.path.join(tmpdir, f"source_{idx}_{source.filename}")
            output_id = f"{batch_id}_{idx}"
            
            with open(src_path, 'wb') as f:
                f.write(await source.read())
            
            try:
                svc = state.get_svc(use_neural=use_neural, device=device)
                output_path = os.path.join(tmp_dir, f"converted_{output_id}.wav")
                
                t0 = time.time()
                svc.convert(
                    source_path=src_path,
                    reference_path=ref_path,
                    output_path=output_path,
                    f0_transpose=f0_transpose,
                    noise_scale=noise_scale,
                )
                
                state.output_cache[output_id] = {
                    "path": output_path,
                    "created_at": time.time(),
                    "tmp_dir": tmp_dir,
                }
                
                results.append({
                    "filename": source.filename,
                    "output_id": output_id,
                    "success": True,
                    "time_s": round(time.time() - t0, 2),
                })
            except Exception as e:
                results.append({
                    "filename": source.filename,
                    "output_id": None,
                    "success": False,
                    "error": str(e),
                })
        
        return JSONResponse({
            "batch_id": batch_id,
            "total": len(sources),
            "successful": sum(1 for r in results if r["success"]),
            "results": results,
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{output_id}")
async def download_output(output_id: str):
    """Download converted audio by output_id."""
    if output_id not in state.output_cache:
        raise HTTPException(status_code=404, detail="Output not found or expired")
    
    cache_entry = state.output_cache[output_id]
    path = cache_entry["path"]
    
    if not os.path.exists(path):
        del state.output_cache[output_id]
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=path,
        media_type="audio/wav",
        filename=f"converted_{output_id}.wav"
    )


@app.post("/api/similarity", response_model=SimilarityResponse)
async def compute_similarity(
    audio1: UploadFile = File(...),
    audio2: UploadFile = File(...),
    use_neural: bool = Form(default=False),
):
    """Compute voice similarity between two audio files."""
    import torch
    
    tmp_dir = tempfile.mkdtemp(prefix="zsvc_sim_")
    
    try:
        path1 = os.path.join(tmp_dir, f"audio1_{audio1.filename}")
        path2 = os.path.join(tmpdir, f"audio2_{audio2.filename}")
        
        with open(path1, 'wb') as f:
            f.write(await audio1.read())
        with open(path2, 'wb') as f:
            f.write(await audio2.read())
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        svc = state.get_svc(use_neural=use_neural, device=device)
        
        similarity = svc.compute_similarity(path1, path2)
        
        return SimilarityResponse(**similarity)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/profiles")
async def save_profile(
    reference: UploadFile = File(..., description="Reference audio for profiling"),
    name: str = Form(default=None, description="Profile name"),
    use_neural: bool = Form(default=False),
):
    """Save a speaker profile from reference audio."""
    import torch
    
    tmp_dir = tempfile.mkdtemp(prefix="zsvc_profile_")
    
    try:
        ref_path = os.path.join(tmp_dir, f"reference_{reference.filename}")
        with open(ref_path, 'wb') as f:
            f.write(await reference.read())
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        svc = state.get_svc(use_neural=use_neural, device=device)
        
        profile_path = svc.save_speaker_profile(ref_path, name=name or reference.filename)
        
        with open(profile_path) as f:
            profile = json.load(f)
        
        return {"success": True, "profile": profile}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/api/profiles")
async def list_profiles():
    """List all saved speaker profiles."""
    import torch
    
    svc = state.get_svc(use_neural=False)
    profiles = svc.list_speaker_profiles()
    
    return {"profiles": profiles, "count": len(profiles)}


@app.delete("/api/cache")
async def clear_cache():
    """Clear expired outputs from cache."""
    import shutil
    
    max_age = 3600
    cleared = 0
    
    for output_id, entry in list(state.output_cache.items()):
        age = time.time() - entry["created_at"]
        if age > max_age:
            tmp_dir = entry.get("tmp_dir")
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            del state.output_cache[output_id]
            cleared += 1
    
    return {"cleared": cleared, "remaining": len(state.output_cache)}


# ── Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot SVC REST API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to listen on (default: 8000)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (default: 1)")
    args = parser.parse_args()

    print("=" * 60)
    print("Zero-Shot SVC REST API Server")
    print("=" * 60)
    print(f"\nServer starting at http://{args.host}:{args.port}")
    print("\nAvailable endpoints:")
    print("  POST /api/convert       - Convert audio")
    print("  POST /api/batch-convert - Batch convert")
    print("  GET  /api/download/{id} - Download result")
    print("  POST /api/similarity    - Voice similarity")
    print("  POST /api/profiles      - Save speaker profile")
    print("  GET  /api/profiles      - List profiles")
    print("  GET  /api/health         - Health check")
    print("  GET  /api/status         - System status")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
