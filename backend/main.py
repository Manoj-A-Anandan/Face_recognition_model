import os
os.environ["INSIGHTFACE_DISABLE_FACE3D"] = "1"

from dotenv import load_dotenv
load_dotenv()

import io
import time
import json
import numpy as np
import cv2
import asyncio
import zipfile
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "event-photos")

ENCODINGS_OBJECT = "encodings.npy"
FILENAMES_OBJECT = "filenames.json"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Set SUPABASE_URL and SUPABASE_KEY in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Wedding Face Matcher")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CV Setup ----------------
FACE_APP = FaceAnalysis(name="buffalo_l")
FACE_APP.prepare(ctx_id=-1, det_size=(640, 640)) 

# ---------------- Utilities ----------------
def image_bytes_to_rgb_array(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def l2_normalize(vecs: np.ndarray):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

def upload_object(bucket: str, path: str, data: bytes, content_type: str = "application/octet-stream"):
    try:
        supabase.storage.from_(bucket).upload(path=path, file=data, file_options={"content-type": content_type, "upsert": "true"})
    except Exception as e:
        print(f"Upload failed for {path}: {e}")

def download_object(bucket: str, path: str) -> bytes:
    try:
        return supabase.storage.from_(bucket).download(path)
    except Exception:
        return None

def list_bucket(bucket: str, prefix: str = ""):
    try:
        res = supabase.storage.from_(bucket).list(prefix)
        return [r["name"] for r in res]
    except Exception:
        return []

def create_signed_url(bucket: str, path: str, expires_in: int = 3600):
    try:
        res = supabase.storage.from_(bucket).create_signed_url(path, expires_in)
        return res.get("signedURL") if isinstance(res, dict) else None
    except Exception:
        return None

# ============================================================
# STRICT UPLOAD (No Duplicates, Fast Memory Processing)
# ============================================================
@app.post("/upload-folder")
async def upload_folder(files: list[UploadFile] = File(...), event_name: str = Form(...)):
    if not files: raise HTTPException(400, "No files sent.")
    
    # Clean Name
    event_name = event_name.strip()
    if not event_name: raise HTTPException(400, "Name cannot be empty.")

    event_prefix = f"event_{event_name}/"
    
    # --- STRICT CHECK ---
    existing = list_bucket(SUPABASE_BUCKET, prefix=event_prefix)
    if existing: 
        raise HTTPException(400, f"Name taken.")
    
    raw_prefix = event_prefix + "raw/"
    allowed_ext = {"jpg","jpeg","png","bmp","webp"}
    
    embeddings = []
    filenames = []
    
    # Fast Worker: Process RAM -> Upload Background
    async def process_and_upload(file: UploadFile):
        try:
            file_bytes = await file.read()
            ext = file.filename.split(".")[-1].lower()
            if ext not in allowed_ext: return None

            rgb = image_bytes_to_rgb_array(file_bytes)
            if rgb is None: return None
            
            loop = asyncio.get_event_loop()
            # Heavy Math on Thread
            faces = await loop.run_in_executor(None, lambda: FACE_APP.get(rgb))
            
            if not faces: return None

            # Upload in background
            obj_path = raw_prefix + file.filename.replace(" ", "_")
            ctype = file.content_type or "image/jpeg"
            await loop.run_in_executor(None, lambda: upload_object(SUPABASE_BUCKET, obj_path, file_bytes, ctype))

            file_embeddings = []
            for face in faces:
                emb = np.array(face.embedding, dtype=np.float32)
                file_embeddings.append(emb)
            
            return (obj_path, file_embeddings)

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            return None

    # Parallel Execution
    tasks = [process_and_upload(f) for f in files]
    results = await asyncio.gather(*tasks)

    for res in results:
        if res:
            obj_path, embs = res
            for emb in embs:
                embeddings.append(emb)
                filenames.append(obj_path)

    if not embeddings: raise HTTPException(400, "Zero faces found.")

    # Save Vector Data
    enc_array = l2_normalize(np.vstack(embeddings).astype(np.float32))
    enc_bytes = io.BytesIO()
    np.save(enc_bytes, enc_array, allow_pickle=False)
    
    upload_object(SUPABASE_BUCKET, event_prefix + ENCODINGS_OBJECT, enc_bytes.getvalue())
    upload_object(SUPABASE_BUCKET, event_prefix + FILENAMES_OBJECT, json.dumps(filenames).encode("utf-8"), content_type="application/json")

    return {"status":"ok", "event_prefix":event_prefix, "images_uploaded":len(results), "faces_found":len(embeddings)}

# ============================================================
@app.get("/list-events")
def list_events():
    try:
        items = list_bucket(SUPABASE_BUCKET)
        events = set()
        for item in items:
            if item.startswith("event_"):
                events.add(item)
        return sorted(list(events))
    except Exception: return []

# ============================================================
@app.get("/check-event-name")
def check_event_name(name: str):
    try:
        items = list_bucket(SUPABASE_BUCKET, prefix=f"event_{name}/")
        return {"exists": len(items)>0}
    except Exception:
        return {"exists": False}

# ============================================================
@app.post("/match-against-event")
async def match_against_event(event_prefix: str = Form(...), file: UploadFile = File(...), threshold: float = Form(0.45), top_k: int = Form(200)):
    if not event_prefix.endswith("/"): event_prefix += "/"
    
    enc_blob = download_object(SUPABASE_BUCKET, event_prefix + ENCODINGS_OBJECT)
    fn_blob = download_object(SUPABASE_BUCKET, event_prefix + FILENAMES_OBJECT)
    
    if enc_blob is None or fn_blob is None: 
        raise HTTPException(404, "encodings not found")

    enc_array = np.load(io.BytesIO(enc_blob))
    filenames = json.loads(fn_blob.decode("utf-8"))

    file_bytes = await file.read()
    rgb = image_bytes_to_rgb_array(file_bytes)
    if rgb is None: raise HTTPException(400,"Invalid query image")

    faces = FACE_APP.get(rgb)
    if not faces: raise HTTPException(400,"No face found")

    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
    face = faces[0]

    q_emb = np.array(face.embedding, dtype=np.float32)
    q_emb = q_emb / np.linalg.norm(q_emb)
    
    dists = np.linalg.norm(enc_array - q_emb, axis=1)
    scores = 1.0/(1.0+dists)
    order = np.argsort(dists)[:top_k]

    results = []
    seen = set()
    
    for idx in order:
        score = float(scores[idx])
        if score >= threshold:
            obj = filenames[idx]
            if obj in seen: continue
            seen.add(obj)
            signed = create_signed_url(SUPABASE_BUCKET, obj)
            results.append({"object_name":obj,"distance":float(dists[idx]),"score":score,"signed_url":signed})

    return {"matches": results}

# ============================================================
# PARALLEL DELETE
# ============================================================
class DeleteRequest(BaseModel):
    event_prefixes: list[str]

@app.post("/delete-events")
async def delete_events(req: DeleteRequest):
    async def delete_single_event(prefix):
        try:
            folder_path = prefix.strip("/") + "/" 
            # Delete raw images loop
            raw_folder = f"{folder_path}raw"
            while True:
                raw_files = supabase.storage.from_(SUPABASE_BUCKET).list(raw_folder)
                files_to_remove = []
                for item in raw_files:
                    name = item.get("name")
                    if name and name != ".emptyFolderPlaceholder":
                        files_to_remove.append(f"{raw_folder}/{name}")
                if not files_to_remove: break
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: supabase.storage.from_(SUPABASE_BUCKET).remove(files_to_remove))
            
            # Delete system files
            root_path = folder_path.rstrip("/")
            root_files = supabase.storage.from_(SUPABASE_BUCKET).list(root_path)
            root_to_remove = []
            for item in root_files:
                name = item.get("name")
                if name and name != ".emptyFolderPlaceholder" and name != "raw": 
                    root_to_remove.append(f"{folder_path}{name}")
            if root_to_remove:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: supabase.storage.from_(SUPABASE_BUCKET).remove(root_to_remove))
            return True
        except Exception as e:
            print(f"Failed to delete {prefix}: {e}")
            return False

    tasks = [delete_single_event(p) for p in req.event_prefixes]
    results = await asyncio.gather(*tasks)
    return {"status": "ok", "deleted": sum(1 for r in results if r)}

# ============================================================
# SERVER ZIP
# ============================================================
class ZipRequest(BaseModel):
    files: list[str]

@app.post("/download-zip")
async def download_zip(req: ZipRequest):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
        for object_path in req.files:
            file_data = download_object(SUPABASE_BUCKET, object_path)
            if file_data:
                filename = object_path.split("/")[-1]
                zf.writestr(filename, file_data)
    zip_buffer.seek(0)
    return StreamingResponse(
        iter([zip_buffer.getvalue()]), 
        media_type="application/zip", 
        headers={"Content-Disposition": "attachment; filename=wedding_matches.zip"}
    )

@app.get("/health")
def health(): return {"status":"ok"}