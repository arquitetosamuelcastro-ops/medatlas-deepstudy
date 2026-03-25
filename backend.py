#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedAtlas DeepStudy — FastAPI Agentic RAG v4
- Upload múltiplo de PDFs com OCR fallback
- BM25 + expansão semântica médica
- Agentic RAG com Anthropic API (LLM real) ou fallback local
- Cadastro/listagem/remoção de fontes
- Chat com memória de sessão via histórico
"""

import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    HAS_TESSERACT = True
    TESSERACT_VERSION = str(pytesseract.get_tesseract_version())
except Exception:
    HAS_TESSERACT = False
    TESSERACT_VERSION = "não instalado"

if sys.platform == "win32" and not HAS_TESSERACT:
    _win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
    ]
    for _p in _win_paths:
        if os.path.exists(_p):
            try:
                import pytesseract as _pt
                _pt.pytesseract.tesseract_cmd = _p
                _pt.get_tesseract_version()
                pytesseract = _pt
                HAS_TESSERACT = True
                TESSERACT_VERSION = str(_pt.get_tesseract_version())
                break
            except Exception:
                pass

OCR_AVAILABLE = HAS_PDFPLUMBER and HAS_PIL and HAS_TESSERACT

SCRIPT_DIR  = Path(__file__).parent.resolve()
CACHE_DIR   = SCRIPT_DIR / "medatlas_cache"
CACHE_DIR.mkdir(exist_ok=True)
INDEX_FILE   = CACHE_DIR / "index.json"
SOURCES_FILE = CACHE_DIR / "sources.json"
PORT = int(os.environ.get("PORT", "8742"))

# ANTHROPIC_API_KEY lida em tempo de execução no runtime

DEFAULT_SOURCES = [
    {"id":"anatpat_unicamp","name":"UNICAMP AnatPat","url":"https://anatpat.unicamp.br/aulas2.html","type":"site","category":"patologia","enabled":True,"notes":"Índice de aulas e assuntos de anatomia patológica."},
    {"id":"radiopaedia","name":"Radiopaedia","url":"https://radiopaedia.org/","type":"site","category":"radiologia","enabled":True,"notes":"Casos, artigos e imagens radiológicas."},
    {"id":"histology_guide","name":"HistologyGuide","url":"https://histologyguide.com/EM-atlas/15-liver-and-gallbladder.html","type":"site","category":"histologia","enabled":True,"notes":"Atlas de histologia e microscopia eletrônica."},
]


# ─── texto ─────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text))
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9\s]", " ", ascii_str.lower())


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).strip()


STOPWORDS = set("""
a ao aos as da das de do dos e em é na nas no nos o os para
pela pelas pelo pelos por que se um uma uns umas
the and for are with from this that have been can will also
do not but or at by an
""".split())


def tokenize(text: str) -> List[str]:
    return [t for t in norm(text).split() if len(t) > 2 and t not in STOPWORDS]


def fix_hyphenation(text: str) -> str:
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def clean_page_text(raw: str) -> str:
    if not raw:
        return ""
    text = fix_hyphenation(raw)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def split_into_chunks(text: str, page: int, source: str,
                      min_len: int = 80, max_words: int = 120, stride: int = 60) -> list:
    chunks = []
    for para in re.split(r"\n{2,}", text):
        p = para.strip()
        if len(p) >= min_len:
            chunks.append({"text": p, "page": page, "source": source, "type": "paragraph"})
    for sent in re.split(r"(?<=[.!?;])\s+", text.replace("\n", " ")):
        s = sent.strip()
        if len(s) >= 100:
            chunks.append({"text": s, "page": page, "source": source, "type": "sentence"})
    words = text.replace("\n", " ").split()
    if len(words) >= 30:
        for i in range(0, len(words), stride):
            window = " ".join(words[i:i+max_words]).strip()
            if len(window) >= min_len:
                chunks.append({"text": window, "page": page, "source": source, "type": "window"})
    seen, deduped = set(), []
    for c in chunks:
        key = norm(c["text"])[:200]
        if key and key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def is_scanned_page(text: str) -> bool:
    return not text or len(text.split()) < 15 or len(text) < 80


def safe_excerpt(text: str, query: str, max_len: int = 520) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= max_len:
        return cleaned
    nq = norm(query)
    nc = norm(cleaned)
    pos = nc.find(nq) if nq else -1
    if pos < 0:
        return cleaned[:max_len].strip() + " ..."
    start = max(0, pos - max_len // 3)
    end = min(len(cleaned), start + max_len)
    excerpt = cleaned[start:end].strip()
    if start > 0:
        excerpt = "... " + excerpt
    if end < len(cleaned):
        excerpt = excerpt + " ..."
    return excerpt


# ─── persistência ─────────────────────────────────────────────────────────────

def load_json(path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text("utf-8"))
        except Exception:
            pass
    return fallback


def save_json(path: Path, data: Dict[str, Any]):
    data["updated"] = int(time.time())
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index() -> Dict[str, Any]:
    return load_json(INDEX_FILE, {"docs": {}, "updated": 0})


def save_index(data: Dict[str, Any]):
    save_json(INDEX_FILE, data)


def load_sources() -> Dict[str, Any]:
    return load_json(SOURCES_FILE, {"sources": DEFAULT_SOURCES[:], "updated": int(time.time())})


def save_sources(data: Dict[str, Any]):
    save_json(SOURCES_FILE, data)


def source_id(name: str, url: str) -> str:
    return hashlib.md5((name.strip() + "|" + url.strip()).encode("utf-8")).hexdigest()[:12]


# ─── BM25 ──────────────────────────────────────────────────────────────────────

class BM25:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.n = len(corpus)
        self.tokenized = [tokenize(d) for d in corpus]
        self.dl = [len(t) for t in self.tokenized]
        self.avgdl = sum(self.dl) / max(1, self.n)
        self.df: Dict[str, int] = {}
        for doc in self.tokenized:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        return math.log((self.n - n_t + 0.5) / (n_t + 0.5) + 1)

    def score(self, query_terms: List[str], doc_idx: int) -> float:
        doc = self.tokenized[doc_idx]
        dl = self.dl[doc_idx]
        tf_map: Dict[str, int] = {}
        for t in doc:
            tf_map[t] = tf_map.get(t, 0) + 1
        total = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            if tf == 0:
                for k in tf_map:
                    if len(term) >= 5 and (k.startswith(term[:5]) or term.startswith(k[:5])):
                        tf = tf_map[k]
                        break
            if tf == 0:
                continue
            idf = self.idf(term)
            num = tf * (1.5 + 1)
            den = tf + 1.5 * (1 - 0.75 + 0.75 * dl / max(1, self.avgdl))
            total += idf * num / den
        return total

    def rank(self, query: str, top_k: int = 30) -> list:
        q_terms = tokenize(query)
        if not q_terms:
            return []
        scores = [(i, self.score(q_terms, i)) for i in range(self.n)]
        scores = [(i, s) for i, s in scores if s > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


MEDICAL_SYNONYMS = {
    "cirrose": ["fibrose hepatica", "hepatopatia cronica", "cirrose alcoolica", "cirrose viral", "hipertensao portal"],
    "nefrolitiase": ["nefrolitíase", "calculo renal", "pedra no rim", "urolitiase", "urolitíase", "colica renal", "hidronefrose"],
    "glomerulonefrite": ["gn", "sindrome nefritica", "hematuria", "proteinuria"],
    "pancreatite": ["amilase", "lipase", "necrose pancreatica"],
    "apendicite": ["fossa iliaca direita", "fecalito", "peritonite"],
}


def expand_query(query: str) -> List[str]:
    base = tokenize(query)
    nq = norm(query)
    extra = []
    for key, syns in MEDICAL_SYNONYMS.items():
        if key in nq or any(norm(s) in nq for s in syns):
            extra.extend(syns)
    return list(dict.fromkeys(base + [t for s in extra for t in tokenize(s)]))


def page_match_score(page_text: str, query_terms: List[str]) -> float:
    ntext = norm(page_text)
    if not ntext:
        return 0.0
    score = 0.0
    for term in query_terms:
        nt = norm(term)
        if nt and nt in ntext:
            score += 3.0 + min(5, ntext.count(nt))
    density = len([t for t in query_terms if norm(t) in ntext]) / max(1, len(ntext.split()))
    score += density * 40
    return round(score, 3)


def search_chunks(query: str, index: Dict[str, Any], top_k: int = 25) -> List[Dict[str, Any]]:
    all_chunks = []
    for doc_id, doc_data in index.get("docs", {}).items():
        for chunk in doc_data.get("chunks", []):
            all_chunks.append({**chunk, "doc_id": doc_id})
    if not all_chunks:
        return []
    corpus = [c["text"] for c in all_chunks]
    bm25 = BM25(corpus)
    q_expanded = expand_query(query)
    ranked = dict(bm25.rank(query + " " + " ".join(q_expanded), top_k=top_k * 6))
    best: Dict[Any, Any] = {}
    type_bonus = {"paragraph": 1.5, "sentence": 1.0, "window": 0.2}
    for idx, chunk in enumerate(all_chunks):
        bscore = ranked.get(idx, 0)
        if bscore <= 0:
            continue
        page_text = chunk.get("page_text") or chunk.get("text") or ""
        pscore = page_match_score(page_text, q_expanded)
        if pscore <= 0:
            continue
        final = round(bscore + pscore + type_bonus.get(chunk.get("type", "window"), 0), 3)
        key = (chunk["doc_id"], chunk["page"])
        cand = {
            "doc_id": chunk["doc_id"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": final,
            "method": chunk.get("method", "text"),
            "matched_terms": [t for t in q_expanded if norm(t) in norm(page_text)][:10],
            "text": safe_excerpt(page_text, query, max_len=520),
        }
        prev = best.get(key)
        if prev is None or cand["score"] > prev["score"]:
            best[key] = cand
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)[:top_k]


# ─── extração PDF ──────────────────────────────────────────────────────────────

def extract_pdf_sync(file_bytes: bytes, filename: str, do_ocr: bool = True) -> Dict[str, Any]:
    result = {
        "name": filename, "pages": 0, "text_pages": 0, "ocr_pages": 0,
        "failed_pages": 0, "chunks": [], "full_text": "", "page_texts": [],
        "page_stats": [], "char_count": 0, "status": "ok",
        "ocr_available": OCR_AVAILABLE, "ocr_requested": do_ocr
    }
    if not HAS_PDFPLUMBER:
        result["status"] = "error: pdfplumber não instalado"
        result["error"] = "Instale: pip install pdfplumber"
        return result
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        result["pages"] = len(pdf.pages)
        all_text_parts = []
        for i, page in enumerate(pdf.pages):
            pnum = i + 1
            stat = {"page": pnum, "method": "text", "chars": 0, "chunks": 0, "error": None}
            try:
                try:
                    raw = page.extract_text(x_tolerance=3, y_tolerance=3, layout=True, x_density=7.25, y_density=13) or ""
                except TypeError:
                    raw = page.extract_text() or ""
                cleaned = clean_page_text(raw)
                use_ocr = do_ocr and OCR_AVAILABLE and is_scanned_page(cleaned)
                if use_ocr:
                    try:
                        img = page.to_image(resolution=180).original
                        gray = img.convert("L")
                        ocr_text = ""
                        for lang in ["por+eng", "por", "eng"]:
                            try:
                                ocr_text = pytesseract.image_to_string(
                                    gray, lang=lang, config="--psm 6 --oem 3"
                                )
                                if len(ocr_text.split()) > 10:
                                    break
                            except Exception:
                                continue
                        ocr_clean = clean_page_text(ocr_text)
                        if len(ocr_clean) > len(cleaned):
                            cleaned = ocr_clean
                            stat["method"] = "ocr"
                            result["ocr_pages"] += 1
                    except Exception as e:
                        stat["error"] = f"OCR: {str(e)[:120]}"
                if cleaned and len(cleaned.split()) >= 5:
                    result["text_pages"] += 1
                if cleaned and len(cleaned) >= 40:
                    page_chunks = split_into_chunks(cleaned, pnum, filename)
                    for ch in page_chunks:
                        ch["page_text"] = cleaned
                        ch["method"] = stat["method"]
                    result["chunks"].extend(page_chunks)
                    result["page_texts"].append({"page": pnum, "text": cleaned, "method": stat["method"]})
                    stat["chunks"] = len(page_chunks)
                    all_text_parts.append(f"[Pág.{pnum}]\n{cleaned}")
                stat["chars"] = len(cleaned)
            except Exception as e:
                stat["method"] = "failed"
                stat["error"] = str(e)[:200]
                result["failed_pages"] += 1
            result["page_stats"].append(stat)
        result["full_text"] = "\n\n".join(all_text_parts)
        result["char_count"] = len(result["full_text"])
        if result["char_count"] == 0:
            result["status"] = "warning: nenhum texto extraído"
    return result


# ─── runtime ───────────────────────────────────────────────────────────────────

from agentic_rag_runtime_v3 import AgenticStudyRuntime  # noqa: E402

runtime = AgenticStudyRuntime(
    search_chunks_fn=search_chunks,
    load_index_fn=load_index,
    load_sources_fn=load_sources,
)

# ─── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="MedAtlas DeepStudy Agentic RAG v4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(SCRIPT_DIR)), name="static")


@app.on_event("startup")
async def startup_event():
    save_sources(load_sources())


@app.get("/", response_class=HTMLResponse)
async def home():
    index_file = SCRIPT_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>index.html não encontrado</h1>", status_code=404)
    return HTMLResponse(index_file.read_text(encoding="utf-8", errors="ignore"))


@app.get("/api/status")
async def api_status():
    idx = load_index()
    src = load_sources()
    return {
        "ok": True,
        "server": "MedAtlas DeepStudy FastAPI Agentic RAG v4",
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "pdfplumber": HAS_PDFPLUMBER,
        "ocr_available": OCR_AVAILABLE,
        "tesseract": TESSERACT_VERSION,
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
        "docs": len(idx.get("docs", {})),
        "sources": len(src.get("sources", [])),
        "cache": str(CACHE_DIR),
    }


@app.get("/api/index")
async def api_index():
    idx = load_index()
    summary = {}
    for doc_id, doc in idx.get("docs", {}).items():
        summary[doc_id] = {
            "name": doc.get("name"),
            "pages": doc.get("pages"),
            "text_pages": doc.get("text_pages"),
            "ocr_pages": doc.get("ocr_pages"),
            "failed_pages": doc.get("failed_pages"),
            "chunk_count": len(doc.get("chunks", [])),
            "char_count": doc.get("char_count"),
            "status": doc.get("status"),
            "uploaded_at": doc.get("uploaded_at"),
            "ocr_available": doc.get("ocr_available"),
            "page_stats": doc.get("page_stats", []),
        }
    return {"docs": summary, "updated": idx.get("updated")}


@app.get("/api/sources")
async def api_sources():
    return load_sources()


@app.post("/api/sources")
async def api_add_source(request: Request):
    body = await request.json()
    name     = (body.get("name") or "").strip()
    url      = (body.get("url") or "").strip()
    stype    = (body.get("type") or "site").strip()
    category = (body.get("category") or "geral").strip()
    notes    = (body.get("notes") or "").strip()
    if not name or not url:
        raise HTTPException(status_code=400, detail="Campos 'name' e 'url' são obrigatórios")
    if not re.match(r"^https?://", url, flags=re.I):
        raise HTTPException(status_code=400, detail="URL deve começar com http:// ou https://")
    data = load_sources()
    sources = data.get("sources", [])
    sid = source_id(name, url)
    existing = next((s for s in sources if s.get("id") == sid or s.get("url") == url), None)
    if existing:
        return {"ok": True, "source": existing, "message": "Fonte já cadastrada"}
    new_source = {
        "id": sid, "name": name, "url": url, "type": stype,
        "category": category, "enabled": True, "notes": notes,
        "created_at": int(time.time()),
    }
    sources.append(new_source)
    data["sources"] = sources
    save_sources(data)
    return {"ok": True, "source": new_source, "total": len(sources)}


@app.delete("/api/sources/{sid}")
async def api_delete_source(sid: str):
    data = load_sources()
    before = len(data.get("sources", []))
    data["sources"] = [s for s in data.get("sources", []) if s.get("id") != sid]
    after = len(data["sources"])
    save_sources(data)
    if after == before:
        raise HTTPException(status_code=404, detail=f"Fonte '{sid}' não encontrada")
    return {"ok": True, "deleted": sid, "total": after}


@app.post("/api/upload")
async def api_upload(files: List[UploadFile] = File(...), ocr: bool = Form(True)):
    if not HAS_PDFPLUMBER:
        raise HTTPException(status_code=500, detail="pdfplumber não instalado")
    results = []
    idx = load_index()
    for up in files:
        file_bytes = await up.read()
        doc_id = hashlib.md5((up.filename + "::").encode("utf-8") + file_bytes).hexdigest()[:12]
        extracted = await asyncio.to_thread(extract_pdf_sync, file_bytes, up.filename, ocr)
        extracted["uploaded_at"] = int(time.time())
        extracted["doc_id"] = doc_id
        idx.setdefault("docs", {})[doc_id] = extracted
        results.append({
            "doc_id": doc_id,
            "name": up.filename,
            "pages": extracted["pages"],
            "text_pages": extracted.get("text_pages", 0),
            "ocr_pages": extracted.get("ocr_pages", 0),
            "failed_pages": extracted.get("failed_pages", 0),
            "chunks": len(extracted["chunks"]),
            "char_count": extracted["char_count"],
            "status": extracted["status"],
            "ocr_available": OCR_AVAILABLE,
            "page_stats": extracted.get("page_stats", [])[:200],
        })
    save_index(idx)
    return {"results": results, "total": len(results)}


@app.get("/api/search")
async def api_search(q: str, k: int = 25):
    idx = load_index()
    results = search_chunks(q, idx, top_k=min(k, 50))
    return {
        "query": q,
        "expanded_terms": expand_query(q)[:20],
        "total_chunks_searched": sum(len(d.get("chunks", [])) for d in idx.get("docs", {}).values()),
        "results": results,
        "count": len(results),
    }


@app.get("/api/delete/{doc_id}")
async def api_delete(doc_id: str):
    idx = load_index()
    if doc_id in idx.get("docs", {}):
        del idx["docs"][doc_id]
        save_index(idx)
        return {"ok": True, "deleted": doc_id}
    raise HTTPException(status_code=404, detail=f"Documento '{doc_id}' não encontrado")


@app.post("/api/chat")
async def api_chat(request: Request):
    body = await request.json()
    message       = (body.get("message") or "").strip()
    current_theme = (body.get("current_theme") or "").strip()
    history       = body.get("history") or []
    if not message and not current_theme:
        raise HTTPException(status_code=400, detail="Informe message ou current_theme")
    result = await asyncio.to_thread(
        runtime.chat,
        message=message,
        current_theme=current_theme,
        history=history,
    )
    return JSONResponse(result)


@app.get("/api/agent")
async def api_agent(q: str):
    result = await asyncio.to_thread(
        runtime.chat,
        message=f"Organize um relatório de estudo completo sobre {q}.",
        current_theme=q,
        history=[],
    )
    return JSONResponse(result)


def run():
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=PORT, reload=False)


if __name__ == "__main__":
    run()
