#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedAtlas DeepStudy — FastAPI Agentic RAG v4.1
CORREÇÕES v4.1:
  [BUG1] search_chunks: removido double-filter (pscore>0 descartava resultados BM25 válidos)
         Agora: bm25_score>0 é suficiente; density calculada como bônus, não como filtro
  [BUG2] api_upload: ocr=Form(True) não convertia string multipart → agora Optional[str] + parse manual
  [BUG3] MEDICAL_SYNONYMS expandido com mais termos clínicos
  [ADD]  /api/status retorna total_chunks para o frontend atualizar KPIs corretamente
  [ADD]  log de startup informativo
"""

import asyncio
import gc
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
from typing import Any, Dict, List, Optional

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

HAS_TESSERACT = False
TESSERACT_VERSION = "não instalado"
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    HAS_TESSERACT = True
    TESSERACT_VERSION = str(pytesseract.get_tesseract_version())
except Exception:
    pass

if sys.platform == "win32" and not HAS_TESSERACT:
    for _p in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
    ]:
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

SCRIPT_DIR   = Path(__file__).parent.resolve()
CACHE_DIR    = SCRIPT_DIR / "medatlas_cache"
CACHE_DIR.mkdir(exist_ok=True)
INDEX_FILE   = CACHE_DIR / "index.json"
SOURCES_FILE = CACHE_DIR / "sources.json"
PORT = int(os.environ.get("PORT", "8742"))

DEFAULT_SOURCES = [
    {"id":"anatpat_unicamp","name":"UNICAMP AnatPat","url":"https://anatpat.unicamp.br/aulas2.html",
     "type":"site","category":"patologia","enabled":True,"notes":"Anatomia patológica macro e microscópica."},
    {"id":"radiopaedia","name":"Radiopaedia","url":"https://radiopaedia.org/",
     "type":"site","category":"radiologia","enabled":True,"notes":"Casos e artigos radiológicos."},
    {"id":"histology_guide","name":"HistologyGuide","url":"https://histologyguide.com/EM-atlas/15-liver-and-gallbladder.html",
     "type":"site","category":"histologia","enabled":True,"notes":"Atlas de histologia e microscopia eletrônica."},
]


# ── texto ──────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text))
    return re.sub(r"[^a-z0-9\s]", " ", nfkd.encode("ascii","ignore").decode("ascii").lower())

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).strip()

STOPWORDS = set("""
a ao aos as da das de do dos e em é na nas no nos o os para
pela pelas pelo pelos por que se um uma uns umas
the and for are with from this that have been can will also do not but or at by an
""".split())

def tokenize(text: str) -> List[str]:
    return [t for t in norm(text).split() if len(t) > 2 and t not in STOPWORDS]

def fix_hyphenation(text: str) -> str:
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

def clean_page_text(raw: str) -> str:
    if not raw: return ""
    text = fix_hyphenation(raw)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()

def split_into_chunks(text: str, page: int, source: str,
                      min_len: int=80, max_words: int=120, stride: int=60) -> list:
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
            seen.add(key); deduped.append(c)
    return deduped

def is_scanned_page(text: str) -> bool:
    return not text or len(text.split()) < 15 or len(text) < 80

def safe_excerpt(text: str, query: str, max_len: int=520) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= max_len: return cleaned
    nq = norm(query)
    nc = norm(cleaned)
    pos = nc.find(nq) if nq else -1
    if pos < 0: return cleaned[:max_len].strip() + " ..."
    start = max(0, pos - max_len // 3)
    end   = min(len(cleaned), start + max_len)
    excerpt = cleaned[start:end].strip()
    if start > 0: excerpt = "... " + excerpt
    if end < len(cleaned): excerpt = excerpt + " ..."
    return excerpt


# ── persistência ───────────────────────────────────────────────────────────────

def load_json(path: Path, fallback: dict) -> dict:
    if path.exists():
        try: return json.loads(path.read_text("utf-8"))
        except Exception: pass
    return fallback

def save_json(path: Path, data: dict):
    data["updated"] = int(time.time())
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_index() -> dict: return load_json(INDEX_FILE, {"docs":{}, "updated":0})
def save_index(data: dict): save_json(INDEX_FILE, data)
def load_sources() -> dict: return load_json(SOURCES_FILE, {"sources":DEFAULT_SOURCES[:], "updated":int(time.time())})
def save_sources(data: dict): save_json(SOURCES_FILE, data)

def source_id(name: str, url: str) -> str:
    return hashlib.md5((name.strip()+"|"+url.strip()).encode("utf-8")).hexdigest()[:12]


# ── BM25 ───────────────────────────────────────────────────────────────────────

class BM25:
    K1=1.5; B=0.75
    def __init__(self, corpus: List[str]):
        self.n = len(corpus)
        self.tokenized = [tokenize(d) for d in corpus]
        self.dl = [len(t) for t in self.tokenized]
        self.avgdl = sum(self.dl) / max(1, self.n)
        self.df: Dict[str,int] = {}
        for doc in self.tokenized:
            for term in set(doc): self.df[term] = self.df.get(term,0)+1

    def idf(self, term: str) -> float:
        n_t = self.df.get(term,0)
        return math.log((self.n - n_t + 0.5) / (n_t + 0.5) + 1)

    def score(self, query_terms: List[str], doc_idx: int) -> float:
        doc = self.tokenized[doc_idx]; dl = self.dl[doc_idx]
        tf_map: Dict[str,int] = {}
        for t in doc: tf_map[t] = tf_map.get(t,0)+1
        total = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            if tf == 0 and len(term) >= 5:
                for k in tf_map:
                    if k.startswith(term[:5]) or term.startswith(k[:5]):
                        tf = tf_map[k]; break
            if tf == 0: continue
            num = tf*(self.K1+1)
            den = tf + self.K1*(1-self.B+self.B*dl/max(1,self.avgdl))
            total += self.idf(term)*num/den
        return total

    def rank(self, query: str, top_k: int=30) -> list:
        q_terms = tokenize(query)
        if not q_terms: return []
        scores = [(i, self.score(q_terms,i)) for i in range(self.n)]
        scores = [(i,s) for i,s in scores if s>0]
        scores.sort(key=lambda x:x[1], reverse=True)
        return scores[:top_k]


# ── sinônimos médicos expandidos ───────────────────────────────────────────────

MEDICAL_SYNONYMS: Dict[str,List[str]] = {
    "cirrose":       ["fibrose hepatica","hepatopatia cronica","cirrose alcoolica","cirrose viral","hipertensao portal","nodulos regeneracao"],
    "hepatite":      ["hepatite viral","hepatite b","hepatite c","hepatite aguda","hepatite cronica","transaminases","alt","ast","hbv","hcv"],
    "esteatose":     ["figado gorduroso","nash","dhgna","nafld","esteatohepatite","esteatose macrovesicular","gordura hepatica"],
    "tuberculose":   ["tb","mycobacterium","granuloma caseoso","bacilo de koch","baar","bk","tuberculose pulmonar","complexo ghon","caverna"],
    "glomerulonefrite":["gn","nefrite","sindrome nefritica","sindrome nefrotica","proteinuria","hematuria","crescentica","membranosa"],
    "infarto":       ["iam","necrose miocardica","stemi","nstemi","coronaria","isquemia miocardica","angina","infarto agudo"],
    "pneumonia":     ["broncopneumonia","consolidacao","hepatizacao","lobar","pneumococo","atipica","pac","curb"],
    "nefrolitiase":  ["calculo renal","litiase renal","pedra rim","urolitiase","colica renal","hidronefrose","oxalato calcio","acido urico","estruvita"],
    "pielonefrite":  ["infeccao urinaria alta","itu alta","abscesso renal","pionefrose","bacteriuria"],
    "apendicite":    ["apendice","fossa iliaca","peritonite","fecalito","apendicite aguda"],
    "pancreatite":   ["pancreatite aguda","pancreatite cronica","lipase","amilase","pseudocisto","necrose pancreatica","balthazar"],
    "diabetes":      ["dm1","dm2","hiperglicemia","insulina","resistencia insulinica","nefropatia diabetica","glicemia","hba1c"],
    "linfoma":       ["hodgkin","nao hodgkin","reed sternberg","ldgcb","linfadenopatia","linfonodo"],
    "avc":           ["acidente vascular cerebral","infarto cerebral","derrame","isquemia cerebral","hemorragia cerebral","stroke"],
    "aterosclerose": ["arteriosclerose","placa ateromatosa","coronariopatia","calcificacao vascular","foam cells","ldl"],
    "necrose":       ["necrose coagulativa","necrose liquefativa","necrose caseosa","gangrena","apoptose","morte celular","picnose"],
    "crohn":         ["doenca de crohn","ileite terminal","dii","skip lesions","granuloma intestinal","fistula"],
    "retocolite":    ["rcui","colite ulcerativa","dii","abscesso criptas","hematoquesia","colite"],
    "embolia":       ["tep","tromboembolismo","trombo pulmonar","tvp","trombose venosa","embolia pulmonar"],
    "chc":           ["carcinoma hepatocelular","hepatocarcinoma","hcc","li rads","afp","alfa fetoproteina"],
    "carcinoma":     ["neoplasia","tumor maligno","adenocarcinoma","metastase","invasao","cancer","malignidade"],
}

def expand_query(query: str) -> List[str]:
    base = tokenize(query)
    nq   = norm(query)
    extra: List[str] = []
    for key, syns in MEDICAL_SYNONYMS.items():
        if key in nq or any(norm(s) in nq for s in syns):
            extra.extend(syns)
    return list(dict.fromkeys(base + [t for s in extra for t in tokenize(s)]))


# ── search_chunks CORRIGIDO ────────────────────────────────────────────────────
# [BUG FIX] Removido double-filter: antes exigia pscore>0 além de bm25>0.
# page_match_score usava norm(term) in ntext, mas termos expandidos raramente
# batem exatamente no texto normalizado → resultados BM25 válidos eram descartados.
# Agora: qualquer chunk com bm25>0 é candidato. Density calculada como bônus.

def search_chunks(query: str, index: Dict[str,Any], top_k: int=25) -> List[Dict[str,Any]]:
    all_chunks: List[Dict[str,Any]] = []
    for doc_id, doc_data in index.get("docs",{}).items():
        for chunk in doc_data.get("chunks",[]):
            all_chunks.append({**chunk, "doc_id": doc_id})
    if not all_chunks:
        return []

    corpus = [c["text"] for c in all_chunks]
    bm25   = BM25(corpus)
    q_exp  = expand_query(query)
    ranked = dict(bm25.rank(query+" "+" ".join(q_exp), top_k=min(len(corpus), top_k*6)))

    nq = norm(query)
    type_bonus = {"paragraph":1.5, "sentence":1.0, "window":0.2}

    candidates: List[Dict[str,Any]] = []
    for idx, chunk in enumerate(all_chunks):
        bscore = ranked.get(idx, 0.0)
        if bscore <= 0:
            continue
        ctext = chunk.get("text") or ""
        nc    = norm(ctext)
        # Bônus: frase exata presente no chunk
        exact_bonus = 5.0 if nq and nq in nc else 0.0
        # Bônus: densidade de termos expandidos no chunk
        matched = [t for t in q_exp if norm(t) in nc]
        density = (len(matched) / max(1, len(nc.split()))) * 30
        final   = round(bscore + exact_bonus + density + type_bonus.get(chunk.get("type","window"), 0.0), 3)
        candidates.append({
            "doc_id":       chunk["doc_id"],
            "source":       chunk.get("source",""),
            "page":         chunk.get("page","?"),
            "score":        final,
            "bm25":         round(bscore,3),
            "method":       chunk.get("method","text"),
            "matched_terms":matched[:10],
            "text":         safe_excerpt(ctext, query, max_len=520),
            "ocr":          chunk.get("method") == "ocr",
        })

    # Dedup por (doc_id, page) — melhor score por página
    best: Dict[Any,Dict[str,Any]] = {}
    for c in sorted(candidates, key=lambda x:x["score"], reverse=True):
        key = (c["doc_id"], c["page"])
        if key not in best:
            best[key] = c

    return sorted(best.values(), key=lambda x:x["score"], reverse=True)[:top_k]


# ── extração PDF ───────────────────────────────────────────────────────────────

def extract_pdf_sync(file_bytes: bytes, filename: str, do_ocr: bool=True) -> Dict[str,Any]:
    result: Dict[str,Any] = {
        "name":filename, "pages":0, "text_pages":0, "ocr_pages":0,
        "failed_pages":0, "chunks":[], "full_text":"", "page_texts":[],
        "page_stats":[], "char_count":0, "status":"ok",
        "ocr_available":OCR_AVAILABLE, "ocr_requested":do_ocr,
    }
    if not HAS_PDFPLUMBER:
        result["status"] = "error: pdfplumber não instalado"; return result
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            result["pages"] = len(pdf.pages)
            all_text_parts: List[str] = []
            for i, page in enumerate(pdf.pages):
                pnum = i+1
                stat = {"page":pnum,"method":"text","chars":0,"chunks":0,"error":None}
                try:
                    try:
                        raw = page.extract_text(x_tolerance=3,y_tolerance=3,layout=True,x_density=7.25,y_density=13) or ""
                    except TypeError:
                        raw = page.extract_text() or ""
                    cleaned = clean_page_text(raw)
                    use_ocr = do_ocr and OCR_AVAILABLE and is_scanned_page(cleaned)
                    if use_ocr:
                        try:
                            img  = page.to_image(resolution=180).original
                            gray = img.convert("L")
                            ocr_text = ""
                            for lang in ["por+eng","por","eng"]:
                                try:
                                    ocr_text = pytesseract.image_to_string(gray,lang=lang,config="--psm 6 --oem 3")
                                    if len(ocr_text.split())>10: break
                                except Exception: continue
                            ocr_clean = clean_page_text(ocr_text)
                            if len(ocr_clean) > len(cleaned):
                                cleaned=ocr_clean; stat["method"]="ocr"; result["ocr_pages"]+=1
                        except Exception as e:
                            stat["error"] = f"OCR:{str(e)[:80]}"
                    if cleaned and len(cleaned.split())>=5: result["text_pages"]+=1
                    if cleaned and len(cleaned)>=40:
                        page_chunks = split_into_chunks(cleaned, pnum, filename)
                        for ch in page_chunks:
                            ch["page_text"] = cleaned
                            ch["method"]    = stat["method"]
                        result["chunks"].extend(page_chunks)
                        result["page_texts"].append({"page":pnum,"text":cleaned,"method":stat["method"]})
                        stat["chunks"] = len(page_chunks)
                        all_text_parts.append(f"[Pág.{pnum}]\n{cleaned}")
                    stat["chars"] = len(cleaned)
                except Exception as e:
                    stat["method"]="failed"; stat["error"]=str(e)[:200]; result["failed_pages"]+=1
                result["page_stats"].append(stat)
            result["full_text"]  = "\n\n".join(all_text_parts)
            result["char_count"] = len(result["full_text"])
            if result["char_count"]==0: result["status"]="warning: nenhum texto extraído"
    except Exception as e:
        import traceback
        result["status"] = f"error:{str(e)}"; result["error"] = traceback.format_exc()
    return result


# ── runtime ────────────────────────────────────────────────────────────────────

from agentic_rag_runtime_v3 import AgenticStudyRuntime

runtime = AgenticStudyRuntime(
    search_chunks_fn=search_chunks,
    load_index_fn=load_index,
    load_sources_fn=load_sources,
)

# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI(title="MedAtlas DeepStudy Agentic RAG v4.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(SCRIPT_DIR)), name="static")


@app.on_event("startup")
async def startup_event():
    save_sources(load_sources())
    idx = load_index()
    n_docs   = len(idx.get("docs",{}))
    n_chunks = sum(len(d.get("chunks",[])) for d in idx.get("docs",{}).values())
    print(f"[MedAtlas v4.1] Iniciado — porta {PORT}", flush=True)
    print(f"[MedAtlas] pdfplumber={HAS_PDFPLUMBER} | OCR={OCR_AVAILABLE} | Tesseract={TESSERACT_VERSION}", flush=True)
    print(f"[MedAtlas] Acervo: {n_docs} docs | {n_chunks} chunks", flush=True)


@app.get("/", response_class=HTMLResponse)
async def home():
    f = SCRIPT_DIR / "index.html"
    if not f.exists():
        return HTMLResponse("<h1>index.html não encontrado</h1>", 404)
    return HTMLResponse(f.read_text(encoding="utf-8", errors="ignore"))


@app.get("/api/status")
async def api_status():
    idx = load_index(); src = load_sources()
    n_chunks = sum(len(d.get("chunks",[])) for d in idx.get("docs",{}).values())
    return {
        "ok": True,
        "server": "MedAtlas DeepStudy FastAPI Agentic RAG v4.1",
        "platform": sys.platform, "python": sys.version.split()[0],
        "pdfplumber": HAS_PDFPLUMBER, "ocr_available": OCR_AVAILABLE,
        "tesseract": TESSERACT_VERSION,
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY","").strip()),
        "docs": len(idx.get("docs",{})), "total_chunks": n_chunks,
        "sources": len(src.get("sources",[])), "cache": str(CACHE_DIR),
    }


@app.get("/api/index")
async def api_index():
    idx = load_index(); summary = {}
    for doc_id, doc in idx.get("docs",{}).items():
        summary[doc_id] = {
            "name": doc.get("name"), "pages": doc.get("pages"),
            "text_pages": doc.get("text_pages"), "ocr_pages": doc.get("ocr_pages"),
            "failed_pages": doc.get("failed_pages"),
            "chunk_count": len(doc.get("chunks",[])), "char_count": doc.get("char_count"),
            "status": doc.get("status"), "uploaded_at": doc.get("uploaded_at"),
            "page_stats": doc.get("page_stats",[]),
        }
    return {"docs": summary, "updated": idx.get("updated")}


@app.get("/api/sources")
async def api_sources(): return load_sources()


@app.post("/api/sources")
async def api_add_source(request: Request):
    body = await request.json()
    name = (body.get("name") or "").strip(); url = (body.get("url") or "").strip()
    if not name or not url:
        raise HTTPException(400, "Campos 'name' e 'url' são obrigatórios")
    if not re.match(r"^https?://", url, re.I):
        raise HTTPException(400, "URL deve começar com http:// ou https://")
    data    = load_sources(); sources = data.get("sources",[])
    sid     = source_id(name, url)
    existing = next((s for s in sources if s.get("id")==sid or s.get("url")==url), None)
    if existing: return {"ok":True, "source":existing, "message":"Fonte já cadastrada"}
    new_src = {
        "id":sid, "name":name, "url":url,
        "type":(body.get("type") or "site").strip(),
        "category":(body.get("category") or "geral").strip(),
        "enabled":True, "notes":(body.get("notes") or "").strip(),
        "created_at":int(time.time()),
    }
    sources.append(new_src); data["sources"] = sources; save_sources(data)
    return {"ok":True, "source":new_src, "total":len(sources)}


@app.delete("/api/sources/{sid}")
async def api_delete_source(sid: str):
    data = load_sources(); before = len(data.get("sources",[]))
    data["sources"] = [s for s in data.get("sources",[]) if s.get("id")!=sid]
    save_sources(data)
    if len(data["sources"])==before: raise HTTPException(404, f"Fonte '{sid}' não encontrada")
    return {"ok":True, "deleted":sid, "total":len(data["sources"])}


@app.post("/api/upload")
async def api_upload(
    files: List[UploadFile] = File(...),
    ocr: Optional[str] = Form(default="true"),   # [BUG FIX] era bool=Form(True) — não convertia string multipart
):
    do_ocr = str(ocr or "true").lower() not in ("false","0","no","off")

    if not HAS_PDFPLUMBER:
        return JSONResponse(status_code=500, content={"error":"pdfplumber não instalado — pip install pdfplumber"})

    results: List[dict] = []
    idx = load_index()
    for up in files:
        try:
            file_bytes = await up.read()
            if not file_bytes:
                results.append({"name":up.filename,"error":"Arquivo vazio"}); continue
            doc_id = hashlib.md5((up.filename+"::").encode()+ file_bytes[:1024]).hexdigest()[:12]
            extracted = await asyncio.to_thread(extract_pdf_sync, file_bytes, up.filename, do_ocr)
            extracted["uploaded_at"] = int(time.time()); extracted["doc_id"] = doc_id
            idx.setdefault("docs",{})[doc_id] = extracted
            save_index(idx)
            results.append({
                "doc_id":doc_id, "name":up.filename,
                "pages":extracted.get("pages",0), "text_pages":extracted.get("text_pages",0),
                "ocr_pages":extracted.get("ocr_pages",0), "failed_pages":extracted.get("failed_pages",0),
                "chunks":len(extracted.get("chunks",[])), "char_count":extracted.get("char_count",0),
                "status":extracted.get("status","ok"), "ocr_available":OCR_AVAILABLE,
                "page_stats":extracted.get("page_stats",[])[:50],
            })
            del file_bytes, extracted; gc.collect()
        except Exception as e:
            results.append({"name":getattr(up,"filename","arquivo"),"error":str(e)[:300]}); gc.collect()
    return JSONResponse({"results":results,"total":len(results),"ok":True})


@app.get("/api/search")
async def api_search(q: str, k: int=25):
    idx = load_index()
    total = sum(len(d.get("chunks",[])) for d in idx.get("docs",{}).values())
    results = search_chunks(q, idx, top_k=min(k,50))
    return {"query":q, "expanded_terms":expand_query(q)[:20],
            "total_chunks_searched":total, "results":results, "count":len(results)}


@app.get("/api/delete/{doc_id}")
async def api_delete(doc_id: str):
    idx = load_index()
    if doc_id not in idx.get("docs",{}):
        raise HTTPException(404, f"Documento '{doc_id}' não encontrado")
    del idx["docs"][doc_id]; save_index(idx)
    return {"ok":True,"deleted":doc_id}


@app.post("/api/chat")
async def api_chat(request: Request):
    body = await request.json()
    message       = (body.get("message") or "").strip()
    current_theme = (body.get("current_theme") or "").strip()
    history       = body.get("history") or []
    if not message and not current_theme:
        raise HTTPException(400, "Informe 'message' ou 'current_theme'")
    result = await asyncio.to_thread(
        runtime.chat, message=message, current_theme=current_theme, history=history
    )
    return JSONResponse(result)


@app.get("/api/agent")
async def api_agent(q: str):
    result = await asyncio.to_thread(
        runtime.chat,
        message=f"Organize um relatório de estudo completo sobre {q}.",
        current_theme=q, history=[],
    )
    return JSONResponse(result)


def run():
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=PORT, reload=False)

if __name__ == "__main__":
    run()
