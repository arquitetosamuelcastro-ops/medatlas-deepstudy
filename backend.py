#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedAtlas DeepStudy — Backend v1.1
Servidor HTTP nativo · pdfplumber · BM25 · OCR opcional
Compatível com Windows, macOS e Linux
"""

import json
import re
import math
import hashlib
import unicodedata
import threading
import time
import traceback
import sys
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from io import BytesIO

# ─────────────────────────────────────────────────────────────────
#  DETECÇÃO DE DEPENDÊNCIAS OPCIONAIS
# ─────────────────────────────────────────────────────────────────
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("[AVISO] pdfplumber não encontrado. Instale: pip install pdfplumber")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytesseract
    # Testa se o binário existe antes de declarar disponível
    pytesseract.get_tesseract_version()
    HAS_TESSERACT = True
    TESSERACT_VERSION = str(pytesseract.get_tesseract_version())
except Exception:
    HAS_TESSERACT = False
    TESSERACT_VERSION = "não instalado"

# Caminho do Tesseract no Windows (padrão de instalação)
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

OCR_AVAILABLE = HAS_TESSERACT and HAS_PIL and HAS_PDFPLUMBER

# ─────────────────────────────────────────────────────────────────
#  CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────
PORT = int(os.environ.get("PORT", 8742))
# Cache na mesma pasta do script — funciona em qualquer OS
SCRIPT_DIR = Path(__file__).parent.resolve()
CACHE_DIR = SCRIPT_DIR / "medatlas_cache"
CACHE_DIR.mkdir(exist_ok=True)
INDEX_FILE = CACHE_DIR / "index.json"
LOCK = threading.Lock()

BM25_K1 = 1.5
BM25_B = 0.75

STOPWORDS = set("""
a ao aos as da das de do dos e em é na nas no nos o os para
pela pelas pelo pelos por que se um uma uns umas
the and for are with from this that have been can will also
do not but or at by an
""".split())


# ─────────────────────────────────────────────────────────────────
#  UTILITÁRIOS DE TEXTO
# ─────────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """Remove acentos e converte para lowercase ASCII."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9\s]", " ", ascii_str.lower())


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).strip()


def tokenize(text: str) -> list:
    tokens = norm(text).split()
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def fix_hyphenation(text: str) -> str:
    """Une palavras partidas com hífen no fim de linha."""
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def clean_page_text(raw: str) -> str:
    """Pipeline completo de limpeza de texto extraído de PDF."""
    if not raw:
        return ""
    text = fix_hyphenation(raw)
    # Quebras simples (não parágrafos) → espaço
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Múltiplos espaços → um
    text = re.sub(r"[ \t]+", " ", text)
    # Mais de 2 quebras → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove caracteres de controle (problemáticos no Windows)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def split_into_chunks(text: str, page: int, source: str,
                      min_len: int = 80,
                      max_words: int = 120,
                      stride: int = 60) -> list:
    """
    Gera chunks por 3 estratégias:
    1. Parágrafos (blocos separados por linha em branco)
    2. Sentenças longas (>100 chars)
    3. Janelas deslizantes de palavras
    """
    chunks = []

    # 1. Parágrafos
    for para in re.split(r"\n{2,}", text):
        p = para.strip()
        if len(p) >= min_len:
            chunks.append({"text": p, "page": page, "source": source, "type": "paragraph"})

    # 2. Sentenças longas
    for sent in re.split(r"(?<=[.!?;])\s+", text.replace("\n", " ")):
        s = sent.strip()
        if len(s) >= 100:
            chunks.append({"text": s, "page": page, "source": source, "type": "sentence"})

    # 3. Janelas deslizantes
    words = text.replace("\n", " ").split()
    if len(words) >= 30:
        for i in range(0, len(words), stride):
            window = " ".join(words[i: i + max_words]).strip()
            if len(window) >= min_len:
                chunks.append({"text": window, "page": page, "source": source, "type": "window"})

    # Dedup por chave normalizada
    seen, deduped = set(), []
    for c in chunks:
        key = norm(c["text"])[:200]
        if key and key not in seen:
            seen.add(key)
            deduped.append(c)

    return deduped


def is_scanned_page(text: str) -> bool:
    """Detecta página com pouco/nenhum texto extraível."""
    if not text:
        return True
    return len(text.split()) < 15 or len(text) < 80


# ─────────────────────────────────────────────────────────────────
#  EXTRAÇÃO DE PDF
# ─────────────────────────────────────────────────────────────────
def extract_pdf(file_bytes: bytes, filename: str, do_ocr: bool = True) -> dict:
    """
    Extrai texto de um PDF página a página.
    - Tenta extração nativa com pdfplumber
    - Fallback para OCR se a página estiver escaneada e OCR disponível
    - Sempre retorna resultado, mesmo que parcial
    """
    result = {
        "name": filename,
        "pages": 0,
        "text_pages": 0,
        "ocr_pages": 0,
        "failed_pages": 0,
        "chunks": [],
        "full_text": "",
        "page_stats": [],
        "char_count": 0,
        "status": "ok",
        "ocr_available": OCR_AVAILABLE,
        "ocr_requested": do_ocr,
    }

    if not HAS_PDFPLUMBER:
        result["status"] = "error: pdfplumber não instalado"
        result["error"] = "Instale: pip install pdfplumber"
        return result

    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            result["pages"] = len(pdf.pages)
            all_text_parts = []

            for i, page in enumerate(pdf.pages):
                pnum = i + 1
                page_stat = {
                    "page": pnum,
                    "method": "text",
                    "chars": 0,
                    "chunks": 0,
                    "error": None
                }

                try:
                    # ── Extração nativa de texto ──
                    try:
                        raw = page.extract_text(
                            x_tolerance=3,
                            y_tolerance=3,
                            layout=True,
                            x_density=7.25,
                            y_density=13,
                        ) or ""
                    except TypeError:
                        # Versões mais antigas do pdfplumber não aceitam todos esses args
                        raw = page.extract_text() or ""

                    cleaned = clean_page_text(raw)

                    # ── OCR fallback (somente se disponível e texto insuficiente) ──
                    use_ocr = do_ocr and OCR_AVAILABLE and is_scanned_page(cleaned)
                    if use_ocr:
                        try:
                            img = page.to_image(resolution=200).original
                            gray = img.convert("L")

                            # Tenta português primeiro, inglês como fallback
                            for lang in ["por+eng", "por", "eng"]:
                                try:
                                    ocr_text = pytesseract.image_to_string(
                                        gray,
                                        lang=lang,
                                        config="--psm 6 --oem 3"
                                    )
                                    break
                                except pytesseract.TesseractError:
                                    continue

                            ocr_cleaned = clean_page_text(ocr_text)
                            if len(ocr_cleaned) > len(cleaned):
                                cleaned = ocr_cleaned
                                page_stat["method"] = "ocr"
                                result["ocr_pages"] += 1

                        except Exception as ocr_err:
                            page_stat["error"] = f"OCR: {str(ocr_err)[:120]}"

                    elif do_ocr and not OCR_AVAILABLE and is_scanned_page(cleaned):
                        page_stat["method"] = "text_only"
                        page_stat["error"] = "OCR indisponível — Tesseract não instalado"

                    # ── Contabiliza páginas com texto ──
                    if cleaned and len(cleaned.split()) >= 5:
                        result["text_pages"] += 1

                    # ── Gera chunks ──
                    if cleaned and len(cleaned) >= 40:
                        page_chunks = split_into_chunks(cleaned, pnum, filename)
                        result["chunks"].extend(page_chunks)
                        page_stat["chunks"] = len(page_chunks)
                        all_text_parts.append(f"[Pág.{pnum}]\n{cleaned}")

                    page_stat["chars"] = len(cleaned)

                except Exception as page_err:
                    page_stat["error"] = str(page_err)[:200]
                    page_stat["method"] = "failed"
                    result["failed_pages"] += 1

                result["page_stats"].append(page_stat)

            result["full_text"] = "\n\n".join(all_text_parts)
            result["char_count"] = len(result["full_text"])

            if result["char_count"] == 0:
                result["status"] = "warning: nenhum texto extraído"

    except Exception as e:
        result["status"] = f"error: {str(e)}"
        result["error"] = traceback.format_exc()

    return result


# ─────────────────────────────────────────────────────────────────
#  CACHE / ÍNDICE
# ─────────────────────────────────────────────────────────────────
def load_index() -> dict:
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text("utf-8"))
        except Exception:
            pass
    return {"docs": {}, "updated": 0}


def save_index(idx: dict):
    idx["updated"] = int(time.time())
    try:
        INDEX_FILE.write_text(
            json.dumps(idx, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        print(f"[AVISO] Não foi possível salvar cache: {e}")


# ─────────────────────────────────────────────────────────────────
#  BM25
# ─────────────────────────────────────────────────────────────────
class BM25:
    """BM25 ranqueamento sobre coleção de textos."""

    def __init__(self, corpus: list):
        self.corpus = corpus
        self.n = len(corpus)
        self.tokenized = [tokenize(d) for d in corpus]
        self.dl = [len(t) for t in self.tokenized]
        self.avgdl = sum(self.dl) / max(1, self.n)
        self.df = {}
        for doc in self.tokenized:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        return math.log((self.n - n_t + 0.5) / (n_t + 0.5) + 1)

    def score(self, query_terms: list, doc_idx: int) -> float:
        doc = self.tokenized[doc_idx]
        dl = self.dl[doc_idx]
        tf_map = {}
        for t in doc:
            tf_map[t] = tf_map.get(t, 0) + 1

        total = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            # Match parcial por prefixo (útil para flexões)
            if tf == 0:
                for k in tf_map:
                    if len(term) >= 5 and (k.startswith(term[:5]) or term.startswith(k[:5])):
                        tf = tf_map[k]
                        break
            if tf == 0:
                continue
            idf = self.idf(term)
            num = tf * (BM25_K1 + 1)
            den = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / max(1, self.avgdl))
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


# ─────────────────────────────────────────────────────────────────
#  EXPANSÃO SEMÂNTICA MÉDICA
# ─────────────────────────────────────────────────────────────────
MEDICAL_SYNONYMS = {
    "cirrose": ["fibrose hepatica", "hepatopatia cronica", "cirrose alcoolica", "cirrose viral", "hipertensao portal", "nodulos regeneracao"],
    "hepatite": ["hepatite viral", "hepatite b", "hepatite c", "hepatite aguda", "hepatite cronica", "transaminases"],
    "esteatose": ["figado gorduroso", "nash", "dhgna", "nafld", "esteatohepatite", "esteatose macrovesicular"],
    "tuberculose": ["tb", "mycobacterium", "granuloma caseoso", "bacilo de koch", "baar", "bk"],
    "glomerulonefrite": ["gn", "nefrite", "sindrome nefritica", "sindrome nefrotica", "proteinuria", "hematuria"],
    "infarto": ["iam", "necrose miocardica", "stemi", "nstemi", "coronaria", "isquemia miocardica", "angina"],
    "pneumonia": ["broncopneumonia", "consolidacao", "hepatizacao", "lobar", "pneumococo", "atipica"],
    "nefrolitiase": ["calculo renal", "litiase renal", "pedra rim", "urolitiase", "colica renal", "hidronefrose"],
    "pielonefrite": ["infeccao urinaria alta", "itu alta", "abscesso renal", "pionefrose", "bacteriuria"],
    "apendicite": ["apendice", "fossa iliaca", "peritonite", "fecalito"],
    "pancreatite": ["pancreatite aguda", "pancreatite cronica", "lipase", "amilase", "pseudocisto", "necrose pancreatica"],
    "diabetes": ["dm1", "dm2", "hiperglicemia", "insulina", "resistencia insulinica", "nefropatia diabetica", "glicemia"],
    "linfoma": ["hodgkin", "nao hodgkin", "reed sternberg", "ldgcb", "linfadenopatia", "linfonodo"],
    "avc": ["acidente vascular cerebral", "infarto cerebral", "derrame", "isquemia cerebral", "hemorragia cerebral", "stroke"],
    "aterosclerose": ["arteriosclerose", "placa ateromatosa", "coronariopatia", "calcificacao vascular", "foam cells"],
    "necrose": ["necrose coagulativa", "necrose liquefativa", "necrose caseosa", "gangrena", "apoptose", "infarto"],
    "crohn": ["doenca de crohn", "ileite terminal", "dii", "skip lesions", "granuloma intestinal"],
    "retocolite": ["rcui", "colite ulcerativa", "dii", "abscesso criptas", "hematoquesia"],
    "embolia": ["tep", "tromboembolismo", "trombo pulmonar", "tvp", "trombose venosa"],
    "carcinoma": ["neoplasia", "tumor maligno", "adenocarcinoma", "metastase", "invasao"],
}


def expand_query(query: str) -> list:
    """Expande query com sinônimos médicos e retorna lista deduplicada de tokens."""
    base = tokenize(query)
    nq = norm(query)
    extra = []
    for key, syns in MEDICAL_SYNONYMS.items():
        if key in nq or any(norm(s) in nq for s in syns):
            extra.extend(syns)
    all_tokens = list(dict.fromkeys(base + [t for s in extra for t in tokenize(s)]))
    return all_tokens


# ─────────────────────────────────────────────────────────────────
#  BUSCA COM BM25 + RERANKING
# ─────────────────────────────────────────────────────────────────
def search_chunks(query: str, index: dict, top_k: int = 25) -> list:
    """
    Busca BM25 com expansão semântica e reranking por:
    - Correspondência exata da query no texto
    - Densidade de termos
    - Tipo de chunk (parágrafo > sentença > janela)
    """
    all_chunks = []
    for doc_id, doc_data in index.get("docs", {}).items():
        for chunk in doc_data.get("chunks", []):
            all_chunks.append({**chunk, "doc_id": doc_id})

    if not all_chunks:
        return []

    corpus = [c["text"] for c in all_chunks]
    bm25 = BM25(corpus)

    q_expanded = expand_query(query)
    expanded_query_str = query + " " + " ".join(q_expanded)
    bm25_raw = dict(bm25.rank(expanded_query_str, top_k=top_k * 4))

    nq = norm(query)
    type_bonus = {"paragraph": 1.5, "sentence": 1.0, "window": 0.0}

    results = []
    for idx, chunk in enumerate(all_chunks):
        bm25_score = bm25_raw.get(idx, 0)
        if bm25_score == 0:
            continue

        nc = norm(chunk["text"])
        # Exact match bonus
        exact = 8.0 if nq and nq in nc else 0.0
        # Density: matched terms / total words
        matched = [t for t in q_expanded if t in nc]
        density = (len(matched) / max(1, len(nc.split()))) * 50
        # Type preference
        bonus = type_bonus.get(chunk.get("type", "window"), 0)

        final = bm25_score + exact + density + bonus

        results.append({
            **chunk,
            "bm25": round(bm25_score, 3),
            "score": round(final, 2),
            "matched_terms": matched[:10],
        })

    # Dedup + sort
    seen, deduped = set(), []
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        key = norm(r["text"])[:200]
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped[:top_k]


# ─────────────────────────────────────────────────────────────────
#  HTTP HANDLER
# ─────────────────────────────────────────────────────────────────
class MedAtlasHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # Silencia logs de acesso padrão

    def log_error(self, format, *args):
        pass  # Silencia erros de conexão esperados

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Requested-With")

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, path: Path):
        if not path.exists():
            self.send_json({"error": "index.html não encontrado na pasta do backend"}, 404)
            return
        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"

            if path in ("/", "/index.html"):
                self.send_html(SCRIPT_DIR / "index.html")
            elif path == "/api/status":
                self._api_status()
            elif path == "/api/index":
                self._api_get_index()
            elif path == "/api/search":
                self._api_search(parse_qs(parsed.query))
            elif path.startswith("/api/delete/"):
                doc_id = path[len("/api/delete/"):]
                self._api_delete(doc_id)
            else:
                self.send_json({"error": f"Rota não encontrada: {path}"}, 404)
        except Exception as e:
            self.send_json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/upload":
                self._api_upload()
            else:
                self.send_json({"error": "Rota POST não encontrada"}, 404)
        except Exception as e:
            self.send_json({"error": str(e), "trace": traceback.format_exc()}, 500)

    # ── ENDPOINTS ──────────────────────────────────────────────────

    def _api_status(self):
        with LOCK:
            idx = load_index()
        docs = idx.get("docs", {})
        total_chunks = sum(len(d.get("chunks", [])) for d in docs.values())
        self.send_json({
            "ok": True,
            "server": "MedAtlas DeepStudy Backend v1.1",
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "pdfplumber": HAS_PDFPLUMBER,
            "ocr_available": OCR_AVAILABLE,
            "tesseract": TESSERACT_VERSION,
            "docs": len(docs),
            "total_chunks": total_chunks,
            "cache": str(CACHE_DIR),
        })

    def _api_get_index(self):
        with LOCK:
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
        self.send_json({"docs": summary, "updated": idx.get("updated")})

    def _api_upload(self):
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))

        if content_length > 200 * 1024 * 1024:
            self.send_json({"error": "Arquivo muito grande (máx 200MB)"}, 413)
            return

        raw_body = self.rfile.read(content_length)

        # Parse multipart/form-data sem dependências externas
        try:
            boundary_raw = content_type.split("boundary=")[-1].strip()
            boundary = boundary_raw.encode("utf-8")
            parts = raw_body.split(b"--" + boundary)

            files = []
            do_ocr = True

            for part in parts:
                if b"Content-Disposition" not in part:
                    continue
                sep = part.find(b"\r\n\r\n")
                if sep < 0:
                    continue
                header_bytes = part[:sep]
                body = part[sep + 4:].rstrip(b"\r\n")

                try:
                    header = header_bytes.decode("utf-8", errors="replace")
                except Exception:
                    header = header_bytes.decode("latin-1", errors="replace")

                if 'name="file"' in header or 'name="files"' in header:
                    fn_match = re.search(r'filename=["\']?([^"\';\r\n]+)["\']?', header)
                    fname = fn_match.group(1).strip() if fn_match else "upload.pdf"
                    if body:
                        files.append((fname, body))
                elif 'name="ocr"' in header:
                    do_ocr = body.strip().lower() in (b"true", b"1", b"yes")

            if not files:
                self.send_json({"error": "Nenhum arquivo PDF recebido no upload"}, 400)
                return

            results = []
            for fname, fbytes in files:
                doc_id = hashlib.md5((fname + "::").encode("utf-8") + fbytes).hexdigest()[:12]
                try:
                    extracted = extract_pdf(fbytes, fname, do_ocr=do_ocr)
                    extracted["uploaded_at"] = int(time.time())
                    extracted["doc_id"] = doc_id

                    with LOCK:
                        idx = load_index()
                        idx.setdefault("docs", {})[doc_id] = extracted
                        save_index(idx)

                    results.append({
                        "doc_id": doc_id,
                        "name": fname,
                        "pages": extracted["pages"],
                        "text_pages": extracted.get("text_pages", 0),
                        "ocr_pages": extracted.get("ocr_pages", 0),
                        "failed_pages": extracted.get("failed_pages", 0),
                        "chunks": len(extracted["chunks"]),
                        "char_count": extracted["char_count"],
                        "status": extracted["status"],
                        "ocr_available": OCR_AVAILABLE,
                    })
                except Exception as e:
                    results.append({
                        "name": fname,
                        "error": str(e),
                        "status": "error",
                    })

            self.send_json({"results": results, "total": len(results)})

        except Exception as e:
            self.send_json({
                "error": f"Falha no processamento do upload: {str(e)}",
                "trace": traceback.format_exc()
            }, 500)

    def _api_search(self, params: dict):
        query = params.get("q", [""])[0].strip()
        top_k = min(int(params.get("k", ["25"])[0]), 50)

        if not query:
            self.send_json({"error": "Parâmetro 'q' obrigatório"}, 400)
            return

        with LOCK:
            idx = load_index()

        total = sum(len(d.get("chunks", [])) for d in idx.get("docs", {}).values())
        results = search_chunks(query, idx, top_k=top_k)
        expanded = expand_query(query)

        self.send_json({
            "query": query,
            "expanded_terms": expanded[:20],
            "total_chunks_searched": total,
            "results": results,
            "count": len(results),
        })

    def _api_delete(self, doc_id: str):
        with LOCK:
            idx = load_index()
            if doc_id in idx.get("docs", {}):
                del idx["docs"][doc_id]
                save_index(idx)
                self.send_json({"ok": True, "deleted": doc_id})
            else:
                self.send_json({"error": f"Documento '{doc_id}' não encontrado"}, 404)


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║       MedAtlas DeepStudy — Backend v1.1                 ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  Plataforma : {sys.platform} / Python {sys.version.split()[0]}")
    print(f"  pdfplumber : {'✓ OK' if HAS_PDFPLUMBER else '✗ NÃO INSTALADO — pip install pdfplumber'}")
    print(f"  Pillow     : {'✓ OK' if HAS_PIL else '✗ NÃO INSTALADO — pip install Pillow'}")
    print(f"  Tesseract  : {'✓ ' + TESSERACT_VERSION if HAS_TESSERACT else '✗ NÃO INSTALADO (OCR desativado)'}")
    print(f"  OCR ativo  : {'✓ SIM' if OCR_AVAILABLE else '✗ NÃO (extração de texto nativo continua funcionando)'}")
    print()

    if not HAS_PDFPLUMBER:
        print("  [ERRO] pdfplumber é obrigatório. Execute:")
        print("         pip install pdfplumber")
        print()
        input("  Pressione Enter para sair...")
        sys.exit(1)

    if not HAS_TESSERACT:
        print("  [INFO] Tesseract OCR não encontrado.")
        print("         PDFs com texto nativo continuarão sendo processados normalmente.")
        print("         Para ativar OCR em páginas escaneadas:")
        if sys.platform == "win32":
            print("         → Baixe e instale: https://github.com/UB-Mannheim/tesseract/wiki")
            print("         → Instale o pacote de idioma português (por)")
        else:
            print("         → sudo apt install tesseract-ocr tesseract-ocr-por tesseract-ocr-eng")
        print()

    print(f"  ► Servidor : http://localhost:{PORT}")
    print(f"  ► Frontend : http://localhost:{PORT}  (abra no navegador)")
    print(f"  ► Cache    : {CACHE_DIR}")
    print()
    print("  Endpoints:")
    print(f"    GET  http://localhost:{PORT}/             → frontend HTML")
    print(f"    GET  http://localhost:{PORT}/api/status   → status")
    print(f"    GET  http://localhost:{PORT}/api/index    → documentos")
    print(f"    GET  http://localhost:{PORT}/api/search?q=TERMO → busca")
    print(f"    POST http://localhost:{PORT}/api/upload   → upload PDF")
    print()
    print("  Ctrl+C para parar.")
    print()

    try:
        server = HTTPServer(("0.0.0.0", PORT), MedAtlasHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Servidor encerrado.")
        server.server_close()
    except OSError as e:
        if "address already in use" in str(e).lower() or "10048" in str(e):
            print(f"\n  [ERRO] Porta {PORT} já está em uso.")
            print(f"         Feche o processo que está usando a porta ou edite PORT no início do arquivo.")
        else:
            print(f"\n  [ERRO] {e}")
        input("\n  Pressione Enter para sair...")
        sys.exit(1)


if __name__ == "__main__":
    main()
