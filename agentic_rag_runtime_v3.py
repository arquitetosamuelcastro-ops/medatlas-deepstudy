#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic Study Runtime v6
Motor principal : Claude API com web_search tool_use
Fallback local  : trechos reais filtrados por relevância
Histórico real  : contexto completo passado ao LLM
"""
from __future__ import annotations
import os, re, sys, unicodedata
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

try:
    import anthropic as _sdk
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

MODEL = "claude-sonnet-4-20250514"

# ── Normalização ─────────────────────────────────────────────────
def normalize_text(t):
    nfkd = unicodedata.normalize("NFKD", str(t))
    return re.sub(r"[^a-z0-9\s]", " ", nfkd.encode("ascii","ignore").decode("ascii").lower())

def norm(t): return re.sub(r"\s+", " ", normalize_text(t)).strip()

def source_query_url(source, query):
    base = source.get("url","").strip()
    if not base: return ""
    nq = quote_plus(query)
    if "anatpat.unicamp.br" in base: return f"{base}?busca={nq}"
    if "radiopaedia.org" in base: return f"https://radiopaedia.org/search?q={nq}"
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}q={nq}"

# ── Intent e Tema ────────────────────────────────────────────────
IVBS = {"organize","busque","busca","pesquise","monte","transforme","compile",
        "execute","reescreva","liste","explique","descreva","detalhe","fale",
        "flashcard","flashcards","slides","roteiro","checklist","memoriza",
        "apresentacao","apresente","crie","gere","faca","juncao","resuma",
        "responda","elabore","desenvolva"}

def classify_intent(msg):
    m = norm(msg)
    if any(x in m for x in ["sim execute","execute proximos","execute os proximos","agora execute","prossiga","continue","executar proximos"]):
        return "execute_next_steps"
    if any(x in m for x in ["slides","topicos para slides"]): return "slides"
    # Flashcards tem prioridade sobre compile (ex: "crie flashcards" nao eh compile)
    if any(x in m for x in ["flashcard","flashcards","memoriza","cartao","cartoes"]): return "flashcards"
    if any(x in m for x in ["checklist","apresentacao","roteiro","monte um roteiro"]): return "presentation"
    if any(x in m for x in ["texto unico","compilar","texto corrido","texto final",
        "material unico","material compilado","juncao","principais pontos","todo o conteudo",
        "todos o conteudo","resuma","compilado","crie um resumo",
        "crie um material","material completo","com base em todo o material",
        "com base em todo o conteudo","resumo completo"]): return "compile_text"
    # "organize" e "compile" isolados sao compile apenas se nao houver palavra de pergunta
    if any(x in m for x in ["organize","compile"]):
        if not any(q in m for q in ["voce","utilizou","usou","como","qual","por que","porque",
                                     "quais","explique","descreva"]): return "compile_text"
    return "query"

def _theme_ok(t):
    w = norm(t).split()
    return bool(w) and w[0] not in IVBS and len(t) <= 60

def _extract_theme(t):
    for pat in [r"(?:sobre o|sobre a|sobre)\s+([A-Za-z\u00c0-\u00ff][A-Za-z\u00c0-\u00ff\s]{1,50}?)(?:\s+com|\s+base|\s+classe|\s+e\s|,|\.|$)",
                r"(?:do|da)\s+([A-Z][A-Za-z\u00c0-\u00ff\s]{1,40}?)(?:\s+com|\s+base|,|\.|$)"]:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            c = m.group(1).strip()
            if c and len(c) > 2 and norm(c).split()[0] not in IVBS: return c
    return ""

def infer_theme(msg, ct, history):
    if ct and ct.strip(): return ct.strip()
    for h in reversed(history or []):
        if h.get("role") == "user":
            c = (h.get("content") or "").strip()
            if c and 3 <= len(c) <= 60 and _theme_ok(c): return c
    for h in reversed(history or []):
        if h.get("role") == "user":
            e = _extract_theme((h.get("content") or "").strip())
            if e: return e
    fm = _extract_theme(msg)
    return fm if fm else msg.strip()[:60] or "tema medico"

def wants_manual(msg, history):
    all_t = " ".join((h.get("content") or "") for h in (history or []) if h.get("role")=="user") + " " + msg
    return "manual do professor" in norm(all_t)

# ── Evidências ───────────────────────────────────────────────────
def clean_excerpt(t, max_len=420):
    t = re.sub(r"\s+", " ", t or "").strip()
    t = re.sub(r"\{[^}]*\}", "", t).replace("$","")
    return t[:max_len].strip(" .;,:") + ("..." if len(t) > max_len else "")

def rank_manual_first(ev):
    return sorted(ev, key=lambda x: (0 if "manual do professor" in norm(x.get("arquivo","") or "") else 1, -(x.get("score") or 0)))

SYNS: Dict[str, List[str]] = {
    "esteatose": ["esteatohepatite","nafld","nash","gordura hepatica","dhgna","dhna"],
    "cirrose": ["fibrose hepatica","hepatopatia cronica","hipertensao portal","cirrose alcoolica","cirrose viral"],
    "nefrolitiase": ["calculo renal","pedra no rim","urolitiase","colica renal","litiase"],
    "pneumonia": ["infeccao pulmonar","consolidacao pulmonar","pac","curb","pneumococo"],
    "iam": ["infarto agudo","infarto do miocardio","sindrome coronariana","stemi","nstemi"],
    "hepatite": ["hepatite viral","hbv","hcv","hav"],
    "diabetes": ["diabete","dm","hiperglicemia","insulina","glicose"],
    "hipertensao": ["pressao arterial","has","anti hipertensivo"],
    "avc": ["acidente vascular","isquemia cerebral","hemorragia cerebral","stroke"],
    "pancreatite": ["amilase","lipase","necrose pancreatica"],
    "apendicite": ["fossa iliaca","fecalito","peritonite"],
}

def _rel_score(trecho, theme):
    nt, nth = norm(trecho), norm(theme)
    if not nt or not nth: return 0.0
    toks = [t for t in nth.split() if len(t) > 3]
    if not toks: return 0.5
    exp = list(toks)
    for k, sv in SYNS.items():
        if k in nth or any(norm(s) in nth for s in sv):
            exp.extend([norm(s) for s in sv]); exp.append(k)
    hits = sum(1 for t in set(exp) if t in nt)
    score = hits / max(1, len(set(exp)))
    if nth in nt: score += 0.5
    for t in toks:
        if t in nt: score += 0.15
    return min(1.0, score)

def filter_ev(ev_list, theme, min_s=0.12):
    sc = [(_rel_score(e.get("trecho",""), theme), e) for e in ev_list]
    sc.sort(key=lambda x: x[0], reverse=True)
    f = [e for r,e in sc if r >= min_s]
    return f if f else [e for _,e in sc[:3]]

def _best_section(nt: str) -> str:
    """Retorna a seção mais específica para um trecho (prioridade alta→baixa)."""
    # Ordem de especificidade — mais específico primeiro
    checks = [
        ("differentials",  ["diferencial","diagnosticos diferenciais","excluir","distinguir"]),
        ("treatment",      ["tratamento","terapia","conduta","prevencao","medicamento","farmaco","cirurgia","transplante"]),
        ("diagnosis",      ["diagnostico","criterio","biopsia","elastografia","histolog","laboratorio","exame","imagem","ultrassom","alt","ast","tgo","tgp"]),
        ("clinical",       ["clinica","sintoma","sinal","manifesta","fadiga","ictericia","ascite","complicac","assintom"]),
        ("pathophysiology",["fisiopatologia","mecanismo","patogenese","fibrose","inflamac","oxidativ","progressao","apoptose"]),
        ("etiology",       ["etiologia","causa","alcool","virus","hepatite","gordura","metaboli","obesidade","resistencia","isquemia","toxico","infeccao"]),
        ("epidemiology",   ["prevalencia","incidencia","epidemiologia","populacao","fatores de risco","mortalidade","morbidade"]),
        ("definition",     ["defin","conceito","caracteriza","condicao","doenca","entidade","tipo de"]),
    ]
    for sec, keywords in checks:
        if any(k in nt for k in keywords):
            return sec
    return "general"

def classify_sections(ev_list):
    s: Dict[str, list] = {k: [] for k in ["manual","definition","epidemiology","etiology",
        "pathophysiology","clinical","diagnosis","treatment","differentials","general"]}
    for ev in ev_list:
        nt  = norm(clean_excerpt(ev.get("trecho","")))
        src = norm(ev.get("arquivo","") or "")
        if "manual do professor" in src:
            s["manual"].append(ev)
        best = _best_section(nt)
        s[best].append(ev)
    return s

def extract_manual_pts(evs):
    if not evs: return []
    joined = " ".join(clean_excerpt(e.get("trecho",""), 500) for e in evs[:6])
    bullets = []
    for pat in [r"rever[^\.:\n;]{10,80}",r"conhecer[^\.:\n;]{10,80}",r"identificar[^\.:\n;]{10,80}",
                r"discutir[^\.:\n;]{10,80}",r"apresentar[^\.:\n;]{10,80}",r"explicar[^\.:\n;]{10,80}",
                r"descrever[^\.:\n;]{10,80}",r"entender[^\.:\n;]{10,80}",r"analisar[^\.:\n;]{10,80}"]:
        for m in re.findall(pat, joined, flags=re.I):
            b = m.strip(" .;-").capitalize()
            if b and b not in bullets: bullets.append(b)
    return bullets[:10]

# ── Fallback Local ───────────────────────────────────────────────
def _dedup(evs):
    seen, out = set(), []
    for e in evs:
        k = norm(e.get("trecho",""))[:150]
        if k and k not in seen: seen.add(k); out.append(e)
    return out

def _all_ev(secs):
    out = []
    for k in ["definition","epidemiology","etiology","pathophysiology","clinical",
              "diagnosis","treatment","differentials","manual","general"]:
        out.extend(secs.get(k,[]))
    return _dedup(out)

def _fmt(e, ml=440):
    t = clean_excerpt(e.get("trecho",""), ml)
    return f"{t}\n  *({e.get('arquivo','PDF')}, pag. {e.get('pagina','?')})*"

def _no_content(theme):
    return (
        f"**Nenhum trecho relevante sobre '{theme}' nos PDFs carregados.**\n\n"
        f"**Sugestões:**\n"
        f"- Carregue PDFs sobre {theme}\n"
        f"- Tente sinônimos (ex: NAFLD para esteatose, IAM para infarto)\n"
        f"- **Configure ANTHROPIC_API_KEY** — o agente buscará nas fontes externas automaticamente"
    )

def _ext_links(external_sources):
    if not external_sources: return ""
    items = []
    for s in external_sources[:6]:
        nome = s.get("fonte",""); busca = s.get("url_busca","") or s.get("url_base","")
        if nome and busca: items.append(f"- [{nome}]({busca})")
    if not items: return ""
    return (
        "\n\n---\n**Fontes externas** *(configure ANTHROPIC_API_KEY para o agente ler e incorporar automaticamente):*\n"
        + "\n".join(items)
    )

def compose_fallback(message, theme, secs, pdf_ev, ext_src):
    all_e = _all_ev(secs)
    m = norm(message)
    ext = _ext_links(ext_src)

    # PERGUNTA DE CONTINUIDADE — verificar PRIMEIRO antes de qualquer outro bloco
    continuity_signals = ["voce utilizou","voce usou","utilizou as fontes","usou as fontes",
        "fontes externas","como voce","por que voce","porque voce","de onde","qual foi",
        "como foi feito","o que voce","foi utilizado","foi usado","utilizou o conteudo",
        "para confeccionar","para fazer","para criar","para gerar","acima voce",
        "acima foi","o resumo acima","o material acima","a resposta acima"]
    if any(x in m for x in continuity_signals):
        import os as _os
        has_key = bool(_os.environ.get("ANTHROPIC_API_KEY","").strip())
        if not has_key:
            return (
                f"## Sobre a resposta anterior\n\n"
                f"**Sem ANTHROPIC_API_KEY configurada**, o agente opera em modo fallback local:\n\n"
                f"- As respostas são montadas **exclusivamente a partir dos trechos dos PDFs** carregados\n"
                f"- As **fontes externas** (Radiopaedia, UNICAMP AnatPat, etc.) são **listadas como links**, "
                f"mas o conteúdo delas **não é lido nem incorporado** automaticamente\n"
                f"- Não há memória real de contexto — cada resposta é independente\n\n"
                f"**Para o agente funcionar como um chat completo** — lendo fontes externas, "
                f"mantendo contexto e sintetizando com inteligência real:\n\n"
                f"```\nset ANTHROPIC_API_KEY=sk-ant-api03-SUA_CHAVE_AQUI\npython start.py\n```\n\n"
                f"Com a chave configurada, o agente usa **Claude + web_search** — "
                f"busca e lê as fontes externas cadastradas e incorpora na resposta automaticamente."
                + ext
            )

    # FLASHCARDS
    if any(x in m for x in ["flashcard","memoriza","cartao"]):
        sm = [("Definição", secs.get("definition",[])),
              ("Epidemiologia/Fatores risco", secs.get("epidemiology",[])),
              ("Etiologia", secs.get("etiology",[])),
              ("Fisiopatologia", secs.get("pathophysiology",[])),
              ("Quadro clínico", secs.get("clinical",[])),
              ("Diagnóstico", secs.get("diagnosis",[])),
              ("Tratamento", secs.get("treatment",[])),
              ("Diagnósticos diferenciais", secs.get("differentials",[]))]
        tot = sum(len(v) for _,v in sm)
        if tot == 0 and all_e:
            c = max(1, len(all_e)//8)
            sm = [(sm[i][0], all_e[i*c:(i+1)*c]) for i in range(min(8,len(all_e)))]
        cards = [f"## Flashcards — {theme}\n"]
        num=1; seen=set()
        for lbl, evs in sm:
            for e in _dedup(evs)[:2]:
                txt=clean_excerpt(e.get("trecho",""),380); k=norm(txt)[:120]
                if not txt or k in seen or len(txt)<40: continue
                seen.add(k); src=e.get("arquivo","PDF"); pg=e.get("pagina","?")
                cards.append(f"**Cartão {num} — {lbl}**\n**P:** O que seus PDFs informam sobre {lbl.lower()} em {theme}?\n**R:** {txt}\n  *({src}, pag. {pg})*\n")
                num+=1
                if num>12: break
            if num>12: break
        if num==1: return _no_content(theme)+ext
        cards.append(f"\n---\n*{num-1} cartões gerados. Configure ANTHROPIC_API_KEY para respostas sintéticas reais.*")
        return "\n".join(cards)+ext

    # SLIDES
    if any(x in m for x in ["slide","topicos para slides"]):
        if not all_e: return _no_content(theme)+ext
        sl=[("Definição e relevância clínica", secs.get("definition",[]) or all_e[:1]),
            ("Epidemiologia", secs.get("epidemiology",[])),
            ("Etiologia", secs.get("etiology",[])),
            ("Fisiopatologia", secs.get("pathophysiology",[])),
            ("Quadro clínico", secs.get("clinical",[])),
            ("Diagnóstico", secs.get("diagnosis",[])),
            ("Tratamento e prevenção", secs.get("treatment",[])),
            ("Diagnósticos diferenciais", secs.get("differentials",[]))]
        lines=[f"## Tópicos para Slides — {theme}\n",f"**Slide 1 — {theme}: título e motivação clínica**\n"]
        sn=2
        for ti,evs in sl:
            u=_dedup(evs or [])
            if not u: continue
            lines.append(f"**Slide {sn} — {ti}**")
            for e in u[:2]:
                txt=clean_excerpt(e.get("trecho",""),260)
                if len(txt)<30: continue
                lines.append(f"  - {txt} *({e.get('arquivo','PDF')}, pag. {e.get('pagina','?')})*")
            lines.append(""); sn+=1
        lines.append(f"**Slide {sn} — Take-home message sobre {theme}**")
        return "\n".join(lines)+ext

    # RESUMO / COMPILAÇÃO
    if any(x in m for x in ["resumo","compile","compilar","texto unico","material unico","material compilado",
        "juncao","organize","compilado","crie um resumo","crie um material","material completo",
        "todo o conteudo","todos o conteudo","texto corrido"]):
        if not all_e: return _no_content(theme)+ext
        so=[("Definição e relevância clínica", secs.get("definition",[])),
            ("Epidemiologia e fatores de risco", secs.get("epidemiology",[])),
            ("Etiologia", secs.get("etiology",[])),
            ("Fisiopatologia", secs.get("pathophysiology",[])),
            ("Quadro clínico e semiologia", secs.get("clinical",[])),
            ("Diagnóstico", secs.get("diagnosis",[])),
            ("Tratamento e prevenção", secs.get("treatment",[])),
            ("Diagnósticos diferenciais", secs.get("differentials",[]))]
        has_sp = any(evs for _,evs in so)
        parts=[f"## Material compilado — {theme}\n"]
        if has_sp:
            globally_used: set = set()
            for ti,evs in so:
                u=_dedup(evs)
                if not u: continue
                sec_lines = []
                shown=0
                for e in u:
                    txt=clean_excerpt(e.get("trecho",""),440)
                    k=norm(txt)[:150]
                    if len(txt)<30 or k in globally_used: continue
                    globally_used.add(k)
                    sec_lines.append(f"- {txt}\n  *({e.get('arquivo','PDF')}, pag. {e.get('pagina','?')})*")
                    shown+=1
                    if shown>=3: break
                if sec_lines:
                    parts.append(f"### {ti}")
                    parts.extend(sec_lines)
                    parts.append("")
        else:
            parts.append(f"*⚠️ PDFs sem seções específicas sobre '{theme}'. Trechos mais próximos:*\n")
            for e in all_e[:10]:
                txt=clean_excerpt(e.get("trecho",""),440)
                if len(txt)<30: continue
                parts.append(f"- {txt}\n  *({e.get('arquivo','PDF')}, pag. {e.get('pagina','?')})*")
        parts.append("\n---\n*Configure ANTHROPIC_API_KEY para síntese acadêmica real com fontes externas.*")
        return "\n\n".join(parts)+ext

    # ROTEIRO / CHECKLIST
    if any(x in m for x in ["roteiro","checklist","apresentacao"]):
        if not all_e: return _no_content(theme)+ext
        mpts=extract_manual_pts(secs.get("manual",[]))
        lines=[f"## Roteiro de apresentação — {theme}\n"]
        if mpts:
            lines.append("**Estrutura do Manual do Professor:**")
            for p in mpts[:10]: lines.append(f"- {p}")
            lines.append("")
        roteiro_used: set = set()
        for lbl,evs in [
            ("Definição", secs.get("definition",[]) or all_e[:1]),
            ("Epidemiologia", secs.get("epidemiology",[])),
            ("Etiologia e fisiopatologia", _dedup(secs.get("etiology",[])+secs.get("pathophysiology",[]))),
            ("Quadro clínico", secs.get("clinical",[])),
            ("Diagnóstico", secs.get("diagnosis",[])),
            ("Tratamento", secs.get("treatment",[])),
            ("Diferenciais", secs.get("differentials",[]))]:
            u=_dedup(evs or [])
            if not u: continue
            chosen=None
            for e in u:
                k=norm(e.get("trecho",""))[:150]
                if k not in roteiro_used:
                    roteiro_used.add(k); chosen=e; break
            if not chosen: continue
            txt=clean_excerpt(chosen.get("trecho",""),320)
            if len(txt)<30: continue
            lines.append(f"**{lbl}:** {txt}\n  *({chosen.get('arquivo','PDF')}, pag. {chosen.get('pagina','?')})*\n")
        return "\n".join(lines)+ext

    # RESPOSTA LIVRE
    if not all_e: return _no_content(theme)+ext
    parts=[f"## {theme} — trechos encontrados nos PDFs\n"]
    shown=0
    for e in all_e:
        txt=clean_excerpt(e.get("trecho",""),460)
        if len(txt)<40: continue
        parts.append(f"- {txt}\n  *({e.get('arquivo','PDF')}, pag. {e.get('pagina','?')})*\n")
        shown+=1
        if shown>=10: break
    if shown==0: return _no_content(theme)+ext
    parts.append("---\n*Configure ANTHROPIC_API_KEY para análise e síntese real.*")
    return "\n".join(parts)+ext


# ── LLM com web_search ───────────────────────────────────────────
def _system_prompt(use_manual):
    note = ("\n\nPRIORIDADE: use o Manual do Professor como estrutura base."
            if use_manual else "")
    return (
        "Você é um agente acadêmico médico sênior que auxilia estudantes de medicina.\n"
        "Recursos disponíveis:\n"
        "  1. Trechos dos PDFs do aluno (fornecidos no contexto)\n"
        "  2. web_search — use para buscar nas fontes externas do aluno e complementar lacunas\n\n"
        "REGRAS:\n"
        "1. Responda EXATAMENTE ao que o aluno solicitou. Sem seções extras não pedidas.\n"
        "2. PDFs = base principal. Use web_search para complementar e buscar nas fontes externas.\n"
        "3. Cite: PDF → (arquivo, pag. X) | Web → (fonte consultada)\n"
        "4. Português brasileiro com acentos corretos.\n"
        "5. Markdown: ## títulos, **negrito**, - listas.\n"
        "6. Mantenha coerência com o histórico da conversa.\n"
        "7. Perguntas de continuidade: responda especificamente, sem repetir o conteúdo anterior."
        + note
    )

def _build_msgs(message, history, pdf_ev, ext_src, mpts, theme):
    # Contexto dos PDFs
    pdf_lines = []
    for e in pdf_ev[:20]:
        src=e.get("arquivo","PDF"); pg=e.get("pagina","?")
        txt=clean_excerpt(e.get("trecho",""),400)
        tag="[MANUAL]" if "manual do professor" in norm(src) else "[PDF]"
        pdf_lines.append(f"{tag} {src} | pag. {pg}:\n{txt}")
    pdf_block = ("TRECHOS DOS PDFs (cite ao usar):\n\n"+"\n\n".join(pdf_lines)
                 if pdf_lines else "(Nenhum PDF indexado — use web_search)")

    # Fontes externas
    ext_lines=[]
    for s in ext_src[:8]:
        n=s.get("fonte",""); u=s.get("url_busca","") or s.get("url_base",""); c=s.get("category","")
        if n: ext_lines.append(f"- {n} ({c}): {u}")
    ext_block = ("\n\nFONTES EXTERNAS DO ALUNO (use web_search nestas URLs):\n"+"\n".join(ext_lines)
                 if ext_lines else "")

    manual_block = ("\n\nROTEIRO DO MANUAL:\n"+"\n".join(f"- {p}" for p in mpts)
                    if mpts else "")

    ctx = f"Tema: {theme}\n\n{pdf_block}{ext_block}{manual_block}\n\n---"

    # Monta histórico no formato Anthropic
    msgs = []
    hist = [h for h in (history or [])[-20:] if (h.get("content") or "").strip()]
    for h in hist:
        r = h.get("role","")
        c = (h.get("content") or "").strip()
        if r == "user": msgs.append({"role":"user","content":c})
        elif r in {"assistant","agent"}: msgs.append({"role":"assistant","content":c})

    if not msgs:
        msgs.append({"role":"user","content": f"{ctx}\n\nSolicitação: {message}"})
    else:
        msgs.append({"role":"user","content": f"[Tema={theme}, {len(pdf_ev)} PDFs disponíveis]\n\n{message}"})
    return msgs

def call_llm(system, messages, ext_src):
    if not HAS_ANTHROPIC:
        print("[Runtime] anthropic não instalado", file=sys.stderr); return None
    key = os.environ.get("ANTHROPIC_API_KEY","").strip()
    if not key:
        print("[Runtime] ANTHROPIC_API_KEY não configurada", file=sys.stderr); return None
    try:
        client = _sdk.Anthropic(api_key=key)
        tools = [{"type":"web_search_20250305","name":"web_search"}]
        resp = client.messages.create(model=MODEL, max_tokens=3000,
                                       system=system, tools=tools, messages=messages)
        parts = []
        has_tool = False
        for b in resp.content:
            if b.type=="text": parts.append(b.text)
            elif b.type=="tool_use": has_tool=True

        # Segunda rodada se usou tools
        if has_tool and resp.stop_reason=="tool_use":
            tool_msgs = list(messages)+[{"role":"assistant","content":resp.content}]
            results = [{"type":"tool_result","tool_use_id":b.id,"content":"Busca realizada."}
                       for b in resp.content if b.type=="tool_use"]
            if results:
                tool_msgs.append({"role":"user","content":results})
                resp2 = client.messages.create(model=MODEL, max_tokens=3000,
                                                system=system, tools=tools, messages=tool_msgs)
                for b in resp2.content:
                    if b.type=="text": parts.append(b.text)

        result = "\n".join(parts).strip()
        if result:
            print(f"[Runtime] LLM OK — {len(result)} chars (tool={has_tool})", file=sys.stderr)
            return result
        return None
    except Exception as e:
        print(f"[Runtime] Erro: {e}", file=sys.stderr); return None


# ── Runtime Principal ────────────────────────────────────────────
class AgenticStudyRuntime:
    def __init__(self, search_chunks_fn, load_index_fn, load_sources_fn):
        self.search_chunks_fn = search_chunks_fn
        self.load_index_fn = load_index_fn
        self.load_sources_fn = load_sources_fn

    def _mode(self):
        key = os.environ.get("ANTHROPIC_API_KEY","").strip()
        return "anthropic-sdk+web_search" if (HAS_ANTHROPIC and key) else "local-fallback"

    def _get_ev(self, query, use_manual, theme, top_k=20):
        raw = self.search_chunks_fn(query, self.load_index_fn(), top_k=top_k)
        ev = [{"arquivo":e.get("source",""),"pagina":e.get("page",""),
               "score":e.get("score",0),"trecho":e.get("text","")} for e in raw]
        ev = filter_ev(ev, theme, 0.12)
        return rank_manual_first(ev) if use_manual else ev

    def _get_src(self, theme):
        out=[]
        for s in self.load_sources_fn().get("sources",[]):
            if s.get("enabled",True):
                out.append({"id":s.get("id"),"fonte":s.get("name"),"url_base":s.get("url"),
                            "url_busca":source_query_url(s,theme),"category":s.get("category","geral")})
        return out

    def chat(self, message, current_theme="", history=None):
        history = history or []
        theme = infer_theme(message, current_theme, history)
        intent = classify_intent(message)
        use_man = wants_manual(message, history)

        rq = message if intent=="query" else theme
        top_k = 24 if intent=="query" else 20

        pdf_ev = self._get_ev(rq, use_man, theme, top_k)
        ext_src = self._get_src(theme)
        secs = classify_sections(pdf_ev)
        mpts = extract_manual_pts(secs["manual"])

        system = _system_prompt(use_man)
        msgs = _build_msgs(message, history, pdf_ev, ext_src, mpts, theme)
        llm = call_llm(system, msgs, ext_src)

        reply = llm if llm else compose_fallback(message, theme, secs, pdf_ev, ext_src)

        return {
            "ok": True,
            "mode": self._mode(),
            "anthropic_sdk": HAS_ANTHROPIC,
            "llm_used": llm is not None,
            "theme": theme,
            "intent": intent,
            "use_manual": use_man,
            "deep_synthesis": reply,
            "pdf_evidence": pdf_ev,
            "external_sources": ext_src,
            "counts": {"pdf_evidence":len(pdf_ev),"external_sources":len(ext_src),
                       "manual_evidence":len(secs.get("manual",[]))},
            "reply": reply,
        }
