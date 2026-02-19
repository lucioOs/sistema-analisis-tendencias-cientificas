# src/taxonomy.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd

# ============================================================
# Taxonomía robusta (nivel ingeniería) sin volverse inmanejable
#
# Objetivo:
# - Un SOLO "núcleo" de asignación para TODO el sistema:
#   Histórico + Live + Predicción + Comparación
#
# Principios:
# 1) Prioridad por señal fuerte: arXiv category -> macro_area
# 2) Fallback por PLN (keywords) con scoring (no "primer match")
# 3) Trazabilidad: macro_area_source, hits, score
# 4) Contrato estable: siempre produce macro_area canónica
#
# IMPORTANTE (compatibilidad):
# - Soporta:
#     assign_macro_area(df, cat_col="primary_category", text_col="text", out_col="macro_area")
# - Y también modo flexible por coalesce:
#     assign_macro_area(df, text_cols_priority=("text_clean","text","title","abstract"))
# ============================================================

# -------------------------
# 1) Nombres canónicos
# -------------------------
CANONICAL_MACRO_AREAS: list[str] = [
    "IA / Machine Learning",
    "NLP / LLM",
    "Visión / CV",
    "Sistemas / Distribuidos",
    "Redes / Seguridad",
    "Bases de datos / IR",
    "Teoría / Algoritmos",
    "Robótica / Control",
    "HCI / Multimedia",
    "Sin clasificar",
]
DEFAULT_MACRO_AREA = "Sin clasificar"

# Orden de desempate (prioridad fija cuando hay empates en score)
PRIORITY_ORDER: list[str] = [a for a in CANONICAL_MACRO_AREAS if a != DEFAULT_MACRO_AREA] + [DEFAULT_MACRO_AREA]

# -------------------------
# 2) Mapeo arXiv -> macro_area (señal fuerte)
# -------------------------
ARXIV_EXACT_TO_MACRO: dict[str, str] = {
    # IA / ML
    "cs.LG": "IA / Machine Learning",
    "stat.ML": "IA / Machine Learning",
    "cs.AI": "IA / Machine Learning",
    # NLP
    "cs.CL": "NLP / LLM",
    # Visión
    "cs.CV": "Visión / CV",
    # Sistemas / Distribuidos
    "cs.DC": "Sistemas / Distribuidos",
    "cs.OS": "Sistemas / Distribuidos",
    "cs.PF": "Sistemas / Distribuidos",
    "cs.SE": "Sistemas / Distribuidos",
    # Redes / Seguridad
    "cs.NI": "Redes / Seguridad",
    "cs.CR": "Redes / Seguridad",
    # DB / IR
    "cs.DB": "Bases de datos / IR",
    "cs.IR": "Bases de datos / IR",
    # Teoría / Algoritmos
    "cs.DS": "Teoría / Algoritmos",
    "cs.CC": "Teoría / Algoritmos",
    "cs.CG": "Teoría / Algoritmos",
    "cs.FL": "Teoría / Algoritmos",
    # Robótica / Control
    "cs.RO": "Robótica / Control",
}

ARXIV_PREFIX_RULES: list[tuple[str, str]] = [
    ("cs.CV", "Visión / CV"),
    ("cs.CL", "NLP / LLM"),
    ("cs.LG", "IA / Machine Learning"),
    ("cs.AI", "IA / Machine Learning"),
    ("stat.ML", "IA / Machine Learning"),
    ("cs.NI", "Redes / Seguridad"),
    ("cs.CR", "Redes / Seguridad"),
    ("cs.DB", "Bases de datos / IR"),
    ("cs.IR", "Bases de datos / IR"),
    ("cs.DC", "Sistemas / Distribuidos"),
    ("cs.OS", "Sistemas / Distribuidos"),
    ("cs.SE", "Sistemas / Distribuidos"),
    ("cs.DS", "Teoría / Algoritmos"),
    ("cs.CC", "Teoría / Algoritmos"),
    ("cs.RO", "Robótica / Control"),
]

# -------------------------
# 3) Keywords con scoring (fallback)
# -------------------------
KEYWORDS: dict[str, list[str]] = {
    "IA / Machine Learning": [
        "machine learning",
        "deep learning",
        "neural network",
        "transformer",
        "reinforcement learning",
        "policy gradient",
        "supervised",
        "unsupervised",
        "fine-tuning",
        "foundation model",
    ],
    "NLP / LLM": [
        "natural language",
        "nlp",
        "language model",
        "llm",
        "tokenization",
        "prompt",
        "instruction tuning",
        "summarization",
        "translation",
        "named entity",
        "rag",
    ],
    "Visión / CV": [
        "computer vision",
        "image",
        "video",
        "object detection",
        "segmentation",
        "optical flow",
        "yolo",
        "diffusion model",
        "captioning",
    ],
    "Sistemas / Distribuidos": [
        "distributed system",
        "microservices",
        "kubernetes",
        "container",
        "cloud",
        "throughput",
        "latency",
        "fault tolerance",
        "consensus",
        "replication",
    ],
    "Redes / Seguridad": [
        "network",
        "routing",
        "sdn",
        "firewall",
        "intrusion detection",
        "malware",
        "cryptography",
        "authentication",
        "privacy",
        "zero trust",
    ],
    "Bases de datos / IR": [
        "database",
        "sql",
        "index",
        "query optimization",
        "information retrieval",
        "search engine",
        "inverted index",
        "vector database",
        "embedding",
    ],
    "Teoría / Algoritmos": [
        "algorithm",
        "complexity",
        "approximation",
        "graph theory",
        "dynamic programming",
        "np-hard",
        "proof",
        "formal verification",
    ],
    "Robótica / Control": [
        "robot",
        "robotics",
        "slam",
        "localization",
        "path planning",
        "control",
        "pid",
        "model predictive control",
        "manipulation",
    ],
    "HCI / Multimedia": [
        "human-computer interaction",
        "hci",
        "user study",
        "usability",
        "multimedia",
        "audio",
        "speech",
        "interaction design",
    ],
}

WEIGHTED_TERMS: dict[str, dict[str, int]] = {
    "NLP / LLM": {"tokenization": 3, "instruction tuning": 3, "named entity": 2, "rag": 3},
    "Visión / CV": {"optical flow": 3, "yolo": 3, "object detection": 2, "segmentation": 2, "diffusion model": 2},
    "Bases de datos / IR": {
        "inverted index": 3,
        "query optimization": 3,
        "information retrieval": 3,
        "vector database": 3,
    },
    "Sistemas / Distribuidos": {"kubernetes": 3, "consensus": 3, "fault tolerance": 3, "replication": 2},
    "Redes / Seguridad": {"cryptography": 3, "intrusion detection": 3, "zero trust": 3, "malware": 2},
    "IA / Machine Learning": {"reinforcement learning": 3, "transformer": 2, "foundation model": 2, "fine-tuning": 2},
    "Robótica / Control": {"slam": 3, "model predictive control": 3, "path planning": 2},
}

AMBIGUOUS_TERMS: set[str] = {"retrieval", "search", "model", "learning", "data"}

_ARXIV_CAT_TOKEN_RE = re.compile(r"(?:(?<=\s)|^|,|;)\s*([a-z]+\.[A-Z]{2})\s*(?=\s|$|,|;)")

# -------------------------
# 4) Normalización / aliases
# -------------------------
_ALIASES: dict[str, str] = {
    "IA/Machine Learning": "IA / Machine Learning",
    "IA / ML": "IA / Machine Learning",
    "NLP": "NLP / LLM",
    "Vision / CV": "Visión / CV",
    "Vision/CV": "Visión / CV",
    "Visión/CV": "Visión / CV",
    "Sistemas/Distribuidos": "Sistemas / Distribuidos",
    "Redes/Seguridad": "Redes / Seguridad",
    "Bases de datos/IR": "Bases de datos / IR",
    "Sin_clasificar": "Sin clasificar",
    "Sin clasificar": "Sin clasificar",
}


def normalize_macro_area(value: Optional[str]) -> str:
    if not value:
        return DEFAULT_MACRO_AREA
    v = str(value).strip()
    v = _ALIASES.get(v, v)
    return v if v in CANONICAL_MACRO_AREAS else DEFAULT_MACRO_AREA


# -------------------------
# 5) Categorías arXiv: extracción + mapeo
# -------------------------
def extract_arxiv_categories(value: Optional[str]) -> list[str]:
    if not value:
        return []
    s = str(value)
    found = _ARXIV_CAT_TOKEN_RE.findall(s) or []
    seen = set()
    out: list[str] = []
    for c in found:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def extract_primary_arxiv_category(cat_value: Optional[str]) -> Optional[str]:
    cats = extract_arxiv_categories(cat_value)
    return cats[0] if cats else None


def arxiv_category_to_macro(primary_category: Optional[str]) -> Optional[str]:
    cat = extract_primary_arxiv_category(primary_category)
    if not cat:
        return None

    if cat in ARXIV_EXACT_TO_MACRO:
        return normalize_macro_area(ARXIV_EXACT_TO_MACRO[cat])

    for prefix, area in ARXIV_PREFIX_RULES:
        if cat.startswith(prefix):
            return normalize_macro_area(area)

    return None


# -------------------------
# 6) Scoring por texto (fallback)
# -------------------------
@dataclass(frozen=True)
class TextScore:
    macro_area: str
    score: int
    hits: tuple[str, ...]


def _compile_patterns() -> dict[str, list[tuple[re.Pattern, str]]]:
    out: dict[str, list[tuple[re.Pattern, str]]] = {}
    for area, words in KEYWORDS.items():
        pats: list[tuple[re.Pattern, str]] = []
        for w in words:
            pats.append((re.compile(re.escape(w), re.IGNORECASE), w))
        out[area] = pats
    return out


def _compile_weighted_patterns() -> dict[str, list[tuple[re.Pattern, int, str]]]:
    out: dict[str, list[tuple[re.Pattern, int, str]]] = {}
    for area, mapping in WEIGHTED_TERMS.items():
        pats: list[tuple[re.Pattern, int, str]] = []
        for term, weight in mapping.items():
            pats.append((re.compile(re.escape(term), re.IGNORECASE), int(weight), term))
        out[area] = pats
    return out


_PATTERNS = _compile_patterns()
_W_PATTERNS = _compile_weighted_patterns()


def score_text_to_macro(text: Optional[str]) -> TextScore:
    if not text:
        return TextScore(DEFAULT_MACRO_AREA, 0, ())

    t = str(text)

    scores: dict[str, int] = {a: 0 for a in PRIORITY_ORDER if a != DEFAULT_MACRO_AREA}
    hits: dict[str, list[str]] = {a: [] for a in scores.keys()}

    # base hits (+1)
    for area, pats in _PATTERNS.items():
        if area not in scores:
            continue
        for pat, word in pats:
            if pat.search(t):
                if word.lower() in AMBIGUOUS_TERMS:
                    continue
                scores[area] += 1
                hits[area].append(word)

    # weighted (+peso)
    for area, wlist in _W_PATTERNS.items():
        if area not in scores:
            continue
        for pat, weight, term in wlist:
            if pat.search(t):
                scores[area] += int(weight)
                hits[area].append(term)

    def _contains_any(words: Iterable[str]) -> bool:
        for w in words:
            if re.search(re.escape(w), t, flags=re.IGNORECASE):
                return True
        return False

    if _contains_any(["tokenization", "prompt", "instruction tuning", "named entity", "translation"]) and "NLP / LLM" in scores:
        scores["NLP / LLM"] += 2
    if _contains_any(["sql", "inverted index", "query optimization", "database", "information retrieval"]) and "Bases de datos / IR" in scores:
        scores["Bases de datos / IR"] += 2

    best_area = DEFAULT_MACRO_AREA
    best_score = 0
    for area in PRIORITY_ORDER:
        if area == DEFAULT_MACRO_AREA:
            continue
        s = int(scores.get(area, 0))
        if s > best_score:
            best_area, best_score = area, s

    best_hits: list[str] = []
    if best_area != DEFAULT_MACRO_AREA:
        seen = set()
        for h in hits[best_area]:
            if h not in seen:
                seen.add(h)
                best_hits.append(h)
            if len(best_hits) >= 5:
                break

    return TextScore(normalize_macro_area(best_area), int(best_score), tuple(best_hits))


# -------------------------
# 7) API pública (DataFrame)
# -------------------------
def ensure_primary_category(
    df: pd.DataFrame,
    *,
    possible_cols: tuple[str, ...] = ("primary_category", "arxiv_cat", "category", "categories"),
    out_col: str = "primary_category",
) -> pd.DataFrame:
    out = df.copy()

    if out_col in out.columns:
        out[out_col] = out[out_col].apply(extract_primary_arxiv_category)
        return out

    found = None
    for c in possible_cols:
        if c in out.columns:
            found = c
            break

    if not found:
        out[out_col] = None
        return out

    out[out_col] = out[found].apply(extract_primary_arxiv_category)
    return out


def _coalesce_text_for_scoring(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.Series([""] * len(df), index=df.index, dtype=str)

    s = df[available[0]].astype(str).fillna("")
    for c in available[1:]:
        s = s + " " + df[c].astype(str).fillna("")
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def assign_macro_area(
    df: pd.DataFrame,
    *,
    cat_col: str = "primary_category",
    # ✅ COMPATIBILIDAD: soporta text_col como en live_runner
    text_col: Optional[str] = None,
    out_col: str = "macro_area",
    source_col: str = "macro_area_source",
    hits_col: str = "macro_area_hits",
    score_col: str = "macro_area_score",
    # modo flexible (si no pasas text_col): coalesce en este orden
    text_cols_priority: tuple[str, ...] = ("text_clean", "text", "title", "abstract"),
) -> pd.DataFrame:
    """
    Asigna macro_area de forma canónica y trazable.

    Prioridad:
      1) arXiv category (exact/prefix) -> source="arxiv_map" score=999
      2) scoring por texto (keywords + weighted) -> source="keywords" score>=1
      3) Sin clasificar -> source="default" score=0

    Compatibilidad:
      - text_col="text" (o el que sea) fuerza usar SOLO esa columna como input de scoring.
      - Si text_col es None, usa coalesce text_cols_priority.
    """
    if df is None or df.empty:
        out = pd.DataFrame(columns=list(df.columns) if df is not None else [])
        for c in (out_col, source_col, hits_col, score_col, cat_col):
            if c not in out.columns:
                out[c] = []  # type: ignore
        return out

    out = df.copy()

    # 1) asegurar category primaria
    if cat_col not in out.columns:
        out = ensure_primary_category(out, possible_cols=(cat_col, "arxiv_cat", "category", "categories"), out_col=cat_col)
    else:
        out[cat_col] = out[cat_col].apply(extract_primary_arxiv_category)

    # 2) inicializar columnas trazables
    out[out_col] = DEFAULT_MACRO_AREA
    out[source_col] = "default"
    out[hits_col] = ""
    out[score_col] = 0

    # 3) map por categoría (señal fuerte)
    macro_from_cat = out[cat_col].apply(arxiv_category_to_macro) if cat_col in out.columns else None
    if macro_from_cat is not None:
        mask_cat = macro_from_cat.notna()
        if mask_cat.any():
            out.loc[mask_cat, out_col] = macro_from_cat[mask_cat].apply(normalize_macro_area)
            out.loc[mask_cat, source_col] = "arxiv_map"
            out.loc[mask_cat, hits_col] = out.loc[mask_cat, cat_col].fillna("").astype(str)
            out.loc[mask_cat, score_col] = 999

    # 4) fallback scoring por texto (solo donde quedó sin clasificar)
    mask_txt = out[out_col].eq(DEFAULT_MACRO_AREA) | out[out_col].isna()
    if mask_txt.any():
        # ✅ si el caller pasa text_col, úsala primero (y solo esa) si existe
        if text_col and text_col in out.columns:
            blob = out.loc[mask_txt, text_col].astype(str).fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
        else:
            blob = _coalesce_text_for_scoring(out.loc[mask_txt], cols=text_cols_priority)

        scored = blob.apply(score_text_to_macro)

        out.loc[mask_txt, out_col] = scored.apply(lambda x: x.macro_area)
        out.loc[mask_txt, score_col] = scored.apply(lambda x: int(x.score))
        out.loc[mask_txt, hits_col] = scored.apply(lambda x: ", ".join(x.hits))
        out.loc[mask_txt, source_col] = scored.apply(lambda x: "keywords" if int(x.score) > 0 else "default")

    # 5) normalización final
    out[out_col] = out[out_col].apply(normalize_macro_area)

    return out


def taxonomy_report(
    df: pd.DataFrame,
    *,
    cat_col: str = "primary_category",
    macro_col: str = "macro_area",
    source_col: str = "macro_area_source",
    top_n: int = 15,
) -> dict:
    out: dict = {}

    if df is None or df.empty:
        out["coverage_macro"] = {}
        out["coverage_source"] = {}
        out["unmapped_categories"] = {}
        return out

    if macro_col in df.columns:
        out["coverage_macro"] = df[macro_col].value_counts(dropna=False).head(int(top_n)).to_dict()

    if source_col in df.columns:
        out["coverage_source"] = df[source_col].value_counts(dropna=False).head(int(top_n)).to_dict()

    if cat_col in df.columns and macro_col in df.columns:
        mask = df[macro_col].eq(DEFAULT_MACRO_AREA) | df[macro_col].isna()
        if mask.any():
            out["unmapped_categories"] = (
                df.loc[mask, cat_col].fillna("None").astype(str).value_counts().head(int(top_n)).to_dict()
            )
        else:
            out["unmapped_categories"] = {}

    return out
