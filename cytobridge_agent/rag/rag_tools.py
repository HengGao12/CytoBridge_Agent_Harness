"""
RAG helper documentation.
"""
import os
import json
import re
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
from pathlib import Path
import logging
import hashlib
from datetime import datetime

from .config import DIRS, FILES, KB_CONFIG, ensure_dir, get_sentence_transformer_model, get_spacy_model, is_spacy_available, MODEL_CONFIG

logger = logging.getLogger(__name__)

# RAG helper comment.
_use_spacy = is_spacy_available()
_nlp = get_spacy_model() if _use_spacy else None

if _use_spacy:
    logger.info("RAG status message")
else:
    fallback = MODEL_CONFIG["spacy"]["fallback_to_regex"]
    if fallback:
        logger.warning("RAG status message")
    else:
        logger.error("RAG status message")

# RAG helper comment.

REFERENCE_SECTION_PATTERNS = [
    r"references?",
    r"bibliography",
    r"works cited",
    r"literature cited",
    r"references and notes",
    r"reference list",
    r"appendix",
    r"supplementary(?: materials?)?",
    r"supplemental(?: materials?)?",
]

_HEADER_FOOTER_PATTERNS = [
    r'^\s*\d+\s*$',
    r'^\s*Page \d+\s*$',
    r'^\s*page\s*\d+\s*$',
    r'^\s*Articles?\s*$',
    r'^\s*NAT(?:URE?)?\s*[A-Za-z\s]*BiOTecHNOlOgy\s*$',
    r'^\s*NATuRe\s*Bio\s*TeCHNolog\s*Y?\s*$',
    r'^\s*[A-Z]+\s+[A-Z]+\s+[A-Z]+\s*\d+\s*$',
    r'^\s*www\.\S+\.com\s*$',
    r'^\s*nature\.com\s*$',
    r'^\s*VOL\s*\.?\s*\d+\s*$',
    r'^\s*VOLUME\s+\d+\s*$',
    r'^\s*No\.?\s*\d+\s*$',
    r'^\s*[A-Za-z\s]+ \d{4}\s*$',
    r'^\s*doi:\s*10\.\d{4}/.*$',
    r'^\s*https?://\S+$',
    r'^\s*[A-Z]+\s+[A-Z]+\s+\d{1,2}\s*$',
    r'^\s*©\s+\d{4}\s+.*$',
    r'^\s*Received:.*$',
    r'^\s*Accepted:.*$',
    r'^\s*Published:.*$',
]
_COMPILED_HEADER_FOOTER_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _HEADER_FOOTER_PATTERNS]


def _normalize_pdf_line(text: str) -> str:
    line = ''.join(ch if ord(ch) >= 32 or ch in '\n\t' else ' ' for ch in str(text or ""))
    line = line.replace('\u00a0', ' ')
    line = re.sub(r'[ \t]+', ' ', line)
    return line.strip()


def _is_noise_line(line: str) -> bool:
    line_stripped = _normalize_pdf_line(line)
    if not line_stripped:
        return False

    if len(line_stripped) < 3:
        return True

    for pat in _COMPILED_HEADER_FOOTER_PATTERNS:
        if pat.match(line_stripped):
            return True

    lower_line = line_stripped.lower()
    header_keywords = ['articles', 'nature', 'biotechnology', 'vol', 'www.', '.com', 'issn', 'doi']
    if len(line_stripped) < 100:
        keyword_count = sum(1 for kw in header_keywords if kw in lower_line)
        if keyword_count >= 2:
            return True
        if re.search(r'nature.*biotechnology', lower_line, re.IGNORECASE):
            return True
        if re.search(r'articles.*nature', lower_line, re.IGNORECASE):
            return True
        if re.search(r'biotechnology.*vol', lower_line, re.IGNORECASE):
            return True
        if re.search(r'\d+\s*[|]\s*[a-z]+', lower_line):
            return True

    words = line_stripped.split()
    if len(words) <= 3 and all(w.isupper() for w in words) and all(len(w) > 1 for w in words):
        if not any(w in ['FIG.', 'TABLE', 'FIG'] for w in words):
            return True

    return False


def _clean_block_text(text: str) -> str:
    if not text:
        return ""
    text = clean_hyphens(text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    return text.strip()


def _is_section_heading(text: str) -> bool:
    text = _normalize_pdf_line(text)
    if not text:
        return False
    if len(text) > 120:
        return False
    if text.endswith(('.', ';', ':')) and len(text.split()) > 6:
        return False
    lower = text.lower()
    if re.fullmatch(r'(?:\d+(?:\.\d+)*)?\s*(abstract|introduction|background|methods?|materials and methods?|results?|discussion|conclusion|conclusions|experiments?|implementation|appendix|supplementary(?: materials?)?|references?)', lower):
        return True
    if len(text.split()) <= 8 and (text.istitle() or text.isupper()):
        return True
    return False


def _section_name_from_heading(text: str) -> str:
    cleaned = _normalize_pdf_line(text)
    cleaned = re.sub(r'^\d+(?:\.\d+)*\s*', '', cleaned)
    return cleaned.strip() or "Unknown"


def is_reference_section_heading(text: str) -> bool:
    candidate = _normalize_pdf_line(text)
    if not candidate or len(candidate.split()) > 5:
        return False
    lower = candidate.lower()
    return any(re.fullmatch(pattern, lower) for pattern in REFERENCE_SECTION_PATTERNS)


def _find_reference_heading_line(lines: List[str]) -> Optional[int]:
    """
    RAG helper documentation.
    RAG helper documentation.
    RAG helper documentation.
    RAG helper documentation.
    """
    for idx, line in enumerate(lines):
        normalized = _normalize_pdf_line(line)
        if not normalized:
            continue
        lower = normalized.lower()
        for pattern in REFERENCE_SECTION_PATTERNS:
            anchored = rf"^(?:\d+(?:\.\d+)*)?\s*{pattern}(?:\b|[:.])"
            if re.match(anchored, lower):
                return idx
    return None


def _finalize_structured_page(page_number: int, blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    filtered_blocks = [dict(block) for block in blocks if str(block.get("text") or "").strip()]
    if not filtered_blocks:
        return None

    parts: List[str] = []
    cursor = 0
    for block in filtered_blocks:
        if parts:
            parts.append("\n\n")
            cursor += 2
        text = str(block["text"])
        block["char_start"] = cursor
        parts.append(text)
        cursor += len(text)
        block["char_end"] = cursor

    page_text = ''.join(parts).strip()
    if len(page_text) < 50:
        return None

    return {
        "page": int(page_number),
        "text": page_text,
        "blocks": filtered_blocks,
    }


def extract_pdf_page_blocks(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    RAG helper documentation.

    Returns:
        [{"page": int, "text": str, "blocks": [...]}]
    """
    logger.info("RAG status message")
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("RAG status message")
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            pages = []
            for i, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                cleaned = clean_page_text(raw_text, i + 1)
                if not cleaned:
                    continue
                block = {
                    "block_id": f"p{i + 1}_b0",
                    "text": cleaned,
                    "lines": [_normalize_pdf_line(line) for line in cleaned.split('\n') if _normalize_pdf_line(line)],
                    "bbox": None,
                    "is_heading": False,
                    "section_hint": None,
                }
                structured_page = _finalize_structured_page(i + 1, [block])
                if structured_page:
                    pages.append(structured_page)
            return pages
        except Exception:
            return []

    pages: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(str(pdf_path))
        if len(doc) == 0:
            return []

        first_page = doc[0]
        page_height = first_page.rect.height
        page_width = first_page.rect.width
        top_margin = page_height * 0.12
        bottom_margin = page_height * 0.88

        for i in range(len(doc)):
            page = doc[i]
            clip_rect = fitz.Rect(0, top_margin, page_width, bottom_margin)
            page_dict = page.get_text("dict", clip=clip_rect)
            raw_blocks = page_dict.get("blocks", []) if isinstance(page_dict, dict) else []

            structured_blocks: List[Dict[str, Any]] = []
            for block_idx, block in enumerate(raw_blocks):
                if int(block.get("type", 0)) != 0:
                    continue
                lines_payload = block.get("lines", []) or []
                lines: List[str] = []
                for line in lines_payload:
                    spans = line.get("spans", []) or []
                    span_text = ''.join(str(span.get("text") or "") for span in spans)
                    normalized = _normalize_pdf_line(span_text)
                    if normalized and not _is_noise_line(normalized):
                        lines.append(normalized)
                if not lines:
                    continue

                block_text = _clean_block_text('\n'.join(lines))
                if len(block_text) < 20:
                    continue

                is_heading = _is_section_heading(block_text)
                structured_blocks.append(
                    {
                        "block_id": f"p{i + 1}_b{block_idx}",
                        "text": block_text,
                        "lines": lines,
                        "bbox": block.get("bbox"),
                        "is_heading": is_heading,
                        "section_hint": _section_name_from_heading(block_text) if is_heading else None,
                    }
                )

            structured_page = _finalize_structured_page(i + 1, structured_blocks)
            if structured_page:
                pages.append(structured_page)

        doc.close()
    except Exception as e:
        logger.warning("RAG status message")
        try:
            doc = fitz.open(str(pdf_path))
            for i in range(len(doc)):
                raw_text = doc[i].get_text() or ""
                cleaned = clean_page_text(raw_text, i + 1)
                if not cleaned:
                    continue
                block = {
                    "block_id": f"p{i + 1}_b0",
                    "text": cleaned,
                    "lines": [_normalize_pdf_line(line) for line in cleaned.split('\n') if _normalize_pdf_line(line)],
                    "bbox": None,
                    "is_heading": False,
                    "section_hint": None,
                }
                structured_page = _finalize_structured_page(i + 1, [block])
                if structured_page:
                    pages.append(structured_page)
            doc.close()
        except Exception:
            return []

    return pages


def extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    RAG helper documentation.
    """
    pages = extract_pdf_page_blocks(pdf_path)
    return [(int(page["page"]), str(page["text"])) for page in pages]


def extract_pdf_text(pdf_path: str) -> str:
    """
    RAG helper documentation.
    RAG helper documentation.
    
    Args:
        pdf_path: PDFFile paths
        
    Returns:
        RAG helper documentation.
    """
    try:
        return extract_text_up_to_references(Path(pdf_path))
    except Exception as e:
        logger.error("RAG status message")
        raise


def clean_page_text(text: str, page_num: int) -> str:
    """
    RAG helper documentation.
    
    Args:
        RAG helper documentation.
        RAG helper documentation.
        
    Returns:
        RAG helper documentation.
    """
    if not text:
        return ""

    lines = text.split('\n')

    # RAG helper comment.
    start_idx = 0
    for i in range(min(15, len(lines))):
        if _is_noise_line(lines[i]):
            start_idx = i + 1
        else:
            break

    # RAG helper comment.
    end_idx = len(lines)
    for i in range(len(lines) - 1, max(start_idx, len(lines) - 15) - 1, -1):
        if _is_noise_line(lines[i]):
            end_idx = i
        else:
            break

    # RAG helper comment.
    selected_lines = lines[start_idx:end_idx]
    
    # RAG helper comment.
    if not selected_lines:
        return ""
    
    # RAG helper comment.
    cleaned = '\n'.join(selected_lines)
    # RAG helper comment.
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # RAG helper comment.
    cleaned = re.sub(r'^\n+', '', cleaned)  # RAG helper comment.
    cleaned = re.sub(r'\n+$', '', cleaned)  # RAG helper comment.
    
    return cleaned.strip()


def clean_hyphens(text: str) -> str:
    """
    RAG helper documentation.
    """
    # RAG helper comment.
    pattern1 = r'(\w+)-\s+(\w+)'
    text = re.sub(pattern1, r'\1\2', text)
    
    pattern2 = r'(\w+)-\n\s*(\w+)'
    text = re.sub(pattern2, r'\1\2', text)
    
    # RAG helper comment.
    # RAG helper comment.
    symbol_map = {
        '≤': '≤', '≥': '≥', '≠': '≠', '±': '±', '×': '×', '÷': '÷',
        'α': 'α', 'β': 'β', 'γ': 'γ', 'δ': 'δ', 'ε': 'ε', 'θ': 'θ',
        'λ': 'λ', 'μ': 'μ', 'π': 'π', 'σ': 'σ', 'τ': 'τ', 'φ': 'φ',
        'ω': 'ω', '∞': '∞', '∑': '∑', '∏': '∏', '∫': '∫',
    }
    for wrong, correct in symbol_map.items():
        text = text.replace(wrong, correct)  # RAG helper comment.
    
    # RAG helper comment.
    # RAG helper comment.
    # RAG helper comment.
    protected = []
    
    def protect_abbrev(m):
        protected.append(m.group(0))
        return f"__PROTECT_{len(protected)-1}__"
    
    # RAG helper comment.
    abbrev_pattern = r'\b(?:e\.g\.|i\.e\.|et al\.|Fig\.|Figs\.|Table\.|Tables\.|Suppl\.|Ref\.|Eq\.|vs\.|cf\.)\b'
    text = re.sub(abbrev_pattern, protect_abbrev, text, flags=re.IGNORECASE)
    
    # RAG helper comment.
    text = re.sub(r'\.([A-Za-z0-9])', r'. \1', text)
    
    # RAG helper comment.
    text = re.sub(r'([!?])([A-Za-z0-9])', r'\1 \2', text)
    
    # RAG helper comment.
    for i, orig in enumerate(protected):
        text = text.replace(f"__PROTECT_{i}__", orig)
    
    return text


def extract_pages_up_to_references(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    RAG helper documentation.
    
    Args:
        pdf_path: PDFFile paths
        
    Returns:
        RAG helper documentation.
    """
    structured_pages = extract_structured_pages_up_to_references(pdf_path)
    return [(int(page["page"]), str(page["text"])) for page in structured_pages]


def extract_structured_pages_up_to_references(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    RAG helper documentation.
    RAG helper documentation.
    """
    pages = extract_pdf_page_blocks(pdf_path)
    if not pages:
        return []

    kept_pages: List[Dict[str, Any]] = []
    for page in pages:
        page_number = int(page.get("page") or 0)
        blocks = list(page.get("blocks") or [])
        if not blocks:
            continue

        kept_blocks: List[Dict[str, Any]] = []
        stop_here = False
        for block in blocks:
            text = str(block.get("text") or "").strip()
            if not text:
                continue

            raw_lines = list(block.get("lines") or [])
            ref_line_idx = _find_reference_heading_line(raw_lines)
            if ref_line_idx == 0 or is_reference_section_heading(text):
                stop_here = True
                break

            if ref_line_idx is not None and ref_line_idx > 0:
                kept_lines = [_normalize_pdf_line(line) for line in raw_lines[:ref_line_idx] if _normalize_pdf_line(line)]
                kept_text = _clean_block_text('\n'.join(kept_lines))
                if kept_text:
                    trimmed_block = dict(block)
                    trimmed_block["text"] = kept_text
                    trimmed_block["lines"] = kept_lines
                    kept_blocks.append(trimmed_block)
                stop_here = True
                break

            kept_blocks.append(dict(block))

        if kept_blocks:
            structured_page = _finalize_structured_page(page_number, kept_blocks)
            if structured_page:
                kept_pages.append(structured_page)

        if stop_here:
            break

    return kept_pages

def extract_text_up_to_references(pdf_path: Path) -> str:
    """
    RAG helper documentation.
    RAG helper documentation.
    
    Args:
        pdf_path: PDFFile paths
        
    Returns:
        RAG helper documentation.
    """
    pages = extract_pages_up_to_references(pdf_path)
    if not pages:
        return ""
    
    # RAG helper comment.
    return "\n".join([text for _, text in pages])


# RAG helper comment.

def is_figure_or_table_paragraph(para: str) -> Optional[str]:
    """
    RAG helper documentation.
    RAG helper documentation.
    """
    if not para:
        return None
    
    # RAG helper comment.
    para = para.strip()
    if len(para) < 5:  # RAG helper comment.
        return None
    
    # RAG helper comment.
    original_para = para
    header_noise_patterns = [
        r'^Articles?\s+NATure\s+BiOTecHNOlOgy\s*',           # "Articles NATure BiOTecHNOlOgy"
        r'^NATuRe\s+Bio\s+TeCHNolog\s+Y?\s*',                # "NATuRe Bio TeCHNolog Y"
        r'^www\.\S+\.com\s*',  # RAG helper comment.
        r'^VOL\s*\.?\s*\d+\s*',                               # "VOL. 37"
        r'^VOLUME\s+\d+\s*',                                  # "VOLUME 37"
        r'^No\.?\s*\d+\s*',                                   # "No. 37"
        r'^[A-Z]+\s+[A-Z]+\s+\d+\s*[|]\s*',                  # "NATURE BIOTECHNOLOGY 37 |"
        r'^\s*\d+\s*[|]\s*[A-Za-z\s]+\s*',                   # "37 | Articles"
        r'^\s*(?:Articles|NATURE|BIOTECHNOLOGY|BIOTEC)\s+',  # RAG helper comment.
    ]
    for pattern in header_noise_patterns:
        para = re.sub(pattern, '', para, flags=re.IGNORECASE).strip()
    
    # RAG helper comment.
    if len(para) < 5:
        return None
    
    # RAG helper comment.
    lower_para = para.lower()
    noise_keywords = ['articles', 'nature', 'biotechnology', 'www.', '.com', 'vol', 'doi:', 'issn']
    noise_count = sum(1 for kw in noise_keywords if kw in lower_para)
    if noise_count >= 2 and len(para) < 200:
        return None
    # RAG helper comment.
    if para != original_para and len(original_para) - len(para) > 20:
        logger.debug("RAG status message")
    # RAG helper comment.
    
    # RAG helper comment.
    # RAG helper comment.
    figure_patterns = [
        # RAG helper comment.
        r'^(?:Figure|FIGURES?|Fig\.?)\s*\d+[A-Za-z]?[\.:]?\s+',
        r'^(?:Supplementary|Supplemental)\s+(?:Figure|Fig\.?)\s*\d+[A-Za-z]?[\.:]?\s+',
        r'^(?:FIGS?\.?)\s*\d+(?:[–-]\d+)?[\.:]?\s+',  # RAG helper comment.
        
        # RAG helper comment.
        r'(?:^|\s)(?:Figure|FIGURES?|Fig\.?)\s*\d+[A-Za-z]?[\.:]?(?:\s|$)',
        r'(?:^|\s)(?:Supplementary|Supplemental)\s+(?:Figure|Fig\.?)\s*\d+[A-Za-z]?[\.:]?(?:\s|$)',
        r'(?:^|\s)(?:FIGS?\.?)\s*\d+(?:[–-]\d+)?[\.:]?(?:\s|$)',
        
        # RAG helper comment.
        r'(?:^|\s)(?:Figure|Fig\.?)\s*\(\s*\d+[A-Za-z]?\s*\)[\.:]?\s*',
        
        # RAG helper comment.
        r'\s+(?:Figure|Fig\.?)\s*\d+[A-Za-z]?[\.:]?\s*$',
    ]
    
    table_patterns = [
        # RAG helper comment.
        r'^(?:Table|TABLES?|Tab\.?)\s*\d+[A-Za-z]?[\.:]?\s+',
        r'^(?:Supplementary|Supplemental)\s+(?:Table|Tab\.?)\s*\d+[A-Za-z]?[\.:]?\s+',
        r'^(?:TABS?\.?)\s*\d+(?:[–-]\d+)?[\.:]?\s+',
        
        # RAG helper comment.
        r'(?:^|\s)(?:Table|TABLES?|Tab\.?)\s*\d+[A-Za-z]?[\.:]?(?:\s|$)',
        r'(?:^|\s)(?:Supplementary|Supplemental)\s+(?:Table|Tab\.?)\s*\d+[A-Za-z]?[\.:]?(?:\s|$)',
        r'(?:^|\s)(?:TABS?\.?)\s*\d+(?:[–-]\d+)?[\.:]?(?:\s|$)',
        
        # RAG helper comment.
        r'(?:^|\s)(?:Table|Tab\.?)\s*\(\s*\d+[A-Za-z]?\s*\)[\.:]?\s*',
        
        # RAG helper comment.
        r'\s+(?:Table|Tab\.?)\s*\d+[A-Za-z]?[\.:]?\s*$',
    ]
    
    # RAG helper comment.
    compiled_figure_patterns = [re.compile(p, re.IGNORECASE) for p in figure_patterns]
    compiled_table_patterns = [re.compile(p, re.IGNORECASE) for p in table_patterns]
    
    # RAG helper comment.
    for pattern in compiled_figure_patterns:
        if pattern.search(para):
            return 'figure'
    
    for pattern in compiled_table_patterns:
        if pattern.search(para):
            return 'table'
    
    # RAG helper comment.
    lower_para = para.lower()
    
    # RAG helper comment.
    figure_keywords = ['figure', 'fig.', 'fig', 'figs', 'figs.', 'supplementary figure', 'supplemental figure']
    table_keywords = ['table', 'tab.', 'tab', 'tables', 'tabs', 'supplementary table', 'supplemental table']
    
    # RAG helper comment.
    has_digit = bool(re.search(r'\d', para[:50]))  # RAG helper comment.
    
    # RAG helper comment.
    for kw in figure_keywords:
        if kw in lower_para:
            # RAG helper comment.
            idx = lower_para.find(kw)
            if idx >= 0:
                after_kw = para[idx + len(kw):].strip()
                if after_kw and (after_kw[0].isdigit() or after_kw.startswith(':') or after_kw.startswith('.')):
                    return 'figure'
    
    for kw in table_keywords:
        if kw in lower_para:
            idx = lower_para.find(kw)
            if idx >= 0:
                after_kw = para[idx + len(kw):].strip()
                if after_kw and (after_kw[0].isdigit() or after_kw.startswith(':') or after_kw.startswith('.')):
                    return 'table'
    
    # RAG helper comment.
    if has_digit and len(para) < 300:
        for kw in figure_keywords:
            if kw in lower_para:
                return 'figure'
        for kw in table_keywords:
            if kw in lower_para:
                return 'table'
    
    return None


def split_sentences(text: str) -> List[str]:
    """
    RAG helper documentation.
    """
    if not text:
        return []
    
    # RAG helper comment.
    # RAG helper comment.
    protected = []
    
    def protect_abbrev(m):
        protected.append(m.group(0))
        return f"__PROTECT_SENT_{len(protected)-1}__"
    
    # RAG helper comment.
    abbrev_pattern = r'\b(?:e\.g\.|i\.e\.|et al\.|Fig\.|Figs\.|Table\.|Tables\.|Suppl\.|Ref\.|Eq\.|vs\.|cf\.)\b'
    text = re.sub(abbrev_pattern, protect_abbrev, text, flags=re.IGNORECASE)
    
    if _use_spacy:
        # RAG helper comment.
        doc = _nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # RAG helper comment.
        sentences = [s for s in sentences if len(s) > 20 or s.endswith(('.', '!', '?'))]
    else:
        # RAG helper comment.
        # RAG helper comment.
        sentence_endings = r'(?<=[。！？；;\.!?])\s+(?=[A-Z"“])|(?<=[。！？；;\.!?])$'
        raw_sents = re.split(sentence_endings, text)
        sentences = [s.strip() for s in raw_sents if s.strip() and len(s.strip()) > 10]
    
    # RAG helper comment.
    for i, orig in enumerate(protected):
        for j, sent in enumerate(sentences):
            sentences[j] = sentences[j].replace(f"__PROTECT_SENT_{i}__", orig)
    
    return sentences


def split_paragraphs(text: str) -> List[str]:
    """
    RAG helper documentation.
    RAG helper documentation.
    
    Args:
        RAG helper documentation.
        
    Returns:
        RAG helper documentation.
    """
    if not text:
        return []
    
    # RAG helper comment.
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # RAG helper comment.
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # RAG helper comment.
    paras = re.split(r'\n\s*\n', text)
    
    # RAG helper comment.
    paragraphs = [p.strip() for p in paras if p.strip()]
    
    # RAG helper comment.
    # RAG helper comment.
    return paragraphs


def protect_math_formulas(text: str) -> Tuple[str, List[str]]:
    """
    RAG helper documentation.
    
    RAG helper documentation.
    - LaTeX: $...$, $$...$$, \[...\], \(...\)
    RAG helper documentation.
    """
    if not text:
        return text, []
    
    protected = []
    formula_count = 0
    
    def protect_latex(match):
        nonlocal formula_count
        formula = match.group(0)
        protected.append(formula)
        placeholder = f"__MATH_FORMULA_{formula_count}__"
        formula_count += 1
        return placeholder
    
    # RAG helper comment.
    patterns = [
        # RAG helper comment.
        (r'\$\$[^\$]+\$\$', re.DOTALL),           # $$...$$
        (r'\\\[[^\]]+\\\]', re.DOTALL),           # \[...\]
        (r'\\begin\{equation\}.*?\\end\{equation\}', re.DOTALL | re.IGNORECASE),
        (r'\\begin\{align\}.*?\\end\{align\}', re.DOTALL | re.IGNORECASE),
        (r'\\begin\{math\}.*?\\end\{math\}', re.DOTALL | re.IGNORECASE),
        
        # RAG helper comment.
        (r'\\\([^\)]+\\\)', re.DOTALL),           # \(...\)
        (r'\$[^\$]+\$', re.DOTALL),  # RAG helper comment.
    ]
    
    protected_text = text
    for pattern, flags in patterns:
        protected_text = re.sub(pattern, protect_latex, protected_text, flags=flags)
    
    return protected_text, protected


def restore_math_formulas(text: str, protected: List[str]) -> str:
    """
    RAG helper documentation.
    """
    result = text
    for i, formula in enumerate(protected):
        result = result.replace(f"__MATH_FORMULA_{i}__", formula)
    return result


def chunk_by_paragraphs(
    pages: List[Union[Tuple[int, str], Dict[str, Any]]],
    chunk_size: int = None,  # RAG helper comment.
    chunk_overlap: int = None  # RAG helper comment.
) -> List[Dict[str, Any]]:
    """
    RAG helper documentation.

    RAG helper documentation.
    RAG helper documentation.
    RAG helper documentation.
    RAG helper documentation.
    RAG helper documentation.

    Args:
        RAG helper documentation.
        RAG helper documentation.
        RAG helper documentation.

    Returns:
        [
            {
                "chunk_id": str,
                "page": int,
                "text": str,
                "type": str,
                "section": Optional[str],
                "block_id": Optional[str],
                "char_start": Optional[int],
                "char_end": Optional[int],
                "sentence_count": int,
            },
            ...
        ]
    """
    if not pages:
        return []

    all_chunks: List[Dict[str, Any]] = []
    chunk_id_counter = 0

    structured_pages: List[Dict[str, Any]] = []
    for item in pages:
        if isinstance(item, dict):
            structured_pages.append(item)
        else:
            page_num, page_text = item
            block_text = clean_hyphens(str(page_text or ""))
            synthetic_page = _finalize_structured_page(
                int(page_num),
                [
                    {
                        "block_id": f"p{page_num}_b0",
                        "text": block_text,
                        "lines": [_normalize_pdf_line(line) for line in block_text.split('\n') if _normalize_pdf_line(line)],
                        "bbox": None,
                        "is_heading": False,
                        "section_hint": None,
                    }
                ],
            )
            if synthetic_page:
                structured_pages.append(synthetic_page)

    current_section: Optional[str] = None
    for page in structured_pages:
        page_num = int(page.get("page") or 0)
        page_text = str(page.get("text") or "")
        blocks = list(page.get("blocks") or [])
        if not blocks:
            continue

        for block in blocks:
            block_text = clean_hyphens(str(block.get("text") or ""))
            if not block_text:
                continue

            block_id = str(block.get("block_id") or f"p{page_num}_b{chunk_id_counter}")
            block_start = block.get("char_start")
            block_end = block.get("char_end")
            block_is_heading = bool(block.get("is_heading"))
            block_section_hint = block.get("section_hint")
            if block_section_hint:
                current_section = str(block_section_hint)

            protected_text, math_formulas = protect_math_formulas(block_text)
            paragraphs = split_paragraphs(protected_text) or [protected_text]
            search_from = int(block_start or 0)

            for para in paragraphs:
                para_with_math = restore_math_formulas(para, math_formulas).strip()
                if len(para_with_math) < 20:
                    continue

                chart_type = is_figure_or_table_paragraph(para_with_math)
                if block_is_heading:
                    chunk_type = "section_heading"
                    chunk_id = f"heading_chunk_{chunk_id_counter}"
                elif chart_type == 'figure':
                    chunk_type = "figure"
                    chunk_id = f"fig_chunk_{chunk_id_counter}"
                elif chart_type == 'table':
                    chunk_type = "table"
                    chunk_id = f"table_chunk_{chunk_id_counter}"
                else:
                    chunk_type = "text"
                    chunk_id = f"chunk_{chunk_id_counter}"

                para_start = page_text.find(para_with_math, search_from)
                if para_start < 0:
                    para_start = int(block_start or 0)
                para_end = para_start + len(para_with_math)
                search_from = max(search_from, para_end)

                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_num,
                        "text": para_with_math,
                        "type": chunk_type,
                        "section": current_section,
                        "block_id": block_id,
                        "char_start": para_start,
                        "char_end": para_end if para_end > para_start else block_end,
                        "sentence_count": len(split_sentences(para_with_math)),
                    }
                )
                chunk_id_counter += 1

    logger.info("RAG status message")
    return all_chunks


# RAG helper comment.
def load_apa_citations() -> Dict[str, str]:
    """
    RAG helper documentation.
    
    Returns:
        RAG helper documentation.
    """
    apa_citations = {}
    apa_file = FILES["apa_citations"]

    if not apa_file.exists():
        logger.warning("RAG status message")
        return apa_citations

    try:
        with open(apa_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                filename = lines[i]
                citation = lines[i + 1]

                if not filename.lower().endswith('.pdf'):
                    filename = f"{filename}.pdf"

                filename = os.path.basename(filename)
                apa_citations[filename] = citation
                logger.debug("RAG status message")

        logger.info("RAG status message")
        return apa_citations

    except Exception as e:
        logger.error("RAG status message")
        return apa_citations


# RAG helper comment.
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    RAG helper documentation.
    
    Args:
        RAG helper documentation.
        RAG helper documentation.
        
    Returns:
        RAG helper documentation.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


# RAG helper comment.
def is_likely_reference_block(text: str) -> bool:
    """
    RAG helper documentation.
    
    Args:
        RAG helper documentation.
        
    Returns:
        RAG helper documentation.
    """
    normalized_text = str(text or "")
    if not normalized_text.strip():
        return False

    lines = [line.strip() for line in normalized_text.split('\n') if line.strip()]
    if not lines:
        return False

    num_lines = len(lines)
    starts_with_digit = sum(1 for line in lines if re.match(r'^\s*(\[\d+\]|\d+[.)])', line))
    has_etal = bool(re.search(r'\bet al\.', normalized_text, flags=re.IGNORECASE))
    years = re.findall(r'\b(?:19|20)\d{2}\b', normalized_text)
    has_year = bool(years)
    has_doi = bool(re.search(r'\bdoi\b|https?://\S+|10\.\d{4,9}/', normalized_text, flags=re.IGNORECASE))
    journal_markers = bool(
        re.search(
            r'\b(?:Nat\.|Nature|Cell|Science|PNAS|Proc\.|bioRxiv|arXiv|Commun\.|J\.)\b',
            normalized_text,
            flags=re.IGNORECASE,
        )
    )
    references_heading = bool(
        re.match(r'^\s*references?\b', lines[0], flags=re.IGNORECASE)
    )

    # RAG helper comment.
    flattened_reference_like = num_lines <= 2 and len(years) >= 3 and (has_etal or has_doi or journal_markers)

    return (
        references_heading
        or flattened_reference_like
        or (starts_with_digit > 0 and (has_etal or has_year or has_doi))
        or (starts_with_digit > num_lines * 0.3 and (has_year or journal_markers))
    )


# RAG helper comment.
def get_cache_key(*args, **kwargs) -> str:
    """
    RAG helper documentation.
    
    Returns:
        RAG helper documentation.
    """
    content = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(content.encode()).hexdigest()


# RAG helper comment.
def get_text_stats(text: str) -> Dict[str, Any]:
    """
    RAG helper documentation.
    
    Args:
        RAG helper documentation.
        
    Returns:
        RAG helper documentation.
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0
        }
    
    words = re.findall(r'\b\w+\b', text)
    sentences = split_sentences(text)  # RAG helper comment.
    paragraphs = split_paragraphs(text)
    
    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs)
    }
