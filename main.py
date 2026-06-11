# hybrid_invoice_extractor.py
# Batch scanned PDF invoice extractor.
# Flow:
#   1) Extract embedded text when available
#   2) Otherwise OCR locally with Tesseract
#   3) Run regex/rule extraction
#   4) Call OpenAI text model only when fields are missing or --llm-mode always
#   5) Write one JSON per PDF + combined JSON/CSV

import os
import re
import json
import time
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

# Reduce CPU thread overuse on Mac
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import fitz  # PyMuPDF
from pydantic import BaseModel

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from PIL import Image, ImageOps, ImageFilter
except Exception:
    Image = None
    ImageOps = None
    ImageFilter = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


IMPORTANT_FIELDS = {
    "invoice_number",
    "client_name",
    "hsn_code",
    "total_quantity",
    "value",
    "currency",
}

REQUIRED_FIELDS_FOR_HIGH_CONFIDENCE = {
    "invoice_number",
    "client_name",
    "hsn_code",
    "total_quantity",
    "value",
    "currency",
}


class ExtractedInvoice(BaseModel):
    is_one_liner: Optional[bool] = None
    invoice_number: Optional[str] = None
    client_name: Optional[str] = None
    hsn_code: Optional[str] = None
    total_quantity: Optional[str] = None
    value: Optional[str] = None
    currency: Optional[str] = None
    confidence: Optional[float] = None
    source_file: Optional[str] = None
    source_pages: Optional[str] = None
    extraction_method: Optional[str] = None
    error: Optional[str] = None


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip(" :-\n\t")


def normalize_amount(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = clean_text(value).upper()
    s = re.sub(r"\b(EUR|GBP|USD|INR)\b", "", s).strip()
    return s or None


def infer_currency(text: str, value: Optional[str] = None) -> Optional[str]:
    combined = f"{value or ''}\n{text or ''}"
    m = re.search(r"\b(EUR|GBP|USD|INR)\b", combined, flags=re.I)
    return m.group(1).upper() if m else None


def has_invoice_data(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    return any(row.get(k) not in (None, "", []) for k in IMPORTANT_FIELDS)


def quality_score(row: Dict[str, Any]) -> float:
    present = sum(1 for k in REQUIRED_FIELDS_FOR_HIGH_CONFIDENCE if row.get(k) not in (None, "", []))
    return present / max(len(REQUIRED_FIELDS_FOR_HIGH_CONFIDENCE), 1)


def row_needs_llm(row: Dict[str, Any], min_quality: float) -> bool:
    return quality_score(row) < min_quality


# -----------------------------
# PDF text/OCR
# -----------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    parts: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            if len(text.strip()) > 50:
                parts.append(f"\n--- PAGE {page_no} ---\n{text}")
    return "\n".join(parts).strip()


def render_page_to_image(pdf_path: str, page_index: int, image_path: str, dpi: int) -> None:
    with fitz.open(pdf_path) as doc:
        page = doc[page_index]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(image_path)


def preprocess_for_tesseract(image_path: str) -> str:
    """Light preprocessing improves scanned invoice OCR without adding heavy deps."""
    if Image is None:
        return image_path
    img = Image.open(image_path)
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    processed_path = str(Path(image_path).with_suffix(".processed.png"))
    img.save(processed_path)
    return processed_path


def ocr_image_tesseract(image_path: str) -> List[str]:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed. Run: pip install pytesseract pillow")
    if shutil.which("tesseract") is None:
        raise RuntimeError("Tesseract binary not found. On Mac run: brew install tesseract")

    processed = preprocess_for_tesseract(image_path)
    config = os.getenv("TESSERACT_CONFIG", "--oem 3 --psm 6 -c preserve_interword_spaces=1")
    timeout_seconds = env_int("TESSERACT_TIMEOUT_SECONDS", 35)
    text = pytesseract.image_to_string(
        processed,
        lang=os.getenv("TESSERACT_LANG", "eng"),
        config=config,
        timeout=timeout_seconds,
    )
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def ocr_pdf_tesseract(pdf_path: str, max_pages: Optional[int]) -> str:
    dpi = env_int("OCR_DPI", 180)
    all_text: List[str] = []

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
    pages_to_process = total_pages if max_pages is None else min(max_pages, total_pages)

    print(f"OCR needed. Engine: tesseract. Pages found: {total_pages}. Processing: {pages_to_process}. DPI: {dpi}.", flush=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        for page_no in range(1, pages_to_process + 1):
            start = time.time()
            print(f"  OCR page {page_no}/{pages_to_process} started", flush=True)
            image_path = os.path.join(temp_dir, f"page_{page_no}.png")
            try:
                render_page_to_image(pdf_path, page_no - 1, image_path, dpi=dpi)
                lines = ocr_image_tesseract(image_path)
                error = None
            except Exception as e:
                lines, error = [], repr(e)

            elapsed = round(time.time() - start, 1)
            if error:
                print(f"  OCR page {page_no}/{pages_to_process} failed/skipped: {error} ({elapsed}s)", flush=True)
            else:
                print(f"  OCR page {page_no}/{pages_to_process} done. Lines: {len(lines)} ({elapsed}s)", flush=True)
            all_text.append(f"\n--- PAGE {page_no} ---\n" + (f"[OCR_ERROR] {error}\n" if error else "") + "\n".join(lines))

    return "\n".join(all_text).strip()


def get_pdf_text(pdf_path: str, max_ocr_pages: Optional[int], force_ocr: bool = False) -> Dict[str, str]:
    if not force_ocr:
        print("Extracting embedded text...", flush=True)
        embedded_text = extract_text_from_pdf(pdf_path)
        if len(embedded_text) >= 200:
            print(f"Embedded text found. Length: {len(embedded_text)}", flush=True)
            return {"text": embedded_text, "method": "embedded_text"}
        print("Embedded text not enough. Switching to local Tesseract OCR.", flush=True)
    else:
        print("Force OCR enabled. Using local Tesseract OCR.", flush=True)

    text = ocr_pdf_tesseract(pdf_path, max_pages=max_ocr_pages)
    return {"text": text, "method": "ocr_tesseract"}


# -----------------------------
# Rule extraction
# -----------------------------



def normalize_ocr_for_rules(text: str) -> str:
    """Normalize common OCR mistakes without changing the saved OCR text."""
    if not text:
        return ""
    t = text
    # OCR sometimes reads Invoice as lnvoice/Invo1ce/lhvoice and N° as N*/N0.
    t = re.sub(r"\b[Ii1l]nvo[i1l]ce\b", "Invoice", t, flags=re.I)
    t = re.sub(r"\bInvolce\b", "Invoice", t, flags=re.I)
    t = re.sub(r"\bInv0ice\b", "Invoice", t, flags=re.I)
    t = re.sub(r"N[\*ºo0]", "N°", t, flags=re.I)
    t = re.sub(r"Amount\s+Payab[1l]e", "Amount Payable", t, flags=re.I)
    t = re.sub(r"Commod[i1l]ty", "Commodity", t, flags=re.I)
    return t


def split_text_pages(text: str) -> List[Dict[str, Any]]:
    """Return page-numbered chunks from OCR/debug text."""
    parts = re.split(r"(?i)\n?--- PAGE\s+(\d+)\s+---\n?", text or "")
    pages: List[Dict[str, Any]] = []
    for i in range(1, len(parts), 2):
        try:
            page_no = int(parts[i])
        except Exception:
            continue
        body = parts[i + 1] if i + 1 < len(parts) else ""
        pages.append({"page_no": page_no, "text": body})
    if not pages and clean_text(text):
        pages.append({"page_no": None, "text": text})
    return pages


def page_has_invoice_signal(page_text: str) -> bool:
    t = normalize_ocr_for_rules(page_text)
    return bool(
        re.search(r"Invoice\s*(?:N[°oº0]?|No\.?|Number|#)?\s*[:\-]?\s*[0-9]{5,}", t, re.I)
        or (re.search(r"Amount\s+Payable|Net\s+Total|VAT\s+Base", t, re.I) and re.search(r"Bill\s*To|Customer\s+VAT|Payment\s+term", t, re.I))
    )


def build_invoice_blocks_from_pages(text: str) -> List[str]:
    """Prefer page-level invoice blocks. This avoids transport/customs pages swallowing invoice pages."""
    pages = split_text_pages(text)
    if not pages:
        return []
    blocks: List[str] = []
    for p in pages:
        if page_has_invoice_signal(p["text"]):
            page_no = p.get("page_no")
            header = f"--- PAGE {page_no} ---\n" if page_no is not None else ""
            blocks.append(header + p["text"])
    return blocks

def parse_page_numbers(block: str) -> Optional[str]:
    nums = re.findall(r"--- PAGE\s+(\d+)\s+---", block, flags=re.I)
    return ",".join(dict.fromkeys(nums)) if nums else None


def split_invoice_blocks(text: str) -> List[str]:
    # First try page-level detection. For scanned export packets, each actual invoice is usually a page.
    page_blocks = build_invoice_blocks_from_pages(text)
    if page_blocks:
        return page_blocks

    # Fallback: Match invoice markers in continuous text. Keep one record per actual invoice number.
    normalized = normalize_ocr_for_rules(text)
    pattern = r"(?i)Invoice\s*(?:N[°oº0]?|No\.?|Number|#)?\s*[:\-]?\s*([0-9]{5,})"
    matches = list(re.finditer(pattern, normalized))
    if not matches:
        return [text] if clean_text(text) else []

    blocks: List[str] = []
    for idx, m in enumerate(matches):
        start = max(0, m.start() - 1200)
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        blocks.append(normalized[start:end])
    return blocks


def first_match(text: str, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, re.I | re.S)
        if m:
            return clean_text(m.group(1) if m.groups() else m.group(0))
    return None


def rule_based_extract(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    text = normalize_ocr_for_rules(text)
    compact = re.sub(r"[ \t]+", " ", text)

    result["invoice_number"] = first_match(compact, [
        r"Invoice\s*(?:N[°oº0]?|No\.?|Number|#)?\s*[:\-]?\s*([0-9]{5,})",
        r"\bINV(?:OICE)?\s*[:#\- ]+([0-9]{5,})",
    ])

    result["client_name"] = first_match(text, [
        r"\b(DALER\s+ROWNEY\s+LTD)\b",
        r"\b(DALER\s+ROWNEY\s+MANUFACTURING)\b",
        r"Bill\s*To(?:.|\n){0,500}?\b(DALER\s+ROWNEY\s+LTD)\b",
        r"(?:Buyer|Consignee|Sold\s*To)\s*[:\n]\s*(.+?)(?:\n|VAT|Customer)",
    ])

    result["hsn_code"] = first_match(text, [
        r"commodity\s*Code:.*?\n\s*([0-9]{6,18})",
        r"Commodity\s*Code:.*?\n\s*([0-9]{6,18})",
        r"commodity\s*Code\s*[:\s]+(?:Net\s*Weight\s*:\(?kg\)?\s*)?(?:Net\s*Total\s*:)?(?:Country\s*of\s*Origin:)?\s*([0-9]{6,18})",
        r"Commodity\s*Code\s*[:\s]+([0-9]{6,18})",
        r"Commod(?:ity)?\s*Code\s*[:\s]+([0-9]{6,18})",
        r"Code\s+des\s+marchandises\s*[:\s]*([0-9]{6,18})",
        r"(?:HSN|H\.S\.?\s*Code|HS\s*Code)\s*[:\s]+([0-9]{6,18})",
        r"\b(48025810|48025590|48025690)\b",
    ])

    qty = first_match(text, [
        r"commodity\s*Code:.*?Net\s*Weight\s*:\(?kg\)?.*?\n\s*[0-9]{6,18}\s+([0-9,.]+)",
        r"Net\s*Weight\s*\(kg\)\s*([0-9,.]+)",
        r"Net\s*Weight\s*[:\s]*(?:KG)?\s*([0-9,.]+\s*(?:KG|KGS)?)",
        r"Inv\s*qty\s*[:\s]*([0-9,.]+\s*(?:KG|KGS|PCS|M2|ROLL|NO|NOS))",
        r"Masse\s+nette\s*\(kg\)\s*([0-9,.]+)",
    ])
    if qty:
        result["total_quantity"] = qty if re.search(r"kg|kgs|pcs|m2|roll|no|nos", qty, re.I) else f"{qty} KG"

    value = first_match(text, [
        r"Amount\s+Payable.*?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})\s*(?:EUR|GBP|USD|INR)?)",
        r"Net\s+Total\s+([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})\s*(?:EUR|GBP|USD|INR)?)",
        r"AMOUNT\s+Payable\s+([0-9,.]+\s*(?:EUR|GBP|USD|INR)?)",
        r"Amount\s+payable\s+([0-9,.]+\s*(?:EUR|GBP|USD|INR)?)",
        r"Invoice\s+Total\s+([0-9,.]+\s*(?:EUR|GBP|USD|INR)?)",
        r"Net\s+Total\s*[:\s]+([0-9,.]+\s*(?:EUR|GBP|USD|INR)?)",
        r"TOTAL\s+AMOUNT\s+([0-9,.]+\s*(?:EUR|GBP|USD|INR)?)",
        r"Valeur\s+statistique\s*([0-9,.]+)",
    ])
    if value:
        result["value"] = normalize_amount(value)

    currency = infer_currency(compact, value)
    if currency:
        result["currency"] = currency

    product_rows = re.findall(r"(?m)^\s*(?:C|F|A)?[A-Z0-9][A-Z0-9\-]{5,}\s+.+?\s+[0-9,.]+\s*(?:KG|KGS)\b", text, re.I)
    if product_rows:
        result["is_one_liner"] = len(product_rows) == 1

    result["source_pages"] = parse_page_numbers(text)
    result["confidence"] = round(max(0.2, quality_score(result)), 2)
    return {k: v for k, v in result.items() if v is not None and v != ""}


# -----------------------------
# OpenAI text-only cleanup
# -----------------------------

def safe_json_loads(content: str) -> Any:
    content = (content or "").strip()
    content = re.sub(r"^```json", "", content, flags=re.I).strip()
    content = re.sub(r"^```", "", content).strip()
    content = re.sub(r"```$", "", content).strip()
    return json.loads(content)


def get_openai_api_key() -> str:
    """Return API key safely stripped. Newlines/spaces make HTTP Authorization header invalid."""
    return (os.getenv("OPENAI_API_KEY") or "").strip()


def llm_available() -> bool:
    return OpenAI is not None and bool(get_openai_api_key())


def llm_extract_many_from_text(text: str, rule_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not get_openai_api_key():
        raise RuntimeError("OPENAI_API_KEY is not set")
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")

    client = OpenAI(api_key=get_openai_api_key(), timeout=float(os.getenv("OPENAI_TIMEOUT", "90")), max_retries=1)
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    api_key = get_openai_api_key()
    print(f"OpenAI key loaded: yes, length={len(api_key)}, starts={api_key[:7]}...", flush=True)

    prompt = f"""
Extract fields from OCR text. Return ONLY valid JSON array. One object per actual invoice only.
Do not create rows for CMR/transport/export declaration pages unless they contain actual invoice fields.

Fields:
is_one_liner, invoice_number, client_name, hsn_code, total_quantity, value, currency, confidence, source_pages

Rules:
- client_name = Bill To / buyer / consignee.
- hsn_code = Commodity Code / Code des marchandises / HSN / HS Code.
- total_quantity = Net Weight / Inv qty.
- value = Amount Payable / Invoice Total / Net Total / Total Amount.
- currency = EUR/GBP/USD/INR if visible.
- source_pages = page numbers where the invoice data came from.
- Missing value must be null.
- Return numbers as strings exactly as read, without inventing values.

Regex/rule hints:
{json.dumps(rule_results, indent=2, ensure_ascii=False)}

OCR text:
{text[:60000]}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You extract invoice data from OCR text and return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    parsed = safe_json_loads(response.choices[0].message.content or "[]")
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def merge_rows(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(fallback)
    for k, v in primary.items():
        if v is not None and v != "":
            out[k] = v
    if out.get("value"):
        out["value"] = normalize_amount(out.get("value"))
    if not out.get("currency"):
        out["currency"] = infer_currency("", out.get("value"))
    return out


def extract_documents_from_pdf(
    pdf_path: str,
    max_ocr_pages: Optional[int] = None,
    save_ocr_text: bool = True,
    llm_mode: str = "missing",
    min_quality: float = 0.85,
    force_ocr: bool = False,
) -> List[ExtractedInvoice]:
    print("Extracting document...", flush=True)
    text_data = get_pdf_text(pdf_path, max_ocr_pages=max_ocr_pages, force_ocr=force_ocr)
    text = text_data["text"]
    method = text_data["method"]
    print(f"Text method: {method}. Text length: {len(text)}", flush=True)

    if save_ocr_text:
        txt_path = str(Path(pdf_path).with_suffix(".ocr.txt"))
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"OCR/debug text saved: {txt_path}", flush=True)
        except Exception as e:
            print(f"Could not save OCR/debug text: {e}", flush=True)

    blocks = split_invoice_blocks(text)
    print(f"Detected invoice blocks: {len(blocks)}", flush=True)

    rule_results = [rule_based_extract(block) for block in blocks]
    rule_results = [r for r in rule_results if has_invoice_data(r)]
    if not rule_results and clean_text(text):
        fallback = rule_based_extract(text)
        if has_invoice_data(fallback):
            rule_results = [fallback]

    print(f"Rule results: {rule_results}", flush=True)

    should_call_llm = False
    if llm_mode == "always":
        should_call_llm = True
    elif llm_mode == "missing":
        should_call_llm = (not rule_results) or any(row_needs_llm(r, min_quality) for r in rule_results)
    elif llm_mode == "never":
        should_call_llm = False

    llm_results: List[Dict[str, Any]] = []
    llm_error: Optional[str] = None
    if should_call_llm:
        if llm_available():
            try:
                print("Calling OpenAI text model for missing/low-confidence fields...", flush=True)
                llm_results = llm_extract_many_from_text(text, rule_results)
                print(f"OpenAI cleanup done. Rows: {len(llm_results)}", flush=True)
            except Exception as e:
                err = repr(e)
                key = get_openai_api_key()
                if key:
                    err = err.replace(key, "[OPENAI_API_KEY_REDACTED]")
                llm_error = f"OpenAI cleanup failed: {err}"
                print(llm_error, flush=True)
        else:
            llm_error = "OpenAI cleanup skipped: OPENAI_API_KEY is not set or openai package is not installed"
            print(llm_error, flush=True)
    else:
        print("OpenAI cleanup not needed based on regex confidence.", flush=True)

    cleaned_llm = [r for r in llm_results if has_invoice_data(r)]
    if cleaned_llm:
        final_rows = [merge_rows(cleaned_llm[i], rule_results[i] if i < len(rule_results) else {}) for i in range(len(cleaned_llm))]
        extraction_method = f"{method}+regex+openai_text"
    else:
        final_rows = rule_results
        extraction_method = f"{method}+regex"

    final_rows = [r for r in final_rows if has_invoice_data(r)]

    if not final_rows:
        print("No extracted rows. Check OCR text and OpenAI cleanup status.", flush=True)
        if llm_error:
            print(llm_error, flush=True)

    rows: List[ExtractedInvoice] = []
    for r in final_rows:
        rows.append(ExtractedInvoice(
            is_one_liner=r.get("is_one_liner"),
            invoice_number=r.get("invoice_number"),
            client_name=r.get("client_name"),
            hsn_code=r.get("hsn_code"),
            total_quantity=r.get("total_quantity"),
            value=normalize_amount(r.get("value")),
            currency=r.get("currency"),
            confidence=r.get("confidence", round(quality_score(r), 2)),
            source_file=os.path.basename(pdf_path),
            source_pages=r.get("source_pages"),
            extraction_method=extraction_method,
            error=llm_error,
        ))
    return rows


# -----------------------------
# Folder processing
# -----------------------------

def safe_filename(value: str) -> str:
    value = Path(value).stem
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value or "document"


def unique_output_path(output_dir: Path, pdf_path: Path, folder_root: Path) -> Path:
    try:
        relative = pdf_path.relative_to(folder_root)
        stem = safe_filename("__".join(relative.with_suffix("").parts))
    except Exception:
        stem = safe_filename(pdf_path.stem)

    candidate = output_dir / f"{stem}.json"
    counter = 2
    while candidate.exists():
        candidate = output_dir / f"{stem}_{counter}.json"
        counter += 1
    return candidate


def process_folder(
    folder_path: str,
    output_json: str,
    max_ocr_pages: Optional[int],
    llm_mode: str,
    min_quality: float,
    force_ocr: bool,
) -> None:
    all_results: List[Dict[str, Any]] = []
    folder_root = Path(folder_path).expanduser().resolve()

    output_arg = Path(output_json).expanduser()
    if output_arg.suffix.lower() == ".json":
        combined_json_path = output_arg
        output_dir = output_arg.parent / output_arg.stem
    else:
        output_dir = output_arg
        combined_json_path = output_dir / "extracted_output_all.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(folder_root.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {folder_root}", flush=True)
        return

    print(f"Found {len(pdf_files)} PDF files in: {folder_root}", flush=True)
    print(f"Writing one JSON per PDF into: {output_dir.resolve()}", flush=True)

    for index, pdf_path in enumerate(pdf_files, start=1):
        print("=" * 80, flush=True)
        print(f"Processing {index}/{len(pdf_files)}: {pdf_path}", flush=True)

        per_file_json = unique_output_path(output_dir, pdf_path, folder_root)

        try:
            rows = extract_documents_from_pdf(
                str(pdf_path),
                max_ocr_pages=max_ocr_pages,
                llm_mode=llm_mode,
                min_quality=min_quality,
                force_ocr=force_ocr,
            )
            if rows:
                file_results = [r.model_dump() for r in rows]
            else:
                file_results = [{
                    "source_file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "error": "No rows extracted. Check the .ocr.txt file. OCR text may be too poor or invoice fields were not detected.",
                }]
        except Exception as e:
            print(f"FAILED: {pdf_path.name}. Error: {e}", flush=True)
            file_results = [{
                "source_file": pdf_path.name,
                "source_path": str(pdf_path),
                "error": repr(e),
            }]

        for item in file_results:
            item.setdefault("source_file", pdf_path.name)
            item.setdefault("source_path", str(pdf_path))

        with open(per_file_json, "w", encoding="utf-8") as f:
            json.dump(file_results, f, indent=2, ensure_ascii=False)
        print(f"Saved per-file JSON: {per_file_json}", flush=True)

        all_results.extend(file_results)

        with open(combined_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Combined progress saved to: {combined_json_path}", flush=True)

    try:
        import pandas as pd
        output_csv = combined_json_path.with_suffix(".csv")
        pd.DataFrame(all_results).to_csv(output_csv, index=False)
        print(f"Saved combined CSV: {output_csv}", flush=True)
    except Exception as e:
        print(f"CSV not created: {e}", flush=True)

    print("Done.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="./documents")
    parser.add_argument(
        "--output",
        default="extracted_json",
        help="Directory for one JSON per PDF. If you pass a .json filename, a folder with that stem is created and a combined JSON is also saved.",
    )
    parser.add_argument("--max-ocr-pages", type=int, default=None)
    parser.add_argument(
        "--llm-mode",
        choices=["missing", "always", "never"],
        default=os.getenv("LLM_MODE", "missing"),
        help="missing = call OpenAI only if regex fields are missing/low confidence; always = always cleanup with OpenAI text; never = regex only.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=float(os.getenv("MIN_QUALITY", "0.85")),
        help="Quality threshold for skipping OpenAI when --llm-mode missing.",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Ignore embedded PDF text and always run Tesseract OCR.",
    )
    args = parser.parse_args()

    process_folder(
        args.folder,
        args.output,
        args.max_ocr_pages,
        args.llm_mode,
        args.min_quality,
        args.force_ocr,
    )


if __name__ == "__main__":
    main()
