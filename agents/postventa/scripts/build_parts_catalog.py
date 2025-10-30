import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv

# Carga .env y, si existe, sobrescribe con secrets.local.txt (compat con nombre anterior)
import os as _os
try:
    load_dotenv(override=False)
    _base = _os.path.dirname(__file__)
    _cands = [
        "secrets.local.txt",
        _os.path.join(_base, "secrets.local.txt"),
        "secrets.loca.txt",  # compatibilidad legado
        _os.path.join(_base, "secrets.loca.txt"),
    ]
    for _p in _cands:
        if _os.path.exists(_p):
            load_dotenv(_p, override=True)
            break
except Exception:
    load_dotenv()

PDF_FILE = os.getenv("PDF_FILE", "SSOT-HIGER.pdf")
BRAND_NAME = os.getenv("BRAND_NAME", "Higer")
MODEL_NAME = os.getenv("MODEL_NAME", "H6C")
OUT_FILE = os.getenv("PARTS_INDEX_FILE", "parts_index.json")
PARTS_PAGES = os.getenv("PARTS_PAGES", "")  # Ej: "240-260,420-460,512"

OEM_RE = re.compile(r"\b[A-Z0-9]{2,}(?:[\-–][A-Z0-9]{2,})+\b")

_REF_CLEAN_RE = re.compile(
    r"\b(?:ref\.?|p/?n|n[oº]\.?|c[oó]digo)\s*[:#=\-]*\s*[A-Za-z0-9./\-]+",
    flags=re.IGNORECASE,
)


def _normalize_token(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    if re.fullmatch(r"[A-Z0-9./-]+", token):
        if len(token) <= 3 or '/' in token or '-' in token:
            return token.upper()
        return token.capitalize()
    if token.lower() in {"de", "del", "la", "el", "los", "las"}:
        return token.lower()
    return token.capitalize()


def _clean_name(text: str) -> str:
    txt = " ".join((text or "").split())
    txt = re.sub(r"\bnota[s]?[:].*$", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\(.*?(?:ver|nota).*?\)", "", txt, flags=re.IGNORECASE)
    txt = _REF_CLEAN_RE.sub("", txt)
    txt = txt.rstrip(' .,:;').strip()
    tokens = [ _normalize_token(tok) for tok in txt.split() ]
    return " ".join(tokens)

def resolve_pdf_path() -> Path:
    script_dir = Path(__file__).parent
    pdf_path = Path(PDF_FILE)
    if pdf_path.exists():
        return pdf_path
    alt = script_dir / PDF_FILE
    if alt.exists():
        return alt
    alt2 = script_dir / 'SSOT-HIGER.pdf'
    if alt2.exists():
        return alt2
    raise FileNotFoundError(f"No se encontró el PDF en '{PDF_FILE}' ni en '{alt}'.")

def parse_page_spec(spec: str):
    pages = set()
    spec = (spec or "").strip()
    if not spec:
        return None
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            a,b = token.split('-',1)
            try:
                a=int(a); b=int(b)
                for p in range(min(a,b), max(a,b)+1):
                    pages.add(p)
            except Exception:
                continue
        else:
            try:
                pages.add(int(token))
            except Exception:
                continue
    return pages or None

def is_parts_page(page, page_text: str, tables) -> bool:
    # Heurística: encabezados típicos o densidad OEM
    text_l = (page_text or "").lower()
    if any(kw in text_l for kw in ["número de parte", "numero de parte", "oem", "part no", "parts list", "catálogo de partes", "catalogo de partes"]):
        return True
    # OEM density in text
    if len(OEM_RE.findall(page_text or "")) >= 3:
        return True
    # Headers in tables
    for tbl in tables or []:
        if not tbl:
            continue
        header = [((c or '').strip().lower()) for c in (tbl[0] if tbl else [])]
        if any(h for h in header) and any(h for h in header if any(k in h for k in ["oem","nº","no.","num","número","numero","p/n","part no","descripción","descripcion"])):
            return True
    return False

def extract_parts(pdf_path: Path) -> list[dict]:
    try:
        import pdfplumber
    except Exception as e:
        print(f"Error: pdfplumber no disponible ({e}).")
        return []

    items_by_oem: dict[str, dict] = {}
    whitelist = parse_page_spec(PARTS_PAGES)
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_no = page.page_number
            if whitelist and page_no not in whitelist:
                continue
            page_text = page.extract_text() or ""
            tables = page.extract_tables() or []
            # si no hay whitelist, saltar páginas que no parezcan de partes
            if not whitelist and not is_parts_page(page, page_text, tables):
                continue
            for tbl in tables:
                if not tbl:
                    continue
                header = None
                if tbl and all(isinstance(c, str) or c is None for c in tbl[0]):
                    header = [((c or '').strip().lower()) for c in tbl[0]]
                    if not any(header):
                        header = None
                rows = tbl[1:] if header else tbl
                for row in rows:
                    if not row:
                        continue
                    cells = [(c or "").strip() for c in row]
                    joined = " | ".join(cells)
                    m = OEM_RE.search(joined)
                    if not m:
                        continue
                    oem = m.group(0)
                    # Heurística de nombre
                    name = ""
                    if header:
                        name_cols = {i for i,h in enumerate(header) if any(k in h for k in ["refacción","refaccion","repuesto","pieza","descripción","descripcion","nombre","part","item"]) }
                        for i,c in enumerate(cells):
                            if i in name_cols and c and oem not in c and len(c) > len(name):
                                name = c
                    if not name:
                        for c in cells:
                            if oem not in c and len(c) > len(name):
                                name = c
                    if not name:
                        continue
                    clean_name = _clean_name(name)
                    if not clean_name:
                        continue
                    key = oem.upper()
                    current = items_by_oem.get(key)
                    candidate = {
                        "part_name": clean_name,
                        "oem": key,
                        "page_label": str(page_no),
                    }
                    if not current:
                        items_by_oem[key] = candidate
                    else:
                        existing = current.get("part_name", "")
                        if len(clean_name) > len(existing):
                            items_by_oem[key] = candidate
    # Ordenar por OEM para determinismo
    return [items_by_oem[k] for k in sorted(items_by_oem.keys())]

def main():
    pdf_path = resolve_pdf_path()
    doc_id = pdf_path.stem
    print(f"Extrayendo catálogo de partes desde: {pdf_path}")
    items = extract_parts(pdf_path)
    out = {
        "doc_id": doc_id,
        "brand": BRAND_NAME,
        "model": MODEL_NAME,
        "items": items,
    }
    out_path = Path(__file__).parent / OUT_FILE
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Guardado catálogo de partes: {out_path} (items={len(items)})")

if __name__ == "__main__":
    main()
