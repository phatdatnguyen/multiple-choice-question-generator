"""Document text extraction, token counting and chunking."""

import hashlib
import os
import re
import zipfile
from xml.etree import ElementTree

import html2text
import tiktoken

try:  # pypdf is the maintained successor to PyPDF2
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - fallback for older installs
    from PyPDF2 import PdfReader

from docx import Document
from pptx import Presentation

SUPPORTED_EXTENSIONS = [
    ".pdf", ".docx", ".pptx", ".htm", ".html", ".epub",
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".rst",
]

# Formats python-docx / python-pptx cannot open, reported with advice instead of
# a confusing parser traceback.
LEGACY_EXTENSIONS = {
    ".doc": "Word 97-2003",
    ".ppt": "PowerPoint 97-2003",
    ".xls": "Excel 97-2003",
    ".rtf": "Rich Text",
    ".pages": "Apple Pages",
    ".key": "Apple Keynote",
}

_TEXT_CACHE = {}
_CACHE_LIMIT = 32


class DocumentError(Exception):
    """Raised when a document cannot be read."""


# --------------------------------------------------------------------------- #
# Extractors
# --------------------------------------------------------------------------- #

def read_pdf_file(file_path):
    reader = PdfReader(file_path)
    # extract_text() returns None for pages with no text layer.
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_word_file(file_path):
    document = Document(file_path)
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def read_powerpoint_file(file_path):
    presentation = Presentation(file_path)
    parts = []
    for number, slide in enumerate(presentation.slides, start=1):
        parts.append(f"--- Slide {number} ---")
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                parts.append(shape.text_frame.text)
            elif getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            parts.append(f"[Notes] {slide.notes_slide.notes_text_frame.text}")
    return "\n".join(parts)


def _html_to_text(markup):
    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    converter.body_width = 0
    return converter.handle(markup)


def read_html_file(file_path):
    return _html_to_text(_read_raw_text(file_path))


def read_epub_file(file_path):
    """Extract an EPUB by reading its spine in reading order."""
    with zipfile.ZipFile(file_path) as archive:
        names = archive.namelist()
        opf_name = next((name for name in names if name.lower().endswith(".opf")), None)

        hrefs = []
        if opf_name:
            root = ElementTree.fromstring(archive.read(opf_name))
            namespace = {"opf": "http://www.idpf.org/2007/opf"}
            manifest = {
                item.get("id"): item.get("href")
                for item in root.iterfind(".//opf:manifest/opf:item", namespace)
            }
            base = os.path.dirname(opf_name)
            for itemref in root.iterfind(".//opf:spine/opf:itemref", namespace):
                href = manifest.get(itemref.get("idref"))
                if href:
                    joined = os.path.normpath(os.path.join(base, href)).replace("\\", "/")
                    if joined in names:
                        hrefs.append(joined)

        if not hrefs:  # No usable spine - fall back to every document in the zip.
            hrefs = sorted(
                name for name in names
                if name.lower().endswith((".xhtml", ".html", ".htm"))
            )

        parts = []
        for href in hrefs:
            markup = archive.read(href).decode("utf-8", errors="replace")
            parts.append(_html_to_text(markup))
    return "\n".join(parts)


def _read_raw_text(file_path):
    """Read a text file, tolerating imperfect encodings rather than crashing."""
    with open(file_path, "rb") as file:
        data = file.read()

    # UTF-16 is only tried when a byte-order mark says so: it decodes almost any
    # even-length input into plausible-looking mojibake otherwise.
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeError:
            pass

    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeError:
            continue
    return data.decode("utf-8", errors="replace")


def read_text_file(file_path):
    return _read_raw_text(file_path)


_EXTRACTORS = {
    ".pdf": read_pdf_file,
    ".docx": read_word_file,
    ".pptx": read_powerpoint_file,
    ".htm": read_html_file,
    ".html": read_html_file,
    ".epub": read_epub_file,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def _file_digest(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_text(file_path, *, use_cache=True):
    """Extract plain text from one document.

    Results are cached by file content, so switching models and regenerating
    does not re-parse a large PDF.
    """
    extension = os.path.splitext(file_path)[1].lower()
    if extension in LEGACY_EXTENSIONS:
        raise DocumentError(
            f"{LEGACY_EXTENSIONS[extension]} files ({extension}) are not supported. "
            f"Please save the file as .docx, .pptx or .pdf and try again."
        )

    key = None
    if use_cache:
        try:
            key = f"{_file_digest(file_path)}:{extension}"
        except OSError:
            key = None
        if key and key in _TEXT_CACHE:
            return _TEXT_CACHE[key]

    extractor = _EXTRACTORS.get(extension, read_text_file)
    try:
        text = extractor(file_path)
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"Could not read {os.path.basename(file_path)}: {exc}") from exc

    text = (text or "").strip()
    if not text:
        raise DocumentError(
            f"No text found in {os.path.basename(file_path)}. "
            f"If it is a scanned PDF it needs OCR first."
        )

    if key:
        if len(_TEXT_CACHE) >= _CACHE_LIMIT:
            _TEXT_CACHE.pop(next(iter(_TEXT_CACHE)))
        _TEXT_CACHE[key] = text
    return text


def build_document_context(file_paths, *, use_cache=True):
    """Extract several documents into one delimited string.

    Returns ``(text, problems)``; unreadable files are reported but do not stop
    the readable ones from being used.
    """
    if isinstance(file_paths, (str, bytes)) or file_paths is None:
        file_paths = [file_paths] if file_paths else []

    sections, problems = [], []
    for file_path in file_paths:
        path = getattr(file_path, "name", file_path)
        try:
            text = extract_text(path, use_cache=use_cache)
        except DocumentError as exc:
            problems.append(str(exc))
            continue
        sections.append(
            "<<<DOCUMENT_CONTENT>>>\n"
            f"File: {os.path.basename(path)}\n"
            f"{text}\n"
            "<<<END_DOCUMENT>>>"
        )
    return "\n\n".join(sections), problems


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #

def get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        # Newer models share the o200k_base vocabulary.
        try:
            return tiktoken.get_encoding("o200k_base")
        except (KeyError, ValueError):
            return tiktoken.get_encoding("cl100k_base")


def count_tokens(text, model):
    """Count tokens in a string."""
    return len(get_encoding(model).encode(str(text or "")))


def chunk_text(text, max_tokens, model):
    """Split text into chunks of at most ``max_tokens`` tokens.

    Splits on blank lines first, then single lines, and only falls back to
    slicing tokens for a single line that is itself too long.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0.")

    encoding = get_encoding(model)
    text = str(text or "")
    if len(encoding.encode(text)) <= max_tokens:
        return [text] if text.strip() else []

    chunks, current, current_tokens = [], [], 0

    def flush():
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current).strip())
            current, current_tokens = [], 0

    for block in _split_blocks(text, max_tokens, encoding):
        block_tokens = len(encoding.encode(block))
        if current_tokens + block_tokens > max_tokens:
            flush()
        current.append(block)
        current_tokens += block_tokens
    flush()
    return [chunk for chunk in chunks if chunk.strip()]


def _split_blocks(text, max_tokens, encoding):
    """Yield blocks that each fit within ``max_tokens``."""
    for paragraph in re.split(r"\n\s*\n", text):
        if not paragraph.strip():
            continue
        if len(encoding.encode(paragraph)) <= max_tokens:
            yield paragraph
            continue
        for line in paragraph.splitlines():
            if not line.strip():
                continue
            tokens = encoding.encode(line)
            if len(tokens) <= max_tokens:
                yield line
            else:
                for start in range(0, len(tokens), max_tokens):
                    yield encoding.decode(tokens[start:start + max_tokens])
