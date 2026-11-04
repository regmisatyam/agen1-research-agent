"""Bounded local-document extraction; run in a cancellable child process."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import zipfile
from xml.etree import ElementTree

MAX_BYTES = 20_000_000
MAX_CHARS = 200_000
TEXT_FORMATS = {'.txt', '.md', '.markdown', '.rst', '.csv', '.tsv', '.json', '.jsonl'}


class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def extract(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Material must be an existing regular file')
    kind = path.suffix.lower()
    if kind not in TEXT_FORMATS | {'.html', '.htm', '.pdf', '.docx'}:
        raise ValueError(f'Unsupported material format {kind or "(no extension)"}; use text, Markdown, PDF, DOCX, CSV, JSON, or HTML')
    with path.open('rb') as f:
        raw = f.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('Material exceeds the 20 MB file limit')
    notes = []
    truncated = False
    if kind == '.pdf':
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("PDF import needs the pdf extra: python -m pip install -e '.[pdf]'") from None
        import io
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted and not reader.decrypt(''):
            raise ValueError('Password-protected PDFs are not supported; provide an unlocked copy')
        pieces, count = [], 0
        for page in reader.pages[:30]:
            part = page.extract_text() or ''
            pieces.append(part[:MAX_CHARS + 1])
            count += len(part) + 1
            if count > MAX_CHARS:
                truncated = True
                break
        text = '\n'.join(pieces)
        truncated |= len(reader.pages) > 30
        notes.append('PDF text extraction: at most the first 30 pages; no OCR or image interpretation.')
    elif kind == '.docx':
        import io
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            info = archive.getinfo('word/document.xml')
            if info.file_size > MAX_BYTES:
                raise ValueError('DOCX document XML exceeds the 20 MB expansion limit')
            xml = archive.read(info)
        if b'<!DOCTYPE' in xml.upper() or b'<!ENTITY' in xml.upper():
            raise ValueError('DOCX XML declarations with entities are not supported')
        root = ElementTree.fromstring(xml)
        ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        text = '\n'.join(''.join(n.text or '' for n in p.iter(ns + 't')) for p in root.iter(ns + 'p'))
        notes.append('DOCX body paragraphs and table text only; images and embedded objects are not interpreted.')
    else:
        text = raw.decode('utf-8-sig')
        if '\x00' in text:
            raise ValueError('Material contains binary data; use a supported text/document format')
        if kind in {'.html', '.htm'}:
            parser = HTMLText()
            parser.feed(text)
            text = '\n'.join(parser.parts)
            notes.append('HTML text only; scripts, styles, and external resources are not executed or fetched.')
    if not text.strip():
        raise ValueError('No readable text found; scanned PDFs require OCR before import')
    truncated |= len(text) > MAX_CHARS
    return {'text': text[:MAX_CHARS], 'truncated': truncated, 'sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw), 'format': kind[1:], 'extraction_notes': notes}


def main():
    try:
        result = extract(json.load(sys.stdin)['path'])
    except Exception as error:
        result = {'error': f'{type(error).__name__}: {error}'}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
