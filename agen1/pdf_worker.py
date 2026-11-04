"""Optional PDF extraction in an interruptible process."""
import json
import sys
from pypdf import PdfReader

reader = PdfReader(sys.argv[1])
chunks = []
size = 0
for page in reader.pages[:30]:
    text = (page.extract_text() or '')[:16000]
    chunks.append(text)
    size += len(text)
    if size >= 16000:
        break
print(json.dumps({'text': '\n'.join(chunks)[:16000]}))
