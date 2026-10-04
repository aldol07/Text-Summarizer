"""Turn URLs and uploaded files into plain text.

URL fetching is a classic SSRF vector, so every hop (including redirects) is
checked: http(s) only, and the hostname must resolve to public IP addresses.
Responses are size-capped and must be HTML or plain text.
"""

import io
import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

MAX_REDIRECTS = 3
MAX_DOWNLOAD_BYTES = 3 * 1024 * 1024
MAX_PDF_PAGES = 60
TIMEOUT_SECONDS = 10.0
USER_AGENT = "TextSummarizerBot/0.1 (+https://github.com/aldol07/Text-Summarizer)"

_NOISE_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe", "svg", "button")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_BACK_MATTER = {"references", "see also", "external links", "notes", "further reading", "bibliography", "sources"}
_CITATION_RE = re.compile(r"\s?\[\s*(?:\d+|citation needed|[a-z])\s*\]")  # Wikipedia-style "[1]"


class IngestError(ValueError):
    """Input could not be turned into text (bad URL, unsupported file, ...)."""


@dataclass(frozen=True)
class ExtractedText:
    title: str
    text: str
    source: str

    def to_dict(self, max_chars: int | None = None) -> dict:
        """JSON payload; text is cut at a paragraph boundary when longer than ``max_chars``."""
        text, truncated = self.text, False
        if max_chars and len(text) > max_chars:
            cut = text.rfind("\n\n", 0, max_chars)
            text, truncated = text[: cut if cut > max_chars // 2 else max_chars].rstrip(), True
        return {"title": self.title, "text": text, "source": self.source, "chars": len(text), "truncated": truncated}


# URLs --------------------------------------------------------------------------
def assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise IngestError("Only http(s) URLs are supported")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise IngestError(f"Could not resolve host {parsed.hostname!r}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise IngestError("URLs pointing to private or local network addresses are not allowed")


def fetch_url(url: str, client: httpx.Client | None = None) -> ExtractedText:
    owns_client = client is None
    client = client or httpx.Client(timeout=TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT})
    try:
        current = url.strip()
        for _ in range(MAX_REDIRECTS + 1):
            assert_public_url(current)
            with client.stream("GET", current, follow_redirects=False) as response:
                if response.is_redirect:
                    current = urljoin(current, response.headers.get("location", ""))
                    continue
                if response.status_code >= 400:
                    raise IngestError(f"The page returned HTTP {response.status_code}")
                content_type = response.headers.get("content-type", "").lower()
                if "html" not in content_type and "text/plain" not in content_type:
                    raise IngestError(f"Unsupported content type {content_type or 'unknown'!r}; expected an HTML page")
                body = _read_capped(response)
                if "text/plain" in content_type:
                    return ExtractedText(
                        title="", text=body.decode(response.encoding or "utf-8", "replace"), source=current
                    )
                title, text = extract_article(body.decode(response.encoding or "utf-8", "replace"))
                return ExtractedText(title=title, text=text, source=current)
        raise IngestError("Too many redirects")
    except httpx.HTTPError as exc:
        raise IngestError(f"Could not fetch the page: {exc.__class__.__name__}") from exc
    finally:
        if owns_client:
            client.close()


def _read_capped(response: httpx.Response) -> bytes:
    chunks, size = [], 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > MAX_DOWNLOAD_BYTES:
            raise IngestError("The page is too large")
        chunks.append(chunk)
    return b"".join(chunks)


def extract_article(html: str) -> tuple[str, str]:
    """Readability-style main-text extraction: prefer <article>/<main>, keep substantial paragraphs."""
    soup = BeautifulSoup(html, "lxml")
    title = _title(soup)
    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    container = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = []
    for element in container.find_all(["p", "h2", "h3", "li"]):
        text = element.get_text(" ", strip=True)
        if element.name == "h2" and text.lower().rstrip(":") in _BACK_MATTER:
            break  # references, external links, ...: not article content
        if len(text) >= 40 or (text and text[-1] in ".!?"):
            paragraphs.append(text)
    text = _CITATION_RE.sub("", "\n\n".join(paragraphs))
    if not paragraphs:  # pages without <p> structure
        text = container.get_text("\n", strip=True)
    if not text.strip():
        raise IngestError("No readable text found on the page")
    return title, text


def _title(soup: BeautifulSoup) -> str:
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else ""


# Files -------------------------------------------------------------------------
def extract_file(filename: str, content: bytes) -> ExtractedText:
    name = (filename or "").lower()
    if name.endswith(".pdf") or content[:5] == b"%PDF-":
        return ExtractedText(title=filename, text=_pdf_text(content), source=filename)
    if name.endswith((".txt", ".md", ".markdown", ".text")):
        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            raise IngestError("The file is empty")
        return ExtractedText(title=filename, text=text, source=filename)
    raise IngestError("Unsupported file type; upload a .pdf, .txt or .md file")


def _pdf_text(content: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise IngestError("Encrypted PDFs are not supported")
        pages = [page.extract_text() or "" for page in reader.pages[:MAX_PDF_PAGES]]
    except PdfReadError as exc:
        raise IngestError("Could not read the PDF") from exc
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", "\n\n".join(p.strip() for p in pages if p.strip()))
    if not text.strip():
        raise IngestError("No text found in the PDF (scanned PDFs need OCR, which is not supported)")
    return text
