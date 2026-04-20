
import shutil
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx
except ImportError:
    docx = None

try:
    import pytesseract
    from PIL import Image

    _tesseract = shutil.which("tesseract") or next(
        (p for p in [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ] if Path(p).exists()),
        None
    )
    if _tesseract:
        pytesseract.pytesseract.tesseract_cmd = _tesseract
except ImportError:
    pytesseract = None
    Image = None


def extract_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    if not fitz:
        return "Error: PyMuPDF (fitz) is not installed."
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text() + "\n\n"
        doc.close()
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"
    return text.strip()


def extract_docx(file_path: str) -> str:
    """Extract text from DOCX."""
    if not docx:
        return "Error: python-docx is not installed."
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        return f"Error extracting DOCX: {str(e)}"


def extract_image(file_path: str) -> str:
    """Extract text from image using Tesseract OCR."""
    if not pytesseract or not Image:
        return "Error: Image OCR dependencies (pytesseract, Pillow) are missing."
    try:
        pytesseract.get_tesseract_version()
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        return "Error: Tesseract OCR is not installed on this system. Cannot process image."
    except Exception as e:
        return f"Error extracting image text: {str(e)}"


def extract_txt(file_path: str) -> str:
    """Read text file directly."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read().strip()
        except Exception as e:
            return f"Error reading text file: {str(e)}"
    except Exception as e:
        return f"Error reading text file: {str(e)}"


def structure_as_markdown(text: str, source_name: str) -> str:
    """Wrap extracted text with a simple source label for ingestion."""
    if not text.strip():
        return ""
    return f"# {source_name}\n\n{text}"


def extract_url(url: str) -> str:
    """Fetch and extract readable text content from a webpage."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return "Error: 'requests' or 'beautifulsoup4' not installed."

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        title = soup.title.string.strip() if soup.title else "No title"

        for tag in soup(["script", "style", "img", "input", "nav", "footer", "header", "aside", "table", "form", "button"]):
            tag.decompose()

        body_text = soup.body.get_text(separator="\n", strip=True) if soup.body else ""
        return f"# {title}\n\nSource: {url}\n\n{body_text}"

    except requests.exceptions.Timeout:
        return f"Error: Request timed out for {url}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {str(e)}"
    except Exception as e:
        return f"Error processing {url}: {str(e)}"


def extract_file(file_path: str) -> str:
    """Dispatcher to appropriate extractor based on file extension."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".tiff"]:
        return extract_image(file_path)
    elif ext == ".docx":
        return extract_docx(file_path)
    elif ext in [".txt", ".md", ".csv", ".json"]:
        return extract_txt(file_path)
    else:
        return f"Error: Unsupported file type '{ext}'"
