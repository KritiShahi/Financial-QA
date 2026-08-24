import re
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def read_html_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text


def clean_text_basic(text: str) -> str:
    text = re.sub(r'page\s*\d+(\s*\|\s*sec.*)?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess(text: str) -> str:
    return clean_text_basic(text)


def extract_text(text, item_start, item_end):
    starts = [i.start() for i in item_start.finditer(text)]
    ends   = [i.start() for i in item_end.finditer(text)]

    positions = []
    for s in starts:
        for e in ends:
            if s < e:
                positions.append([s, e])
                break

    if not positions:
        return ""

    item_position = max(positions, key=lambda p: p[1] - p[0])
    return text[item_position[0]:item_position[1]]


def section_text(text, section):
    try:
        if section == 1:
            start = re.compile(r"item\s*[1][\.\;\:\-\_]*\s*\b", re.IGNORECASE)
            end   = re.compile(r"item\s*1a[\.\;\:\-\_]\s*Risk|item\s*2[\.\,\;\:\-\_]\s*Prop", re.IGNORECASE)
            return extract_text(text, start, end)

        if section == 2:
            start = re.compile(r"(?<!,\s)item\s*1a[\.\;\:\-\_]\s*Risk", re.IGNORECASE)
            end   = re.compile(r"item\s*2[\.\;\:\-\_]\s*Prop|item\s*[1][\.\;\:\-\_]*\s*\b", re.IGNORECASE)
            return extract_text(text, start, end)

        if section == 3:
            start = re.compile(r"item\s*[7][\.\;\:\-\_]*\s*\bM", re.IGNORECASE)
            end   = re.compile(r"item\s*7a[\.\;\:\-\_]\sQuanti|item\s*8[\.\,\;\:\-\_]\s*", re.IGNORECASE)
            return extract_text(text, start, end)

        if section == 9:
            start = re.compile(r"item\s*[9][\.\;\:\-\_]*\s*Ch", re.IGNORECASE)
            end   = re.compile(r"item\s*9a[\.\;\:\-\_]\sControls|item\s*9b[\.\,\;\:\-\_]\s*", re.IGNORECASE)
            return extract_text(text, start, end)

        if section == 10:
            start = re.compile(r"item\s*^10$[\.\;\:\-\_]*\s*\bDir", re.IGNORECASE)
            end   = re.compile(r"item\s*11[\.\;\:\-\_]\sComp|item\s*12[\.\,\;\:\-\_]\s*", re.IGNORECASE)
            return extract_text(text, start, end)

        if section == 4:
            start = re.compile(r"item\s*[3][\.\;\:\-\_]*\s*\bLeg", re.IGNORECASE)
            end   = re.compile(r"item\s*4[\.\;\:\-\_]\sMin|item\s*5[\.\,\;\:\-\_]\s*", re.IGNORECASE)
            return extract_text(text, start, end)

        if section == 12:
            start = re.compile(r"item\s*\d{2}[\.\;\:\-\_]*\s*\bSec", re.IGNORECASE)
            end   = re.compile(r"item\s*13[\.\;\:\-\_]\sRel|item\s*14[\.\,\;\:\-\_]\s*", re.IGNORECASE)
            return extract_text(text, start, end)

    except Exception:
        return ""

    return ""


def sec_data_func(full_text: str, section: int) -> str:
    data = section_text(full_text, section)
    data = re.sub(r'\n', ' ', str(data))
    data = re.sub(r'[^A-Za-z0-9]', ' ', data)
    return preprocess(data)
