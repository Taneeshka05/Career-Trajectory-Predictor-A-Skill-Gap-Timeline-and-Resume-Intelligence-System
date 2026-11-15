"""
parser.py
Advanced multi-format document parser and resume entity extractor.
Supports .txt, .pdf (via pypdf), and .docx (via native zipfile + XML).
Zero external C-dependencies required.
"""

import io
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import date
from typing import Dict, List, Optional, Any
from core import config as cfg
from core.taxonomy import SKILL_ALIASES
from core.logger_setup import get_logger

log = get_logger("parser")

# Precompiled regex patterns
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
LINKEDIN_REGEX = re.compile(r"linkedin\.com/in/[A-Za-z0-9_-]+", re.IGNORECASE)
GITHUB_REGEX = re.compile(r"github\.com/[A-Za-z0-9_-]+", re.IGNORECASE)

DEGREE_PATTERNS = [
    (re.compile(r"\b(?:m\.?c\.?a|master of computer applications)\b", re.I), "MCA"),
    (re.compile(r"\b(?:b\.?c\.?a|bachelor of computer applications)\b", re.I), "BCA"),
    (re.compile(r"\b(?:b\.?tech|b\.?e|bachelor of technology|bachelor of engineering)\b", re.I), "B.Tech/B.E."),
    (re.compile(r"\b(?:m\.?tech|m\.?e|master of technology|master of engineering)\b", re.I), "M.Tech/M.E."),
    (re.compile(r"\b(?:b\.?sc|bachelor of science)\b", re.I), "B.Sc"),
    (re.compile(r"\b(?:m\.?sc|master of science)\b", re.I), "M.Sc"),
    (re.compile(r"\b(?:ph\.?d|doctor of philosophy)\b", re.I), "Ph.D."),
    (re.compile(r"\b(?:mba|master of business administration)\b", re.I), "MBA"),
]

CERTIFICATION_PATTERNS = [
    (re.compile(r"\baws\s+certified\b", re.I), "AWS Certified"),
    (re.compile(r"\bazure\s+certified\b", re.I), "Azure Certified"),
    (re.compile(r"\bgcp\s+certified|google\s+cloud\s+certified\b", re.I), "GCP Certified"),
    (re.compile(r"\bcubernetes\s+certified|cka|ckad\b", re.I), "Kubernetes Certified (CKA/CKAD)"),
    (re.compile(r"\btensorflow\s+developer\b", re.I), "TensorFlow Developer Certificate"),
    (re.compile(r"\bcertified\s+scrum\s+master|csm\b", re.I), "Certified Scrum Master (CSM)"),
    (re.compile(r"\bpmp|project\s+management\s+professional\b", re.I), "PMP Certification"),
    (re.compile(r"\bistqb\b", re.I), "ISTQB Certified Tester"),
    (re.compile(r"\boracle\s+certified\b", re.I), "Oracle Certified Java Associate/Professional"),
]

YEARS_PATTERNS = [
    re.compile(r"(\d{1,2})\+?\s*(?:years|yrs|year|yr)\s*(?:of\s*)?(?:experience|exp)?", re.I),
    re.compile(r"(?:experience|exp)\s*:\s*(\d{1,2})\+?\s*(?:years|yrs)", re.I),
]

SECTION_HEADINGS = {
    "experience": {
        "experience", "professional experience", "work experience", "employment history",
        "work history", "career history",
    },
    "education": {"education", "academic background", "qualifications"},
    "skills": {
        "skills", "technical skills", "core skills", "technical proficiencies",
        "skills and technologies", "technologies",
    },
    "projects": {"projects", "personal projects", "academic projects"},
    "certifications": {"certifications", "certificates", "licenses"},
    "summary": {"professional summary", "career objective", "profile summary", "profile"},
    "achievements": {"achievements", "awards and achievements"},
    "internships": {"internships", "internship experience"},
}
ROLE_TERMS_REGEX = re.compile(
    r"\b(?:engineer|developer|analyst|designer|manager|consultant|administrator|"
    r"intern|technician|scientist|specialist|assistant|coordinator|architect|"
    r"accountant|researcher|teacher|nurse|executive|product\s+owner|"
    r"project\s+manager|sales\s+representative)\b",
    re.I,
)
SKILL_VALIDATION_PATTERNS = {
    re.compile(r"\b" + re.escape(term) + r"\b", re.I): canonical
    for term, canonical in (
        [(skill, skill) for skill in cfg.ALL_SKILLS]
        + list(SKILL_ALIASES.items())
    )
}
DATE_RANGE_REGEX = re.compile(
    r"\b(?P<start_month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?|0?[1-9]|1[0-2])?[\s,./-]*"
    r"(?P<start>19\d{2}|20\d{2})\s*(?:-|–|—|to)\s*"
    r"(?P<end_month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?|0?[1-9]|1[0-2])?[\s,./-]*"
    r"(?P<end>19\d{2}|20\d{2}|present|current|now)\b",
    re.I,
)


def is_valid_resume(text: str) -> bool:
    """Require resume structure and multiple independent career-history signals."""
    if not text or len(text.split()) < 15:
        return False

    sections = set()
    for line in text.splitlines():
        heading = re.sub(r"^[\s•\-*#]+|[\s:|—–-]+$", "", line.strip().lower())
        for section, headings in SECTION_HEADINGS.items():
            if any(
                heading == item or heading.startswith(item + ":")
                for item in headings
            ):
                sections.add(section)

    has_contact = bool(EMAIL_REGEX.search(text) or PHONE_REGEX.search(text) or LINKEDIN_REGEX.search(text))
    has_degree = any(pattern.search(text) for pattern, _ in DEGREE_PATTERNS)
    has_experience = bool(
        re.search(r"\b\d{1,2}\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)\b", text, re.I)
    )
    has_role = bool(ROLE_TERMS_REGEX.search(text))
    has_employment_dates = _years_from_experience_dates(text) > 0
    matched_skills = {
        canonical for pattern, canonical in SKILL_VALIDATION_PATTERNS.items()
        if pattern.search(text)
    }

    professional_signals = sum((has_degree, has_experience, has_role, has_employment_dates))
    supporting_signals = sum((has_contact, len(matched_skills) >= 2))
    has_resume_sections = len(sections) >= 2 and bool(
        sections & {"experience", "education", "skills", "projects", "internships"}
    )
    if has_resume_sections and professional_signals >= 1 and professional_signals + supporting_signals >= 2:
        return True
    if has_contact and len(matched_skills) >= 3 and (
        {"education", "skills"} <= sections or {"projects", "skills"} <= sections
    ):
        return True

    # Allow concise narrative resumes without headings, but require several
    # independent resume-specific signals rather than generic words like "CV".
    return has_degree and has_experience and len(matched_skills) >= 2


def _years_from_experience_dates(text: str) -> int:
    heading = re.search(
        r"(?im)^\s*(?:professional\s+|work\s+)?experience(?:\s+history)?\s*:?\s*$",
        text,
    )
    if not heading:
        return 0

    remainder = text[heading.end():]
    next_section = re.search(
        r"(?im)^\s*(?:education|academic\s+background|technical\s+skills|skills|"
        r"projects|certifications|certificates|summary|career\s+objective)\s*:?\s*$",
        remainder,
    )
    experience_text = remainder[:next_section.start()] if next_section else remainder

    periods = []
    today = date.today()
    current_month = today.year * 12 + today.month - 1

    def month_number(value: Optional[str], default: int) -> int:
        if not value:
            return default
        if value.isdigit():
            return int(value)
        return {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }[value[:3].lower()]

    for match in DATE_RANGE_REGEX.finditer(experience_text):
        start = int(match.group("start")) * 12 + month_number(match.group("start_month"), 1) - 1
        end_value = match.group("end")
        end = (
            current_month if end_value.lower() in {"present", "current", "now"}
            else int(end_value) * 12 + month_number(match.group("end_month"), 1) - 1
        )
        if start <= end <= current_month:
            periods.append((start, end + 1))

    if not periods:
        return 0

    periods.sort()
    merged = []
    for start, end in periods:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    months = sum(end - start for start, end in merged)
    return min(months // 12, 25)


def extract_text_from_pdf(file_bytes_or_stream) -> str:
    """Extracts text from PDF stream using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_bytes_or_stream)
        text_parts = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text()
            if txt:
                text_parts.append(txt)
        return "\n".join(text_parts)
    except Exception as e:
        log.warning("PDF extraction failed: %s", e)
        return ""


def extract_text_from_docx(file_bytes_or_stream) -> str:
    """
    Extracts text from DOCX document using pure Python standard library
    zipfile and XML parsing without requiring python-docx.
    """
    try:
        if isinstance(file_bytes_or_stream, (bytes, bytearray)):
            stream = io.BytesIO(file_bytes_or_stream)
        else:
            stream = file_bytes_or_stream
        
        with zipfile.ZipFile(stream) as docx_zip:
            xml_content = docx_zip.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for p in tree.iterfind(".//w:p", ns):
                texts = [node.text for node in p.iterfind(".//w:t", ns) if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            return "\n".join(paragraphs)
    except Exception as e:
        log.warning("DOCX extraction failed: %s", e)
        return ""


def parse_document(file_obj, filename: str) -> str:
    """
    Parses an uploaded file (stream or bytes) based on its filename extension.
    Supports .txt, .pdf, and .docx.
    """
    fn = filename.lower()
    if fn.endswith(".pdf"):
        return extract_text_from_pdf(file_obj)
    elif fn.endswith(".docx"):
        return extract_text_from_docx(file_obj)
    else:
        # Default text read
        if hasattr(file_obj, "read"):
            content = file_obj.read()
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="ignore")
            return str(content)
        return str(file_obj)


def extract_resume_metadata(text: str) -> Dict[str, Any]:
    """
    Extracts structured entities from resume text:
    - Contact info (emails, phones, links)
    - Degrees / Education
    - Detected years of experience
    - Certifications
    """
    emails = EMAIL_REGEX.findall(text)
    phones = PHONE_REGEX.findall(text)
    linkedin = LINKEDIN_REGEX.findall(text)
    github = GITHUB_REGEX.findall(text)

    degrees = []
    for pattern, name in DEGREE_PATTERNS:
        if pattern.search(text):
            degrees.append(name)

    certs = []
    for pattern, name in CERTIFICATION_PATTERNS:
        if pattern.search(text):
            certs.append(name)

    years = 0
    for pat in YEARS_PATTERNS:
        matches = pat.findall(text)
        if matches:
            try:
                vals = [int(m) for m in matches if m.isdigit()]
                if vals:
                    years = max(years, min(max(vals), 25))
            except Exception:
                pass
    years = max(years, _years_from_experience_dates(text))

    return {
        "emails": list(dict.fromkeys(emails)),
        "phones": list(dict.fromkeys(phones)),
        "linkedin": list(dict.fromkeys(linkedin)),
        "github": list(dict.fromkeys(github)),
        "degrees": list(dict.fromkeys(degrees)),
        "certifications": list(dict.fromkeys(certs)),
        "extracted_years": years,
    }
