import io
import zipfile
from datetime import date
import pytest
from core.parser import (
    extract_text_from_docx,
    parse_document,
    extract_resume_metadata,
    is_valid_resume,
)


def _create_dummy_docx_bytes(paragraphs: list) -> bytes:
    """Creates a minimal in-memory .docx file for testing without external tools."""
    xml_parts = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                 '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>']
    for p in paragraphs:
        xml_parts.append(f'<w:p><w:r><w:t>{p}</w:t></w:r></w:p>')
    xml_parts.append('</w:body></w:document>')
    xml_content = "".join(xml_parts).encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", xml_content)
    return buf.getvalue()


def test_docx_parser_extracts_paragraphs():
    paragraphs = ["First line of resume", "Software Engineer with Python skills", "MCA Graduate"]
    docx_bytes = _create_dummy_docx_bytes(paragraphs)
    text = extract_text_from_docx(docx_bytes)
    for p in paragraphs:
        assert p in text


def test_parse_document_dispatcher():
    # Plain text
    txt_file = io.BytesIO(b"Candidate with 3 years experience")
    assert "3 years experience" in parse_document(txt_file, "resume.txt")

    # Docx
    docx_bytes = _create_dummy_docx_bytes(["AWS Certified Cloud Developer"])
    assert "AWS Certified" in parse_document(docx_bytes, "resume.docx")


def test_extract_resume_metadata():
    resume_text = (
        "Alice Smith - Software Engineer\n"
        "Email: alice.dev@example.com | Phone: 987-654-3210\n"
        "GitHub: github.com/alicedev | LinkedIn: linkedin.com/in/alicedev\n"
        "Education: MCA from Delhi University. 4 years of experience.\n"
        "Certifications: AWS Certified and TensorFlow Developer.\n"
    )
    meta = extract_resume_metadata(resume_text)
    assert "alice.dev@example.com" in meta["emails"]
    assert "MCA" in meta["degrees"]
    assert any("AWS Certified" in c for c in meta["certifications"])
    assert meta["extracted_years"] == 4


def test_experience_is_derived_from_employment_dates():
    resume_text = (
        "Professional Experience\n"
        "Software Engineer\n"
        "Jan 2020 - Dec 2022\n"
        "Data Analyst\n"
        "Jan 2023 - Present\n"
        "Education\n"
        "BCA, 2016 - 2020\n"
    )
    meta = extract_resume_metadata(resume_text)
    expected_years = min(date.today().year - 2020, 25)
    assert meta["extracted_years"] == expected_years


def test_resume_validation_rejects_unrelated_documents():
    random_document = (
        "Quarterly facilities report describing the building maintenance schedule, "
        "energy usage, invoices, and planned repairs for the upcoming season."
    )
    assert not is_valid_resume(random_document)


def test_resume_title_and_generic_sections_are_not_enough():
    unrelated = (
        "CV: Quarterly Operations Report\n"
        "Experience\n"
        "This document describes customer experience with a new service.\n"
        "Skills\n"
        "The report covers communication, planning, and general observations "
        "from the latest survey across several business departments."
    )
    assert not is_valid_resume(unrelated)


def test_resume_validation_accepts_resume_content():
    resume_text = (
        "Jordan Smith | jordan.smith@example.com\n"
        "Professional Experience\n"
        "Software Engineer, Example Co, 2021 - Present\n"
        "Developed Python services and improved SQL query performance.\n"
        "Education\n"
        "BCA, Example University\n"
        "Technical Skills\n"
        "Python, SQL, Docker"
    )
    assert is_valid_resume(resume_text)
    assert is_valid_resume(
        "BCA graduate with 1 years of experience. Technical skills include python, sql, "
        "excel. Created dashboards using excel. Responsible for data cleaning using python."
    )


def test_resume_validation_accepts_fresher_resume_without_work_history():
    fresher_resume = (
        "Avery Doe | avery.doe@example.com\n"
        "Education\n"
        "Bachelor of Information Technology, Example University\n"
        "Skills\n"
        "Python, Java, SQL, Excel\n"
        "Projects\n"
        "Built an inventory application as an academic project."
    )
    assert is_valid_resume(fresher_resume)


def test_resume_validation_rejects_books_and_publications():
    book_text = (
        "Designing for Behavior Change\n"
        "SECOND EDITION\n"
        "Applying Psychology and Behavioral Economics\n"
        "Stephen Wendel\n\n"
        "Designing for Behavior Change\n"
        "by Stephen Wendel\n"
        "Copyright 2020 Stephen Wendel. All rights reserved.\n"
        "Printed in Canada.\n"
        "Published by O'Reilly Media, Inc.\n"
        "ISBN: 978-1-492-05603-4\n\n"
        "Stephen Wendel has a Ph.D. and over 25 years of experience in behavioral economics.\n"
        "He utilizes Python and SQL to analyze behavior."
    )
    assert not is_valid_resume(book_text)


def test_resume_validation_detailed_reason():
    from core.parser import validate_resume_text
    is_valid, reason = validate_resume_text("ISBN: 978-1-492-05603-4 Second Edition All rights reserved")
    assert not is_valid
    assert "ISBN" in reason or "Non-resume" in reason
