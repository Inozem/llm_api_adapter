import base64

import pytest

from src.llm_api_adapter.models.messages.file_parts import (
    DocumentPart,
    FilePart,
    ImagePart,
)


# ---------------------------------------------------------------------------
# FilePart — validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_file_part_url_auto_detects_media_type():
    fp = FilePart(url="https://example.com/img.jpg")
    assert fp.media_type == "image/jpeg"


@pytest.mark.unit
def test_file_part_url_no_extension_leaves_media_type_none():
    fp = FilePart(url="https://api.example.com/img?id=1")
    assert fp.media_type is None


@pytest.mark.unit
def test_file_part_data_with_media_type_ok():
    fp = FilePart(data=b"x", media_type="image/jpeg")
    assert fp.media_type == "image/jpeg"


@pytest.mark.unit
def test_file_part_no_source_raises():
    with pytest.raises(ValueError, match="requires either url or data"):
        FilePart()


@pytest.mark.unit
def test_file_part_both_sources_raises():
    with pytest.raises(ValueError, match="not both"):
        FilePart(url="https://example.com/img.jpg", data=b"x")


@pytest.mark.unit
def test_file_part_data_without_media_type_raises():
    with pytest.raises(ValueError, match="requires media_type"):
        FilePart(data=b"x")


# ---------------------------------------------------------------------------
# FilePart — helpers
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_file_part_is_url_true_for_https():
    fp = FilePart(url="https://example.com/img.png")
    assert fp._is_url() is True


@pytest.mark.unit
def test_file_part_is_url_false_for_data_uri():
    fp = FilePart(url="data:image/png;base64,abc")
    assert fp._is_url() is False


@pytest.mark.unit
def test_file_part_get_media_type_explicit():
    fp = FilePart(data=b"x", media_type="image/webp")
    assert fp._get_media_type() == "image/webp"


@pytest.mark.unit
def test_file_part_get_media_type_from_data_uri():
    fp = FilePart(url="data:image/gif;base64,abc")
    assert fp._get_media_type() == "image/gif"


@pytest.mark.unit
def test_file_part_to_data_uri_from_bytes():
    raw = b"hello"
    fp = FilePart(data=raw, media_type="image/jpeg")
    uri = fp._to_data_uri()
    assert uri == f"data:image/jpeg;base64,{base64.b64encode(raw).decode()}"


@pytest.mark.unit
def test_file_part_to_data_uri_passthrough_for_data_uri():
    uri = "data:image/png;base64,abc123"
    fp = FilePart(url=uri)
    assert fp._to_data_uri() == uri


@pytest.mark.unit
def test_file_part_get_b64_data_from_bytes():
    raw = b"hello"
    fp = FilePart(data=raw, media_type="image/jpeg")
    assert fp._get_b64_data() == base64.b64encode(raw).decode()


@pytest.mark.unit
def test_file_part_get_b64_data_from_data_uri():
    fp = FilePart(url="data:image/png;base64,abc123")
    assert fp._get_b64_data() == "abc123"


# ---------------------------------------------------------------------------
# ImagePart — validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_image_part_url_auto_detects_media_type():
    img = ImagePart(url="https://example.com/photo.jpg")
    assert img.media_type == "image/jpeg"


@pytest.mark.unit
def test_image_part_data_with_image_media_type_ok():
    img = ImagePart(data=b"x", media_type="image/png")
    assert img.media_type == "image/png"


@pytest.mark.unit
def test_image_part_url_with_non_image_extension_raises():
    with pytest.raises(ValueError, match="image/\\*"):
        ImagePart(url="https://example.com/doc.pdf")


@pytest.mark.unit
def test_image_part_url_with_no_extension_raises():
    with pytest.raises(ValueError, match="Cannot detect"):
        ImagePart(url="https://api.example.com/img")


@pytest.mark.unit
def test_image_part_data_with_non_image_media_type_raises():
    with pytest.raises(ValueError, match="image/\\*"):
        ImagePart(data=b"x", media_type="application/pdf")


@pytest.mark.unit
def test_image_part_data_uri_in_url_resolves_media_type():
    img = ImagePart(url="data:image/webp;base64,abc")
    assert img._get_media_type() == "image/webp"


# ---------------------------------------------------------------------------
# DocumentPart — validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_document_part_url_auto_detects_pdf_media_type():
    doc = DocumentPart(url="https://example.com/report.pdf")
    assert doc.media_type == "application/pdf"


@pytest.mark.unit
def test_document_part_data_with_pdf_media_type_ok():
    doc = DocumentPart(data=b"%PDF", media_type="application/pdf")
    assert doc.media_type == "application/pdf"


@pytest.mark.unit
def test_document_part_url_without_extension_raises():
    with pytest.raises(ValueError, match="Cannot detect document"):
        DocumentPart(url="https://api.example.com/report")


@pytest.mark.unit
def test_document_part_url_with_non_pdf_extension_raises():
    with pytest.raises(ValueError, match="application/pdf"):
        DocumentPart(url="https://example.com/photo.jpg")


@pytest.mark.unit
def test_document_part_data_with_non_pdf_media_type_raises():
    with pytest.raises(ValueError, match="application/pdf"):
        DocumentPart(data=b"image", media_type="image/jpeg")


@pytest.mark.unit
def test_document_part_data_uri_resolves_pdf_media_type():
    doc = DocumentPart(url="data:application/pdf;base64,JVBERg==")
    assert doc._get_media_type() == "application/pdf"
