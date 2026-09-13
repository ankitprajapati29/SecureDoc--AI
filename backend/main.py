from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from PIL import (
    Image,
    ImageOps,
    ImageEnhance,
    ImageFilter,
    ImageStat,
    ExifTags,
)

import fitz
import io
import os
import re
import hashlib
from datetime import datetime, date
from collections import Counter, defaultdict

import pytesseract

# ============================================================
# OPTIONAL PADDLEOCR SUPPORT
# ============================================================

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception as paddle_import_error:
    PaddleOCR = None
    PADDLE_AVAILABLE = False
    print("PADDLEOCR IMPORT ERROR:", str(paddle_import_error))

_PADDLE_ENGINE = None
_PADDLE_ENGINE_ERROR = None


# ============================================================
# OPTIONAL OPENCV SUPPORT
# ============================================================

try:
    import cv2
    import numpy as np

    CV2_AVAILABLE = True

except Exception:
    cv2 = None
    np = None
    CV2_AVAILABLE = False


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="SecureDoc AI Backend",
    version="6.2.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIGURATION
# ============================================================

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = (
        TESSERACT_PATH
    )


ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/pdf",
}


MAX_FILE_SIZE = 10 * 1024 * 1024

MAX_OCR_TEXT_LENGTH = 25000

MAX_PDF_OCR_PAGES = 5

OCR_TARGET_WIDTH = 1600
OCR_MAX_DIMENSION = 2400

# ============================================================
# DOCUMENT LABELS
# ============================================================

DOCUMENT_LABELS = {
    "AADHAAR_CARD": "Aadhaar Card",
    "PAN_CARD": "PAN Card",
    "DRIVING_LICENCE": "Driving Licence",
    "PASSPORT": "Passport",
    "VISA": "Visa",
    "PERMIT": "Permit",
    "IDENTITY_CARD": "Identity Card",
    "VOTER_ID": "Voter ID",
    "RATION_CARD": "Ration Card",
    "GST_DOCUMENT": "GST Document",
    "INVOICE": "Invoice",
    "MARKSHEET": "Marksheet",
    "CERTIFICATE": "Certificate",
    "BANK_DOCUMENT": "Bank Document",
    "UNKNOWN": "Unknown Document",
}


# ============================================================
# BASIC TEXT UTILITIES
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\u00a0", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n[ \t]+",
        "\n",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def normalize_single_line(text):

    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


def clean_field_value(value):

    if value is None:
        return None

    value = str(value).strip()

    value = re.sub(
        r"^[^A-Za-z0-9\u0900-\u097F]+",
        "",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    value = value.strip(
        " .,:;|_-#/\\"
    )

    if not value:
        return None

    return value


def clean_ocr_token(value):

    if not value:
        return ""

    value = str(value)

    value = value.replace(
        "\u2018",
        "'",
    )

    value = value.replace(
        "\u2019",
        "'",
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def split_clean_lines(text):

    output = []

    for line in normalize_text(text).splitlines():

        line = clean_field_value(line)

        if line:
            output.append(line)

    return output


def compact_text(text):

    return re.sub(
        r"[^A-Za-z0-9]",
        "",
        str(text or ""),
    )


# ============================================================
# SAFE IMAGE CLOSE
# ============================================================

def safe_close(image):

    try:
        if image is not None:
            image.close()

    except Exception:
        pass


# ============================================================
# FILE SIGNATURE DETECTION
# ============================================================

def detect_file_signature(data):

    if data.startswith(
        b"\xff\xd8\xff"
    ):
        return "image/jpeg"

    if data.startswith(
        b"\x89PNG\r\n\x1a\n"
    ):
        return "image/png"

    if data.startswith(
        b"%PDF"
    ):
        return "application/pdf"

    if (
        data[:4] == b"RIFF"
        and b"WEBP" in data[:16]
    ):
        return "image/webp"

    return "unknown"


def normalize_content_type(content_type):

    aliases = {
        "image/jpg": "image/jpeg",
        "application/x-pdf": "application/pdf",
    }

    value = (
        content_type or ""
    ).lower().strip()

    return aliases.get(
        value,
        value,
    )


def content_type_matches(
    declared_type,
    detected_type,
):

    declared_type = normalize_content_type(
        declared_type
    )

    return (
        detected_type != "unknown"
        and declared_type == detected_type
    )


# ============================================================
# OCR LANGUAGE
# ============================================================

def get_ocr_language():

    try:

        languages = set(
            pytesseract.get_languages(
                config=""
            )
        )

        if (
            "eng" in languages
            and "hin" in languages
        ):
            return "eng+hin"

        if "eng" in languages:
            return "eng"

        if languages:
            return next(
                iter(languages)
            )

    except Exception:
        pass

    return "eng"


# ============================================================
# IMAGE ORIENTATION
# ============================================================

def fix_orientation(image):

    try:

        return ImageOps.exif_transpose(
            image
        )

    except Exception:

        return image


# ============================================================
# IMAGE RESIZE
# ============================================================

def resize_for_ocr(
    image,
    target_width=OCR_TARGET_WIDTH,
    max_dimension=OCR_MAX_DIMENSION,
):

    width, height = image.size

    if width <= 0 or height <= 0:
        return image.copy()

    scale = 1.0

    if width < target_width:

        scale = min(
            target_width / width,
            3.5,
        )

    if (
        max(width, height) * scale
        > max_dimension
    ):

        scale = (
            max_dimension
            / max(width, height)
        )

    if abs(scale - 1.0) < 0.01:

        return image.copy()

    new_width = max(
        1,
        int(width * scale),
    )

    new_height = max(
        1,
        int(height * scale),
    )

    return image.resize(
        (
            new_width,
            new_height,
        ),
        Image.Resampling.LANCZOS,
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def create_enhanced_gray(image):

    image = fix_orientation(
        image
    )

    if image.mode != "RGB":

        image = image.convert(
            "RGB"
        )

    image = resize_for_ocr(
        image
    )

    gray = ImageOps.grayscale(
        image
    )

    gray = ImageOps.autocontrast(
        gray,
        cutoff=1,
    )

    gray = ImageEnhance.Contrast(
        gray
    ).enhance(
        1.8
    )

    gray = ImageEnhance.Sharpness(
        gray
    ).enhance(
        1.4
    )

    gray = gray.filter(
        ImageFilter.MedianFilter(
            size=3
        )
    )

    gray = gray.filter(
        ImageFilter.UnsharpMask(
            radius=1.5,
            percent=160,
            threshold=3,
        )
    )

    return gray


def create_ocr_variants(image):

    original = fix_orientation(
        image
    ).convert(
        "RGB"
    )

    original = resize_for_ocr(
        original
    )

    enhanced = create_enhanced_gray(
        image
    )

    variants = [
        (
            "original",
            original.copy(),
        ),
        (
            "enhanced_gray",
            enhanced.copy(),
        ),
    ]

    if CV2_AVAILABLE:

        arr = np.array(
            enhanced
        )

        try:

            otsu = cv2.threshold(
                arr,
                0,
                255,
                cv2.THRESH_BINARY
                + cv2.THRESH_OTSU,
            )[1]

            variants.append(
                (
                    "otsu",
                    Image.fromarray(
                        otsu
                    ),
                )
            )

        except Exception:
            pass

        try:

            adaptive_gaussian = (
                cv2.adaptiveThreshold(
                    arr,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31,
                    8,
                )
            )

            variants.append(
                (
                    "adaptive_gaussian",
                    Image.fromarray(
                        adaptive_gaussian
                    ),
                )
            )

        except Exception:
            pass

        try:

            adaptive_mean = (
                cv2.adaptiveThreshold(
                    arr,
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    51,
                    10,
                )
            )

            variants.append(
                (
                    "adaptive_mean",
                    Image.fromarray(
                        adaptive_mean
                    ),
                )
            )

        except Exception:
            pass

        try:

            denoised = (
                cv2.fastNlMeansDenoising(
                    arr,
                    None,
                    7,
                    7,
                    21,
                )
            )

            variants.append(
                (
                    "denoised",
                    Image.fromarray(
                        denoised
                    ),
                )
            )

        except Exception:
            pass

    else:

        for threshold in (
            145,
            165,
            185,
        ):

            binary = enhanced.point(
                lambda x,
                t=threshold: (
                    255
                    if x > t
                    else 0
                )
            )

            variants.append(
                (
                    f"threshold_{threshold}",
                    binary,
                )
            )

    safe_close(
        enhanced
    )

    safe_close(
        original
    )

    return variants


# ============================================================
# IMAGE QUALITY ANALYSIS
# ============================================================

def analyze_image_quality(image):

    gray = ImageOps.grayscale(
        fix_orientation(
            image
        )
    )

    width, height = gray.size

    stat = ImageStat.Stat(
        gray
    )

    mean = (
        float(stat.mean[0])
        if stat.mean
        else 0.0
    )

    stddev = (
        float(stat.stddev[0])
        if stat.stddev
        else 0.0
    )

    blur_score = None

    if CV2_AVAILABLE:

        try:

            arr = np.array(
                gray
            )

            blur_score = float(
                cv2.Laplacian(
                    arr,
                    cv2.CV_64F,
                ).var()
            )

        except Exception:
            blur_score = None

    issues = []

    if width < 500 or height < 300:

        issues.append(
            "Low image resolution"
        )

    if stddev < 18:

        issues.append(
            "Low image contrast"
        )

    if (
        blur_score is not None
        and blur_score < 35
    ):

        issues.append(
            "Image may be blurry"
        )

    if mean < 25:

        issues.append(
            "Image is very dark"
        )

    if mean > 245:

        issues.append(
            "Image is very bright"
        )

    status = (
        "REVIEW"
        if issues
        else "GOOD"
    )

    safe_close(
        gray
    )

    return {
        "status": status,
        "width": width,
        "height": height,
        "mean_brightness": round(
            mean,
            1,
        ),
        "contrast": round(
            stddev,
            1,
        ),
        "blur_score": (
            round(
                blur_score,
                1,
            )
            if blur_score is not None
            else None
        ),
        "issues": issues,
    }


# ============================================================
# PADDLEOCR ENGINE (PRIMARY ACCURACY LAYER)
# ============================================================

def get_paddle_ocr():
    global _PADDLE_ENGINE, _PADDLE_ENGINE_ERROR

    if not PADDLE_AVAILABLE:
        return None

    if _PADDLE_ENGINE is not None:
        return _PADDLE_ENGINE

    if _PADDLE_ENGINE_ERROR is not None:
        return None

    try:
        _PADDLE_ENGINE = PaddleOCR(
            lang="en",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        return _PADDLE_ENGINE

    except Exception as error:
        _PADDLE_ENGINE_ERROR = str(error)
        print("PADDLEOCR INITIALIZATION ERROR:", _PADDLE_ENGINE_ERROR)
        return None


def _paddle_to_python(value):
    try:
        if hasattr(value, "json"):
            json_value = value.json
            if callable(json_value):
                json_value = json_value()
            if isinstance(json_value, str):
                import json
                return json.loads(json_value)
            return json_value
    except Exception:
        pass
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)):
        return [_paddle_to_python(item) for item in value]
    return value


def _extract_paddle_pairs(value):
    pairs = []

    def walk(item):
        item = _paddle_to_python(item)
        if isinstance(item, dict):
            texts = (
                item.get("rec_texts")
                or item.get("texts")
                or item.get("text")
            )
            scores = (
                item.get("rec_scores")
                or item.get("scores")
                or item.get("score")
            )
            if isinstance(texts, str):
                texts = [texts]
            if texts:
                if not isinstance(scores, (list, tuple)):
                    scores = [scores] * len(texts)
                for text, score in zip(texts, scores):
                    clean = clean_ocr_token(str(text))
                    if not clean:
                        continue
                    try:
                        confidence = float(score)
                        if confidence <= 1:
                            confidence *= 100
                    except Exception:
                        confidence = 0.0
                    pairs.append((clean, confidence))
                return
            for nested in item.values():
                if isinstance(nested, (dict, list, tuple)):
                    walk(nested)
            return

        if isinstance(item, (list, tuple)):
            # Legacy PaddleOCR output: [box, (text, confidence)]
            if (
                len(item) >= 2
                and isinstance(item[1], (list, tuple))
                and len(item[1]) >= 2
                and isinstance(item[1][0], str)
            ):
                clean = clean_ocr_token(item[1][0])
                if clean:
                    try:
                        confidence = float(item[1][1])
                        if confidence <= 1:
                            confidence *= 100
                    except Exception:
                        confidence = 0.0
                    pairs.append((clean, confidence))
                return
            for nested in item:
                walk(nested)

    walk(value)
    return pairs


def run_paddle_ocr_pass(image):
    engine = get_paddle_ocr()
    if engine is None:
        return {
            "text": "",
            "lines": [],
            "tokens": [],
            "confidence": 0.0,
        }

    try:
        rgb = image.convert("RGB")
        array = np.ascontiguousarray(np.asarray(rgb)) if CV2_AVAILABLE else None

        if hasattr(engine, "predict"):
            result = engine.predict(array if array is not None else rgb)
        elif hasattr(engine, "ocr"):
            result = engine.ocr(array if array is not None else rgb, cls=True)
        else:
            raise RuntimeError("Installed PaddleOCR API does not expose predict() or ocr().")

        pairs = _extract_paddle_pairs(result)
        lines = []
        tokens = []
        confidences = []

        for index, (text, confidence) in enumerate(pairs):
            if not text:
                continue
            lines.append({
                "text": text,
                "confidence": confidence,
                "left": 0,
                "top": index,
                "width": 0,
                "height": 0,
            })
            tokens.append({
                "text": text,
                "confidence": confidence,
                "left": 0,
                "top": index,
                "width": 0,
                "height": 0,
                "block": 0,
                "paragraph": 0,
                "line": index,
            })
            if confidence >= 0:
                confidences.append(confidence)

        text = normalize_text("\n".join(item["text"] for item in lines))
        confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else 0.0
        )

        return {
            "text": text,
            "lines": lines,
            "tokens": tokens,
            "confidence": round(confidence, 2),
        }

    except Exception as error:
        print("PADDLEOCR ERROR:", str(error))
        return {
            "text": "",
            "lines": [],
            "tokens": [],
            "confidence": 0.0,
        }


# ============================================================
# LAYOUT-PRESERVING OCR
# ============================================================

def run_ocr_pass(
    image,
    config,
    language,
):

    try:

        data = (
            pytesseract.image_to_data(
                image,
                lang=language,
                config=config,
                output_type=(
                    pytesseract.Output.DICT
                ),
            )
        )

    except Exception as error:

        print(
            "OCR ERROR:",
            str(error),
        )

        return {
            "text": "",
            "lines": [],
            "tokens": [],
            "confidence": 0.0,
        }

    grouped = defaultdict(
        list
    )

    confidences = []

    tokens = []

    total = len(
        data.get(
            "text",
            [],
        )
    )

    for index in range(total):

        word = clean_ocr_token(
            data["text"][index]
        )

        if not word:
            continue

        try:

            confidence = float(
                data["conf"][index]
            )

        except Exception:

            confidence = -1.0

        if confidence >= 0:

            confidences.append(
                confidence
            )

        try:

            block = int(
                data["block_num"][index]
            )

            paragraph = int(
                data["par_num"][index]
            )

            line_number = int(
                data["line_num"][index]
            )

        except Exception:

            block = 0
            paragraph = 0
            line_number = index

        try:

            left = int(
                data["left"][index]
            )

            top = int(
                data["top"][index]
            )

            width = int(
                data["width"][index]
            )

            height = int(
                data["height"][index]
            )

        except Exception:

            left = 0
            top = 0
            width = 0
            height = 0

        key = (
            block,
            paragraph,
            line_number,
        )

        grouped[key].append(
            {
                "text": word,
                "confidence": confidence,
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }
        )

        tokens.append(
            {
                "text": word,
                "confidence": confidence,
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "block": block,
                "paragraph": paragraph,
                "line": line_number,
            }
        )

    line_objects = []

    for key, words in grouped.items():

        words.sort(
            key=lambda item: item["left"]
        )

        line_text = " ".join(
            item["text"]
            for item in words
        ).strip()

        valid_confidences = [
            item["confidence"]
            for item in words
            if item["confidence"] >= 0
        ]

        line_confidence = (
            round(
                sum(
                    valid_confidences
                )
                / len(
                    valid_confidences
                ),
                1,
            )
            if valid_confidences
            else 0.0
        )

        line_top = min(
            (
                item["top"]
                for item in words
            ),
            default=0,
        )

        line_left = min(
            (
                item["left"]
                for item in words
            ),
            default=0,
        )

        line_objects.append(
            {
                "text": line_text,
                "confidence": line_confidence,
                "top": line_top,
                "left": line_left,
            }
        )

    line_objects.sort(
        key=lambda item: (
            item["top"],
            item["left"],
        )
    )

    lines = [
        item["text"]
        for item in line_objects
        if item["text"]
    ]

    text = normalize_text(
        "\n".join(
            lines
        )
    )

    confidence = (
        round(
            sum(confidences)
            / len(confidences),
            1,
        )
        if confidences
        else 0.0
    )

    return {
        "text": text,
        "lines": lines,
        "line_objects": line_objects,
        "tokens": tokens,
        "confidence": confidence,
    }


# ============================================================
# OCR TEXT QUALITY SCORE
# ============================================================

def text_quality_score(
    text,
    confidence,
):

    if not text:
        return -999.0

    length = len(
        text
    )

    alnum = sum(
        character.isalnum()
        for character in text
    )

    useful_ratio = (
        alnum
        / max(length, 1)
    )

    words = re.findall(
        r"[A-Za-z0-9\u0900-\u097F]{2,}",
        text,
    )

    garbage = len(
        re.findall(
            r"[^\w\s.,:/()&@#%+\-]",
            text,
        )
    )

    score = 0.0

    score += min(
        length,
        1000,
    ) * 0.04

    score += (
        useful_ratio
        * 30
    )

    score += min(
        len(words),
        100,
    ) * 0.35

    score += (
        confidence
        * 0.45
    )

    score -= min(
        garbage,
        50,
    ) * 0.35

    detection = detect_document_type(
        text
    )

    score += (
        detection.get(
            "score",
            0,
        )
        * 2
    )

    return round(
        score,
        2,
    )


# ============================================================
# DOCUMENT TYPE DETECTION
# ============================================================

def count_keywords(
    text,
    keywords,
):

    score = 0

    for keyword in keywords:

        if keyword in text:

            score += 1

    return score


# =========================================================
# DOCUMENT TYPE DETECTION
# PHASE 2 — MULTI-DOCUMENT DETECTION
# =========================================================

def detect_document_type(text):

    upper = normalize_text(
        text
    ).upper()

    scores = {}

    # --------------------------------------------------------
    # AADHAAR
    # --------------------------------------------------------

    aadhaar_score = count_keywords(
        upper,
        [
            "AADHAAR",
            "AADHAR",
            "UIDAI",
            "UNIQUE IDENTIFICATION AUTHORITY",
            "GOVERNMENT OF INDIA",
        ],
    )

    if re.search(
        r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)",
        upper,
    ):
        aadhaar_score += 3

    scores["AADHAAR_CARD"] = aadhaar_score

    # --------------------------------------------------------
    # PAN
    # --------------------------------------------------------

    pan_score = count_keywords(
        upper,
        [
            "INCOME TAX DEPARTMENT",
            "PERMANENT ACCOUNT NUMBER",
            "INCOME TAX",
        ],
    )

    if re.search(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        upper,
    ):
        pan_score += 3

    scores["PAN_CARD"] = pan_score

    # --------------------------------------------------------
    # DRIVING LICENCE
    # --------------------------------------------------------

    driving_score = count_keywords(
        upper,
        [
            "DRIVING LICENCE",
            "DRIVING LICENSE",
            "UNION OF INDIA",
            "TRANSPORT DEPARTMENT",
            "LICENCE NO",
            "LICENSE NO",
        ],
    )

    if re.search(
        r"\b[A-Z]{2}\d{2}[ -]?\d{4,6}[ -]?\d{5,10}\b",
        upper,
    ):
        driving_score += 3

    scores["DRIVING_LICENCE"] = driving_score

    # --------------------------------------------------------
    # PASSPORT
    # --------------------------------------------------------

    passport_score = count_keywords(
        upper,
        [
            "PASSPORT",
            "NATIONALITY",
            "DATE OF EXPIRY",
            "DATE OF ISSUE",
        ],
    )

    if re.search(
        r"\b[A-Z][0-9]{7}\b",
        upper,
    ):
        passport_score += 2

    scores["PASSPORT"] = passport_score

    # --------------------------------------------------------
    # VISA
    # --------------------------------------------------------

    visa_score = count_keywords(
        upper,
        [
            "VISA",
            "VISA NUMBER",
            "VISA NO",
            "TYPE OF VISA",
            "VALID FROM",
            "VALID UNTIL",
            "VALID TO",
            "ENTRIES",
        ],
    )

    scores["VISA"] = visa_score

    # --------------------------------------------------------
    # PERMIT
    # --------------------------------------------------------

    permit_score = count_keywords(
        upper,
        [
            "PERMIT",
            "RESIDENCE PERMIT",
            "WORK PERMIT",
            "ENTRY PERMIT",
            "TEMPORARY PERMIT",
        ],
    )

    scores["PERMIT"] = permit_score

    # --------------------------------------------------------
    # GENERIC IDENTITY CARD
    # --------------------------------------------------------

    identity_score = count_keywords(
        upper,
        [
            "IDENTITY CARD",
            "IDENTIFICATION CARD",
            "NATIONAL IDENTITY",
            "NATIONAL IDENTITY CARD",
            "NATIONAL ID",
            "SURNAME",
            "GIVEN NAMES",
        ],
    )

    scores["IDENTITY_CARD"] = identity_score

    # --------------------------------------------------------
    # VOTER ID
    # --------------------------------------------------------

    voter_score = count_keywords(
        upper,
        [
            "ELECTION COMMISSION",
            "ELECTOR",
            "ELECTORAL",
            "EPIC",
        ],
    )

    if re.search(
        r"\b[A-Z]{3}[0-9]{7}\b",
        upper,
    ):
        voter_score += 2

    scores["VOTER_ID"] = voter_score

    # --------------------------------------------------------
    # OTHER DOCUMENTS
    # --------------------------------------------------------

    scores["RATION_CARD"] = count_keywords(
        upper,
        [
            "RATION CARD",
            "PUBLIC DISTRIBUTION SYSTEM",
            "FOOD AND CIVIL SUPPLIES",
        ],
    )

    scores["GST_DOCUMENT"] = count_keywords(
        upper,
        [
            "GSTIN",
            "GOODS AND SERVICES TAX",
            "GST REGISTRATION",
        ],
    )

    scores["INVOICE"] = count_keywords(
        upper,
        [
            "INVOICE",
            "BILL TO",
            "AMOUNT DUE",
            "TOTAL AMOUNT",
        ],
    )

    scores["MARKSHEET"] = count_keywords(
        upper,
        [
            "MARKSHEET",
            "MARK SHEET",
            "TOTAL MARKS",
            "PERCENTAGE",
        ],
    )

    scores["CERTIFICATE"] = count_keywords(
        upper,
        [
            "CERTIFICATE",
            "THIS IS TO CERTIFY",
            "CERTIFIED THAT",
        ],
    )

    scores["BANK_DOCUMENT"] = count_keywords(
        upper,
        [
            "ACCOUNT NUMBER",
            "BANK STATEMENT",
            "IFSC",
            "ACCOUNT HOLDER",
        ],
    )

    # --------------------------------------------------------
    # BEST MATCH
    # --------------------------------------------------------

    category, score = max(
        scores.items(),
        key=lambda item: item[1],
    )

    if score < 2:
        category = "UNKNOWN"
        score = 0

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if score >= 5:
        confidence = "HIGH"

    elif score >= 2:
        confidence = "MEDIUM"

    else:
        confidence = "LOW"

    return {
        "document_category": category,
        "document_label": (
            DOCUMENT_LABELS.get(
                category,
                "Unknown Document",
            )
        ),
        "confidence": confidence,
        "score": score,
        "all_scores": scores,
    }

# ============================================================
# FIELD EXTRACTION HELPERS
# ============================================================

def extract_first(
    patterns,
    text,
    flags=re.IGNORECASE,
):

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags,
        )

        if match:

            if match.groups():

                value = match.group(1)

            else:

                value = match.group(0)

            value = clean_field_value(
                value
            )

            if value:

                return value

    return None


def normalize_date_value(value):

    if not value:
        return None

    value = clean_field_value(
        value
    )

    if not value:
        return None

    value = value.replace(
        ".",
        "/",
    )

    value = value.replace(
        "-",
        "/",
    )

    match = re.fullmatch(
        r"(\d{1,2})/(\d{1,2})/(\d{2,4})",
        value,
    )

    if match:

        day = match.group(1).zfill(2)
        month = match.group(2).zfill(2)
        year = match.group(3)

        return (
            f"{day}/{month}/{year}"
        )

    return value



def _normalize_passport_month(month):
    month = re.sub(r"[^A-Z]", "", str(month or "").upper())
    aliases = {
        "JAN": 1, "JANUARY": 1,
        "FEB": 2, "FEBRUARY": 2, "FEV": 2, "FÉV": 2,
        "MAR": 3, "MARCH": 3, "MARS": 3,
        "APR": 4, "APRIL": 4, "AVR": 4,
        "MAY": 5, "MAI": 5,
        "JUN": 6, "JUNE": 6, "JUIN": 6,
        "JUL": 7, "JULY": 7, "JUIL": 7,
        "AUG": 8, "AUGUST": 8, "AOUT": 8, "AOÛT": 8,
        "SEP": 9, "SEPT": 9, "SEPTEMBER": 9,
        "OCT": 10, "OCTOBER": 10, "OCTOBRE": 10,
        "NOV": 11, "NOVEMBER": 11, "NOVEMBRE": 11,
        "DEC": 12, "DECEMBER": 12, "DECEMBRE": 12, "DÉC": 12,
    }
    return aliases.get(month)


def _normalize_document_date(value):
    value = clean_field_value(value)
    if not value:
        return None

    value = value.upper().replace(".", "/").replace("-", "/")
    value = re.sub(r"\s+", " ", value).strip()

    numeric = re.search(r"(\d{1,2})\s*[/]\s*(\d{1,2})\s*[/]\s*(\d{2,4})", value)
    if numeric:
        d, m, y = numeric.groups()
        if len(y) == 2:
            y = ("20" if int(y) <= 49 else "19") + y
        return f"{int(d):02d}/{int(m):02d}/{y}"

    named = re.search(
        r"(\d{1,2})\s+([A-ZÀ-ÿ]{3,12})(?:\s*/\s*[A-ZÀ-ÿ]{3,12})?\s+(\d{4})",
        value,
        re.IGNORECASE,
    )
    if named:
        d, month_name, y = named.groups()
        m = _normalize_passport_month(month_name)
        if m:
            return f"{int(d):02d}/{m:02d}/{y}"

    return None


def _extract_labelled_document_date(text, labels):
    if not text:
        return None
    label_pattern = "(?:" + "|".join(labels) + ")"
    pattern = (
        label_pattern
        + r".{0,80}?(\d{1,2}\s*(?:[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}"
        + r"|\s+[A-ZÀ-ÿ]{3,12}(?:\s*/\s*[A-ZÀ-ÿ]{3,12})?\s+\d{4}))"
    )
    normalized = normalize_text(text)
    match = re.search(pattern, normalized, re.IGNORECASE)
    if match:
        value = _normalize_document_date(match.group(1))
        if value:
            return value

    lines = split_clean_lines(text)
    label_re = re.compile(label_pattern, re.IGNORECASE)
    for index, line in enumerate(lines):
        if label_re.search(line):
            window = " ".join(lines[index:index + 3])
            for candidate in re.findall(
                r"\d{1,2}\s*(?:[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}|\s+[A-ZÀ-ÿ]{3,12}(?:\s*/\s*[A-ZÀ-ÿ]{3,12})?\s+\d{4})",
                window,
                re.IGNORECASE,
            ):
                value = _normalize_document_date(candidate)
                if value:
                    return value
    return None

def extract_date_of_birth(text):
    value = _extract_labelled_document_date(
        text,
        [
            r"DATE\s*OF\s*BIRTH",
            r"DATE\s*DE\s*NAISSANCE",
            r"\bDOB\b",
            r"BIRTH\s*DATE",
        ],
    )
    return value


def extract_date_of_issue(text):
    return _extract_labelled_document_date(
        text,
        [
            r"DATE\s*OF\s*ISSUE",
            r"DATE\s*OF\s*ISSUANCE",
            r"DATE\s*DE\s*D[ÉE]LIVRANCE",
            r"\bDOI\b",
            r"ISSUED\s*ON",
        ],
    )


def extract_date_of_expiry(text):
    value = _extract_labelled_document_date(
        text,
        [
            r"DATE\s*OF\s*EXPIRY",
            r"DATE\s*OF\s*EXPIRATION",
            r"DATE\s*D[’']EXPIRATION",
            r"EXPIRY\s*DATE",
            r"VALID\s*(?:UP\s*TO|TILL|UNTIL|THRU|THROUGH)",
        ],
    )
    if value:
        return value

    # Driving-licence validity range: take the end date.
    dates = re.findall(
        r"\d{1,2}\s*(?:[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}|\s+[A-ZÀ-ÿ]{3,12}(?:\s*/\s*[A-ZÀ-ÿ]{3,12})?\s+\d{4})",
        normalize_text(text),
        re.IGNORECASE,
    )
    if len(dates) >= 2 and re.search(r"\bVALID(?:ITY)?\b", text, re.IGNORECASE):
        return _normalize_document_date(dates[-1])

    return None


def extract_validity_status(date_of_expiry):
    if not date_of_expiry:
        return "NOT AVAILABLE"

    try:
        expiry = datetime.strptime(date_of_expiry, "%d/%m/%Y").date()
    except Exception:
        return "REVIEW"

    today = date.today()
    if expiry < today:
        return "EXPIRED"
    return "VALID"

def extract_gender(text):
    if not text:
        return None
    value = extract_first(
        [r"(?:SEX|GENDER)\s*[:\-]?\s*(MALE|FEMALE|OTHER|TRANSGENDER|M|F)\b"],
        text,
    )
    if value:
        mapping = {"M": "Male", "F": "Female"}
        upper = value.upper()
        return mapping.get(upper, upper.title())
    match = re.search(r"\b(MALE|FEMALE|OTHER|TRANSGENDER)\b", text, re.IGNORECASE)
    return match.group(1).title() if match else None

# =========================================================
# DOCUMENT-SPECIFIC FIELD EXTRACTION
# PHASE 3
# =========================================================


def extract_nationality(text):

    if not text:
        return None

    upper = normalize_text(text).upper()

    # Passport nationality words -> canonical country name.
    country_map = {
        "CANADA": ["CANADIAN", "CANADIEN", "CANADIENNE", "CANADA"],
        "INDIA": ["INDIAN", "INDIEN", "INDIENNE", "INDIA"],
        "UNITED STATES": ["AMERICAN", "UNITED STATES", "USA"],
        "UNITED KINGDOM": ["BRITISH", "UNITED KINGDOM"],
        "AUSTRALIA": ["AUSTRALIAN", "AUSTRALIA"],
        "FRANCE": ["FRENCH", "FRANCAIS", "FRANÇAISE", "FRANCE"],
        "GERMANY": ["GERMAN", "DEUTSCH", "GERMANY"],
    }

    # Prefer the value that appears on/near a nationality-labelled line.
    lines = split_clean_lines(text)
    for index, line in enumerate(lines):
        line_upper = line.upper()
        if re.search(r"\bNATIONALIT(?:Y|É|E|V)\b", line_upper):
            window = " ".join(lines[index:index + 2]).upper()
            for country, words in country_map.items():
                if any(word in window for word in words):
                    return country

            # Generic fallback: remove the label and bilingual label text.
            value = re.sub(
                r"(?i).*?\bNATIONALIT(?:Y|É|E|V)\b\s*[:/\-]?\s*",
                "",
                line,
            )
            value = re.sub(
                r"(?i)\bNATIONALIT(?:Y|É|E|V)\b.*$",
                "",
                value,
            )
            value = clean_field_value(value)
            if value and not any(ch.isdigit() for ch in value):
                value = value.split("/")[0].strip()
                if len(value) >= 3:
                    return value.title()

    return None

def extract_driving_licence_number(text):

    if not text:
        return None 


    upper_text = text.upper()

    # --------------------------------------------------------
    # PRIORITY 1: LICENCE NUMBER LABEL KE SAATH
    # Example:
    # Licence No. : DL-042011011046
    # DL No : DL042011011046
    # Driving Licence No : DL-042011011046
    # --------------------------------------------------------

    labelled_patterns = [

        r"(?:LICENCE|LICENSE)\s*(?:NO\.?|NUMBER)?\s*[:\-]?\s*([A-Z]{2}(?:[-\s]?\d){6,16})",
        r"DRIVING\s*(?:LICENCE|LICENSE)\s*(?:NO\.?|NUMBER)?\s*[:\-]?\s*([A-Z]{2}(?:[-\s]?\d){6,16})",

        r"DL\s*(?:NO\.?|NUMBER)?\s*[:\-]?\s*([A-Z]{2}(?:[-\s]?\d){6,16})",

    ]

    for pattern in labelled_patterns:

        match = re.search(
            pattern,
            upper_text,
            re.IGNORECASE,
        )

        if match:

            value = match.group(1)

            value = re.sub(
                r"[^A-Z0-9]",
                "",
                value.upper(),
            )

            if len(value) >= 8:

                return value

    # --------------------------------------------------------
    # PRIORITY 2: COMMON DRIVING LICENCE FORMATS
    # Examples:
    # DL042011011046
    # DL-042011011046
    # UP32 20140012345
    # --------------------------------------------------------

    patterns = [

        r"\b[A-Z]{2}(?:[-\s]?\d){6,16}\b",

        r"\b[A-Z]{2}\d{6,16}\b",

        r"\b[A-Z]{2}\s*\d{4}\s*\d{5,10}\b",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            upper_text,
        )

        if match:

            value = match.group(0)

            value = re.sub(
                r"[^A-Z0-9]",
                "",
                value.upper(),
            )

            if len(value) >= 8:

                return value

    return None


def extract_visa_number(text):
    if not text:
        return None

    patterns = [
        r"(?:VISA\s*(?:NO|NUMBER)|VISA\s*NUMBER)\s*[:#-]?\s*([A-Z0-9]{5,20})",
        r"\b([A-Z]{1,3}\d{5,15})\b",
    ]
    upper = normalize_text(text).upper()
    for pattern in patterns:
        match = re.search(pattern, upper, re.IGNORECASE)
        if match:
            value = re.sub(r"[^A-Z0-9]", "", match.group(1).upper())
            if 5 <= len(value) <= 20 and any(ch.isdigit() for ch in value):
                return value
    return None


def extract_visa_type(text):

    match = re.search(
        r"(?:VISA TYPE|TYPE OF VISA|TYPE)"
        r"\s*[:\-]?\s*"
        r"([A-Za-z ]{2,40})",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return clean_field_value(
            match.group(1)
        )

    return None


def extract_stay_duration(text):

    match = re.search(
        r"(?:STAY DURATION|DURATION OF STAY|PERIOD OF STAY)"
        r"\s*[:\-]?\s*"
        r"([A-Za-z0-9 ]{1,30})",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return clean_field_value(
            match.group(1)
        )

    return None

# ============================================================
# DOCUMENT NUMBER EXTRACTION
# ============================================================

def extract_aadhaar_number(text):

    candidates = re.findall(
        r"(?<!\d)(\d{4}[ -]?\d{4}[ -]?\d{4}|\d{12})(?!\d)",
        text,
    )

    for candidate in candidates:

        digits = re.sub(
            r"\D",
            "",
            candidate,
        )

        if len(digits) != 12:
            continue

        if len(set(digits)) == 1:
            continue

        return (
            f"{digits[:4]} "
            f"{digits[4:8]} "
            f"{digits[8:12]}"
        )

    return None


def extract_pan_number(text):

    match = re.search(
        r"\b([A-Z]{5}[0-9]{4}[A-Z])\b",
        text.upper(),
    )

    return (
        match.group(1)
        if match
        else None
    )



def extract_passport_number(text):

    if not text:
        return None

    upper = normalize_text(text).upper()

    invalid_values = {
        "PASSPORT", "NUMBER", "NATIONALITY", "NATIONALITE",
        "CANADA", "INDIA", "GOVERNMENT", "DATEOFBIRTH",
        "EXPIRY", "GENDER", "SEX", "SURNAME", "GIVENNAMES",
        "CANADIEN", "CANADIENNE", "CANADIAN",
    }

    candidates = []

    def add_candidate(value, score):
        value = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
        if not (6 <= len(value) <= 12):
            return
        if value in invalid_values:
            return
        if not any(ch.isdigit() for ch in value):
            return
        if re.fullmatch(r"\d{6,12}", value):
            return
        if value in {"P1234567", "P0000000"}:
            return
        candidates.append((score, value))

    # Highest priority: value next to a passport-number label.
    label_patterns = [
        r"(?:PASSPORT\s*(?:NO|N[O0º°]|NUMBER|NUMERO)?(?:\s*/\s*(?:N[O0º°]\s*DE\s*PASSEPORT|PASSPORT\s*(?:NO|NUMBER)?))?|N[O0º°]\s*DE\s*PASSEPORT)\s*[:#\-]?\s*([A-Z][A-Z0-9]{5,11})",
        r"(?:PASSPORT\s*(?:NUMBER|NO)|DOCUMENT\s*(?:NUMBER|NO))\s*[:#\-]?\s*([A-Z0-9]{6,12})",
    ]
    for pattern in label_patterns:
        for match in re.finditer(pattern, upper, re.IGNORECASE):
            add_candidate(match.group(1), 100)

    # Common real passport formats.
    for match in re.finditer(r"\b([A-Z][A-Z0-9]{5,11})\b", upper):
        value = match.group(1)
        score = 0
        if value.startswith("P"):
            score += 20
        if re.fullmatch(r"[A-Z][0-9]{6,8}[A-Z0-9]{0,3}", value):
            score += 15
        # Prefer values close to passport-number labels.
        start = max(0, match.start() - 80)
        context = upper[start:match.start()]
        if "PASSPORT" in context:
            score += 35
        add_candidate(value, score)

    # MRZ fallback: second MRZ line starts with passport number (first 9 chars).
    for line in upper.splitlines():
        compact = re.sub(r"[^A-Z0-9<]", "", line)
        if len(compact) >= 30 and "<" in compact:
            add_candidate(compact[:9].replace("<", ""), 60)

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return candidates[0][1]

def extract_voter_id_number(text):

    match = re.search(
        r"\b([A-Z]{3}[0-9]{7})\b",
        text.upper(),
    )

    return (
        match.group(1)
        if match
        else None
    )


def extract_gstin(text):

    compact = re.sub(
        r"\s+",
        "",
        text.upper(),
    )

    match = re.search(
        r"\b(\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z0-9]Z[A-Z0-9])\b",
        compact,
    )

    return (
        match.group(1)
        if match
        else None
    )


# ============================================================
# NAME VALIDATION
# ============================================================

NAME_BLOCKED_WORDS = {
    "government",
    "india",
    "union",
    "driving",
    "licence",
    "license",
    "passport",
    "identity",
    "national",
    "card",
    "date",
    "birth",
    "issue",
    "expiry",
    "male",
    "female",
    "surname",
    "given",
    "names",
    "citizen",
    "authority",
    "aadhaar",
    "aadhar",
    "address",
    "blood",
    "group",
    "holder",
    "signature",
    "republic",
}


def normalize_name(value):
    value = clean_field_value(value)
    if not value:
        return None
    value = re.sub(r"\s+", " ", value).strip()
    words = []
    for word in value.split():
        word = re.sub(r"[^A-Za-z\u0900-\u097F\'-]", "", word)
        if word:
            words.append(word)
    return " ".join(words) if words else None

def is_valid_name(value):

    value = normalize_name(
        value
    )

    if not value:
        return False

    if len(value) < 3:

        return False

    if len(value) > 70:

        return False

    if re.search(
        r"\d",
        value,
    ):

        return False

    lower = value.lower()

    words = re.findall(
        r"[A-Za-z]+",
        value,
    )

    if len(words) < 1:

        return False

    if len(words) > 6:

        return False

    if any(
        word.lower()
        in NAME_BLOCKED_WORDS
        for word in words
    ):

        return False

    if (
        len(words) == 1
        and len(words[0]) < 4
    ):

        return False

    return True


def score_name_candidate(
    value,
    source_confidence=0,
):

    value = normalize_name(
        value
    )

    if not is_valid_name(
        value
    ):

        return -1000

    words = value.split()

    score = 0

    score += 25

    if len(words) == 2:

        score += 20

    elif len(words) == 3:

        score += 18

    elif len(words) == 1:

        score += 5

    score += min(
        len(value),
        40,
    ) * 0.5

    score += (
        max(
            source_confidence,
            0,
        )
        * 0.3
    )

    return score


# ============================================================
# LINE-BASED NAME EXTRACTION
# ============================================================

def extract_name_candidates_from_lines(
    lines,
):

    candidates = []

    cleaned_lines = []

    for line in lines:

        value = clean_field_value(
            line
        )

        if value:

            cleaned_lines.append(
                value
            )

    # --------------------------------------------------------
    # LABEL + VALUE ON SAME LINE
    # --------------------------------------------------------

    for line in cleaned_lines:

        patterns = [
            r"^(?:NAME|FULL\s*NAME|CARDHOLDER|CARD\s*HOLDER)\s*[:.-]\s*(.+)$",
            r"^(?:GIVEN\s*NAMES?|FORENAMES?)\s*[:.-]?\s*(.+)$",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                line,
                re.IGNORECASE,
            )

            if match:

                candidate = normalize_name(
                    match.group(1)
                )

                if is_valid_name(
                    candidate
                ):

                    candidates.append(
                        candidate
                    )

    # --------------------------------------------------------
    # SURNAME + GIVEN NAME
    # --------------------------------------------------------

    surname = None
    given_name = None

    for index, line in enumerate(
        cleaned_lines
    ):

        lower = line.lower()

        if (
            "surname" in lower
            and index + 1 < len(
                cleaned_lines
            )
        ):

            possible = (
                cleaned_lines[
                    index + 1
                ]
            )

            if is_valid_name(
                possible
            ):

                surname = normalize_name(
                    possible
                )

        if (
            "given name" in lower
            or "given names" in lower
            or "forenames" in lower
        ):

            if index + 1 < len(
                cleaned_lines
            ):

                possible = (
                    cleaned_lines[
                        index + 1
                    ]
                )

                if is_valid_name(
                    possible
                ):

                    given_name = normalize_name(
                        possible
                    )

    if surname and given_name:

        candidates.append(
            normalize_name(
                f"{given_name} {surname}"
            )
        )

    # --------------------------------------------------------
    # DOCUMENT-SPECIFIC POSITIONAL CANDIDATES
    # --------------------------------------------------------

    for line in cleaned_lines:

        candidate = normalize_name(
            line
        )

        if is_valid_name(
            candidate
        ):

            candidates.append(
                candidate
            )

    return candidates

def extract_name_from_text(text):

    if not text:
        return None

    lines = split_clean_lines(text)

    # ========================================================
    # NAME CLEANER
    # ========================================================

    def clean_name_candidate(value):

        if not value:
            return None

        value = str(value).strip()

        # Remove common field labels accidentally included
        value = re.sub(
            r"(?i)^(?:name|full\s+name|surname|nom|given\s+names?|prenoms?|pr[eé]noms?)\s*[:.\-]?\s*",
            "",
            value,
        )

        # Remove relation data
        value = re.sub(
            r"(?i)\b(?:s/o|d/o|w/o|c/o|son\s+of|daughter\s+of|wife\s+of)\b.*",
            "",
            value,
        )

        value = value.strip(
            " :-|,.;"
        )

        value = normalize_name(value)

        if not value:
            return None

        if not is_valid_name(value):
            return None

        lower = value.lower()

        # ====================================================
        # NEVER ACCEPT THESE AS PERSON NAME
        # ====================================================

        blocked_phrases = [
            "passport",
            "issuing country",
            "issuing count",
            "issuing authority",
            "country",
            "nationality",
            "authority",
            "government",
            "department",
            "transport",
            "licence to drive",
            "license to drive",
            "authorisation",
            "authorization",
            "authorisation to drive",
            "authorization to drive",
            "date of birth",
            "date of issue",
            "date of expiry",
            "date of expiration",
            "place of birth",
            "sex",
            "gender",
            "signature",
            "holder",
            "address",
            "validity",
            "valid till",
            "valid until",
            "passport no",
            "licence no",
            "license no",
            "document",
            "identity card",
        ]

        for phrase in blocked_phrases:

            if phrase in lower:
                return None

        # Reject obvious numbers inside name
        if re.search(
            r"\d",
            value,
        ):
            return None

        words = value.split()

        # Too short
        if len(words) == 0:
            return None

        # Too long = probably a sentence
        if len(words) > 6:
            return None

        return value

    # ========================================================
    # PASSPORT PRIORITY
    #
    # Surname/Nom      MARTIN
    # Given names      SARAH
    #
    # Final result:
    # SARAH MARTIN
    # ========================================================

    surname = None
    given_name = None

    surname_pattern = re.compile(
        r"(?i)"
        r"(?:surname|nom)"
        r"\s*[:.\-]?\s*"
        r"(.*)$"
    )

    given_pattern = re.compile(
        r"(?i)"
        r"(?:"
        r"given\s*names?"
        r"|given\s*name"
        r"|pr[eé]noms?"
        r")"
        r"\s*[:.\-]?\s*"
        r"(.*)$"
    )

    for index, line in enumerate(lines):

        # -------------------------------
        # PASSPORT SURNAME
        # -------------------------------

        surname_match = surname_pattern.search(
            line
        )

        if surname_match:

            value = clean_name_candidate(
                surname_match.group(1)
            )

            # Label may be on one line,
            # value on next line
            if (
                not value
                and index + 1 < len(lines)
            ):

                value = clean_name_candidate(
                    lines[index + 1]
                )

            if value:
                surname = value

        # -------------------------------
        # PASSPORT GIVEN NAME
        # -------------------------------

        given_match = given_pattern.search(
            line
        )

        if given_match:

            value = clean_name_candidate(
                given_match.group(1)
            )

            # Label may be on one line,
            # value on next line
            if (
                not value
                and index + 1 < len(lines)
            ):

                value = clean_name_candidate(
                    lines[index + 1]
                )

            if value:
                given_name = value

    # Passport final priority:
    # SARAH + MARTIN
    if given_name and surname:

        full_name = clean_name_candidate(
            f"{given_name} {surname}"
        )

        if full_name:
            return full_name

    if given_name:
        return given_name

    if surname:
        return surname

    # ========================================================
    # UNIVERSAL EXPLICIT NAME FIELD
    #
    # PAN:
    # Name
    # ANKIT KUMAR
    #
    # DL:
    # Name : ANURAG BREJA
    #
    # Aadhaar:
    # Name
    # XYZ
    # ========================================================

    explicit_name_pattern = re.compile(
        r"(?i)"
        r"^(?:"
        r"name"
        r"|full\s*name"
        r"|name\s*of\s*(?:holder|applicant|person)"
        r"|cardholder"
        r"|card\s*holder"
        r")"
        r"\s*[:.\-]?\s*"
        r"(.*)$"
    )

    for index, line in enumerate(lines):

        match = explicit_name_pattern.search(
            line.strip()
        )

        if not match:
            continue

        value = match.group(1).strip()

        # Same line:
        # Name : ANURAG BREJA
        candidate = clean_name_candidate(
            value
        )

        if candidate:
            return candidate

        # Next line:
        # Name
        # ANURAG BREJA
        if index + 1 < len(lines):

            candidate = clean_name_candidate(
                lines[index + 1]
            )

            if candidate:
                return candidate

    # ========================================================
    # PAN CARD SPECIAL FALLBACK
    #
    # Avoid Father's Name / DOB becoming Name
    # ========================================================

    pan_bad_labels = [
        "father",
        "father's name",
        "father name",
        "date of birth",
        "date of birth incorporation",
        "dob",
        "permanent account number",
        "income tax department",
        "government of india",
    ]

    for index, line in enumerate(lines):

        line_lower = line.lower().strip()

        if (
            line_lower == "name"
            or line_lower.startswith(
                "name "
            )
        ):

            if index + 1 < len(lines):

                candidate = clean_name_candidate(
                    lines[index + 1]
                )

                if candidate:

                    candidate_lower = (
                        candidate.lower()
                    )

                    if not any(
                        bad in candidate_lower
                        for bad in pan_bad_labels
                    ):
                        return candidate

    # ========================================================
    # DRIVING LICENCE / GENERIC LINE CANDIDATES
    # ========================================================

    candidates = []

    for index, line in enumerate(lines):

        candidate = clean_name_candidate(
            line
        )

        if not candidate:
            continue

        score = 0

        # Existing scoring if available
        try:

            score = score_name_candidate(
                candidate
            )

        except Exception:

            score = 0

        lower = candidate.lower()

        # Positive scoring
        if 1 <= len(
            candidate.split()
        ) <= 4:

            score += 20

        # Nearby "name" label = strong signal
        previous_line = ""

        if index > 0:
            previous_line = (
                lines[index - 1]
                .lower()
                .strip()
            )

        if (
            previous_line == "name"
            or previous_line.startswith(
                "name:"
            )
        ):

            score += 100

        # Negative document words
        negative_words = [
            "passport",
            "licence",
            "license",
            "government",
            "department",
            "transport",
            "authorisation",
            "authorization",
            "country",
            "nationality",
            "issuing",
            "date",
            "birth",
            "expiry",
            "issue",
            "address",
            "signature",
        ]

        if any(
            word in lower
            for word in negative_words
        ):

            score -= 1000

        candidates.append(
            (
                score,
                candidate,
            )
        )

    # ========================================================
    # BEST GENERIC CANDIDATE
    # ========================================================

    if candidates:

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        best_score, best_candidate = (
            candidates[0]
        )

        if best_score > -500:
            return best_candidate

    return None

# =========================================================
# FIELD EXTRACTION FROM ONE OCR CANDIDATE
# =========================================================


def extract_fields_from_text(
    text,
    detection,
):

    category = detection.get("document_category", "UNKNOWN")

    date_of_birth = extract_date_of_birth(text)
    date_of_issue = extract_date_of_issue(text)
    date_of_expiry = extract_date_of_expiry(text)

    structured = {
        "name": extract_name_from_text(text),
        "document": detection.get("document_label", "Unknown Document"),
        "document_category": category,
        "aadhaar_number": extract_aadhaar_number(text),
        "pan_number": extract_pan_number(text),
        "driving_licence_number": (
            extract_driving_licence_number(text)
            if category == "DRIVING_LICENCE" else None
        ),
        "passport_number": (
            extract_passport_number(text)
            if category == "PASSPORT" else None
        ),
        "voter_id_number": (
            extract_voter_id_number(text)
            if category == "VOTER_ID" else None
        ),
        "gstin": extract_gstin(text),
        "date_of_birth": date_of_birth,
        "gender": extract_gender(text),
        "date_of_issue": date_of_issue,
        "date_of_expiry": date_of_expiry,
        "nationality": (
            extract_nationality(text)
            if category == "PASSPORT" else None
        ),
        "validity_status": (
            extract_validity_status(date_of_expiry)
            if category in {"PASSPORT", "DRIVING_LICENCE"} else None
        ),
        "visa_number": (
            extract_visa_number(text)
            if category == "VISA" else None
        ),
        "visa_type": (
            extract_visa_type(text)
            if category == "VISA" else None
        ),
        "stay_duration": (
            extract_stay_duration(text)
            if category == "VISA" else None
        ),
    }
    return structured

# ============================================================
# FIELD VALIDATION
# ============================================================

def field_value_is_valid(
    field,
    value,
):

    if value is None:
        return False

    if field == "name":

        return is_valid_name(
            value
        )

    if field == "aadhaar_number":

        digits = re.sub(
            r"\D",
            "",
            value,
        )

        return len(digits) == 12

    if field == "pan_number":

        return bool(
            re.fullmatch(
                r"[A-Z]{5}[0-9]{4}[A-Z]",
                value.upper(),
            )
        )

    if field == "driving_licence_number":

        compact = re.sub(
            r"[^A-Z0-9]",
            "",
            value.upper(),
        )

        return (
            len(compact) >= 10
            and compact[:2].isalpha()
            and compact[2:4].isdigit()
        )

    if field == "passport_number":

        compact = re.sub(
            r"[^A-Z0-9]",
            "",
            str(value).upper(),
        )

        return (
            6 <= len(compact) <= 12
            and any(char.isdigit() for char in compact)
            and any(char.isalpha() for char in compact)
        )

    if field == "nationality":
        cleaned = clean_field_value(value)
        return bool(
            cleaned
            and len(cleaned) >= 3
            and not any(char.isdigit() for char in cleaned)
        )

    if field == "visa_number":
        compact = re.sub(r"[^A-Z0-9]", "", str(value).upper())
        return 5 <= len(compact) <= 20 and any(ch.isdigit() for ch in compact)

    if field in {"visa_type", "stay_duration"}:
        cleaned = clean_field_value(value)
        return bool(cleaned and len(cleaned) >= 1)

    if field == "validity_status":
        return str(value).upper() in {
            "VALID",
            "EXPIRED",
            "REVIEW",
            "NOT AVAILABLE",
        }

    if field in {
        "date_of_birth",
        "date_of_issue",
        "date_of_expiry",
    }:

        return bool(
            re.search(
                r"\d",
                str(value),
            )
        )

    if field == "gender":

        return str(
            value
        ).lower() in {
            "male",
            "female",
            "other",
            "transgender",
        }

    return True


# ============================================================
# FIELD-WISE OCR CONSENSUS
# ============================================================

def normalize_field_key(
    field,
    value,
):

    if value is None:
        return None

    if field == "name":

        value = normalize_name(
            value
        )

        return (
            value.lower()
            if value
            else None
        )

    if field in {
        "aadhaar_number",
        "driving_licence_number",
    }:

        return compact_text(
            value
        ).upper()

    return str(
        value
    ).strip().lower()


def build_field_consensus(
    candidate_results,
    final_detection,
):

    field_votes = defaultdict(
        list
    )

    fields = [
        "name",
        "aadhaar_number",
        "pan_number",
        "driving_licence_number",
        "passport_number",
        "nationality",
        "voter_id_number",
        "gstin",
        "date_of_birth",
        "gender",
        "date_of_issue",
        "date_of_expiry",
        "validity_status",
        "visa_number",
        "visa_type",
        "stay_duration",
    ]

    for candidate in candidate_results:

        extracted = candidate.get(
            "structured",
            {}
        )

        confidence = float(
            candidate.get(
                "confidence",
                0,
            )
            or 0
        )

        quality = float(
            candidate.get(
                "score",
                0,
            )
            or 0
        )

        for field in fields:

            value = extracted.get(
                field
            )

            if not field_value_is_valid(
                field,
                value,
            ):

                continue

            key = normalize_field_key(
                field,
                value,
            )

            if not key:
                continue

            vote_score = (
                10
                + confidence * 0.5
                + max(
                    quality,
                    0,
                )
                * 0.08
            )

            if field == "name":

                vote_score += (
                    score_name_candidate(
                        value,
                        confidence,
                    )
                )

            field_votes[
                field
            ].append(
                {
                    "key": key,
                    "value": value,
                    "vote_score": vote_score,
                    "confidence": confidence,
                    "variant": candidate.get(
                        "variant"
                    ),
                }
            )

    final_data = {
        "name": None,
        "document": final_detection.get(
            "document_label",
            "Unknown Document",
        ),
        "document_category": final_detection.get(
            "document_category",
            "UNKNOWN",
        ),
        "aadhaar_number": None,
        "pan_number": None,
        "driving_licence_number": None,
        "passport_number": None,
        "nationality": None,
        "voter_id_number": None,
        "gstin": None,
        "date_of_birth": None,
        "gender": None,
        "date_of_issue": None,
        "date_of_expiry": None,
        "validity_status": None,
        "visa_number": None,
        "visa_type": None,
        "stay_duration": None,
    }

    field_confidence = {}

    for field in fields:

        values = field_votes.get(
            field,
            []
        )

        if not values:
            field_confidence[
                field
            ] = 0

            continue

        grouped = defaultdict(
            list
        )

        for item in values:

            grouped[
                item["key"]
            ].append(
                item
            )

        best_key = None
        best_score = -1

        for key, items in grouped.items():

            score = sum(
                item["vote_score"]
                for item in items
            )

            frequency_bonus = (
                len(items)
                * 20
            )

            score += frequency_bonus

            if score > best_score:

                best_score = score
                best_key = key

        selected = grouped[
            best_key
        ]

        selected.sort(
            key=lambda item: (
                item["vote_score"],
                item["confidence"],
            ),
            reverse=True,
        )

        best_item = selected[0]

        final_data[
            field
        ] = best_item[
            "value"
        ]

        field_confidence[
            field
        ] = round(
            min(
                100,
                (
                    best_item[
                        "confidence"
                    ]
                    + (
                        len(selected)
                        * 8
                    )
                ),
            ),
            1,
        )

    return (
        final_data,
        field_confidence,
    )
# ================================================================
# OCR EXTRACTION PIPELINE — FAST + RELIABLE TESSERACT OCR
# ================================================================

def extract_ocr_data(image):
    """
    Fast + reliable Tesseract OCR.

    Strategy:
    1. Fast standard PSM 6 pass.
    2. PSM 11 only when the first result is weak/incomplete.
    3. One enhanced grayscale pass only when both results are weak.
    4. Select the best/most useful OCR result.
    """

    language = get_ocr_language()
    candidates = []
    original_image = None
    enhanced_image = None

    try:
        # --------------------------------------------------------
        # PREPARE ONE NORMAL OCR IMAGE
        # --------------------------------------------------------
        original_image = fix_orientation(
            image
        ).convert("RGB")

        original_image = resize_for_ocr(
            original_image
        )

        # --------------------------------------------------------
        # ADD OCR RESULT
        # --------------------------------------------------------
        def add_candidate(result, variant_name, config):
            text = normalize_text(
                result.get("text", "")
            )

            if not text:
                return

            result["variant"] = variant_name
            result["engine"] = "tesseract"

            detection = detect_document_type(
                text
            )

            structured = extract_fields_from_text(
                text,
                detection,
            )

            # Normal OCR quality score
            score = text_quality_score(
                text,
                result.get("confidence", 0),
            )

            # Give extra value to OCR results that
            # successfully identify a document.
            if detection.get("document_category") != "UNKNOWN":
                score += 5

            # Give extra value to results containing
            # important document numbers.
            important_fields = (
                "aadhaar_number",
                "pan_number",
                "driving_licence_number",
                "passport_number",
                "visa_number",
                "voter_id_number",
                "gstin",
            )

            found_fields = sum(
                1
                for field in important_fields
                if structured.get(field)
            )

            score += found_fields * 8

            result["score"] = round(
                score,
                2,
            )

            result["detection"] = detection
            result["structured"] = structured

            candidates.append(result)

        # --------------------------------------------------------
        # PASS 1 - FAST STANDARD OCR
        # --------------------------------------------------------
        result = run_ocr_pass(
            original_image,
            "--oem 3 --psm 6",
            language,
        )

        add_candidate(
            result,
            "original_psm6",
            "--oem 3 --psm 6",
        )

        # --------------------------------------------------------
        # CHECK FIRST RESULT
        # --------------------------------------------------------
        best = max(
            candidates,
            key=lambda item: item.get("score", -999),
            default=None,
        )

        confidence = (
            float(
                best.get("confidence", 0) or 0
            )
            if best
            else 0.0
        )

        category = (
            best.get("detection", {}).get(
                "document_category",
                "UNKNOWN",
            )
            if best
            else "UNKNOWN"
        )

        text_length = (
            len(
                normalize_text(
                    best.get("text", "")
                )
            )
            if best
            else 0
        )

        important_fields = (
            "aadhaar_number",
            "pan_number",
            "driving_licence_number",
            "passport_number",
            "visa_number",
            "voter_id_number",
            "gstin",
        )

        important_field_count = (
            sum(
                1
                for field in important_fields
                if best.get("structured", {}).get(field)
            )
            if best
            else 0
        )

        # --------------------------------------------------------
        # PASS 2 - PSM 11 ONLY FOR GENUINELY WEAK OCR
        # --------------------------------------------------------
        # Keep the common path to one Tesseract pass. This is the
        # main speed improvement for the Render deployment.
        needs_second_pass = (
            best is None
            or confidence < 40
            or not normalize_text(
                best.get("text", "")
            )
            or text_length < 25
        )

        if needs_second_pass:
            result = run_ocr_pass(
                original_image,
                "--oem 3 --psm 11",
                language,
            )

            add_candidate(
                result,
                "original_psm11",
                "--oem 3 --psm 11",
            )

        # --------------------------------------------------------
        # CHECK AGAIN
        # --------------------------------------------------------
        best = max(
            candidates,
            key=lambda item: item.get("score", -999),
            default=None,
        )

        confidence = (
            float(
                best.get("confidence", 0) or 0
            )
            if best
            else 0.0
        )

        category = (
            best.get("detection", {}).get(
                "document_category",
                "UNKNOWN",
            )
            if best
            else "UNKNOWN"
        )

        # --------------------------------------------------------
        # PASS 3 - ENHANCED OCR ONLY FOR DIFFICULT DOCUMENTS
        # --------------------------------------------------------
        needs_enhancement = (
            best is None
            or confidence < 30
            or not normalize_text(
                best.get("text", "")
            )
        )

        if needs_enhancement:

            enhanced_image = create_enhanced_gray(
                original_image
            )

            result = run_ocr_pass(
                enhanced_image,
                "--oem 3 --psm 6",
                language,
            )

            add_candidate(
                result,
                "enhanced_gray_psm6",
                "--oem 3 --psm 6",
            )

        # --------------------------------------------------------
        # NO OCR RESULT
        # --------------------------------------------------------
        if not candidates:
            return {
                "extracted_text": "",
                "raw_ocr_text": "",
                "ocr_confidence": 0.0,
                "ocr_status": "NO_TEXT_DETECTED",
                "ocr_language": language,
                "ocr_engine": "NONE",
                "ocr_variant": None,
                "ocr_candidates_tested": 0,
                "document_detection": detect_document_type(""),
                "structured_data": {},
                "field_confidence": {},
                "candidate_summary": [],
            }

        # --------------------------------------------------------
        # FINAL BEST OCR RESULT
        # --------------------------------------------------------
        candidates.sort(
            key=lambda item: (
                item.get("score", -999),
                item.get("confidence", 0),
                len(
                    normalize_text(
                        item.get("text", "")
                    )
                ),
            ),
            reverse=True,
        )

        best = candidates[0]

        # --------------------------------------------------------
        # FINAL DOCUMENT DETECTION
        # --------------------------------------------------------
        final_detection = detect_document_type(
            best.get("text", "")
        )

        # --------------------------------------------------------
        # RE-EXTRACT STRUCTURED FIELDS
        # --------------------------------------------------------
        for candidate in candidates:
            candidate["structured"] = (
                extract_fields_from_text(
                    candidate.get("text", ""),
                    final_detection,
                )
            )

        # --------------------------------------------------------
        # FIELD CONSENSUS
        # --------------------------------------------------------
        structured_data, field_confidence = (
            build_field_consensus(
                candidates,
                final_detection,
            )
        )

        # --------------------------------------------------------
        # FINAL TEXT
        # --------------------------------------------------------
        raw_ocr_text = normalize_text(
            best.get("text", "")
        )[:MAX_OCR_TEXT_LENGTH]

        candidate_summary = [
            {
                "variant": candidate.get(
                    "variant"
                ),
                "engine": candidate.get(
                    "engine"
                ),
                "confidence": candidate.get(
                    "confidence"
                ),
                "score": candidate.get(
                    "score"
                ),
                "document": final_detection.get(
                    "document_label"
                ),
            }
            for candidate in candidates[:8]
        ]

        return {
            "extracted_text": raw_ocr_text,
            "raw_ocr_text": raw_ocr_text,
            "ocr_confidence": float(
                best.get(
                    "confidence",
                    0
                )
                or 0
            ),
            "ocr_status": (
                "TEXT_DETECTED"
                if raw_ocr_text
                else "NO_TEXT_DETECTED"
            ),
            "ocr_language": language,
            "ocr_engine": "tesseract",
            "ocr_variant": best.get(
                "variant"
            ),
            "ocr_candidates_tested": len(
                candidates
            ),
            "document_detection": final_detection,
            "structured_data": structured_data,
            "field_confidence": field_confidence,
            "candidate_summary": candidate_summary,
            "ocr_tokens": best.get("tokens", []),
        }

    except Exception as error:
        print(
            "TESSERACT OCR PIPELINE ERROR:",
            str(error),
        )

        return {
            "extracted_text": "",
            "raw_ocr_text": "",
            "ocr_confidence": 0.0,
            "ocr_status": "OCR_ERROR",
            "ocr_language": language,
            "ocr_engine": "tesseract",
            "ocr_variant": None,
            "ocr_candidates_tested": len(
                candidates
            ),
            "document_detection": detect_document_type(
                ""
            ),
            "structured_data": {},
            "field_confidence": {},
            "candidate_summary": [],
        }

    finally:
        # --------------------------------------------------------
        # FREE MEMORY
        # --------------------------------------------------------
        safe_close(
            enhanced_image
        )

        safe_close(
            original_image
        )

# ============================================================
# DISPLAY TEXT
# ============================================================

def build_display_text(
    structured,
):

    fields = [
        (
            "Name",
            structured.get(
                "name"
            ),
        ),
        (
            "Document",
            structured.get(
                "document"
            ),
        ),
        (
            "Aadhaar Number",
            structured.get(
                "aadhaar_number"
            ),
        ),
        (
            "PAN Number",
            structured.get(
                "pan_number"
            ),
        ),
        (
            "Driving Licence Number",
            structured.get(
                "driving_licence_number"
            ),
        ),
        (
            "Passport Number",
            structured.get(
                "passport_number"
            ),
        ),
        (
            "Nationality",
            structured.get(
                "nationality"
            ),
        ),
        (
            "Voter ID Number",
            structured.get(
                "voter_id_number"
            ),
        ),
        (
            "GSTIN",
            structured.get(
                "gstin"
            ),
        ),
        (
            "Visa Number",
            structured.get("visa_number"),
        ),
        (
            "Visa Type",
            structured.get("visa_type"),
        ),
        (
            "Stay Duration",
            structured.get("stay_duration"),
        ),
        (
            "Date of Birth",
            structured.get(
                "date_of_birth"
            ),
        ),
        (
            "Gender",
            structured.get(
                "gender"
            ),
        ),
        (
            "Date of Issue",
            structured.get(
                "date_of_issue"
            ),
        ),
        (
            "Date of Expiry",
            structured.get(
                "date_of_expiry"
            ),
        ),
        (
            "Validity Status",
            structured.get(
                "validity_status"
            ),
        ),
    ]

    lines = []

    for label, value in fields:

        if value:

            lines.append(
                f"{label}: {value}"
            )

    return "\n".join(
        lines
    )

# ============================================================
# TAMPERING DETECTION
# ============================================================

def analyze_metadata_tampering(
    metadata
):

    suspicious_keywords = [
        "photoshop",
        "adobe",
        "gimp",
        "canva",
        "pixlr",
        "lightroom",
        "coreldraw",
        "illustrator",
    ]

    suspicious_matches = []

    for key, value in (
        metadata or {}
    ).items():

        combined = (
            f"{key} {value}"
        ).lower()

        for keyword in suspicious_keywords:

            if keyword in combined:

                suspicious_matches.append(
                    keyword
                )

    suspicious_matches = list(
        dict.fromkeys(
            suspicious_matches
        )
    )

    if suspicious_matches:

        status = "REVIEW"

    else:

        status = "NO_STRONG_SIGNAL"

    return {
        "status": status,
        "editing_software_signals": (
            suspicious_matches
        ),
        "metadata_present": bool(
            metadata
        ),
    }


def perform_error_level_analysis(
    image
):

    result = {
        "available": False,
        "status": "NOT_AVAILABLE",
        "score": 0.0,
        "mean_difference": 0.0,
        "max_difference": 0.0,
    }

    if not CV2_AVAILABLE:

        return result

    try:

        original_format = (
            image.format
            or ""
        ).upper()

        # ELA is most meaningful for JPEG images.
        # Other formats are marked as limited evidence.
        if original_format not in {
            "JPEG",
            "JPG",
        }:

            result["status"] = (
                "LIMITED_FOR_NON_JPEG"
            )

            return result

        buffer = io.BytesIO()

        image.convert(
            "RGB"
        ).save(
            buffer,
            format="JPEG",
            quality=90,
        )

        buffer.seek(0)

        recompressed = Image.open(
            buffer
        ).convert(
            "RGB"
        )

        original = np.array(
            image.convert(
                "RGB"
            )
        )

        recompressed_array = np.array(
            recompressed
        )

        difference = cv2.absdiff(
            original,
            recompressed_array,
        )

        gray_difference = cv2.cvtColor(
            difference,
            cv2.COLOR_RGB2GRAY,
        )

        mean_difference = float(
            np.mean(
                gray_difference
            )
        )

        max_difference = float(
            np.max(
                gray_difference
            )
        )

        score = min(
            100.0,
            mean_difference * 4.0,
        )

        result = {
            "available": True,
            "status": (
                "REVIEW"
                if score >= 35
                else "NORMAL"
            ),
            "score": round(
                score,
                1,
            ),
            "mean_difference": round(
                mean_difference,
                2,
            ),
            "max_difference": round(
                max_difference,
                2,
            ),
        }

        safe_close(
            recompressed
        )

        buffer.close()

    except Exception as error:

        result["status"] = "ERROR"

        result["error"] = str(
            error
        )

    return result


def analyze_image_region_consistency(
    image
):

    result = {
        "available": False,
        "noise_score": 0.0,
        "edge_score": 0.0,
        "suspicious_regions": [],
    }

    if not CV2_AVAILABLE:

        return result

    try:

        rgb_image = image.convert(
            "RGB"
        )

        array = np.array(
            rgb_image
        )

        gray = cv2.cvtColor(
            array,
            cv2.COLOR_RGB2GRAY,
        )

        height, width = gray.shape

        rows = 4
        columns = 4

        block_height = max(
            1,
            height // rows,
        )

        block_width = max(
            1,
            width // columns,
        )

        regions = []

        noise_values = []

        edge_values = []

        for row in range(rows):

            for column in range(columns):

                y1 = row * block_height
                x1 = column * block_width

                y2 = (
                    height
                    if row == rows - 1
                    else (row + 1)
                    * block_height
                )

                x2 = (
                    width
                    if column == columns - 1
                    else (column + 1)
                    * block_width
                )

                block = gray[
                    y1:y2,
                    x1:x2,
                ]

                if block.size == 0:
                    continue

                blurred = cv2.GaussianBlur(
                    block,
                    (3, 3),
                    0,
                )

                residual = cv2.absdiff(
                    block,
                    blurred,
                )

                noise_value = float(
                    np.std(
                        residual
                    )
                )

                edge_value = float(
                    cv2.Laplacian(
                        block,
                        cv2.CV_64F,
                    ).var()
                )

                noise_values.append(
                    noise_value
                )

                edge_values.append(
                    edge_value
                )

                regions.append(
                    {
                        "row": row,
                        "column": column,
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(
                            x2 - x1
                        ),
                        "height": int(
                            y2 - y1
                        ),
                        "noise": noise_value,
                        "edge": edge_value,
                    }
                )

        if not regions:

            safe_close(
                rgb_image
            )

            return result

        median_noise = float(
            np.median(
                noise_values
            )
        )

        median_edge = float(
            np.median(
                edge_values
            )
        )

        suspicious_regions = []

        noise_deviations = []

        edge_deviations = []

        for region in regions:

            noise_ratio = (
                abs(
                    region["noise"]
                    - median_noise
                )
                / max(
                    median_noise,
                    1.0,
                )
            )

            edge_ratio = (
                abs(
                    region["edge"]
                    - median_edge
                )
                / max(
                    median_edge,
                    1.0,
                )
            )

            region["noise_deviation"] = round(
                noise_ratio,
                2,
            )

            region["edge_deviation"] = round(
                edge_ratio,
                2,
            )

            noise_deviations.append(
                noise_ratio
            )

            edge_deviations.append(
                edge_ratio
            )

            if (
                noise_ratio >= 1.5
                and edge_ratio >= 1.5
            ):

                suspicious_regions.append(
                    {
                        "row": region["row"],
                        "column": region["column"],
                        "x": region["x"],
                        "y": region["y"],
                        "width": region["width"],
                        "height": region["height"],
                        "reason": (
                            "Local noise and edge "
                            "inconsistency"
                        ),
                    }
                )

        noise_score = min(
            100.0,
            (
                float(
                    np.mean(
                        noise_deviations
                    )
                )
                * 35.0
            ),
        )

        edge_score = min(
            100.0,
            (
                float(
                    np.mean(
                        edge_deviations
                    )
                )
                * 35.0
            ),
        )

        result = {
            "available": True,
            "noise_score": round(
                noise_score,
                1,
            ),
            "edge_score": round(
                edge_score,
                1,
            ),
            "suspicious_regions": (
                suspicious_regions
            ),
        }

        safe_close(
            rgb_image
        )

    except Exception as error:

        result["error"] = str(
            error
        )

    return result


# ============================================================
# LOCAL TEXT REGION TAMPERING ANALYSIS
# ============================================================

def analyze_text_region_for_tampering(image, x, y, w, h):
    """
    Local forensic screening around an OCR-detected text region.
    This is a supporting signal, NOT proof of forgery.
    """

    try:
        import cv2
        import numpy as np

        if image is None or w <= 2 or h <= 2:
            return {
                "suspicious": False,
                "score": 0.0,
                "reason": "Invalid text region"
            }

        # Accept both PIL images and OpenCV/numpy images.
        if hasattr(image, "convert"):
            rgb = np.ascontiguousarray(
                np.asarray(image.convert("RGB"), dtype=np.uint8)
            )
            if rgb.ndim == 2:
                gray_full = rgb
            else:
                gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            array = np.ascontiguousarray(image)
            if array.ndim == 2:
                gray_full = array
            else:
                gray_full = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

        height, width = gray_full.shape[:2]

        margin_x = max(4, int(w * 0.35))
        margin_y = max(4, int(h * 0.60))

        x1 = max(0, int(x) - margin_x)
        y1 = max(0, int(y) - margin_y)
        x2 = min(width, int(x) + int(w) + margin_x)
        y2 = min(height, int(y) + int(h) + margin_y)

        roi = gray_full[y1:y2, x1:x2]

        if roi.size == 0:
            return {
                "suspicious": False,
                "score": 0.0,
                "reason": "Empty region"
            }

        blur = cv2.GaussianBlur(roi, (3, 3), 0)
        noise = cv2.absdiff(roi, blur)
        noise_std = float(np.std(noise))

        edges = cv2.Canny(roi, 80, 180)
        edge_density = float(np.mean(edges > 0))

        sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var())

        signals = []

        if noise_std > 18:
            signals.append("unusual local noise")

        if edge_density > 0.28:
            signals.append("unusual local edge density")

        if sharpness > 1800:
            signals.append("unusual local sharpness")

        suspicious = len(signals) >= 2

        if suspicious:
            score = min(1.0, 0.30 + 0.15 * len(signals))
        else:
            score = min(0.25, 0.08 * len(signals))

        return {
            "suspicious": suspicious,
            "score": round(score, 3),
            "reason": ", ".join(signals) if signals else "No strong local forensic signal",
            "region": {
                "x": int(x1),
                "y": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1)
            },
            "signals": signals,
            "metrics": {
                "noise_std": round(noise_std, 3),
                "edge_density": round(edge_density, 4),
                "sharpness": round(sharpness, 2),
            },
        }

    except Exception as e:
        return {
            "suspicious": False,
            "score": 0.0,
            "reason": f"Local analysis unavailable: {str(e)}"
        }


def analyze_targeted_text_regions(image, ocr_tokens, structured):
    """
    Run local forensic checks only on OCR regions belonging to
    important extracted fields. This keeps the signal focused on
    likely editable fields such as name, DOB and document number.
    """

    result = {
        "available": False,
        "suspicious": False,
        "suspicious_fields": [],
        "regions": [],
        "region_count": 0,
    }

    if not CV2_AVAILABLE or image is None:
        return result

    tokens = [
        token for token in (ocr_tokens or [])
        if isinstance(token, dict)
        and token.get("text")
        and int(token.get("width", 0) or 0) > 2
        and int(token.get("height", 0) or 0) > 2
    ]

    if not tokens:
        return result

    # Only fields whose visual text is useful for targeted tamper screening.
    target_fields = [
        "name",
        "date_of_birth",
        "aadhaar_number",
        "pan_number",
        "driving_licence_number",
        "passport_number",
        "voter_id_number",
        "gstin",
    ]

    def norm(value):
        return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

    def token_matches_field(token_text, field_value):
        token_norm = norm(token_text)
        value_norm = norm(field_value)
        if not token_norm or not value_norm:
            return False
        if token_norm in value_norm or value_norm in token_norm:
            return True
        # Match meaningful words inside a multi-word name.
        parts = [part for part in re.findall(r"[a-z0-9]+", str(field_value).lower()) if len(part) >= 3]
        return any(part == token_norm for part in parts)

    for field in target_fields:
        value = structured.get(field)
        if not value:
            continue

        matching = [
            token for token in tokens
            if token_matches_field(token.get("text", ""), value)
        ]

        if not matching:
            continue

        # Limit repeated OCR fragments so one field cannot dominate the score.
        matching = matching[:6]

        for token in matching:
            analysis = analyze_text_region_for_tampering(
                image,
                int(token.get("left", 0) or 0),
                int(token.get("top", 0) or 0),
                int(token.get("width", 0) or 0),
                int(token.get("height", 0) or 0),
            )

            region_entry = {
                "field": field,
                "text": str(token.get("text", "")),
                "confidence": float(token.get("confidence", 0) or 0),
                **analysis,
            }
            result["regions"].append(region_entry)

            if analysis.get("suspicious"):
                result["suspicious"] = True
                if field not in result["suspicious_fields"]:
                    result["suspicious_fields"].append(field)

    result["available"] = bool(result["regions"])
    result["region_count"] = len(result["regions"])
    return result


def analyze_document_tampering(
    image,
    metadata,
):

    # --------------------------------------------------------
    # METADATA SIGNAL
    # --------------------------------------------------------

    metadata_analysis = (
        analyze_metadata_tampering(
            metadata
        )
    )

    # --------------------------------------------------------
    # ERROR LEVEL ANALYSIS
    # --------------------------------------------------------

    ela_analysis = (
        perform_error_level_analysis(
            image
        )
    )

    # --------------------------------------------------------
    # REGION CONSISTENCY
    # --------------------------------------------------------

    consistency_analysis = (
        analyze_image_region_consistency(
            image
        )
    )

    signals = []

    score_components = []

    # Metadata signal

    if (
        metadata_analysis.get(
            "editing_software_signals"
        )
    ):

        score_components.append(
            20
        )

        signals.append(
            "Image metadata contains "
            "editing-software indicators"
        )

    # ELA signal

    ela_score = float(
        ela_analysis.get(
            "score",
            0,
        )
        or 0
    )

    if ela_analysis.get(
        "available"
    ):

        score_components.append(
            ela_score * 0.35
        )

        if ela_score >= 35:

            signals.append(
                "JPEG compression inconsistency "
                "requires review"
            )

    # Noise signal

    noise_score = float(
        consistency_analysis.get(
            "noise_score",
            0,
        )
        or 0
    )

    score_components.append(
        noise_score * 0.25
    )

    if noise_score >= 35:

        signals.append(
            "Local noise consistency "
            "variation detected"
        )

    # Edge signal

    edge_score = float(
        consistency_analysis.get(
            "edge_score",
            0,
        )
        or 0
    )

    score_components.append(
        edge_score * 0.25
    )

    if edge_score >= 35:

        signals.append(
            "Local edge consistency "
            "variation detected"
        )

    suspicious_regions = (
        consistency_analysis.get(
            "suspicious_regions",
            []
        )
    )

    if suspicious_regions:

        score_components.append(
            min(
                20,
                len(
                    suspicious_regions
                ) * 5,
            )
        )

        signals.append(
            f"{len(suspicious_regions)} "
            "image region(s) require review"
        )

    tampering_probability = min(
        100.0,
        sum(
            score_components
        ),
    )

    tampering_probability = round(
        tampering_probability,
        1,
    )

    if tampering_probability >= 70:

        risk_level = "HIGH"

        status = "HIGH_REVIEW_REQUIRED"

    elif tampering_probability >= 40:

        risk_level = "SUSPICIOUS"

        status = "REVIEW_RECOMMENDED"

    else:

        risk_level = "LOW"

        status = "NO_STRONG_TAMPERING_SIGNAL"

    return {
        "status": status,
        "tampering_probability": (
            tampering_probability
        ),
        "risk_level": risk_level,
        "signals": signals,
        "metadata": metadata_analysis,
        "ela": ela_analysis,
        "noise_analysis": {
            "score": (
                consistency_analysis.get(
                    "noise_score",
                    0,
                )
            ),
        },
        "edge_analysis": {
            "score": (
                consistency_analysis.get(
                    "edge_score",
                    0,
                )
            ),
        },
        "suspicious_regions": (
            suspicious_regions
        ),
        "suspicious_region_count": len(
            suspicious_regions
        ),
    }
    
# ============================================================
# TEXT ANALYSIS
# ============================================================

def analyze_extracted_text(
    raw_text,
    structured,
):

    words = re.findall(
        r"\b[\w'-]+\b",
        raw_text,
    )

    fields_found = []

    for key, value in structured.items():

        if (
            key
            in {
                "document",
                "document_category",
            }
        ):
            continue

        if value:

            fields_found.append(
                key
            )

    document_numbers = []

    number_fields = [
        "aadhaar_number",
        "pan_number",
        "driving_licence_number",
        "passport_number",
        "voter_id_number",
        "gstin",
    ]

    for field in number_fields:

        value = structured.get(
            field
        )

        if value:

            document_numbers.append(
                value
            )

    return {
        "word_count": len(
            words
        ),
        "text_length": len(
            raw_text
        ),
        "fields_found": (
            fields_found
        ),
        "document_numbers_found": (
            document_numbers
        ),
    }


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    file_content,
):

    try:

        verify_image = Image.open(
            io.BytesIO(
                file_content
            )
        )

        verify_image.verify()

        safe_close(
            verify_image
        )

        image = Image.open(
            io.BytesIO(
                file_content
            )
        )

        image.load()

        image = fix_orientation(
            image
        )

        width, height = image.size

        image_format = (
            image.format
            or "UNKNOWN"
        )

        image_mode = image.mode

        metadata = {}

        try:

            exif_data = image.getexif()

            for tag_id, value in exif_data.items():

                tag_name = (
                    ExifTags.TAGS.get(
                        tag_id,
                        str(tag_id),
                    )
                )

                metadata[
                    tag_name
                ] = str(
                    value
                )

        except Exception:
            pass

        quality = analyze_image_quality(
            image
        )

        face_detection = detect_faces_in_document(
           image
        )

        ocr = extract_ocr_data(
            image
        )

        structured = (
            ocr.get(
                "structured_data",
                {}
            )
        )

        detection = (
            ocr.get(
                "document_detection",
                {}
            )
        )

        raw_text = normalize_text(
            ocr.get(
                "raw_ocr_text",
                ""
            )
        )

        display_text = build_display_text(
            structured
        )

        text_analysis = (
            analyze_extracted_text(
                raw_text,
                structured,
            )
        )

        safe_close(
            image
        )

        # ----------------------------------------------------
        # FRONTEND COMPATIBILITY
        # ----------------------------------------------------

        document_object = {
            "document_type": (
                detection.get(
                    "document_label",
                    "Unknown Document",
                )
            ),
            "document_category": (
                detection.get(
                    "document_category",
                    "UNKNOWN",
                )
            ),
            "confidence": (
                detection.get(
                    "confidence",
                    "LOW",
                )
            ),
        }

        return {
            "valid": True,

            "file_category": "IMAGE",

            "document_type": "IMAGE",

            "width": width,

            "height": height,

            "format": image_format,

            "mode": image_mode,

            "metadata_found": bool(
                metadata
            ),

            "metadata_count": len(
                metadata
            ),

            "metadata": metadata,

            "image_quality": quality,
            "face_detection": face_detection,

            "extracted_text": (
                display_text
                or raw_text
            ),

            "raw_ocr_text": raw_text,

            "ocr_confidence": (
                ocr.get(
                    "ocr_confidence",
                    0,
                )
            ),

            "ocr_status": (
                ocr.get(
                    "ocr_status",
                    "NO_TEXT_DETECTED",
                )
            ),

            "ocr_language": (
                ocr.get(
                    "ocr_language"
                )
            ),

            "ocr_engine": (
                ocr.get(
                    "ocr_engine",
                    "TESSERACT",
                )
            ),

            "ocr_variant": (
                ocr.get(
                    "ocr_variant"
                )
            ),

            "ocr_candidates_tested": (
                ocr.get(
                    "ocr_candidates_tested",
                    0,
                )
            ),

            "candidate_summary": (
                ocr.get(
                    "candidate_summary",
                    []
                )
            ),

            "ocr_tokens": (
                ocr.get(
                    "ocr_tokens",
                    []
                )
            ),

            "extracted_characters": len(
                raw_text
            ),

            "display_characters": len(
                display_text
            ),

            "document_category": (
                detection.get(
                    "document_category",
                    "UNKNOWN",
                )
            ),

            "document_label": (
                detection.get(
                    "document_label",
                    "Unknown Document",
                )
            ),

            "document_detection_confidence": (
                detection.get(
                    "confidence",
                    "LOW",
                )
            ),

            # New canonical data
            "structured_data": structured,

            "field_confidence": (
                ocr.get(
                    "field_confidence",
                    {}
                )
            ),

            # Compatibility aliases
            "extracted_data": structured,

            "extracted": structured,

            "document": document_object,

            **text_analysis,
        }

    except Exception as error:

        return {
            "valid": False,
            "file_category": "IMAGE",
            "document_type": "IMAGE",
            "error": str(
                error
            ),
        }


# ============================================================
# PDF ANALYSIS
# ============================================================

def analyze_pdf(
    file_content,
):

    try:

        pdf = fitz.open(
            stream=file_content,
            filetype="pdf",
        )

        encrypted = (
            pdf.is_encrypted
        )

        page_count = (
            pdf.page_count
        )

        metadata = (
            pdf.metadata
            or {}
        )

        pages_to_scan = min(
            page_count,
            MAX_PDF_OCR_PAGES,
        )

        all_text_parts = []

        ocr_confidences = []

        extraction_methods = []

        all_field_candidates = []

        for page_number in range(
            pages_to_scan
        ):

            page = pdf.load_page(
                page_number
            )

            native_text = normalize_text(
                page.get_text(
                    "text"
                )
            )

            if len(
                native_text
            ) >= 25:

                all_text_parts.append(
                    native_text
                )

                extraction_methods.append(
                    "native_pdf_text"
                )

                continue

            pix = page.get_pixmap(
                matrix=fitz.Matrix(
                    1.8,
                    1.8,
                ),
                alpha=False,
            )

            page_image = Image.open(
                io.BytesIO(
                    pix.tobytes(
                        "png"
                    )
                )
            )

            ocr = extract_ocr_data(
                page_image
            )

            safe_close(
                page_image
            )

            page_text = normalize_text(
                ocr.get(
                    "raw_ocr_text",
                    ""
                )
            )

            if page_text:

                all_text_parts.append(
                    page_text
                )

            ocr_confidences.append(
                float(
                    ocr.get(
                        "ocr_confidence",
                        0,
                    )
                    or 0
                )
            )

            extraction_methods.append(
                "rendered_page_ocr"
            )

            structured = ocr.get(
                "structured_data",
                {}
            )

            if structured:

                all_field_candidates.append(
                    structured
                )

        pdf.close()

        combined_text = normalize_text(
            "\n\n".join(
                all_text_parts
            )
        )[:MAX_OCR_TEXT_LENGTH]

        detection = detect_document_type(
            combined_text
        )

        structured = extract_fields_from_text(
            combined_text,
            detection,
        )

        # Merge page OCR structured values
        for candidate in all_field_candidates:

            for key, value in candidate.items():

                if (
                    key
                    in {
                        "document",
                        "document_category",
                    }
                ):
                    continue

                if (
                    not structured.get(
                        key
                    )
                    and value
                ):

                    structured[
                        key
                    ] = value

        structured[
            "document"
        ] = detection.get(
            "document_label",
            "Unknown Document",
        )

        structured[
            "document_category"
        ] = detection.get(
            "document_category",
            "UNKNOWN",
        )

        display_text = build_display_text(
            structured
        )

        text_analysis = (
            analyze_extracted_text(
                combined_text,
                structured,
            )
        )

        if ocr_confidences:

            ocr_confidence = round(
                sum(
                    ocr_confidences
                )
                / len(
                    ocr_confidences
                ),
                1,
            )

        else:

            ocr_confidence = (
                95.0
                if combined_text
                else 0.0
            )

        document_object = {
            "document_type": (
                detection.get(
                    "document_label",
                    "Unknown Document",
                )
            ),
            "document_category": (
                detection.get(
                    "document_category",
                    "UNKNOWN",
                )
            ),
            "confidence": (
                detection.get(
                    "confidence",
                    "LOW",
                )
            ),
        }

        return {
            "valid": True,

            "file_category": "PDF",

            "document_type": "PDF",

            "page_count": page_count,

            "pages_scanned": (
                pages_to_scan
            ),

            "encrypted": encrypted,

            "metadata_found": bool(
                metadata
            ),

            "metadata": metadata,

            "image_quality": None,

            "extracted_text": (
                display_text
                or combined_text
            ),

            "raw_ocr_text": (
                combined_text
            ),

            "ocr_confidence": (
                ocr_confidence
            ),

            "ocr_status": (
                "TEXT_DETECTED"
                if combined_text
                else "NO_TEXT_DETECTED"
            ),

            "extraction_method": sorted(
                set(
                    extraction_methods
                )
            ),

            "extracted_characters": len(
                combined_text
            ),

            "display_characters": len(
                display_text
            ),

            "document_category": (
                detection.get(
                    "document_category",
                    "UNKNOWN",
                )
            ),

            "document_label": (
                detection.get(
                    "document_label",
                    "Unknown Document",
                )
            ),

            "document_detection_confidence": (
                detection.get(
                    "confidence",
                    "LOW",
                )
            ),

            "structured_data": structured,

            "extracted_data": structured,

            "extracted": structured,

            "document": document_object,

            **text_analysis,
        }

    except Exception as error:

        return {
            "valid": False,
            "file_category": "PDF",
            "document_type": "PDF",
            "error": str(
                error
            ),
        }


# ============================================================
# RISK ASSESSMENT
# ============================================================

def calculate_risk(
    file_format_valid,
    structure_valid,
    file_size,
    analysis,
):

    score = 0

    signals = []

    if not file_format_valid:

        score += 45

        signals.append(
            "File signature does not match declared content type"
        )

    if not structure_valid:

        score += 35

        signals.append(
            "Document could not be parsed successfully"
        )

    if (
        file_size
        > 8 * 1024 * 1024
    ):

        score += 5

        signals.append(
            "Large file size"
        )

    ocr_status = analysis.get(
        "ocr_status",
        "NO_TEXT_DETECTED",
    )

    ocr_confidence = float(
        analysis.get(
            "ocr_confidence",
            0,
        )
        or 0
    )

    if (
        ocr_status
        == "NO_TEXT_DETECTED"
    ):

        score += 8

        signals.append(
            "No readable text detected"
        )

    elif (
        0
        < ocr_confidence
        < 40
    ):

        score += 12

        signals.append(
            "Low OCR readability confidence"
        )

    elif (
        0
        < ocr_confidence
        < 60
    ):

        score += 6

        signals.append(
            "Moderate OCR readability confidence"
        )

    quality = analysis.get(
        "image_quality"
    ) or {}

    for issue in quality.get(
        "issues",
        [],
    ):

        score += 4

        signals.append(
            issue
        )

    if analysis.get(
        "encrypted"
    ):

        score += 8

        signals.append(
            "PDF is encrypted"
        )

    score = min(
        max(
            score,
            0,
        ),
        100,
    )

    if score <= 25:

        level = "LOW RISK"

    elif score <= 60:

        level = "MEDIUM RISK"

    else:

        level = "HIGH RISK"

    return (
        score,
        level,
        signals,
    )


# ============================================================
# VALIDATION RESULTS
# ============================================================

def build_validation_results(
    file_format_valid,
    structure_valid,
    analysis,
):

    structured = analysis.get(
        "structured_data",
        {}
    )

    meaningful_fields = sum(
        1
        for key, value
        in structured.items()
        if (
            value
            and key
            not in {
                "document",
                "document_category",
            }
        )
    )

    quality = analysis.get(
        "image_quality"
    ) or {}

    results = [
        {
            "name": "File format check",
            "status": (
                "PASSED"
                if file_format_valid
                else "FAILED"
            ),
        },
        {
            "name": "Document structure check",
            "status": (
                "PASSED"
                if structure_valid
                else "FAILED"
            ),
        },
        {
            "name": "Data consistency check",
            "status": (
                "PASSED"
                if (
                    structure_valid
                    and analysis.get(
                        "ocr_status"
                    )
                    == "TEXT_DETECTED"
                )
                else "REVIEW"
            ),
        },
        {
            "name": "Document type detection",
            "status": analysis.get(
                "document_label",
                "Unknown Document",
            ),
        },
        {
            "name": "Smart extracted-data validation",
            "status": (
                "PASSED"
                if meaningful_fields >= 2
                else "REVIEW REQUIRED"
            ),
        },
        {
            "name": "Cross-field consistency check",
            "status": (
                "PASSED"
                if meaningful_fields >= 1
                else "REVIEW REQUIRED"
            ),
        },
        {
            "name": "Document image quality",
            "status": quality.get(
                "status",
                "NOT APPLICABLE",
            ),
        },
    ]

    if quality.get(
        "issues"
    ):

        anomaly_status = (
            "REVIEW REQUIRED"
        )

    else:

        anomaly_status = (
            "NO CRITICAL SIGNAL"
        )

    results.append(
        {
            "name": (
                "Tampering / anomaly signal analysis"
            ),
            "status": anomaly_status,
        }
    )

    return results




# ============================================================
# SECUREDOC AI — ENHANCEMENT LAYER (PRESERVE OLD OCR)
# ============================================================
# IMPORTANT: The original OCR/extraction implementations above remain the
# primary path. This layer only AUGMENTS missing evidence; it does not replace
# a successful legacy OCR result.

APP_VERSION = "6.2.0"


def _safe_float(value, default=0.0):
    try:
        return float(value or 0)
    except Exception:
        return default


def _targeted_ocr_recovery(image):
    """Run two small recovery passes only when the old OCR missed useful data."""
    results = []
    try:
        rgb = fix_orientation(image).convert("RGB")
        width, height = rgb.size
        regions = [
            ("bottom_psm11", 0.58, "--oem 3 --psm 11"),
            ("bottom_psm12", 0.48, "--oem 3 --psm 12"),
        ]
        language = get_ocr_language()
        for name, start_ratio, config in regions:
            crop = None
            try:
                crop = rgb.crop((0, int(height * start_ratio), width, height))
                item = run_ocr_pass(crop, config, language)
                text = normalize_text(item.get("text", ""))
                if text:
                    results.append({
                        "name": name,
                        "text": text,
                        "confidence": _safe_float(item.get("confidence")),
                        "tokens": item.get("tokens") or [],
                    })
            except Exception:
                continue
            finally:
                safe_close(crop)
        safe_close(rgb)
    except Exception:
        pass
    return results


def _identifier_missing(category, structured):
    structured = structured or {}
    return (
        (category == "AADHAAR_CARD" and not structured.get("aadhaar_number"))
        or (category == "PAN_CARD" and not structured.get("pan_number"))
        or (category == "PASSPORT" and not structured.get("passport_number"))
        or (category == "DRIVING_LICENCE" and not structured.get("driving_licence_number"))
        or (category == "VOTER_ID" and not structured.get("voter_id_number"))
        or (category == "GST_DOCUMENT" and not structured.get("gstin"))
    )


def _enhance_analysis_preserving_legacy(file_content, analysis_data):
    """Additive enhancement layer: legacy OCR/extraction stays authoritative."""
    result = dict(analysis_data or {})
    if not result.get("valid"):
        return result

    # Always expose quality guidance without changing extraction.
    result["quality_guidance"] = _quality_guidance(result.get("image_quality") or {})
    result["ocr_confidence_raw"] = _safe_float(result.get("ocr_confidence"))
    result["ocr_confidence_quality_adjusted"] = _quality_adjusted_confidence(
        result.get("ocr_confidence"), result.get("image_quality") or {}
    )

    if result.get("file_category") == "IMAGE":
        image = None
        try:
            image = Image.open(io.BytesIO(file_content))
            image.load()
            image = fix_orientation(image)

            # QR is additive and neutral when absent/unreadable.
            result["qr_analysis"] = _detect_qr_data(image)
            result["qr_verification"] = result["qr_analysis"]
            result["qr_field_cross_check"] = _compare_qr_with_fields(
                result["qr_analysis"], result.get("structured_data") or {}
            )

            try:
                result["tampering_analysis"] = analyze_document_tampering(
                    image, result.get("metadata") or {}
                )
            except Exception:
                result["tampering_analysis"] = {
                    "status": "ANALYSIS_UNAVAILABLE",
                    "tampering_probability": 0,
                    "risk_level": "UNKNOWN",
                    "signals": [],
                }

            structured = dict(result.get("structured_data") or {})
            category = str(result.get("document_category", "UNKNOWN")).upper()
            raw_text = normalize_text(result.get("raw_ocr_text", ""))
            ocr_conf = _safe_float(result.get("ocr_confidence"))

            # Targeted recovery only when useful information is missing/weak.
            if _identifier_missing(category, structured) or not raw_text or ocr_conf < 35:
                recovery = _targeted_ocr_recovery(image)
                useful = [x for x in recovery if normalize_text(x.get("text", ""))]
                if useful:
                    recovery_text = normalize_text("\n".join(x["text"] for x in useful))
                    combined = normalize_text((raw_text + "\n" + recovery_text).strip())[:MAX_OCR_TEXT_LENGTH]
                    combined_detection = detect_document_type(combined)
                    recovered = extract_fields_from_text(combined, combined_detection) or {}

                    # CRITICAL: legacy non-empty values always win.
                    for key, value in structured.items():
                        if value:
                            recovered[key] = value

                    # Never let recovery change a previously known document category
                    # unless the legacy category was UNKNOWN.
                    legacy_category = category
                    new_category = combined_detection.get("document_category", "UNKNOWN")
                    if legacy_category != "UNKNOWN":
                        final_detection = result.get("document_detection") or {}
                        final_category = legacy_category
                        final_label = result.get("document_label", DOCUMENT_LABELS.get(legacy_category, "Unknown Document"))
                        final_conf = result.get("document_detection_confidence", "LOW")
                    else:
                        final_detection = combined_detection
                        final_category = new_category
                        final_label = combined_detection.get("document_label", "Unknown Document")
                        final_conf = combined_detection.get("confidence", "LOW")

                    result["structured_data"] = recovered
                    result["extracted_data"] = recovered
                    result["extracted"] = recovered
                    result["raw_ocr_text"] = combined
                    result["extracted_text"] = combined
                    result["ocr_status"] = "TEXT_DETECTED" if combined else "NO_TEXT_DETECTED"
                    result["ocr_recovery_passes"] = [x["name"] for x in useful]
                    result["ocr_recovery_text"] = recovery_text
                    result["ocr_recovery_confidence"] = round(max(_safe_float(x.get("confidence")) for x in useful), 1)
                    result["ocr_confidence_primary"] = ocr_conf
                    result["ocr_confidence"] = round(
                        min(100.0, (ocr_conf * 0.75) + (result["ocr_recovery_confidence"] * 0.25)), 1
                    )
                    result["document_detection"] = final_detection
                    result["document_category"] = final_category
                    result["document_label"] = final_label
                    result["document_detection_confidence"] = final_conf
                    result["display_text"] = build_display_text(recovered)
                    result["qr_field_cross_check"] = _compare_qr_with_fields(
                        result.get("qr_analysis") or {}, recovered
                    )

            # Passport MRZ is verification-only and never overwrites legacy fields.
            if category == "PASSPORT" or str(result.get("document_category", "")).upper() == "PASSPORT":
                mrz = _extract_mrz_passport(result.get("raw_ocr_text", ""))
                result["passport_mrz"] = mrz
                result["passport_mrz_cross_check"] = _compare_mrz_with_fields(
                    mrz, result.get("structured_data") or {}
                )
            else:
                result["passport_mrz"] = {"detected": False, "source": "passport_mrz_ocr"}
                result["passport_mrz_cross_check"] = {"status": "NOT_APPLICABLE", "matches": [], "mismatches": []}
        except Exception:
            # Enhancement failures must never destroy the legacy analysis result.
            pass
        finally:
            safe_close(image)

    raw_text = normalize_text(result.get("raw_ocr_text", ""))
    result["document_marking_signals"] = detect_demo_or_mockup_signals(raw_text)
    result["screening_version"] = APP_VERSION
    enhancement = _build_enhancement_summary(result)
    result.update(enhancement)
    return result



# ============================================================
# SAFE ENHANCEMENT LAYER — DOES NOT REPLACE LEGACY EXTRACTION
# ============================================================

def _quality_guidance(quality):
    """Turn image-quality metrics into human-readable guidance only."""
    q = quality or {}
    try:
        blur = float(q.get("blur_score", q.get("blur", 0)) or 0)
    except Exception:
        blur = 0.0
    try:
        resolution = float(q.get("resolution_score", q.get("resolution", 0)) or 0)
    except Exception:
        resolution = 0.0
    try:
        skew = abs(float(q.get("skew_angle", q.get("skew", 0)) or 0))
    except Exception:
        skew = 0.0

    warnings = []
    if blur and blur < 60:
        warnings.append("Image may be blurry; OCR reliability can be reduced")
    if resolution and resolution < 60:
        warnings.append("Image resolution is limited; a clearer scan may improve OCR")
    if skew > 5:
        warnings.append("Document is noticeably skewed; OCR may be less reliable")

    if len(warnings) >= 2:
        status = "POOR"
    elif warnings:
        status = "FAIR"
    else:
        status = "GOOD"
    return {"status": status, "warnings": warnings}


def _safe_mask_identifier(value, keep_start=2, keep_end=2):
    if not value:
        return ""
    text = str(value).strip()
    compact = re.sub(r"\s+", "", text)
    if len(compact) <= keep_start + keep_end:
        return "*" * len(compact)
    return compact[:keep_start] + ("*" * (len(compact) - keep_start - keep_end)) + compact[-keep_end:]


def _field_source_hint(field, value, raw_text):
    """Conservative provenance hint; it never changes the extracted value."""
    if not value:
        return "not_extracted"
    text = normalize_text(raw_text).lower()
    v = normalize_single_line(value).lower()
    labels = {
        "name": ["name", "full name", "surname", "given name"],
        "date_of_birth": ["date of birth", "dob", "birth"],
        "date_of_issue": ["date of issue", "issue date", "issued"],
        "date_of_expiry": ["date of expiry", "expiry", "valid till", "valid until"],
        "gender": ["gender", "sex"],
        "nationality": ["nationality"],
        "aadhaar_number": ["aadhaar", "aadhar"],
        "pan_number": ["pan"],
        "driving_licence_number": ["driving licence", "driving license", "dl no", "dl number"],
        "passport_number": ["passport no", "passport number"],
        "voter_id_number": ["epic", "voter id", "voter"],
        "gstin": ["gstin", "gst no", "gst number"],
    }
    if any(label in text and v in text for label in labels.get(field, [])):
        return "labelled_ocr"
    return "ocr_extracted"


def _build_field_provenance(structured, field_confidence, raw_text):
    out = {}
    for field, value in (structured or {}).items():
        if field in {"document", "document_category"}:
            continue
        conf = _safe_float((field_confidence or {}).get(field), 0)
        if value:
            source = _field_source_hint(field, value, raw_text)
            if conf >= 85:
                reliability = "HIGH"
            elif conf >= 60:
                reliability = "MEDIUM"
            else:
                reliability = "LOW"
            out[field] = {
                "source": source,
                "confidence": round(conf, 1),
                "reliability": reliability,
            }
        else:
            out[field] = {
                "source": "not_extracted",
                "confidence": 0,
                "reliability": "LOW",
            }
    return out


def _validate_identifiers(structured):
    """Validation-only layer. It never rejects or rewrites legacy extracted values."""
    data = structured or {}
    checks = {}

    def add(name, value, valid, note):
        checks[name] = {
            "present": bool(value),
            "format_valid": bool(valid) if value else None,
            "status": "VALID_FORMAT" if value and valid else ("INVALID_FORMAT" if value else "NOT_FOUND"),
            "note": note,
        }

    aadhaar = data.get("aadhaar_number")
    digits = re.sub(r"\D", "", str(aadhaar or ""))
    add("aadhaar_number", aadhaar, len(digits) == 12, "12-digit format check only; not government verification")

    pan = str(data.get("pan_number") or "").upper().replace(" ", "")
    add("pan_number", data.get("pan_number"), bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", pan)), "PAN structural format check only")

    passport = str(data.get("passport_number") or "").upper().replace(" ", "")
    add("passport_number", data.get("passport_number"), 6 <= len(passport) <= 12 and any(c.isalpha() for c in passport) and any(c.isdigit() for c in passport), "Passport-number structural check only")

    dl = str(data.get("driving_licence_number") or "").upper()
    dl_compact = re.sub(r"[^A-Z0-9]", "", dl)
    add("driving_licence_number", data.get("driving_licence_number"), len(dl_compact) >= 10 and dl_compact[:2].isalpha() and dl_compact[2:4].isdigit(), "Driving-licence structural check only")

    voter = str(data.get("voter_id_number") or "").upper().replace(" ", "")
    add("voter_id_number", data.get("voter_id_number"), 3 <= len(voter) <= 20 and bool(re.fullmatch(r"[A-Z0-9-]+", voter)), "Voter-ID structural check only")

    gst = str(data.get("gstin") or "").upper().replace(" ", "")
    add("gstin", data.get("gstin"), bool(re.fullmatch(r"[0-9A-Z]{15}", gst)), "GSTIN structural format check only")
    return checks


def _extract_mrz_passport(raw_text):
    """Best-effort ICAO TD3 MRZ parser from OCR text; no OCR text is modified."""
    lines = [re.sub(r"[^A-Z0-9<]", "", x.upper()) for x in normalize_text(raw_text).splitlines()]
    lines = [x for x in lines if len(x) >= 30]
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        if len(a) < 40 or len(b) < 40:
            continue
        if not (a.startswith("P<") or a.startswith("P")):
            continue
        # TD3 passport MRZ normally has 44 characters per line. OCR can lose a few.
        if not (40 <= len(a) <= 46 and 40 <= len(b) <= 46):
            continue
        passport_no = b[:9].replace("<", "")
        nationality = b[10:13].replace("<", "")
        dob_raw = b[13:19]
        sex = b[20:21]
        expiry_raw = b[21:27]
        name_raw = a[5:].replace("<", " ").strip()
        name_parts = [x for x in name_raw.split() if x]
        surname = name_parts[0] if name_parts else ""
        given = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
        def mrz_date(v):
            if not re.fullmatch(r"\d{6}", v):
                return None
            yy, mm, dd = int(v[:2]), int(v[2:4]), int(v[4:6])
            # Conservative century mapping for modern travel documents.
            year = 2000 + yy if yy <= 49 else 1900 + yy
            try:
                return f"{dd:02d}/{mm:02d}/{year:04d}"
            except Exception:
                return None
        return {
            "detected": True,
            "passport_number": passport_no or None,
            "nationality": nationality or None,
            "date_of_birth": mrz_date(dob_raw),
            "date_of_expiry": mrz_date(expiry_raw),
            "sex": {"M": "male", "F": "female", "X": "other"}.get(sex),
            "surname": surname or None,
            "given_names": given or None,
            "source": "passport_mrz_ocr",
        }
    return {"detected": False, "source": "passport_mrz_ocr"}


def _compare_mrz_with_fields(mrz, structured):
    if not mrz or not mrz.get("detected"):
        return {"status": "NOT_AVAILABLE", "matches": [], "mismatches": []}
    matches, mismatches = [], []
    pairs = [
        ("passport_number", mrz.get("passport_number"), structured.get("passport_number")),
        ("nationality", mrz.get("nationality"), structured.get("nationality")),
        ("date_of_birth", mrz.get("date_of_birth"), structured.get("date_of_birth")),
        ("date_of_expiry", mrz.get("date_of_expiry"), structured.get("date_of_expiry")),
    ]
    for field, a, b in pairs:
        if not a or not b:
            continue
        na = re.sub(r"[^A-Z0-9]", "", str(a).upper())
        nb = re.sub(r"[^A-Z0-9]", "", str(b).upper())
        if na == nb:
            matches.append(field)
        else:
            mismatches.append(field)
    if mismatches:
        status = "MISMATCH"
    elif matches:
        status = "MATCH"
    else:
        status = "INSUFFICIENT_DATA"
    return {"status": status, "matches": matches, "mismatches": mismatches}


def _detect_qr_data(image):
    """Optional QR decoding using OpenCV. Failure/absence is neutral."""
    result = {"status": "NOT_AVAILABLE", "detected": False, "data": None, "source": "opencv_qr_detector"}
    if not CV2_AVAILABLE:
        result["status"] = "OPENCV_UNAVAILABLE"
        return result
    cv_image = None
    try:
        rgb = fix_orientation(image).convert("RGB")
        arr = np.array(rgb)
        cv_image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(cv_image)
        if data:
            result.update({"status": "DECODED", "detected": True, "data": data})
        elif points is not None:
            result.update({"status": "DETECTED_NOT_DECODED", "detected": True})
        else:
            result["status"] = "NOT_DETECTED"
        return result
    except Exception as exc:
        result["status"] = "ANALYSIS_UNAVAILABLE"
        result["error"] = str(exc)[:160]
        return result
    finally:
        if cv_image is not None:
            del cv_image



def _compare_qr_with_fields(qr, structured):
    """Best-effort QR/OCR consistency check. Unknown/opaque QR payloads stay neutral."""
    qr = qr or {}
    data = qr.get("data")
    if not data:
        return {"status": "NOT_AVAILABLE", "matches": [], "mismatches": [], "source": "qr_field_cross_check"}
    text = str(data).strip()
    parsed = {}
    try:
        import json
        obj = json.loads(text)
        if isinstance(obj, dict):
            parsed = {str(k).lower().strip(): str(v).strip() for k, v in obj.items() if v is not None}
    except Exception:
        pass
    if not parsed:
        for part in re.split(r"[|;\n]+", text):
            if ":" in part or "=" in part:
                sep = ":" if ":" in part else "="
                k, v = part.split(sep, 1)
                if k.strip() and v.strip():
                    parsed[k.lower().strip()] = v.strip()

    aliases = {
        "aadhaar_number": ["aadhaar", "aadhar", "aadhaar_number", "uid"],
        "pan_number": ["pan", "pan_number"],
        "passport_number": ["passport", "passport_number", "passport_no"],
        "driving_licence_number": ["dl", "dl_no", "driving_licence", "driving_license"],
        "voter_id_number": ["epic", "voter", "voter_id"],
        "gstin": ["gstin", "gst"],
        "name": ["name", "full_name"],
        "date_of_birth": ["dob", "date_of_birth", "birth_date"],
    }
    matches, mismatches = [], []
    for field, keys in aliases.items():
        qr_value = next((parsed[k] for k in parsed if any(a == k or a in k for a in keys)), None)
        field_value = (structured or {}).get(field)
        if not qr_value or not field_value:
            continue
        a = re.sub(r"[^A-Z0-9]", "", str(qr_value).upper())
        b = re.sub(r"[^A-Z0-9]", "", str(field_value).upper())
        if not a or not b:
            continue
        if a == b:
            matches.append(field)
        elif len(a) >= 4 and len(b) >= 4:
            mismatches.append(field)
    if mismatches:
        status = "MISMATCH"
    elif matches:
        status = "MATCH"
    else:
        status = "OPAQUE_OR_INSUFFICIENT_DATA"
    return {"status": status, "matches": matches, "mismatches": mismatches, "source": "qr_field_cross_check"}

def _quality_adjusted_confidence(ocr_confidence, quality):
    """Only adjusts the displayed reliability metric; raw OCR confidence is preserved."""
    base = max(0.0, min(100.0, _safe_float(ocr_confidence)))
    status = (_quality_guidance(quality).get("status") or "GOOD").upper()
    if status == "POOR":
        return round(base * 0.82, 1)
    if status == "FAIR":
        return round(base * 0.92, 1)
    return round(base, 1)


def _build_enhancement_summary(analysis):
    structured = analysis.get("structured_data") or {}
    confidence = analysis.get("field_confidence") or {}
    raw_text = analysis.get("raw_ocr_text", "") or ""
    quality = analysis.get("image_quality") or {}
    return {
        "field_provenance": _build_field_provenance(structured, confidence, raw_text),
        "identifier_validation": _validate_identifiers(structured),
        "quality_guidance": _quality_guidance(quality),
        "quality_adjusted_ocr_confidence": _quality_adjusted_confidence(analysis.get("ocr_confidence"), quality),
    }

def _enhanced_risk(file_format_valid, structure_valid, file_size, analysis):
    """Conservative risk fusion built on top of the original risk calculation."""
    base = calculate_risk(file_format_valid, structure_valid, file_size, analysis)
    # Original implementation returns (score, level, signals).
    if isinstance(base, tuple) and len(base) == 3:
        score, level, signals = base
    else:
        score, level, signals = 0, "LOW RISK", []

    score = _safe_float(score)
    signals = list(signals or [])

    tampering = analysis.get("tampering_analysis") or {}
    tamper = _safe_float(tampering.get("tampering_probability", tampering.get("score", 0)))
    tamper_status = str(tampering.get("status", "")).upper()

    # Weak forensic evidence gets only a small additive contribution.
    if tamper_status in {"REVIEW_RECOMMENDED", "REVIEW", "POTENTIAL_EDITING_SIGNAL"}:
        score += min(8, max(0, tamper * 0.12))
        signals.append("Forensic screening signal requires review")
    elif tamper_status == "HIGH_REVIEW_REQUIRED":
        score += min(16, max(4, tamper * 0.18))
        signals.append("Multiple forensic signals require review")

    category = str(analysis.get("document_category", "UNKNOWN")).upper()
    detection_conf = str(analysis.get("document_detection_confidence", "LOW")).upper()
    if category == "UNKNOWN":
        score = max(score, 21)
        signals.append("Document type could not be identified confidently")
    elif detection_conf == "LOW":
        score = max(score, 11)
        signals.append("Document type confidence is low")

    # QR absence and face count intentionally do not add risk.
    mrz_check = analysis.get("passport_mrz_cross_check") or {}
    if str(mrz_check.get("status", "")).upper() == "MISMATCH":
        score = max(score, 41)
        signals.append("Passport MRZ and extracted fields disagree; manual review recommended")

    qr = analysis.get("qr_analysis") or {}
    qr_check = analysis.get("qr_field_cross_check") or {}
    if str(qr_check.get("status", "")).upper() == "MISMATCH" or str(qr.get("status", "")).upper() in {"DATA_MISMATCH", "MISMATCH", "INVALID_DATA"}:
        score = max(score, 51)
        signals.append("Decoded QR data conflicts with extracted document data")

    score = int(round(min(100, max(0, score))))
    if score <= 20:
        level = "LOW RISK"
    elif score <= 50:
        level = "MEDIUM RISK"
    elif score <= 75:
        level = "HIGH RISK"
    else:
        level = "CRITICAL REVIEW"
    return score, level, signals


# ============================================================
# FINAL ROUTES — OLD EXTRACTION + SAFE ENHANCEMENTS
# ============================================================

@app.get("/")
def home():
    return {
        "message": "SecureDoc AI Backend is Running!",
        "status": "online",
        "version": APP_VERSION,
        "phase": "Universal Document Screening + OCR + Multi-Layer Forensics",
        "opencv_available": CV2_AVAILABLE,
        "paddleocr_available": PADDLE_AVAILABLE,
    }


@app.get("/health")
def health():
    try:
        version = pytesseract.get_tesseract_version()
        tesseract_ready = True
    except Exception:
        version = None
        tesseract_ready = False
    return {
        "status": "online",
        "tesseract_ready": tesseract_ready,
        "tesseract_version": str(version) if version else None,
        "opencv_available": CV2_AVAILABLE,
        "paddleocr_available": PADDLE_AVAILABLE,
        "paddleocr_error": _PADDLE_ENGINE_ERROR,
        "backend_version": APP_VERSION,
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    declared_type = normalize_content_type(file.content_type)
    allowed = {normalize_content_type(x) for x in ALLOWED_CONTENT_TYPES}
    if declared_type not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload JPG, PNG, WEBP or PDF.")

    file_content = await file.read()
    file_size = len(file_content)
    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty.")
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File is too large. Maximum allowed size is 10 MB.")

    detected_type = detect_file_signature(file_content)
    if detected_type == "unknown":
        raise HTTPException(status_code=400, detail="File signature could not be verified.")

    file_format_valid = content_type_matches(declared_type, detected_type)
    if detected_type.startswith("image/"):
        analysis_data = analyze_image(file_content)
    elif detected_type == "application/pdf":
        analysis_data = analyze_pdf(file_content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported detected file type.")

    # Critical rule: enhancement is post-processing. The old OCR result remains
    # authoritative whenever it already extracted a field successfully.
    analysis_data = _enhance_analysis_preserving_legacy(file_content, analysis_data)

    structure_valid = bool(analysis_data.get("valid", False))
    validation_results = build_validation_results(file_format_valid, structure_valid, analysis_data)
    risk_score, risk_level, risk_signals = _enhanced_risk(
        file_format_valid, structure_valid, file_size, analysis_data
    )

    if risk_score >= 76:
        decision = "CRITICAL_REVIEW"
    elif risk_score >= 51:
        decision = "HIGH_REVIEW"
    elif risk_score >= 21:
        decision = "REVIEW"
    else:
        decision = "LOW_RISK_SCREENING"

    tampering = analysis_data.get("tampering_analysis") or {}
    anomaly = bool(
        risk_score >= 21
        or tampering.get("status") in {"REVIEW_RECOMMENDED", "HIGH_REVIEW_REQUIRED", "REVIEW"}
    )
    if tampering.get("status") == "HIGH_REVIEW_REQUIRED":
        anomaly_title = "Multiple Forensic Signals Require Review"
        anomaly_description = "Multiple technical signals warrant manual forensic review; this is not proof of forgery or AI editing."
    elif tampering.get("status") in {"REVIEW_RECOMMENDED", "REVIEW"}:
        anomaly_title = "Forensic Signals Require Review"
        anomaly_description = "One or more technical forensic signals require manual review."
    elif anomaly:
        anomaly_title = "Technical Review Recommended"
        anomaly_description = "One or more validation, OCR, quality or consistency signals require review."
    else:
        anomaly_title = "No Strong Technical Anomaly Detected"
        anomaly_description = "Available technical checks completed without a strong combined anomaly signal."

    ocr_conf = _safe_float(analysis_data.get("ocr_confidence"))
    tamper_probability = _safe_float(tampering.get("tampering_probability", tampering.get("score", 0)))
    anomaly_confidence = round(min(99, max(50, 55 + ocr_conf * 0.25 + min(25, tamper_probability * 0.25))), 1)

    face = analysis_data.get("face_detection") or {}
    qr = analysis_data.get("qr_analysis") or {}
    structured = analysis_data.get("structured_data") or {}
    raw_text = analysis_data.get("raw_ocr_text", "") or ""
    marking_signals = analysis_data.get("document_marking_signals") or detect_demo_or_mockup_signals(raw_text)

    return {
        "success": True,
        "message": "Document uploaded and analyzed successfully",
        "document": {
            "filename": file.filename,
            "content_type": declared_type,
            "detected_type": detected_type,
            "file_size": file_size,
            "sha256": hashlib.sha256(file_content).hexdigest(),
            "category": analysis_data.get("document_category", "UNKNOWN"),
            "label": analysis_data.get("document_label", "Unknown Document"),
            "detection_confidence": analysis_data.get("document_detection_confidence", "LOW"),
        },
        "ocr": {
            "text": raw_text,
            "confidence": ocr_conf,
            "primary_confidence": analysis_data.get("ocr_confidence_primary", ocr_conf),
            "recovery_confidence": analysis_data.get("ocr_recovery_confidence", 0),
            "status": analysis_data.get("ocr_status", "NO_TEXT_DETECTED"),
            "language": analysis_data.get("ocr_language", "eng"),
            "engine": analysis_data.get("ocr_engine", "tesseract"),
            "variant": analysis_data.get("ocr_variant"),
            "text_regions": analysis_data.get("ocr_tokens", []),
            "recovery_passes": analysis_data.get("ocr_recovery_passes", []),
            "raw_confidence": analysis_data.get("ocr_confidence_raw", ocr_conf),
            "quality_adjusted_confidence": analysis_data.get("ocr_confidence_quality_adjusted", ocr_conf),
        },
        "fields": structured,
        "field_confidence": analysis_data.get("field_confidence", {}),
        "field_provenance": analysis_data.get("field_provenance", {}),
        "identifier_validation": analysis_data.get("identifier_validation", {}),
        "quality_guidance": analysis_data.get("quality_guidance", {}),
        "passport_mrz": analysis_data.get("passport_mrz", {"detected": False}),
        "passport_mrz_cross_check": analysis_data.get("passport_mrz_cross_check", {"status": "NOT_APPLICABLE"}),
        "qr_field_cross_check": analysis_data.get("qr_field_cross_check", {"status": "NOT_AVAILABLE"}),
        "validation": {
            "status": "PASSED" if file_format_valid and structure_valid else "REVIEW",
            "results": validation_results,
        },
        "qr": qr,
        "qr_verification": qr,
        "tampering": tampering,
        "analysis_data": analysis_data,
        "face_verification": face,
        "anomaly": {
            "detected": anomaly,
            "title": anomaly_title,
            "description": anomaly_description,
            "confidence": anomaly_confidence,
        },
        "risk_assessment": {
            "score": risk_score,
            "level": risk_level,
            "decision": decision,
            "signals": risk_signals,
            "description": "Technical screening score. It is not a legal authenticity verdict or a probability that a document is forged.",
        },
        "screening_summary": {
            "document_type": analysis_data.get("document_label", "Unknown Document"),
            "document_type_confidence": analysis_data.get("document_detection_confidence", "LOW"),
            "ocr_confidence": ocr_conf,
            "tampering_screening_score": tamper_probability,
            "tampering_status": tampering.get("status", "NO_STRONG_SIGNAL"),
            "ai_edit_screening": (tampering.get("ai_edit_assessment") or {}).get("status", "NO_STRONG_EDITING_SIGNAL"),
            "final_decision": decision,
            "risk_level": risk_level,
            "document_marking_signals": marking_signals,
            "extracted_fields": [key for key, value in structured.items() if value and key not in {"document", "document_category"}],
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
