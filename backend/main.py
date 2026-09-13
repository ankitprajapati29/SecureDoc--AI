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
    version="5.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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

    print("DRIVING LICENCE OCR TEXT:", repr(text))

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

    value = clean_field_value(
        value
    )

    if not value:
        return None

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    words = []

    for word in value.split():

        word = re.sub(
            r"[^A-Za-z'-]",
            "",
            word,
        )

        if word:

            words.append(
                word
            )

    if not words:

        return None

    return " ".join(
        words
    )


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
# SECUREDOC AI - FINAL UNIVERSAL FORENSIC SCREENING LAYER
# ============================================================
# This layer intentionally preserves the proven OCR/extraction pipeline
# above and replaces only the weak/duplicated decision logic below.
# It is document-agnostic: Aadhaar is one validator among many.
# IMPORTANT: forensic signals indicate screening evidence, not legal proof.

import math


# Keep the original working implementations available as fallbacks.
_LEGACY_DETECT_DOCUMENT_TYPE = detect_document_type
_LEGACY_ANALYZE_IMAGE = analyze_image
_LEGACY_ANALYZE_PDF = analyze_pdf


# ============================================================
# UNIVERSAL DOCUMENT DETECTION
# ============================================================

def _norm_upper(value):
    return re.sub(r"\s+", " ", normalize_text(value)).upper().strip()


def _has_any(text, values):
    upper = _norm_upper(text)
    return any(v.upper() in upper for v in values)


def _count_patterns(text, patterns):
    total = 0
    for pattern in patterns:
        try:
            total += len(re.findall(pattern, text, re.IGNORECASE))
        except Exception:
            pass
    return total


def detect_document_type(text):
    """Universal weighted document classifier using keywords, labels and patterns."""
    text = normalize_text(text)
    upper = _norm_upper(text)
    compact = re.sub(r"[^A-Z0-9]", "", upper)

    scores = {}
    evidence = {}

    def add(category, points, reason):
        scores[category] = scores.get(category, 0.0) + float(points)
        evidence.setdefault(category, []).append(reason)

    # Aadhaar / UIDAI
    if _has_any(upper, ["AADHAAR", "AADHAR", "UIDAI", "UNIQUE IDENTIFICATION AUTHORITY"]):
        add("AADHAAR_CARD", 7, "Aadhaar/UIDAI terminology")
    if _has_any(upper, ["GOVERNMENT OF INDIA"]):
        add("AADHAAR_CARD", 1.5, "Government of India header")
    if re.search(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)", upper):
        add("AADHAAR_CARD", 4, "12-digit grouped identity number pattern")
    if _has_any(upper, ["MY AADHAAR", "मेरा आधार", "आधार पहचान"]):
        add("AADHAAR_CARD", 4, "Aadhaar-specific wording")

    # PAN
    if _has_any(upper, ["PERMANENT ACCOUNT NUMBER", "INCOME TAX DEPARTMENT", "INCOME TAX"]):
        add("PAN_CARD", 6, "Income Tax/PAN terminology")
    if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", upper):
        add("PAN_CARD", 5, "PAN alphanumeric pattern")

    # Driving licence
    if _has_any(upper, ["DRIVING LICENCE", "DRIVING LICENSE", "TRANSPORT DEPARTMENT", "LICENCE NO", "LICENSE NO"]):
        add("DRIVING_LICENCE", 6, "Driving-licence terminology")
    if re.search(r"\b[A-Z]{2}[- ]?[0-9]{2}[- ]?[0-9]{4,13}\b", upper):
        add("DRIVING_LICENCE", 3, "Indian licence-number style pattern")

    # Passport
    if _has_any(upper, ["PASSPORT", "REPUBLIC OF INDIA", "DATE OF EXPIRY", "DATE OF ISSUE", "PASSPORT NO"]):
        add("PASSPORT", 6, "Passport terminology")
    if re.search(r"\b[A-Z][0-9]{7}\b", upper):
        add("PASSPORT", 4, "Passport number pattern")
    if re.search(r"P<[A-Z]{3}", compact):
        add("PASSPORT", 4, "MRZ passport marker")

    # Visa / permit
    if _has_any(upper, ["VISA", "VISA NUMBER", "TYPE OF VISA", "VALID UNTIL", "VALID FROM"]):
        add("VISA", 5, "Visa terminology")
    if _has_any(upper, ["PERMIT", "RESIDENCE PERMIT", "WORK PERMIT", "ENTRY PERMIT"]):
        add("PERMIT", 5, "Permit terminology")

    # Voter ID
    if _has_any(upper, ["ELECTION COMMISSION", "ELECTOR", "ELECTORAL", "EPIC"]):
        add("VOTER_ID", 6, "Election/electoral terminology")
    if re.search(r"\b[A-Z]{3}[0-9]{7}\b", upper):
        add("VOTER_ID", 2.5, "EPIC-like identifier pattern")

    # Ration card
    if _has_any(upper, ["RATION CARD", "PUBLIC DISTRIBUTION SYSTEM", "FOOD AND CIVIL SUPPLIES"]):
        add("RATION_CARD", 6, "Ration/PDS terminology")

    # GST
    if _has_any(upper, ["GSTIN", "GOODS AND SERVICES TAX", "GST REGISTRATION"]):
        add("GST_DOCUMENT", 6, "GST terminology")
    if re.search(r"\b[0-9]{2}[A-Z0-9]{5}[0-9]{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b", upper):
        add("GST_DOCUMENT", 5, "GSTIN pattern")

    # Invoice / commercial
    if _has_any(upper, ["INVOICE", "TAX INVOICE", "BILL TO", "AMOUNT DUE", "TOTAL AMOUNT"]):
        add("INVOICE", 5, "Invoice terminology")

    # Marksheet / certificate
    if _has_any(upper, ["MARKSHEET", "MARK SHEET", "STATEMENT OF MARKS", "TOTAL MARKS", "PERCENTAGE", "GRADE"]):
        add("MARKSHEET", 5, "Academic marks terminology")
    if _has_any(upper, ["CERTIFICATE", "THIS IS TO CERTIFY", "CERTIFIED THAT", "CERTIFICATE NO"]):
        add("CERTIFICATE", 4.5, "Certificate terminology")

    # Bank documents
    if _has_any(upper, ["BANK STATEMENT", "ACCOUNT HOLDER", "ACCOUNT NUMBER", "IFSC", "BRANCH"]):
        add("BANK_DOCUMENT", 4.5, "Banking terminology")

    # Generic identity card
    if _has_any(upper, ["IDENTITY CARD", "IDENTIFICATION CARD", "NATIONAL ID", "GIVEN NAMES", "SURNAME"]):
        add("IDENTITY_CARD", 4, "Identity-card terminology")

    # Generic official document cues
    generic_fields = _count_patterns(text, [
        r"\b(?:NAME|FULL NAME|DATE OF BIRTH|DOB|GENDER|ADDRESS|DATE OF ISSUE|DATE OF EXPIRY)\b",
        r"\b(?:ID|IDENTIFICATION|NUMBER|NO\.?|SERIAL|REGISTRATION)\b",
    ])
    if generic_fields >= 3:
        add("IDENTITY_CARD", min(4, generic_fields * 0.5), "Multiple identity/document fields")

    if not scores:
        return {
            "document_category": "UNKNOWN",
            "document_label": "Unknown Document",
            "confidence": "LOW",
            "score": 0,
            "all_scores": {},
            "evidence": [],
        }

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    category, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # Require actual evidence; never classify from one generic field.
    if top_score < 3:
        category = "UNKNOWN"
        top_score = 0.0

    margin = top_score - second_score
    if category == "UNKNOWN":
        confidence = "LOW"
    elif top_score >= 8 and margin >= 2:
        confidence = "HIGH"
    elif top_score >= 5:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "document_category": category,
        "document_label": DOCUMENT_LABELS.get(category, "Unknown Document"),
        "confidence": confidence,
        "score": round(top_score, 2),
        "all_scores": {k: round(v, 2) for k, v in scores.items()},
        "evidence": evidence.get(category, []),
    }


# ============================================================
# NUMBER / DATE NORMALIZATION
# ============================================================

def _digits(value):
    return re.sub(r"\D", "", str(value or ""))


def _safe_date(value):
    if not value:
        return None
    value = str(value).strip().replace(".", "/").replace("-", "/")
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except Exception:
            pass
    return None


# ============================================================
# AADHAAR VERHOEFF VALIDATION
# ============================================================

_VERHOEFF_D_FINAL = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
    (6, 3, 4, 2, 8, 0, 7, 5, 1, 9),
    (3, 6, 2, 7, 5, 1, 9, 8, 0, 4),
)
_VERHOEFF_P_FINAL = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def is_valid_aadhaar_number(value):
    digits = _digits(value)
    if len(digits) != 12 or digits[0] == "0":
        return False
    checksum = 0
    for index, char in enumerate(reversed(digits)):
        checksum = _VERHOEFF_D_FINAL[checksum][_VERHOEFF_P_FINAL[index % 8][int(char)]]
    return checksum == 0


# ============================================================
# UNIVERSAL DOCUMENT VALIDATION
# ============================================================

def validate_document_specific(category, structured, qr_result=None):
    structured = structured or {}
    checks = []
    failures = []
    warnings = []

    def check(name, status, detail=""):
        item = {"name": name, "status": status}
        if detail:
            item["detail"] = detail
        checks.append(item)
        if status == "FAILED":
            failures.append(item)
        elif status == "REVIEW":
            warnings.append(item)

    # Common date checks
    dob = _safe_date(structured.get("date_of_birth"))
    issue = _safe_date(structured.get("date_of_issue"))
    expiry = _safe_date(structured.get("date_of_expiry"))
    today = date.today()

    if structured.get("date_of_birth"):
        if not dob:
            check("Date of birth format", "FAILED", "Date could not be parsed")
        elif dob > today:
            check("Date of birth logic", "FAILED", "Date of birth is in the future")
        elif dob.year < 1900:
            check("Date of birth logic", "REVIEW", "Unusual historical date")
        else:
            check("Date of birth logic", "PASSED")

    if issue and expiry:
        if expiry < issue:
            check("Issue/expiry date order", "FAILED", "Expiry date is before issue date")
        else:
            check("Issue/expiry date order", "PASSED")
    elif structured.get("date_of_expiry") and not expiry:
        check("Expiry date format", "FAILED", "Expiry date could not be parsed")

    if expiry:
        check("Validity date", "PASSED" if expiry >= today else "REVIEW", "Expired document" if expiry < today else "Currently not expired")

    category = category or "UNKNOWN"

    if category == "AADHAAR_CARD":
        number = _digits(structured.get("aadhaar_number") or structured.get("document_number"))
        if len(number) != 12:
            check("Aadhaar number format", "FAILED", "A complete 12-digit Aadhaar number was not extracted")
        elif is_valid_aadhaar_number(number):
            check("Aadhaar checksum", "PASSED", "Verhoeff checksum is valid")
        else:
            check("Aadhaar checksum", "FAILED", "Extracted 12-digit number fails Verhoeff checksum")

    elif category == "PAN_CARD":
        number = str(structured.get("pan_number") or "").replace(" ", "").upper()
        if not number:
            check("PAN number extraction", "REVIEW", "PAN number not confidently extracted")
        elif re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", number):
            check("PAN format", "PASSED")
        else:
            check("PAN format", "FAILED", "Extracted value does not match PAN structure")

    elif category == "PASSPORT":
        number = str(structured.get("passport_number") or "").replace(" ", "").upper()
        if number and re.fullmatch(r"[A-Z][0-9]{7}", number):
            check("Passport number format", "PASSED")
        elif number:
            check("Passport number format", "FAILED", "Passport number pattern is malformed")
        else:
            check("Passport number extraction", "REVIEW")

    elif category == "DRIVING_LICENCE":
        number = str(structured.get("driving_licence_number") or "").replace(" ", "").upper()
        if number and len(number) >= 8:
            check("Driving licence number", "PASSED")
        elif number:
            check("Driving licence number", "FAILED", "Licence identifier is too short")
        else:
            check("Driving licence number extraction", "REVIEW")

    elif category == "VOTER_ID":
        number = str(structured.get("voter_id_number") or "").replace(" ", "").upper()
        if number and re.fullmatch(r"[A-Z]{3}[0-9]{7}", number):
            check("Voter/EPIC number format", "PASSED")
        elif number:
            check("Voter/EPIC number format", "REVIEW", "Identifier extracted but pattern is uncertain")
        else:
            check("Voter/EPIC number extraction", "REVIEW")

    elif category == "GST_DOCUMENT":
        number = str(structured.get("gstin") or "").replace(" ", "").upper()
        if number and re.fullmatch(r"[0-9]{2}[A-Z0-9]{5}[0-9]{4}[A-Z][A-Z0-9]Z[A-Z0-9]", number):
            check("GSTIN format", "PASSED")
        elif number:
            check("GSTIN format", "FAILED", "Extracted GSTIN does not match expected structure")
        else:
            check("GSTIN extraction", "REVIEW")

    elif category == "UNKNOWN":
        check("Document-specific validation", "REVIEW", "Document type was not identified with sufficient confidence")
    else:
        # Generic documents should not be marked invalid merely because
        # a specialized checksum is unavailable.
        fields = [v for k, v in structured.items() if v and k not in {"document", "document_category"}]
        if len(fields) >= 2:
            check("Generic document field validation", "PASSED", "Multiple meaningful fields were extracted")
        else:
            check("Generic document field validation", "REVIEW", "Insufficient structured fields for strong validation")

    # QR is evidence only when actually decoded.
    qr_result = qr_result or {}
    if qr_result.get("status") == "DATA_MISMATCH":
        check("QR/data consistency", "FAILED", "Decoded QR content conflicts with extracted document data")
    elif qr_result.get("decoded"):
        check("QR/data consistency", "PASSED" if qr_result.get("data_consistent") is not False else "REVIEW")
    elif qr_result.get("status") == "NOT_DECODED":
        check("QR verification", "REVIEW", "A QR-like region was not successfully decoded")

    failed = len(failures)
    review = len(warnings)
    if failed:
        overall = "FAILED"
    elif review:
        overall = "REVIEW"
    else:
        overall = "PASSED"

    return {
        "overall_status": overall,
        "checks": checks,
        "failed_count": failed,
        "review_count": review,
        "passed_count": sum(1 for c in checks if c["status"] == "PASSED"),
        "applicable": category != "UNKNOWN",
    }


# ============================================================
# QR VERIFICATION - DOCUMENT AGNOSTIC
# ============================================================

def analyze_qr_signal(image, structured):
    result = {
        "available": False,
        "detected": False,
        "decoded": False,
        "status": "NOT_AVAILABLE",
        "data_consistent": None,
        "data": None,
        "matched_fields": [],
        "mismatched_fields": [],
    }
    if not CV2_AVAILABLE or cv2 is None or np is None or not hasattr(cv2, "QRCodeDetector"):
        return result

    try:
        array = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
        detector = cv2.QRCodeDetector()
        data = ""
        points = None
        try:
            data, points, _ = detector.detectAndDecode(array)
        except Exception:
            data = ""

        if not data and hasattr(detector, "detectAndDecodeMulti"):
            try:
                ok, decoded_info, points, _ = detector.detectAndDecodeMulti(array)
                if ok and decoded_info:
                    data = next((str(x) for x in decoded_info if x), "")
            except Exception:
                pass

        result["available"] = True
        result["detected"] = points is not None
        if not data:
            result["status"] = "NOT_DECODED" if result["detected"] else "NOT_PRESENT"
            return result

        result["decoded"] = True
        result["status"] = "DECODED"
        result["data"] = data[:4000]
        compact_qr = re.sub(r"[^A-Z0-9]", "", data.upper())

        comparisons = []
        for key in [
            "aadhaar_number", "document_number", "pan_number",
            "passport_number", "driving_licence_number", "voter_id_number", "gstin"
        ]:
            value = structured.get(key)
            if value:
                compact_value = re.sub(r"[^A-Z0-9]", "", str(value).upper())
                if compact_value and compact_value in compact_qr:
                    result["matched_fields"].append(key)
                    comparisons.append(True)
                elif compact_value:
                    result["mismatched_fields"].append(key)
                    comparisons.append(False)

        # Name/date comparison is softer because QR payloads may be encoded.
        name = structured.get("name")
        if name and len(str(name)) >= 4:
            name_tokens = [re.sub(r"[^A-Z]", "", x.upper()) for x in str(name).split() if len(x) >= 3]
            if name_tokens and any(token in compact_qr for token in name_tokens):
                result["matched_fields"].append("name")
            elif name_tokens:
                result["mismatched_fields"].append("name")

        if comparisons:
            result["data_consistent"] = all(comparisons)
            if not result["data_consistent"]:
                result["status"] = "DATA_MISMATCH"
        return result
    except Exception as error:
        result["status"] = "ERROR"
        result["error"] = str(error)
        return result


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces_in_document(image):
    result = {
        "available": False,
        "face_count": 0,
        "status": "NOT_AVAILABLE",
        "description": "Face detection is not available.",
        "faces": [],
    }
    if not CV2_AVAILABLE or cv2 is None or np is None:
        return result
    try:
        if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
            result["status"] = "OPENCV_INCOMPATIBLE"
            result["description"] = "Installed OpenCV build does not expose the standard Haar detector API."
            return result
        rgb = image.convert("RGB")
        array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            result["status"] = "MODEL_LOAD_ERROR"
            return result
        faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30))
        min_area = max(1600, int(gray.shape[0] * gray.shape[1] * 0.008))
        boxes = []
        for x, y, w, h in faces:
            area = int(w) * int(h)
            if area >= min_area:
                boxes.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
        count = len(boxes)
        if count == 0:
            status = "NO_FACE_DETECTED"
            description = "No face detected in the document image."
        elif count == 1:
            status = "ONE_FACE_DETECTED"
            description = "One face detected in the document image."
        else:
            status = "MULTIPLE_FACES_DETECTED"
            description = f"{count} faces detected; manual review may be appropriate."
        return {
            "available": True,
            "face_count": count,
            "status": status,
            "description": description,
            "faces": boxes,
        }
    except Exception as error:
        result["status"] = "PROCESSING_FAILED"
        result["description"] = f"Face detection failed: {str(error)}"
        result["error"] = str(error)
        return result


# ============================================================
# METADATA / EDITOR SOFTWARE SIGNALS
# ============================================================

def analyze_metadata_tampering(metadata):
    metadata = metadata or {}
    values = []
    for key, value in metadata.items():
        if value is None:
            continue
        values.append((str(key).lower(), str(value).lower()))

    software_signals = []
    suspicious = [
        "photoshop", "adobe photoshop", "gimp", "paint.net", "pixlr",
        "canva", "picsart", "snapseed", "lightroom", "affinity",
        "illustrator", "coreldraw", "imagemagick", "microsoft paint",
    ]
    for key, value in values:
        combined = key + " " + value
        for name in suspicious:
            if name in combined and name not in software_signals:
                software_signals.append(name)

    status = "REVIEW" if software_signals else "NO_STRONG_SIGNAL"
    return {
        "status": status,
        "metadata_present": bool(values),
        "editing_software_signals": software_signals,
        "signals": software_signals,
        "evidence": [f"Metadata references editing software: {x}" for x in software_signals],
    }


# ============================================================
# IMAGE FORENSIC METRICS
# ============================================================

def _safe_crop_array(image, box):
    if not CV2_AVAILABLE or np is None:
        return None
    try:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
        h, w = arr.shape[:2]
        x, y, bw, bh = [int(v) for v in box]
        x1 = max(0, min(w - 1, x))
        y1 = max(0, min(h - 1, y))
        x2 = max(x1 + 1, min(w, x + max(1, bw)))
        y2 = max(y1 + 1, min(h, y + max(1, bh)))
        return arr[y1:y2, x1:x2]
    except Exception:
        return None


def _region_metrics_array(arr):
    if arr is None or arr.size == 0 or cv2 is None:
        return None
    try:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        noise = float(np.std(cv2.GaussianBlur(gray, (3, 3), 0) - gray))
        edge = float(np.mean(cv2.Canny(gray, 80, 160) > 0))
        blur = float(lap.var())
        texture = float(np.std(gray))
        return {"noise": noise, "edge": edge, "blur": blur, "texture": texture}
    except Exception:
        return None


def _median(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return 0.0
    return float(np.median(values)) if np is not None else sorted(values)[len(values)//2]


def _robust_z(value, population):
    population = [float(v) for v in population if v is not None and math.isfinite(float(v))]
    if len(population) < 4:
        return 0.0
    med = _median(population)
    mad = _median([abs(v - med) for v in population])
    if mad < 1e-6:
        sd = float(np.std(population)) if np is not None else 0.0
        return abs(value - med) / max(sd, 1e-6)
    return abs(value - med) / (1.4826 * mad)


def perform_error_level_analysis(image):
    """JPEG recompression residual. Interpreted only as supporting evidence."""
    result = {
        "available": False,
        "status": "NOT_AVAILABLE",
        "score": 0.0,
        "mean_difference": 0.0,
        "p95_difference": 0.0,
        "max_difference": 0.0,
    }
    if not CV2_AVAILABLE or np is None or cv2 is None:
        return result
    try:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        # Resize only for forensic speed, preserving enough detail.
        h, w = rgb.shape[:2]
        scale = min(1.0, 1800.0 / max(h, w))
        if scale < 0.99:
            rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return result
        rec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if rec is None:
            return result
        diff = cv2.absdiff(bgr, rec)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        mean_diff = float(np.mean(gray_diff))
        p95 = float(np.percentile(gray_diff, 95))
        max_diff = float(np.max(gray_diff))
        # Relative residual score, deliberately conservative.
        score = min(100.0, max(0.0, (mean_diff - 1.2) * 9.0 + max(0.0, p95 - 8.0) * 1.1))
        result.update({
            "available": True,
            "status": "REVIEW" if score >= 35 else "NO_STRONG_SIGNAL",
            "score": round(score, 2),
            "mean_difference": round(mean_diff, 3),
            "p95_difference": round(p95, 3),
            "max_difference": round(max_diff, 3),
        })
        return result
    except Exception as error:
        result["status"] = "ERROR"
        result["error"] = str(error)
        return result


def analyze_image_region_consistency(image):
    """Compare local forensic metrics against peer regions to find outliers."""
    result = {
        "available": False,
        "noise_score": 0.0,
        "edge_score": 0.0,
        "texture_score": 0.0,
        "suspicious_regions": [],
        "signals": [],
        "metrics": {},
    }
    if not CV2_AVAILABLE or np is None or cv2 is None:
        return result
    try:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
        h, w = arr.shape[:2]
        if w < 120 or h < 120:
            return result

        # 4x4 grid, excluding very small cells.
        cells = []
        for gy in range(4):
            for gx in range(4):
                x = int(gx * w / 4)
                y = int(gy * h / 4)
                x2 = int((gx + 1) * w / 4)
                y2 = int((gy + 1) * h / 4)
                crop = arr[y:y2, x:x2]
                metrics = _region_metrics_array(crop)
                if metrics:
                    cells.append({"x": x, "y": y, "width": x2-x, "height": y2-y, **metrics})

        if len(cells) < 8:
            return result

        suspicious = []
        noise_values = [c["noise"] for c in cells]
        edge_values = [c["edge"] for c in cells]
        blur_values = [c["blur"] for c in cells]
        texture_values = [c["texture"] for c in cells]

        for c in cells:
            zn = _robust_z(c["noise"], noise_values)
            ze = _robust_z(c["edge"], edge_values)
            zb = _robust_z(c["blur"], blur_values)
            zt = _robust_z(c["texture"], texture_values)
            composite = (zn + ze + min(zb, 6) * 0.45 + zt) / 3.45
            # Require more than one metric to be unusual.
            unusual = sum(v >= 3.0 for v in (zn, ze, zb, zt)) >= 2
            if unusual and composite >= 2.2:
                suspicious.append({
                    "x": c["x"], "y": c["y"], "width": c["width"], "height": c["height"],
                    "score": round(min(10.0, composite), 2),
                    "noise_z": round(zn, 2), "edge_z": round(ze, 2),
                    "blur_z": round(zb, 2), "texture_z": round(zt, 2),
                })

        result["available"] = True
        result["noise_score"] = round(min(100.0, _robust_z(_median(noise_values), noise_values) * 10), 2)
        result["edge_score"] = round(min(100.0, _robust_z(_median(edge_values), edge_values) * 10), 2)
        result["texture_score"] = round(min(100.0, _robust_z(_median(texture_values), texture_values) * 10), 2)
        result["suspicious_regions"] = suspicious[:12]
        if suspicious:
            result["signals"].append(f"{len(suspicious)} local image region(s) differ materially from peer regions")
        result["metrics"] = {
            "noise_mean": round(_median(noise_values), 3),
            "noise_std": round(float(np.std(noise_values)), 3),
            "edge_mean": round(_median(edge_values), 4),
            "edge_std": round(float(np.std(edge_values)), 4),
            "texture_mean": round(_median(texture_values), 3),
            "texture_std": round(float(np.std(texture_values)), 3),
        }
        return result
    except Exception as error:
        result["error"] = str(error)
        return result


# ============================================================
# TARGETED TEXT / FACE REGION FORENSICS
# ============================================================

def analyze_text_region_for_tampering(image, x, y, w, h):
    if not CV2_AVAILABLE or np is None:
        return {"suspicious": False, "score": 0.0, "reason": "OpenCV unavailable"}
    try:
        margin_x = max(4, int(w * 0.35))
        margin_y = max(4, int(h * 0.75))
        crop = _safe_crop_array(image, (x - margin_x, y - margin_y, w + 2*margin_x, h + 2*margin_y))
        metrics = _region_metrics_array(crop)
        if not metrics:
            return {"suspicious": False, "score": 0.0, "reason": "Region unavailable"}
        # Absolute threshold is intentionally high because text naturally has edges.
        local_score = 0.0
        if metrics["blur"] < 18:
            local_score += 0.8
        if metrics["noise"] > 18:
            local_score += 0.7
        if metrics["edge"] > 0.45:
            local_score += 0.6
        return {
            "suspicious": local_score >= 1.4,
            "score": round(local_score, 2),
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        }
    except Exception as error:
        return {"suspicious": False, "score": 0.0, "reason": str(error)}


def analyze_targeted_text_regions(image, ocr_tokens, structured):
    result = {
        "available": bool(ocr_tokens),
        "regions_analyzed": 0,
        "suspicious_regions": [],
        "score": 0.0,
        "signals": [],
    }
    if not ocr_tokens:
        return result

    suspicious = []
    for token in ocr_tokens[:80]:
        text = str(token.get("text", "")).strip()
        if len(text) < 2:
            continue
        x = int(token.get("left", 0) or 0)
        y = int(token.get("top", 0) or 0)
        w = int(token.get("width", 0) or 0)
        h = int(token.get("height", 0) or 0)
        if w <= 2 or h <= 2:
            continue
        info = analyze_text_region_for_tampering(image, x, y, w, h)
        result["regions_analyzed"] += 1
        if info.get("suspicious"):
            suspicious.append({
                "text": text[:80], "x": x, "y": y, "width": w, "height": h,
                "score": info.get("score", 0), "metrics": info.get("metrics", {}),
            })

    result["suspicious_regions"] = suspicious[:20]
    if suspicious:
        result["score"] = round(min(100.0, len(suspicious) * 4.0), 2)
        result["signals"].append(f"{len(suspicious)} OCR text region(s) show unusual local image characteristics")
    return result


# ============================================================
# EXPLICIT DEMO / MOCKUP / INVALID-FOR-OFFICIAL-USE SIGNALS
# ============================================================

def detect_demo_or_mockup_signals(raw_text):
    text = _norm_upper(raw_text)
    hits = []
    patterns = [
        (r"\bSAMPLE\b", "SAMPLE marking detected"),
        (r"\bSPECIMEN\b", "SPECIMEN marking detected"),
        (r"NOT\s+VALID\s+FOR\s+OFFICIAL\s+USE", "Document states it is not valid for official use"),
        (r"FOR\s+DEMONSTRATION", "Demonstration marking detected"),
        (r"DEMO\s+(?:CARD|DOCUMENT|COPY)", "Demo document marking detected"),
        (r"MOCK\s*[- ]?UP", "Mock-up marking detected"),
        (r"SAMPLE\s+ONLY", "Sample-only marking detected"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            hits.append(label)
    return hits


# ============================================================
# CONSISTENCY ENGINE
# ============================================================

def analyze_cross_field_consistency(structured):
    structured = structured or {}
    checks = []
    issues = []

    dob = _safe_date(structured.get("date_of_birth"))
    issue = _safe_date(structured.get("date_of_issue"))
    expiry = _safe_date(structured.get("date_of_expiry"))

    if dob and issue and issue < dob:
        checks.append({"name": "DOB vs issue date", "status": "FAILED"})
        issues.append("Issue date occurs before date of birth")
    elif dob and issue:
        checks.append({"name": "DOB vs issue date", "status": "PASSED"})

    if issue and expiry:
        if expiry < issue:
            checks.append({"name": "Issue date vs expiry date", "status": "FAILED"})
            issues.append("Expiry date occurs before issue date")
        else:
            checks.append({"name": "Issue date vs expiry date", "status": "PASSED"})

    # Conflicting name values, when multiple extraction aliases exist.
    names = []
    for key in ("name", "full_name", "holder_name", "applicant_name"):
        if structured.get(key):
            names.append(normalize_single_line(structured[key]).lower())
    if len(set(names)) > 1:
        checks.append({"name": "Name consistency", "status": "FAILED"})
        issues.append("Multiple extracted name fields disagree")

    if structured.get("gender"):
        gender = str(structured["gender"]).lower()
        if gender not in {"male", "female", "other", "transgender", "m", "f"}:
            checks.append({"name": "Gender consistency", "status": "REVIEW"})
            issues.append("Gender value is unusual or ambiguous")
        else:
            checks.append({"name": "Gender consistency", "status": "PASSED"})

    status = "FAILED" if issues else ("PASSED" if checks else "REVIEW")
    return {"status": status, "checks": checks, "issues": issues, "count": len(issues)}



# ============================================================
# VISUAL OVERLAY / STAMP SIGNAL
# ============================================================

def detect_visual_overlay_signal(image):
    """Detect large colored diagonal/overlay markings without OCR dependency."""
    result = {
        "available": False,
        "detected": False,
        "score": 0.0,
        "signals": [],
        "red_fraction": 0.0,
        "diagonal_line_count": 0,
    }
    if not CV2_AVAILABLE or cv2 is None or np is None:
        return result
    try:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        r, g, b = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
        red = ((r > 145) & (r > g * 1.25) & (r > b * 1.25) & ((r - g) > 45) & ((r - b) > 45)).astype(np.uint8) * 255
        red_fraction = float(np.mean(red > 0))
        edges = cv2.Canny(red, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=55, minLineLength=max(90, int(min(rgb.shape[:2]) * 0.12)), maxLineGap=25)
        diagonal_count = 0
        if lines is not None:
            for line in lines[:, 0]:
                x1, y1, x2, y2 = [int(v) for v in line]
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                if length < 90:
                    continue
                angle = abs(math.degrees(math.atan2(dy, dx)))
                angle = angle if angle <= 90 else 180 - angle
                if 12 <= angle <= 78:
                    diagonal_count += 1

        # A large red fraction plus repeated long diagonals is characteristic
        # of a stamped SAMPLE/DEMO overlay. It is a screening signal, not proof.
        score = 0.0
        if red_fraction >= 0.008:
            score += 12
        if red_fraction >= 0.018:
            score += 8
        if diagonal_count >= 3:
            score += 8
        if diagonal_count >= 8:
            score += 7
        detected = score >= 20
        if detected:
            result["signals"].append("Large colored diagonal overlay/stamp detected")

        result.update({
            "available": True,
            "detected": detected,
            "score": round(min(35.0, score), 1),
            "red_fraction": round(red_fraction, 5),
            "diagonal_line_count": diagonal_count,
        })
        return result
    except Exception as error:
        result["error"] = str(error)
        return result


# ============================================================
# TAMPERING FUSION ENGINE
# ============================================================

def analyze_document_tampering(image, metadata, ocr_tokens=None, structured=None):
    metadata = metadata or {}
    structured = structured or {}
    signals = []
    evidence = []
    score_components = []

    metadata_result = analyze_metadata_tampering(metadata)
    if metadata_result.get("editing_software_signals"):
        signals.append("Editing software referenced in image metadata")
        evidence.extend(metadata_result.get("evidence", []))
        score_components.append(("metadata", 22))

    overlay = detect_visual_overlay_signal(image)
    if overlay.get("detected"):
        signals.extend(overlay.get("signals", []))
        evidence.append({"type": "visual_overlay", "metrics": overlay})
        score_components.append(("visual_overlay", min(35, overlay.get("score", 0))))

    ela = perform_error_level_analysis(image)
    if ela.get("status") == "REVIEW":
        signals.append("Unusual recompression residual pattern")
        evidence.append({"type": "ela", "score": ela.get("score", 0), "metrics": ela})
        score_components.append(("ela", min(20, 8 + ela.get("score", 0) * 0.18)))

    region = analyze_image_region_consistency(image)
    region_count = len(region.get("suspicious_regions", []))
    if region_count:
        signals.append(f"{region_count} local image region(s) are forensic outliers")
        evidence.append({"type": "region_consistency", "regions": region.get("suspicious_regions", [])})
        score_components.append(("regional", min(25, 6 + region_count * 3.2)))

    text_region = analyze_targeted_text_regions(image, ocr_tokens or [], structured)
    text_count = len(text_region.get("suspicious_regions", []))
    if text_count:
        signals.append(f"{text_count} OCR text region(s) show local forensic irregularity")
        evidence.append({"type": "text_regions", "regions": text_region.get("suspicious_regions", [])})
        score_components.append(("text_regions", min(22, 5 + text_count * 2.5)))

    raw_text = "\n".join(str(t.get("text", "")) for t in (ocr_tokens or []))
    demo_hits = detect_demo_or_mockup_signals(raw_text)
    if demo_hits:
        for hit in demo_hits:
            signals.append(hit)
            evidence.append({"type": "document_marking", "signal": hit})
        # Explicit sample/mock-up text is strong screening evidence that the
        # uploaded artifact is not an official-use copy, but it is not proof of AI editing.
        score_components.append(("demo_marking", min(35, 12 + len(demo_hits) * 8)))

    score = min(100.0, sum(v for _, v in score_components))
    if score >= 55:
        level = "HIGH"
        status = "HIGH_REVIEW_REQUIRED"
    elif score >= 25:
        level = "MEDIUM"
        status = "REVIEW_RECOMMENDED"
    else:
        level = "LOW"
        status = "NO_STRONG_SIGNAL"

    return {
        "score": round(score, 1),
        "level": level,
        "risk_level": level,
        "status": status,
        "detected": score >= 25,
        "signals": signals,
        "evidence": evidence,
        "score_components": [{"signal": k, "points": round(v, 2)} for k, v in score_components],
        "metadata": metadata_result,
        "visual_overlay": overlay,
        "ela": ela,
        "region_consistency": region,
        "text_region_analysis": text_region,
        "demo_mockup_signals": demo_hits,
        "ai_edit_assessment": {
            "status": "POTENTIAL_EDITING_SIGNAL" if score >= 25 else "NO_STRONG_EDITING_SIGNAL",
            "note": "AI-generated or AI-edited content cannot be proven from these heuristics alone; this result is a technical screening signal.",
        },
        "disclaimer": "Forensic screening signals are supporting evidence, not legal proof of forgery or AI editing.",
    }



# ============================================================
# OCR RECOVERY LAYER - DO NOT DISTURB THE WORKING FAST PATH
# ============================================================
# The old OCR path remains primary. This recovery path runs only when
# important identifiers are missing. It fixes the exact failure where a
# readable document was classified correctly but its bottom identifier was
# skipped by PSM 6.

_LEGACY_EXTRACT_OCR_DATA = extract_ocr_data


def _targeted_identifier_ocr(image):
    results = []
    try:
        rgb = fix_orientation(image).convert("RGB")
        width, height = rgb.size
        crops = [
            ("bottom_psm11", rgb.crop((0, int(height * 0.62), width, height)), "--oem 3 --psm 11"),
            ("bottom_psm12", rgb.crop((0, int(height * 0.50), width, height)), "--oem 3 --psm 12"),
        ]
        language = get_ocr_language()
        for name, crop, config in crops:
            try:
                result = run_ocr_pass(crop, config, language)
                text = normalize_text(result.get("text", ""))
                if text:
                    results.append({"name": name, "text": text, "confidence": result.get("confidence", 0), "tokens": result.get("tokens", [])})
            finally:
                safe_close(crop)
        safe_close(rgb)
    except Exception as error:
        print("TARGETED OCR RECOVERY ERROR:", str(error))
    return results


def _merge_identifier_fields(structured, text, category):
    structured = dict(structured or {})
    upper = _norm_upper(text)

    # Aadhaar: allow spaces/dashes and common OCR separators.
    aadhaar_matches = re.findall(r"(?<!\d)(\d{4})[ -]?(\d{4})[ -]?(\d{4})(?!\d)", upper)
    if aadhaar_matches and category == "AADHAAR_CARD":
        value = " ".join(aadhaar_matches[-1])
        structured["aadhaar_number"] = value
        structured["document_number"] = value

    pan = re.findall(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", upper)
    if pan and category == "PAN_CARD":
        structured["pan_number"] = pan[-1]
        structured["document_number"] = pan[-1]

    passport = re.findall(r"\b([A-Z][0-9]{7})\b", upper)
    if passport and category == "PASSPORT":
        # Avoid interpreting random alphanumeric OCR as a passport unless
        # passport terminology was also detected.
        if _has_any(upper, ["PASSPORT", "NATIONALITY", "DATE OF EXPIRY"]):
            structured["passport_number"] = passport[0]
            structured["document_number"] = passport[0]

    voter = re.findall(r"\b([A-Z]{3}[0-9]{7})\b", upper)
    if voter and category == "VOTER_ID":
        structured["voter_id_number"] = voter[0]
        structured["document_number"] = voter[0]

    gst = re.findall(r"\b([0-9]{2}[A-Z0-9]{5}[0-9]{4}[A-Z][A-Z0-9]Z[A-Z0-9])\b", upper)
    if gst and category == "GST_DOCUMENT":
        structured["gstin"] = gst[0]
        structured["document_number"] = gst[0]

    return structured


def extract_ocr_data(image):
    result = _LEGACY_EXTRACT_OCR_DATA(image)
    if not result:
        result = {"raw_ocr_text": "", "structured_data": {}, "ocr_tokens": [], "ocr_confidence": 0}

    text = normalize_text(result.get("raw_ocr_text", ""))
    structured = result.get("structured_data") or {}
    category = result.get("document_category") or result.get("document_detection", {}).get("document_category", "UNKNOWN")

    needs_recovery = (
        not text
        or category == "UNKNOWN"
        or (category == "AADHAAR_CARD" and not structured.get("aadhaar_number"))
        or (category == "PAN_CARD" and not structured.get("pan_number"))
        or (category == "PASSPORT" and not structured.get("passport_number"))
        or (category == "VOTER_ID" and not structured.get("voter_id_number"))
        or (category == "GST_DOCUMENT" and not structured.get("gstin"))
    )

    if not needs_recovery:
        return result

    recovered = _targeted_identifier_ocr(image)
    if not recovered:
        return result

    recovered_text = "\n".join(item["text"] for item in recovered)
    combined = normalize_text((text + "\n" + recovered_text).strip())[:MAX_OCR_TEXT_LENGTH]

    # Re-run the universal detector on the combined evidence.
    detection = detect_document_type(combined)
    category = detection.get("document_category", category)
    merged = _merge_identifier_fields(structured, combined, category)

    # Let the existing high-quality extractor contribute any fields that were
    # visible only in the recovery crop.
    try:
        recovered_structured = extract_fields_from_text(combined, detection)
        for key, value in (recovered_structured or {}).items():
            if value and not merged.get(key):
                merged[key] = value
    except Exception:
        pass

    # Recompute display text while preserving the legacy result shape.
    result["raw_ocr_text"] = combined
    result["extracted_text"] = combined
    result["ocr_status"] = "TEXT_DETECTED" if combined else result.get("ocr_status", "NO_TEXT_DETECTED")
    result["document_detection"] = detection
    result["document_category"] = category
    result["document_label"] = detection.get("document_label", DOCUMENT_LABELS.get(category, "Unknown Document"))
    result["document_detection_confidence"] = detection.get("confidence", "LOW")
    result["structured_data"] = merged
    result["extracted_data"] = merged
    result["extracted"] = merged
    result["ocr_recovery_passes"] = [item["name"] for item in recovered]
    result["ocr_recovery_text"] = recovered_text
    result["ocr_tokens"] = list(result.get("ocr_tokens") or [])
    for item in recovered:
        result["ocr_tokens"].extend(item.get("tokens") or [])
    result["ocr_confidence"] = max(float(result.get("ocr_confidence", 0) or 0), max((float(x.get("confidence", 0) or 0) for x in recovered), default=0.0))
    return result


# ============================================================
# FINAL IMAGE ANALYSIS WRAPPER - PRESERVE WORKING OCR
# ============================================================

def analyze_image(file_content):
    base = _LEGACY_ANALYZE_IMAGE(file_content)
    if not base.get("valid"):
        return base

    try:
        image = Image.open(io.BytesIO(file_content))
        image.load()
        image = fix_orientation(image)

        structured = base.get("structured_data") or {}
        raw_text = base.get("raw_ocr_text", "") or ""
        category = base.get("document_category", "UNKNOWN")
        ocr_tokens = base.get("ocr_tokens") or []

        qr = analyze_qr_signal(image, structured)
        validation = validate_document_specific(category, structured, qr)
        cross = analyze_cross_field_consistency(structured)
        tampering = analyze_document_tampering(
            image,
            base.get("metadata") or {},
            ocr_tokens,
            structured,
        )
        face = detect_faces_in_document(image)

        # If explicit document markings say SAMPLE/NOT VALID, expose them
        # separately so the frontend can show why review is needed.
        marking_signals = detect_demo_or_mockup_signals(raw_text)

        base["qr_analysis"] = qr
        base["qr_verification"] = qr
        base["document_validation"] = validation
        base["cross_field_consistency"] = cross
        base["tampering_analysis"] = tampering
        base["face_detection"] = face
        base["document_marking_signals"] = marking_signals

        # Canonical status fields.
        base["screening_ready"] = bool(
            base.get("ocr_status") == "TEXT_DETECTED" and category != "UNKNOWN"
        )
        base["screening_evidence"] = {
            "document_type": base.get("document_label", "Unknown Document"),
            "document_type_confidence": base.get("document_detection_confidence", "LOW"),
            "validation_status": validation.get("overall_status"),
            "cross_field_status": cross.get("status"),
            "tampering_status": tampering.get("status"),
        }

        safe_close(image)
        return base
    except Exception as error:
        base["forensic_error"] = str(error)
        base.setdefault("tampering_analysis", {
            "score": 0, "level": "UNKNOWN", "status": "ERROR", "detected": False,
            "signals": ["Forensic layer failed; manual review required"], "evidence": [],
        })
        return base


# ============================================================
# FINAL PDF ANALYSIS WRAPPER
# ============================================================

def analyze_pdf(file_content):
    base = _LEGACY_ANALYZE_PDF(file_content)
    if not base.get("valid"):
        return base

    # Native PDF text may contain enough evidence, but forensic image analysis
    # is still useful for rendered pages. Only inspect a small number of pages
    # to keep the deployment responsive.
    try:
        pdf = fitz.open(stream=file_content, filetype="pdf")
        page_forensics = []
        aggregate_tamper = []
        aggregate_regions = []
        face_counts = 0

        for page_index in range(min(pdf.page_count, MAX_PDF_OCR_PAGES)):
            page = pdf.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
            page_image = Image.open(io.BytesIO(pix.tobytes("png")))
            page_image.load()

            page_qr = analyze_qr_signal(page_image, base.get("structured_data") or {})
            page_tamper = analyze_document_tampering(
                page_image,
                base.get("metadata") or {},
                [],
                base.get("structured_data") or {},
            )
            page_face = detect_faces_in_document(page_image)
            face_counts += int(page_face.get("face_count", 0) or 0)
            aggregate_tamper.append(float(page_tamper.get("score", 0) or 0))
            aggregate_regions.extend(page_tamper.get("region_consistency", {}).get("suspicious_regions", []))
            page_forensics.append({
                "page": page_index + 1,
                "tampering": page_tamper,
                "qr": page_qr,
                "face_detection": page_face,
            })
            safe_close(page_image)

        pdf.close()
        structured = base.get("structured_data") or {}
        qr = {
            "available": any(p.get("qr", {}).get("available") for p in page_forensics),
            "decoded": any(p.get("qr", {}).get("decoded") for p in page_forensics),
            "status": "DATA_MISMATCH" if any(p.get("qr", {}).get("status") == "DATA_MISMATCH" for p in page_forensics) else ("DECODED" if any(p.get("qr", {}).get("decoded") for p in page_forensics) else "NOT_PRESENT"),
            "pages": page_forensics,
        }
        validation = validate_document_specific(base.get("document_category", "UNKNOWN"), structured, qr)
        cross = analyze_cross_field_consistency(structured)
        avg_tamper = sum(aggregate_tamper) / len(aggregate_tamper) if aggregate_tamper else 0
        tamper = {
            "score": round(avg_tamper, 1),
            "level": "HIGH" if avg_tamper >= 55 else "MEDIUM" if avg_tamper >= 25 else "LOW",
            "status": "HIGH_REVIEW_REQUIRED" if avg_tamper >= 55 else "REVIEW_RECOMMENDED" if avg_tamper >= 25 else "NO_STRONG_SIGNAL",
            "detected": avg_tamper >= 25,
            "signals": [],
            "evidence": [],
            "page_forensics": page_forensics,
            "suspicious_regions": aggregate_regions[:30],
            "disclaimer": "Forensic screening signals are supporting evidence, not legal proof of forgery or AI editing.",
        }
        base["qr_analysis"] = qr
        base["qr_verification"] = qr
        base["document_validation"] = validation
        base["cross_field_consistency"] = cross
        base["tampering_analysis"] = tamper
        base["face_detection"] = {
            "available": True,
            "face_count": face_counts,
            "status": "MULTIPLE_FACES_DETECTED" if face_counts > 1 else "ONE_FACE_DETECTED" if face_counts == 1 else "NO_FACE_DETECTED",
            "description": f"{face_counts} face(s) detected across scanned PDF pages.",
        }
        return base
    except Exception as error:
        base["forensic_error"] = str(error)
        return base


# ============================================================
# FINAL RISK FUSION
# ============================================================

def calculate_risk(file_format_valid, structure_valid, file_size, analysis):
    components = []
    signals = []
    evidence = []

    def add(name, points, reason=None):
        points = max(0.0, float(points))
        if points > 0:
            components.append({"name": name, "points": round(points, 2)})
        if reason:
            signals.append(reason)

    if not file_format_valid:
        add("file_integrity", 45, "File signature does not match the declared content type")
    if not structure_valid:
        add("document_structure", 40, "Document could not be parsed successfully")

    ocr_status = analysis.get("ocr_status", "NO_TEXT_DETECTED")
    ocr_conf = float(analysis.get("ocr_confidence", 0) or 0)
    if ocr_status in {"NO_TEXT_DETECTED", "OCR_ERROR"}:
        add("ocr", 22, "No reliable OCR text was obtained")
    elif ocr_conf < 35:
        add("ocr", 18, "OCR readability is low")
    elif ocr_conf < 55:
        add("ocr", 8, "OCR readability is moderate")

    detection = analysis.get("document_category", "UNKNOWN")
    detection_conf = str(analysis.get("document_detection_confidence", "LOW")).upper()
    if detection == "UNKNOWN":
        add("document_detection", 18, "Document type could not be identified confidently")
    elif detection_conf == "LOW":
        add("document_detection", 5, "Document type confidence is low")

    quality = analysis.get("image_quality") or {}
    issues = quality.get("issues") or []
    if issues:
        add("image_quality", min(12, len(issues) * 3), "Image quality issues may reduce screening reliability")

    validation = analysis.get("document_validation") or {}
    failed = int(validation.get("failed_count", 0) or 0)
    reviews = int(validation.get("review_count", 0) or 0)
    if failed:
        add("document_validation", min(40, failed * 20), f"{failed} document-specific validation check(s) failed")
        evidence.extend(validation.get("checks", []))
    elif reviews:
        add("document_validation_review", min(12, reviews * 4), f"{reviews} document validation check(s) require review")

    cross = analysis.get("cross_field_consistency") or {}
    if cross.get("status") == "FAILED":
        add("cross_consistency", 22, "Cross-field consistency checks found contradictions")
        evidence.extend(cross.get("issues", []))
    elif cross.get("status") == "REVIEW":
        add("cross_consistency_review", 5, "Cross-field consistency evidence is incomplete")

    qr = analysis.get("qr_analysis") or {}
    if qr.get("status") == "DATA_MISMATCH":
        add("qr_mismatch", 35, "Decoded QR data conflicts with extracted document data")
        evidence.append(qr)

    tampering = analysis.get("tampering_analysis") or {}
    tamper_score = float(tampering.get("score", 0) or 0)
    if tamper_score >= 25:
        add("forensics", min(45, tamper_score * 0.65), "Forensic image signals require review")
        evidence.extend(tampering.get("signals", []))

    markings = analysis.get("document_marking_signals") or []
    if markings:
        add("official_use_marking", min(30, 10 + len(markings) * 7), "Document contains sample/demo/not-for-official-use marking")
        evidence.extend(markings)

    face = analysis.get("face_detection") or {}
    if face.get("face_count", 0) > 1:
        add("face_count", 8, "Multiple faces detected in the document image")

    # A technical risk score is not a probability of forgery. It is a screening index.
    raw_score = sum(c["points"] for c in components)
    score = int(round(min(100.0, max(0.0, raw_score))))

    if score <= 20:
        level = "LOW RISK"
    elif score <= 50:
        level = "MEDIUM RISK"
    elif score <= 75:
        level = "HIGH RISK"
    else:
        level = "CRITICAL REVIEW"

    if score >= 76:
        screening_status = "HIGH RISK"
    elif score >= 51:
        screening_status = "SUSPICIOUS"
    elif score >= 21:
        screening_status = "REVIEW"
    else:
        screening_status = "LOW RISK"

    if validation.get("overall_status") == "FAILED" or cross.get("status") == "FAILED" or qr.get("status") == "DATA_MISMATCH":
        # Contradictory/failed evidence must not be hidden by a low average score.
        if score < 51:
            score = 51
            level = "HIGH RISK"
            screening_status = "SUSPICIOUS"

    return score, level, signals, {
        "components": components,
        "evidence": evidence[:60],
        "screening_status": screening_status,
        "raw_score": round(raw_score, 2),
    }


# ============================================================
# FINAL VALIDATION UI DATA
# ============================================================

def build_validation_results(file_format_valid, structure_valid, analysis):
    structured = analysis.get("structured_data") or {}
    validation = analysis.get("document_validation") or {}
    cross = analysis.get("cross_field_consistency") or {}
    tampering = analysis.get("tampering_analysis") or {}
    qr = analysis.get("qr_analysis") or {}
    quality = analysis.get("image_quality") or {}

    results = [
        {"name": "File format / signature check", "status": "PASSED" if file_format_valid else "FAILED"},
        {"name": "Document structure check", "status": "PASSED" if structure_valid else "FAILED"},
        {"name": "OCR readability check", "status": "PASSED" if analysis.get("ocr_status") == "TEXT_DETECTED" and float(analysis.get("ocr_confidence", 0) or 0) >= 45 else "REVIEW REQUIRED"},
        {"name": "Document type detection", "status": analysis.get("document_label", "Unknown Document")},
        {"name": "Document type confidence", "status": str(analysis.get("document_detection_confidence", "LOW")).upper()},
        {"name": "Document-specific validation", "status": validation.get("overall_status", "REVIEW")},
        {"name": "Cross-field consistency", "status": cross.get("status", "REVIEW")},
        {"name": "QR verification", "status": qr.get("status", "NOT APPLICABLE")},
        {"name": "Image quality", "status": quality.get("status", "NOT APPLICABLE")},
        {"name": "Tampering / forensic screening", "status": "HIGH REVIEW" if tampering.get("status") == "HIGH_REVIEW_REQUIRED" else "REVIEW REQUIRED" if tampering.get("status") == "REVIEW_RECOMMENDED" else "NO STRONG SIGNAL"},
    ]
    return results


# ============================================================
# DISPLAY TEXT - KEEP FRONTEND FRIENDLY
# ============================================================

def build_display_text(structured):
    structured = structured or {}
    labels = [
        ("Name", "name"),
        ("Document", "document"),
        ("Document Number", "document_number"),
        ("Aadhaar Number", "aadhaar_number"),
        ("PAN Number", "pan_number"),
        ("Driving Licence Number", "driving_licence_number"),
        ("Passport Number", "passport_number"),
        ("Voter ID Number", "voter_id_number"),
        ("GSTIN", "gstin"),
        ("Date of Birth", "date_of_birth"),
        ("Gender", "gender"),
        ("Nationality", "nationality"),
        ("Date of Issue", "date_of_issue"),
        ("Date of Expiry", "date_of_expiry"),
        ("Validity Status", "validity_status"),
        ("Address", "address"),
    ]
    lines = []
    for label, key in labels:
        value = structured.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


# ============================================================
# FINAL ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "SecureDoc AI Backend is Running!",
        "status": "online",
        "version": "6.1.0",
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
        "backend_version": "6.1.0",
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    declared_type = normalize_content_type(file.content_type)
    if declared_type not in {normalize_content_type(x) for x in ALLOWED_CONTENT_TYPES}:
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

    structure_valid = bool(analysis_data.get("valid", False))
    validation_results = build_validation_results(file_format_valid, structure_valid, analysis_data)
    risk_score, risk_level, risk_signals, risk_meta = calculate_risk(
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
        or tampering.get("status") in {"REVIEW_RECOMMENDED", "HIGH_REVIEW_REQUIRED"}
    )

    if tampering.get("status") == "HIGH_REVIEW_REQUIRED":
        anomaly_title = "Multiple Forensic Signals Require Review"
        anomaly_description = "Multiple technical signals warrant manual forensic review; this is not proof of forgery or AI editing."
    elif tampering.get("status") == "REVIEW_RECOMMENDED":
        anomaly_title = "Forensic Signals Require Review"
        anomaly_description = "One or more local, metadata, recompression or document-marking signals require manual review."
    elif anomaly:
        anomaly_title = "Technical Review Recommended"
        anomaly_description = "One or more validation, OCR, quality or consistency signals require review."
    else:
        anomaly_title = "No Strong Technical Anomaly Detected"
        anomaly_description = "Available technical checks completed without a strong combined anomaly signal."

    ocr_conf = float(analysis_data.get("ocr_confidence", 0) or 0)
    tamper_probability = float(tampering.get("score", 0) or 0)
    anomaly_confidence = round(min(99, max(50, 55 + ocr_conf * 0.25 + min(25, tamper_probability * 0.25))), 1)

    document_validation = analysis_data.get("document_validation") or {}
    cross = analysis_data.get("cross_field_consistency") or {}
    validation_status = (
        "PASSED"
        if file_format_valid and structure_valid
        and document_validation.get("overall_status") == "PASSED"
        and cross.get("status") == "PASSED"
        else "REVIEW"
    )

    face = analysis_data.get("face_detection") or {}
    qr = analysis_data.get("qr_analysis") or {}
    structured = analysis_data.get("structured_data") or {}

    risk_breakdown = {
        "file_integrity": "PASS" if file_format_valid else "FAIL",
        "ocr_quality": "GOOD" if ocr_conf >= 70 else "MODERATE" if ocr_conf >= 45 else "LOW",
        "document_structure": "VALID" if structure_valid else "INVALID",
        "document_type": analysis_data.get("document_label", "Unknown Document"),
        "document_type_confidence": analysis_data.get("document_detection_confidence", "LOW"),
        "document_validation": document_validation.get("overall_status", "REVIEW"),
        "cross_field_consistency": cross.get("status", "REVIEW"),
        "qr_consistency": qr.get("status", "N/A"),
        "tampering_signals": tampering.get("risk_level", "LOW"),
        "face_detection": f"{int(face.get('face_count', 0) or 0)} FACE",
    }

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
            "status": analysis_data.get("ocr_status", "NO_TEXT_DETECTED"),
            "language": analysis_data.get("ocr_language", "eng"),
            "engine": analysis_data.get("ocr_engine", "tesseract"),
            "variant": analysis_data.get("ocr_variant"),
            "text_regions": analysis_data.get("ocr_tokens", []),
        },
        "fields": structured,
        "validation": {
            "status": validation_status,
            "results": validation_results,
            "document_specific": document_validation,
            "cross_field": cross,
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
        "risk_breakdown": risk_breakdown,
        "risk_assessment": {
            "score": risk_score,
            "level": risk_level,
            "screening_status": risk_meta.get("screening_status"),
            "decision": decision,
            "signals": risk_signals,
            "components": risk_meta.get("components", []),
            "evidence": risk_meta.get("evidence", []),
            "description": "Technical screening score combining file integrity, OCR, document-specific validation, consistency, QR evidence, image quality and forensic signals. It is not a legal authenticity verdict.",
        },
        "screening_summary": {
            "document_type": analysis_data.get("document_label", "Unknown Document"),
            "document_type_confidence": analysis_data.get("document_detection_confidence", "LOW"),
            "ocr_confidence": ocr_conf,
            "tampering_probability": tamper_probability,
            "tampering_status": tampering.get("status", "NO_STRONG_SIGNAL"),
            "ai_edit_screening": (tampering.get("ai_edit_assessment") or {}).get("status", "NO_STRONG_EDITING_SIGNAL"),
            "final_decision": decision,
            "risk_level": risk_level,
            "document_marking_signals": marking_signals,
            "extracted_fields": [key for key, value in structured.items() if value and key not in {"document", "document_category"}],
        },
    }



# ============================================================
# FINAL CONSOLIDATED SCREENING OVERRIDES
# ============================================================
# These overrides intentionally preserve the old working OCR/PDF/image
# pipeline and replace only the problematic final decision layers:
# document classification, OCR recovery fallback, forensic weighting,
# and risk fusion.
#
# Design principles:
#   - Technical artifacts are not automatically treated as forgery.
#   - Unknown document type means REVIEW, not VALID and not FAKE.
#   - QR absence is NOT a risk signal.
#   - Face count is informational, not a forgery signal.
#   - A single weak forensic signal cannot create high risk.
#   - Document-specific checks are applied only to the detected type.
#   - AI-edit wording is explicitly probabilistic/forensic-screening only.

def detect_document_type(text):
    """Universal evidence-weighted document classifier.

    Uses document terminology, field labels and identifier patterns together.
    Generic terms such as 'Government of India' are deliberately weak so they
    cannot by themselves turn an arbitrary image into an Aadhaar document.
    """
    text = normalize_text(text)
    upper = _norm_upper(text)
    compact = re.sub(r"[^A-Z0-9]", "", upper)

    scores = defaultdict(float)
    evidence = defaultdict(list)

    def add(category, points, reason):
        scores[category] += float(points)
        evidence[category].append(reason)

    def has(*terms):
        return any(term.upper() in upper for term in terms)

    # ------------------------- AADHAAR -------------------------
    if has("AADHAAR", "AADHAR", "UIDAI", "UNIQUE IDENTIFICATION AUTHORITY"):
        add("AADHAAR_CARD", 9, "Aadhaar/UIDAI terminology")
    if has("MY AADHAAR", "मेरा आधार", "आधार पहचान"):
        add("AADHAAR_CARD", 4, "Aadhaar-specific wording")
    if has("AADHAAR IS PROOF OF IDENTITY"):
        add("AADHAAR_CARD", 5, "Aadhaar identity notice")
    if re.search(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)", upper):
        add("AADHAAR_CARD", 3.5, "12-digit grouped identifier pattern")
    if has("GOVERNMENT OF INDIA") and (
        has("UIDAI", "AADHAAR", "AADHAR")
        or re.search(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)", upper)
    ):
        add("AADHAAR_CARD", 1.5, "Government of India + Aadhaar evidence")

    # --------------------------- PAN ---------------------------
    if has("PERMANENT ACCOUNT NUMBER", "INCOME TAX DEPARTMENT"):
        add("PAN_CARD", 9, "PAN/Income Tax terminology")
    elif has("INCOME TAX"):
        add("PAN_CARD", 5, "Income Tax terminology")
    if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", upper):
        add("PAN_CARD", 5, "PAN identifier pattern")

    # -------------------- DRIVING LICENCE ---------------------
    if has("DRIVING LICENCE", "DRIVING LICENSE"):
        add("DRIVING_LICENCE", 9, "Driving licence terminology")
    if has("TRANSPORT DEPARTMENT", "LICENCE NO", "LICENSE NO"):
        add("DRIVING_LICENCE", 4, "Transport/licence field labels")
    if re.search(r"\b[A-Z]{2}\d{2}[ -]?\d{4,6}[ -]?\d{5,10}\b", upper):
        add("DRIVING_LICENCE", 4, "Driving licence number pattern")

    # ------------------------- PASSPORT -----------------------
    if has("PASSPORT"):
        add("PASSPORT", 10, "Passport terminology")
    if has("REPUBLIC OF INDIA", "NATIONALITY") and has("DATE OF EXPIRY", "DATE OF ISSUE"):
        add("PASSPORT", 3, "Passport-style identity/date fields")
    if re.search(r"\b[A-Z][0-9]{7}\b", upper) and has("PASSPORT", "NATIONALITY"):
        add("PASSPORT", 3, "Passport number pattern with passport evidence")

    # --------------------------- VISA --------------------------
    if has("VISA NUMBER", "VISA NO", "TYPE OF VISA", "VALID FROM", "VALID UNTIL", "VALID TO"):
        add("VISA", 7, "Visa-specific field labels")
    elif has("VISA"):
        add("VISA", 5, "Visa terminology")

    # -------------------------- PERMIT ------------------------
    if has("RESIDENCE PERMIT", "WORK PERMIT", "ENTRY PERMIT", "TEMPORARY PERMIT"):
        add("PERMIT", 8, "Permit-specific terminology")
    elif has("PERMIT"):
        add("PERMIT", 4, "Permit terminology")

    # ------------------------ VOTER ID ------------------------
    if has("ELECTION COMMISSION", "ELECTORAL", "ELECTOR PHOTO ID", "EPIC"):
        add("VOTER_ID", 8, "Election/elector terminology")
    if re.search(r"\b[A-Z]{3}[0-9]{7}\b", upper) and has("ELECTOR", "EPIC", "ELECTION"):
        add("VOTER_ID", 4, "Voter identifier pattern")

    # ----------------------- RATION CARD ----------------------
    if has("RATION CARD", "PUBLIC DISTRIBUTION SYSTEM", "FOOD AND CIVIL SUPPLIES"):
        add("RATION_CARD", 8, "Ration/PDS terminology")

    # ------------------------ GST DOCUMENT --------------------
    if has("GSTIN", "GOODS AND SERVICES TAX", "GST REGISTRATION"):
        add("GST_DOCUMENT", 8, "GST terminology")
    if re.search(r"\b[0-9]{2}[A-Z0-9]{5}[0-9]{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b", upper):
        add("GST_DOCUMENT", 5, "GSTIN pattern")

    # -------------------------- INVOICE -----------------------
    if has("INVOICE", "BILL TO", "INVOICE NO", "AMOUNT DUE"):
        add("INVOICE", 7, "Invoice terminology")
    if has("TOTAL AMOUNT", "SUBTOTAL", "TAX", "QUANTITY"):
        add("INVOICE", 2, "Invoice-style financial fields")

    # ------------------------- MARKSHEET ----------------------
    if has("MARKSHEET", "MARK SHEET", "STATEMENT OF MARKS", "TOTAL MARKS"):
        add("MARKSHEET", 8, "Marksheet terminology")
    if has("PERCENTAGE", "GRADE", "SEMESTER", "SUBJECT"):
        add("MARKSHEET", 2, "Academic result fields")

    # ------------------------ CERTIFICATE ---------------------
    if has("CERTIFICATE", "THIS IS TO CERTIFY", "CERTIFIED THAT"):
        add("CERTIFICATE", 7, "Certificate terminology")
    if has("CERTIFICATE NO", "DATE OF ISSUE"):
        add("CERTIFICATE", 2, "Certificate fields")

    # ---------------------- BANK DOCUMENT ---------------------
    if has("BANK STATEMENT", "ACCOUNT HOLDER", "IFSC", "ACCOUNT NUMBER"):
        add("BANK_DOCUMENT", 7, "Banking terminology")
    if has("TRANSACTION", "BALANCE", "BRANCH"):
        add("BANK_DOCUMENT", 2, "Bank statement fields")

    # ---------------------- GENERIC ID CARD -------------------
    if has("IDENTITY CARD", "IDENTIFICATION CARD", "NATIONAL IDENTITY", "NATIONAL ID"):
        add("IDENTITY_CARD", 6, "Identity-card terminology")
    if has("SURNAME", "GIVEN NAMES", "DATE OF BIRTH", "GENDER"):
        add("IDENTITY_CARD", 1.5, "Identity fields")

    if not scores:
        return {
            "document_category": "UNKNOWN",
            "document_label": "Unknown Document",
            "confidence": "LOW",
            "score": 0,
            "all_scores": {},
            "evidence": [],
            "ambiguous": False,
        }

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_category, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # Require meaningful evidence. A weak single clue is not enough.
    if best_score < 5:
        best_category = "UNKNOWN"
        best_score = 0.0

    margin = best_score - second_score
    ambiguous = bool(
        best_category != "UNKNOWN"
        and second_score >= 5
        and margin < max(2.5, best_score * 0.18)
    )

    if best_category == "UNKNOWN":
        confidence = "LOW"
    elif ambiguous:
        confidence = "MEDIUM"
    elif best_score >= 10 and margin >= 3:
        confidence = "HIGH"
    elif best_score >= 7:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "document_category": best_category,
        "document_label": DOCUMENT_LABELS.get(best_category, "Unknown Document"),
        "confidence": confidence,
        "score": round(best_score, 2),
        "all_scores": {k: round(v, 2) for k, v in scores.items()},
        "evidence": evidence.get(best_category, [])[:12],
        "ambiguous": ambiguous,
        "runner_up": (
            DOCUMENT_LABELS.get(ranked[1][0], ranked[1][0])
            if len(ranked) > 1 else None
        ),
    }


def _merge_final_ocr_results(base_result, extra_results):
    """Merge OCR candidates without destroying the legacy result shape."""
    base_result = dict(base_result or {})
    base_text = normalize_text(base_result.get("raw_ocr_text", ""))
    candidates = []

    if base_text:
        candidates.append({
            "text": base_text,
            "confidence": float(base_result.get("ocr_confidence", 0) or 0),
            "tokens": list(base_result.get("ocr_tokens") or []),
            "variant": base_result.get("ocr_variant"),
            "engine": base_result.get("ocr_engine", "tesseract"),
        })

    candidates.extend(extra_results or [])

    if not candidates:
        return base_result

    def candidate_score(item):
        text = normalize_text(item.get("text", ""))
        conf = float(item.get("confidence", 0) or 0)
        det = detect_document_type(text)
        structured = {}
        try:
            structured = extract_fields_from_text(text, det) or {}
        except Exception:
            pass
        useful_fields = sum(
            1 for k in (
                "aadhaar_number", "pan_number", "passport_number",
                "driving_licence_number", "voter_id_number", "gstin",
                "name", "date_of_birth"
            ) if structured.get(k)
        )
        return min(120.0, conf * 0.7 + min(len(text), 500) * 0.03 + useful_fields * 8 + (8 if det["document_category"] != "UNKNOWN" else 0))

    best = max(candidates, key=candidate_score)
    final_text = normalize_text(best.get("text", ""))[:MAX_OCR_TEXT_LENGTH]
    detection = detect_document_type(final_text)

    try:
        structured = extract_fields_from_text(final_text, detection) or {}
    except Exception:
        structured = {}

    # Preserve fields already obtained by the legacy extractor if the new
    # candidate loses something due to OCR variation.
    old_structured = base_result.get("structured_data") or {}
    for key, value in old_structured.items():
        if value and not structured.get(key):
            structured[key] = value

    base_result.update({
        "raw_ocr_text": final_text,
        "extracted_text": final_text,
        "ocr_status": "TEXT_DETECTED" if final_text else "NO_TEXT_DETECTED",
        "ocr_confidence": round(float(best.get("confidence", 0) or 0), 2),
        "ocr_variant": best.get("variant"),
        "ocr_engine": best.get("engine", "tesseract"),
        "structured_data": structured,
        "extracted_data": structured,
        "extracted": structured,
        "document_detection": detection,
        "document_category": detection.get("document_category", "UNKNOWN"),
        "document_label": detection.get("document_label", "Unknown Document"),
        "document_detection_confidence": detection.get("confidence", "LOW"),
        "ocr_tokens": best.get("tokens") or base_result.get("ocr_tokens") or [],
    })
    return base_result


def extract_ocr_data(image):
    """Legacy-first OCR with controlled multi-pass recovery.

    The old fast Tesseract path remains the primary path. Additional passes
    happen only when text is missing, weak, or the document type/identifier is
    unresolved.
    """
    try:
        result = _LEGACY_EXTRACT_OCR_DATA(image) or {}
    except Exception as error:
        print("LEGACY OCR ERROR:", str(error))
        result = {}

    text = normalize_text(result.get("raw_ocr_text", ""))
    category = result.get("document_category") or result.get("document_detection", {}).get("document_category", "UNKNOWN")
    confidence = float(result.get("ocr_confidence", 0) or 0)

    needs_recovery = (
        not text
        or confidence < 45
        or category == "UNKNOWN"
        or len(text) < 30
    )

    if not needs_recovery:
        # Reclassify with the final universal detector so the runtime always
        # uses the same classifier as the final screening layer.
        detection = detect_document_type(text)
        result["document_detection"] = detection
        result["document_category"] = detection.get("document_category", "UNKNOWN")
        result["document_label"] = detection.get("document_label", "Unknown Document")
        result["document_detection_confidence"] = detection.get("confidence", "LOW")
        return result

    extras = []
    language = get_ocr_language()

    try:
        rgb = fix_orientation(image).convert("RGB")
        normal = resize_for_ocr(rgb)

        passes = [
            ("recovery_psm11", normal, "--oem 3 --psm 11"),
            ("recovery_psm12", normal, "--oem 3 --psm 12"),
        ]

        # Only use enhanced/binary variants when the normal image did not
        # provide enough evidence. This avoids degrading already-good scans.
        variants = create_ocr_variants(rgb)
        for name, variant in variants:
            if name == "original":
                continue
            passes.append((f"recovery_{name}_psm6", variant, "--oem 3 --psm 6"))

        seen = set()
        for name, candidate_image, config in passes:
            try:
                candidate = run_ocr_pass(candidate_image, config, language)
                candidate_text = normalize_text(candidate.get("text", ""))
                key = re.sub(r"\s+", " ", candidate_text).lower()
                if candidate_text and key not in seen:
                    seen.add(key)
                    extras.append({
                        "text": candidate_text,
                        "confidence": float(candidate.get("confidence", 0) or 0),
                        "tokens": candidate.get("tokens") or [],
                        "variant": name,
                        "engine": "tesseract",
                    })
            except Exception as error:
                print("OCR RECOVERY PASS ERROR:", name, str(error))

        safe_close(rgb)
        safe_close(normal)
        for _, variant in variants:
            safe_close(variant)
    except Exception as error:
        print("OCR RECOVERY ERROR:", str(error))

    merged = _merge_final_ocr_results(result, extras)

    # Identifier-focused recovery from the existing targeted OCR layer.
    merged_text = normalize_text(merged.get("raw_ocr_text", ""))
    merged_category = merged.get("document_category", "UNKNOWN")
    identifier_missing = (
        (merged_category == "AADHAAR_CARD" and not merged.get("structured_data", {}).get("aadhaar_number"))
        or (merged_category == "PAN_CARD" and not merged.get("structured_data", {}).get("pan_number"))
        or (merged_category == "PASSPORT" and not merged.get("structured_data", {}).get("passport_number"))
        or (merged_category == "VOTER_ID" and not merged.get("structured_data", {}).get("voter_id_number"))
        or (merged_category == "GST_DOCUMENT" and not merged.get("structured_data", {}).get("gstin"))
    )

    if identifier_missing or not merged_text:
        try:
            targeted = _targeted_identifier_ocr(image)
            if targeted:
                targeted_text = normalize_text(
                    merged_text + "\n" + "\n".join(x.get("text", "") for x in targeted)
                )[:MAX_OCR_TEXT_LENGTH]
                detection = detect_document_type(targeted_text)
                structured = extract_fields_from_text(targeted_text, detection) or {}
                old = merged.get("structured_data") or {}
                for key, value in old.items():
                    if value and not structured.get(key):
                        structured[key] = value
                structured = _merge_identifier_fields(
                    structured, targeted_text, detection.get("document_category", "UNKNOWN")
                )
                merged.update({
                    "raw_ocr_text": targeted_text,
                    "extracted_text": targeted_text,
                    "ocr_status": "TEXT_DETECTED" if targeted_text else "NO_TEXT_DETECTED",
                    "document_detection": detection,
                    "document_category": detection.get("document_category", "UNKNOWN"),
                    "document_label": detection.get("document_label", "Unknown Document"),
                    "document_detection_confidence": detection.get("confidence", "LOW"),
                    "structured_data": structured,
                    "extracted_data": structured,
                    "extracted": structured,
                    "ocr_recovery_passes": [
                        x.get("name") for x in targeted
                    ],
                    "ocr_recovery_text": "\n".join(
                        x.get("text", "") for x in targeted
                    ),
                    "ocr_tokens": list(merged.get("ocr_tokens") or []) + [
                        token
                        for x in targeted
                        for token in (x.get("tokens") or [])
                    ],
                })
        except Exception as error:
            print("TARGETED IDENTIFIER MERGE ERROR:", str(error))

    return merged


def analyze_document_tampering(image, metadata, ocr_tokens=None, structured=None):
    """Conservative multi-signal forensic screening.

    ELA, local noise and edge variation are inherently noisy on screenshots,
    scans, JPEGs and photographed documents. They are therefore weak evidence
    unless corroborated by independent signals.
    """
    metadata = metadata or {}
    structured = structured or {}
    ocr_tokens = ocr_tokens or []

    signals = []
    evidence = []
    components = []

    def add(name, points, message, strength="weak", details=None):
        points = max(0.0, float(points))
        if points <= 0:
            return
        components.append({
            "signal": name,
            "points": round(points, 2),
            "strength": strength,
        })
        signals.append(message)
        if details is not None:
            evidence.append({
                "type": name,
                "strength": strength,
                "details": details,
            })

    metadata_result = analyze_metadata_tampering(metadata)
    if metadata_result.get("editing_software_signals"):
        add(
            "metadata_editing_software",
            12,
            "Image metadata references editing software; this is a supporting clue, not proof of alteration.",
            "moderate",
            metadata_result.get("evidence", []),
        )

    overlay = detect_visual_overlay_signal(image)
    if overlay.get("detected"):
        # A diagonal SAMPLE/NOT VALID style overlay is usually a document
        # marking, not evidence that the underlying document was AI-edited.
        add(
            "visual_overlay",
            min(8, float(overlay.get("score", 0) or 0) * 0.20),
            "A visual overlay/stamp was detected; review the document marking separately.",
            "weak",
            overlay,
        )

    ela = perform_error_level_analysis(image)
    ela_score = float(ela.get("score", 0) or 0)
    if ela.get("status") == "REVIEW" and ela_score >= 45:
        add(
            "ela",
            min(10, max(2, (ela_score - 40) * 0.16)),
            "JPEG recompression residuals are unusual enough to warrant supporting review.",
            "weak",
            ela,
        )

    region = analyze_image_region_consistency(image)
    region_count = len(region.get("suspicious_regions", []) or [])
    region_score = float(region.get("noise_score", 0) or 0)
    edge_score = float(region.get("edge_score", 0) or 0)

    if region_count >= 2 and (region_score >= 45 or edge_score >= 45):
        add(
            "regional_consistency",
            min(12, 4 + region_count * 1.5),
            f"{region_count} local image region(s) show elevated forensic variation.",
            "moderate",
            {
                "noise_score": region_score,
                "edge_score": edge_score,
                "suspicious_regions": region.get("suspicious_regions", [])[:20],
            },
        )

    text_region = analyze_targeted_text_regions(image, ocr_tokens, structured)
    text_count = len(text_region.get("suspicious_regions", []) or [])
    if text_count >= 2:
        add(
            "text_region_consistency",
            min(12, 4 + text_count * 1.5),
            f"{text_count} OCR text region(s) show local forensic irregularity.",
            "moderate",
            text_region.get("suspicious_regions", [])[:20],
        )

    raw_text = normalize_text(
        "\n".join(str(t.get("text", "")) for t in ocr_tokens)
    )
    demo_hits = detect_demo_or_mockup_signals(raw_text)

    # Explicit markings are handled as a separate document-status signal.
    # They are intentionally NOT added to AI-edit/tampering score.
    marking_status = "NONE"
    if demo_hits:
        marking_status = "OFFICIAL_USE_REVIEW"
        signals.extend(
            [f"Document marking detected: {hit}" for hit in demo_hits]
        )
        evidence.append({
            "type": "document_marking",
            "strength": "strong_for_document_status",
            "signals": demo_hits,
        })

    # Independent corroboration: at least two different forensic families
    # are needed before the score can move into a meaningful tampering range.
    strongish = [
        c for c in components
        if c["strength"] in {"moderate", "strong"}
    ]

    raw_score = sum(c["points"] for c in components)
    if len(strongish) >= 2:
        raw_score += 6
    elif len(strongish) == 0:
        # Weak-only evidence is deliberately capped.
        raw_score = min(raw_score, 12)

    score = round(min(100.0, max(0.0, raw_score)), 1)

    if len(strongish) >= 2 and score >= 30:
        level = "MEDIUM"
        status = "REVIEW_RECOMMENDED"
    elif len(strongish) >= 2 and score >= 50:
        level = "HIGH"
        status = "HIGH_REVIEW_REQUIRED"
    elif score >= 18 and len(strongish) >= 1:
        level = "LOW-MEDIUM"
        status = "REVIEW_RECOMMENDED"
    else:
        level = "LOW"
        status = "NO_STRONG_SIGNAL"

    # Correct ordering for high threshold after combined evidence.
    if len(strongish) >= 2 and score >= 50:
        level = "HIGH"
        status = "HIGH_REVIEW_REQUIRED"
    elif len(strongish) >= 2 and score >= 30:
        level = "MEDIUM"
        status = "REVIEW_RECOMMENDED"

    if score < 18:
        ai_status = "NO_STRONG_EDITING_SIGNAL"
    elif len(strongish) >= 2:
        ai_status = "COMBINED_FORENSIC_SIGNAL_REVIEW"
    else:
        ai_status = "WEAK_FORENSIC_SIGNAL"

    return {
        "score": score,
        "tampering_probability": score,
        "level": level,
        "risk_level": level,
        "status": status,
        "detected": bool(len(strongish) >= 2 and score >= 30),
        "signals": signals[:40],
        "evidence": evidence[:40],
        "score_components": components,
        "independent_signal_count": len(strongish),
        "metadata": metadata_result,
        "visual_overlay": overlay,
        "ela": ela,
        "region_consistency": region,
        "text_region_analysis": text_region,
        "demo_mockup_signals": demo_hits,
        "document_marking_status": marking_status,
        "ai_edit_assessment": {
            "status": ai_status,
            "note": "This is heuristic forensic screening. It cannot prove that a document was AI-generated or AI-edited.",
        },
        "disclaimer": "Forensic screening signals are supporting evidence, not legal proof of forgery or AI editing.",
    }


def calculate_risk(file_format_valid, structure_valid, file_size, analysis):
    """Evidence-weighted screening score.

    Strong failures can raise risk quickly. Weak forensic artifacts are capped
    so normal scans/screenshots do not become 80-90 risk by themselves.
    """
    analysis = analysis or {}
    components = []
    signals = []
    evidence = []

    def add(name, points, reason=None):
        points = max(0.0, float(points))
        if points > 0:
            components.append({"name": name, "points": round(points, 2)})
        if reason:
            signals.append(reason)

    if not file_format_valid:
        add("file_integrity", 45, "File signature does not match the declared content type.")
    if not structure_valid:
        add("document_structure", 40, "Document could not be parsed successfully.")

    ocr_status = str(analysis.get("ocr_status", "NO_TEXT_DETECTED"))
    ocr_conf = float(analysis.get("ocr_confidence", 0) or 0)
    if ocr_status in {"NO_TEXT_DETECTED", "OCR_ERROR"}:
        add("ocr", 22, "No reliable OCR text was obtained.")
    elif ocr_conf < 35:
        add("ocr", 18, "OCR readability is low.")
    elif ocr_conf < 55:
        add("ocr", 8, "OCR readability is moderate.")

    category = analysis.get("document_category", "UNKNOWN")
    detection_conf = str(
        analysis.get("document_detection_confidence", "LOW")
    ).upper()

    if category == "UNKNOWN":
        add(
            "document_detection",
            18,
            "Document type could not be identified confidently; manual review is required.",
        )
    elif detection_conf == "LOW":
        add("document_detection", 5, "Document type confidence is low.")
    elif detection_conf == "MEDIUM":
        add("document_detection_review", 2, "Document type was identified with medium confidence.")

    quality = analysis.get("image_quality") or {}
    quality_issues = quality.get("issues") or []
    if quality_issues:
        add(
            "image_quality",
            min(10, len(quality_issues) * 2),
            "Image quality may reduce screening reliability.",
        )

    validation = analysis.get("document_validation") or {}
    failed = int(validation.get("failed_count", 0) or 0)
    reviews = int(validation.get("review_count", 0) or 0)

    if failed:
        # Failed document-specific checks are meaningful only for checks that
        # actually apply to the detected document type.
        add(
            "document_validation",
            min(45, failed * 18),
            f"{failed} applicable document validation check(s) failed.",
        )
        evidence.extend(validation.get("checks", [])[:20])
    elif reviews:
        add(
            "document_validation_review",
            min(10, reviews * 3),
            f"{reviews} applicable document validation check(s) require review.",
        )

    cross = analysis.get("cross_field_consistency") or {}
    if cross.get("status") == "FAILED":
        add(
            "cross_consistency",
            22,
            "Cross-field consistency checks found contradictions.",
        )
        evidence.extend(cross.get("issues", [])[:20])
    elif cross.get("status") == "REVIEW":
        add(
            "cross_consistency_review",
            4,
            "Cross-field consistency evidence is incomplete.",
        )

    qr = analysis.get("qr_analysis") or {}
    qr_status = str(qr.get("status", "NOT_AVAILABLE")).upper()
    if qr_status == "DATA_MISMATCH":
        add(
            "qr_mismatch",
            35,
            "Decoded QR data conflicts with extracted document data.",
        )
        evidence.append(qr)
    # NOT_PRESENT / NOT_DECODED / NOT_AVAILABLE are deliberately zero-risk.

    tampering = analysis.get("tampering_analysis") or {}
    tamper_score = float(
        tampering.get("score", tampering.get("tampering_probability", 0)) or 0
    )
    independent = int(tampering.get("independent_signal_count", 0) or 0)

    if independent >= 2 and tamper_score >= 50:
        add(
            "forensics",
            min(35, tamper_score * 0.55),
            "Multiple independent forensic signal families require review.",
        )
        evidence.extend(tampering.get("evidence", [])[:20])
    elif independent >= 2 and tamper_score >= 30:
        add(
            "forensics",
            min(20, tamper_score * 0.35),
            "Combined forensic signals warrant review.",
        )
        evidence.extend(tampering.get("evidence", [])[:15])
    elif tamper_score >= 18:
        # Weak/single forensic evidence has intentionally limited influence.
        add(
            "forensics_weak",
            min(8, tamper_score * 0.20),
            "A weak forensic signal was detected; this alone is not evidence of forgery.",
        )

    markings = analysis.get("document_marking_signals") or []
    if markings:
        # A SAMPLE/NOT VALID marking means the uploaded artifact may be a
        # demonstration/mock-up. It is not treated as an AI-editing signal.
        add(
            "document_marking",
            min(18, 8 + len(markings) * 3),
            "Document contains sample/demo/not-for-official-use marking.",
        )
        evidence.extend(markings[:10])

    # Face count is informational only.
    raw_score = sum(item["points"] for item in components)
    score = int(round(min(100.0, max(0.0, raw_score))))

    if score <= 20:
        level = "LOW RISK"
        screening_status = "LOW RISK"
    elif score <= 50:
        level = "MEDIUM RISK"
        screening_status = "REVIEW"
    elif score <= 75:
        level = "HIGH RISK"
        screening_status = "SUSPICIOUS"
    else:
        level = "CRITICAL REVIEW"
        screening_status = "HIGH RISK"

    # Unknown documents must never be presented as a confident VALID result.
    if category == "UNKNOWN":
        if score < 21:
            score = 21
        level = "MEDIUM RISK"
        screening_status = "REVIEW"

    # Explicit hard contradictions deserve review even if the weighted total
    # happens to remain low.
    if (
        validation.get("overall_status") == "FAILED"
        or cross.get("status") == "FAILED"
        or qr_status == "DATA_MISMATCH"
    ) and score < 51:
        score = 51
        level = "HIGH RISK"
        screening_status = "SUSPICIOUS"

    return score, level, signals, {
        "components": components,
        "evidence": evidence[:60],
        "screening_status": screening_status,
        "raw_score": round(raw_score, 2),
        "scoring_policy": "Evidence-weighted; weak forensic artifacts are capped and QR absence/face count are non-risk signals.",
    }


def build_validation_results(file_format_valid, structure_valid, analysis):
    """Build accurate UI validation labels without implying unsupported checks."""
    validation = analysis.get("document_validation") or {}
    cross = analysis.get("cross_field_consistency") or {}
    tampering = analysis.get("tampering_analysis") or {}
    qr = analysis.get("qr_analysis") or {}
    quality = analysis.get("image_quality") or {}

    ocr_conf = float(analysis.get("ocr_confidence", 0) or 0)
    category = analysis.get("document_category", "UNKNOWN")
    detection_conf = str(
        analysis.get("document_detection_confidence", "LOW")
    ).upper()

    if category == "UNKNOWN":
        document_status = "REVIEW REQUIRED"
    else:
        document_status = analysis.get("document_label", "Unknown Document")

    qr_status = str(qr.get("status", "NOT_AVAILABLE")).upper()
    if qr_status == "NOT_PRESENT":
        qr_ui = "NOT PRESENT"
    elif qr_status == "NOT_DECODED":
        qr_ui = "DETECTED — NOT DECODED"
    elif qr_status == "DATA_MISMATCH":
        qr_ui = "MISMATCH"
    elif qr_status == "DECODED":
        qr_ui = "DECODED"
    else:
        qr_ui = "NOT AVAILABLE"

    if tampering.get("status") == "HIGH_REVIEW_REQUIRED":
        forensic_ui = "HIGH REVIEW"
    elif tampering.get("status") == "REVIEW_RECOMMENDED":
        forensic_ui = "REVIEW REQUIRED"
    else:
        forensic_ui = "NO STRONG SIGNAL"

    return [
        {
            "name": "File format / signature check",
            "status": "PASSED" if file_format_valid else "FAILED",
        },
        {
            "name": "Document structure check",
            "status": "PASSED" if structure_valid else "FAILED",
        },
        {
            "name": "OCR readability check",
            "status": "PASSED" if analysis.get("ocr_status") == "TEXT_DETECTED" and ocr_conf >= 45 else "REVIEW REQUIRED",
        },
        {
            "name": "Document type detection",
            "status": document_status,
        },
        {
            "name": "Document type confidence",
            "status": detection_conf,
        },
        {
            "name": "Document-specific validation",
            "status": validation.get("overall_status", "REVIEW") if category != "UNKNOWN" else "REVIEW",
        },
        {
            "name": "Cross-field consistency",
            "status": cross.get("status", "REVIEW"),
        },
        {
            "name": "QR verification",
            "status": qr_ui,
        },
        {
            "name": "Image quality",
            "status": quality.get("status", "NOT APPLICABLE"),
        },
        {
            "name": "Tampering / forensic screening",
            "status": forensic_ui,
        },
    ]


# ============================================================
# FINAL OCR IDENTIFIER RECOVERY OVERRIDE
# ============================================================
def _targeted_identifier_ocr(image):
    """Fast identifier recovery for small numbers near the lower document edge."""
    results = []
    rgb = None
    try:
        rgb = fix_orientation(image).convert("RGB")
        width, height = rgb.size
        language = get_ocr_language()
        for name, start_fraction in [("identifier_lower", 0.55), ("identifier_bottom", 0.72)]:
            crop = None
            enlarged = None
            try:
                crop = rgb.crop((0, int(height * start_fraction), width, height))
                enlarged = crop.resize((max(1, crop.width * 2), max(1, crop.height * 2)), Image.Resampling.LANCZOS)
                for pass_no, config in enumerate(("--oem 3 --psm 11", "--oem 3 --psm 12"), start=1):
                    try:
                        result = run_ocr_pass(enlarged, config, language)
                        text = normalize_text(result.get("text", ""))
                        if text:
                            results.append({
                                "name": f"{name}_psm_{pass_no}",
                                "text": text,
                                "confidence": float(result.get("confidence", 0) or 0),
                                "tokens": result.get("tokens") or [],
                            })
                    except Exception as error:
                        print("IDENTIFIER OCR PASS ERROR:", str(error))
            finally:
                safe_close(enlarged)
                safe_close(crop)
    except Exception as error:
        print("IDENTIFIER OCR RECOVERY ERROR:", str(error))
    finally:
        safe_close(rgb)
    return results


# Final OCR wrapper: keep the original fast Tesseract pipeline as the primary path.
_FINAL_EXTRACT_OCR_DATA = _LEGACY_EXTRACT_OCR_DATA

def extract_ocr_data(image):
    result = _FINAL_EXTRACT_OCR_DATA(image) or {}
    text = normalize_text(result.get("raw_ocr_text", ""))
    category = result.get("document_category") or result.get("document_detection", {}).get("document_category", "UNKNOWN")
    structured = result.get("structured_data") or {}
    confidence = float(result.get("ocr_confidence", 0) or 0)

    # Reclassify the primary OCR result using the final detector.
    detection = detect_document_type(text)
    category = detection.get("document_category", category)
    try:
        refreshed = extract_fields_from_text(text, detection) or {}
        for key, value in structured.items():
            if value and not refreshed.get(key):
                refreshed[key] = value
        structured = refreshed
    except Exception:
        pass

    result.update({
        "document_detection": detection,
        "document_category": category,
        "document_label": detection.get("document_label", "Unknown Document"),
        "document_detection_confidence": detection.get("confidence", "LOW"),
        "structured_data": structured,
        "extracted_data": structured,
        "extracted": structured,
    })

    identifier_missing = (
        (category == "AADHAAR_CARD" and not structured.get("aadhaar_number"))
        or (category == "PAN_CARD" and not structured.get("pan_number"))
        or (category == "PASSPORT" and not structured.get("passport_number"))
        or (category == "DRIVING_LICENCE" and not structured.get("driving_licence_number"))
        or (category == "VOTER_ID" and not structured.get("voter_id_number"))
        or (category == "GST_DOCUMENT" and not structured.get("gstin"))
    )

    # Only two targeted OCR passes are added when an important identifier is
    # missing. This fixes small bottom-of-card numbers without making every
    # upload run a large preprocessing/OCR matrix.
    if identifier_missing or not text or confidence < 35:
        targeted = _targeted_identifier_ocr(image)
        if targeted:
            combined = normalize_text(
                (text + "\n" + "\n".join(x.get("text", "") for x in targeted)).strip()
            )[:MAX_OCR_TEXT_LENGTH]
            detection = detect_document_type(combined)
            merged = extract_fields_from_text(combined, detection) or {}
            for key, value in structured.items():
                if value and not merged.get(key):
                    merged[key] = value
            merged = _merge_identifier_fields(merged, combined, detection.get("document_category", "UNKNOWN"))
            result.update({
                "raw_ocr_text": combined,
                "extracted_text": combined,
                "ocr_status": "TEXT_DETECTED" if combined else "NO_TEXT_DETECTED",
                "document_detection": detection,
                "document_category": detection.get("document_category", "UNKNOWN"),
                "document_label": detection.get("document_label", "Unknown Document"),
                "document_detection_confidence": detection.get("confidence", "LOW"),
                "structured_data": merged,
                "extracted_data": merged,
                "extracted": merged,
                "ocr_recovery_passes": [x.get("name") for x in targeted],
                "ocr_recovery_text": "\n".join(x.get("text", "") for x in targeted),
                "ocr_tokens": list(result.get("ocr_tokens") or []) + [
                    token for item in targeted for token in (item.get("tokens") or [])
                ],
            })
            result["ocr_confidence"] = max(
                float(result.get("ocr_confidence", 0) or 0),
                max((float(x.get("confidence", 0) or 0) for x in targeted), default=0.0),
            )

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
