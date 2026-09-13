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


# ============================================================
# FINAL CONSOLIDATED ENGINE
# ============================================================
# The previous version had duplicate public functions later in the
# file. The definitions at the bottom silently replaced earlier logic.
# This section is the single canonical implementation used by /upload.
# ============================================================

_VERHOEFF_D = (
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

_VERHOEFF_P = (
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
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) != 12:
        return False
    checksum = 0
    for index, char in enumerate(reversed(digits)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[index % 8][int(char)]]
    return checksum == 0


def _parse_date_value(value):
    if not value:
        return None
    value = re.sub(r"\s+", "", str(value).strip().replace("/", "-").replace(".", "-"))
    for fmt in ("%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y"):
        try:
            parsed = datetime.strptime(value, fmt).date()
            if 1900 <= parsed.year <= 2100:
                return parsed
        except Exception:
            pass
    return None


def _date_is_valid(value):
    return _parse_date_value(value) is not None


# ============================================================
# UNIVERSAL DOCUMENT DETECTION
# ============================================================

FINAL_DOCUMENT_RULES = {
    "AADHAAR_CARD": {
        "keywords": ["AADHAAR", "AADHAR", "UIDAI", "UNIQUE IDENTIFICATION AUTHORITY", "MY AADHAAR", "मेरा आधार", "आधार"],
        "strong": [r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)"],
        "support": [r"GOVERNMENT\s+OF\s+INDIA", r"DATE\s+OF\s+BIRTH", r"DOB", r"MALE|FEMALE"],
    },
    "PAN_CARD": {
        "keywords": ["PERMANENT ACCOUNT NUMBER", "INCOME TAX DEPARTMENT", "INCOME TAX"],
        "strong": [r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"],
        "support": [r"PERMANENT\s+ACCOUNT", r"SIGNATURE", r"DATE\s+OF\s+BIRTH"],
    },
    "PASSPORT": {
        "keywords": ["PASSPORT", "REPUBLIC OF INDIA", "NATIONALITY", "PLACE OF BIRTH", "DATE OF EXPIRY"],
        "strong": [r"\b[A-Z][0-9]{7}\b", r"P<[A-Z]{3}"],
        "support": [r"SURNAME", r"GIVEN\s+NAMES?", r"PERSONAL\s+NUMBER", r"TYPE\s*/\s*TYPE"],
    },
    "DRIVING_LICENCE": {
        "keywords": ["DRIVING LICENCE", "DRIVING LICENSE", "LICENCE TO DRIVE", "LICENSE TO DRIVE", "TRANSPORT DEPARTMENT", "MINISTRY OF ROAD TRANSPORT"],
        "strong": [r"\b[A-Z]{2}[- ]?\d{2}[- ]?[A-Z0-9]{6,16}\b"],
        "support": [r"LICEN[CS]E\s+NO", r"VALID\s+TILL", r"DATE\s+OF\s+ISSUE", r"DOB"],
    },
    "VOTER_ID": {
        "keywords": ["ELECTION COMMISSION OF INDIA", "ELECTION COMMISSION", "ELECTOR PHOTO IDENTITY CARD", "ELECTOR", "EPIC"],
        "strong": [r"\b[A-Z]{3}[0-9]{7}\b"],
        "support": [r"VOTER", r"POLLING", r"CONSTITUENCY"],
    },
    "VISA": {
        "keywords": ["VISA", "VISA NUMBER", "TYPE OF VISA", "VALID FROM", "VALID UNTIL", "VALID TO", "ENTRIES"],
        "strong": [],
        "support": [r"DURATION", r"NUMBER OF ENTRIES", r"ISSUED"],
    },
    "PERMIT": {
        "keywords": ["RESIDENCE PERMIT", "WORK PERMIT", "ENTRY PERMIT", "PERMIT"],
        "strong": [r"(?:PERMIT|DOCUMENT)\s*(?:NO|NUMBER)\s*[:#-]?\s*[A-Z0-9-]{5,20}"],
        "support": [r"VALID\s+FROM", r"VALID\s+UNTIL", r"DATE\s+OF\s+EXPIRY"],
    },
    "IDENTITY_CARD": {
        "keywords": ["IDENTITY CARD", "IDENTIFICATION CARD", "NATIONAL IDENTITY CARD", "NATIONAL ID", "EMPLOYEE ID", "STUDENT ID", "COLLEGE ID", "UNIVERSITY ID"],
        "strong": [r"(?:ID|IDENTITY|CARD)\s*(?:NO|NUMBER)\s*[:#-]?\s*[A-Z0-9-]{4,20}"],
        "support": [r"NAME", r"DATE\s+OF\s+BIRTH", r"GENDER", r"ADDRESS"],
    },
    "RATION_CARD": {
        "keywords": ["RATION CARD", "PUBLIC DISTRIBUTION SYSTEM", "FOOD AND CIVIL SUPPLIES", "FAMILY ID"],
        "strong": [], "support": [r"FAMILY", r"HOUSEHOLD", r"FPS", r"RATIONS"],
    },
    "GST_DOCUMENT": {
        "keywords": ["GOODS AND SERVICES TAX", "GST REGISTRATION", "GSTIN", "GST CERTIFICATE"],
        "strong": [r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b"],
        "support": [r"TAXPAYER", r"REGISTRATION", r"GST"],
    },
    "INVOICE": {
        "keywords": ["INVOICE", "TAX INVOICE", "BILL TO", "AMOUNT DUE", "TOTAL AMOUNT", "SUBTOTAL"],
        "strong": [r"(?:INVOICE|BILL)\s*(?:NO|NUMBER)\s*[:#-]?\s*[A-Z0-9/-]{3,30}"],
        "support": [r"QTY", r"RATE", r"TOTAL", r"GST"],
    },
    "MARKSHEET": {
        "keywords": ["MARKSHEET", "MARK SHEET", "STATEMENT OF MARKS", "MARKS OBTAINED", "TOTAL MARKS", "PERCENTAGE", "GRADE"],
        "strong": [], "support": [r"SUBJECT", r"RESULT", r"ROLL\s*(?:NO|NUMBER)", r"TOTAL"],
    },
    "CERTIFICATE": {
        "keywords": ["CERTIFICATE", "THIS IS TO CERTIFY", "CERTIFIED THAT", "CERTIFICATE OF"],
        "strong": [r"CERTIFICATE\s*(?:NO|NUMBER)\s*[:#-]?\s*[A-Z0-9/-]{4,30}"],
        "support": [r"ISSUED", r"SIGNATURE", r"AUTHORIZED"],
    },
    "BANK_DOCUMENT": {
        "keywords": ["BANK STATEMENT", "ACCOUNT HOLDER", "IFSC", "ACCOUNT NUMBER", "BANK", "BRANCH"],
        "strong": [r"\b[A-Z]{4}0[A-Z0-9]{6}\b"],
        "support": [r"ACCOUNT\s+NUMBER", r"TRANSACTION", r"BALANCE", r"DEBIT", r"CREDIT"],
    },
}


def detect_document_type(text):
    text = normalize_text(text)
    upper = text.upper()
    scores = {}
    evidence = {}

    for category, rules in FINAL_DOCUMENT_RULES.items():
        keyword_hits = [k for k in rules["keywords"] if k.upper() in upper]
        strong_hits = []
        support_hits = []

        for pattern in rules["strong"]:
            try:
                if re.search(pattern, upper, re.IGNORECASE):
                    strong_hits.append(pattern)
            except Exception:
                pass

        for pattern in rules["support"]:
            try:
                if re.search(pattern, upper, re.IGNORECASE):
                    support_hits.append(pattern)
            except Exception:
                pass

        score = len(keyword_hits) * 3 + len(support_hits) * 1.5 + len(strong_hits) * 6

        # Unique identifier bonuses prevent a document becoming UNKNOWN just
        # because the OCR missed a header word.
        if category == "AADHAAR_CARD" and re.search(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)", upper):
            score += 8
        if category == "PAN_CARD" and re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", upper):
            score += 8
        if category == "PASSPORT" and re.search(r"P<[A-Z]{3}", upper):
            score += 8
        if category == "GST_DOCUMENT" and re.search(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b", upper):
            score += 8

        scores[category] = round(score, 2)
        evidence[category] = {
            "keyword_hits": keyword_hits,
            "strong_pattern_hits": len(strong_hits),
            "support_pattern_hits": len(support_hits),
        }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_category, best_score = ranked[0] if ranked else ("UNKNOWN", 0)
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    if best_score < 3:
        best_category = "UNKNOWN"
        best_score = 0

    margin = best_score - second_score
    if best_score >= 12 and margin >= 3:
        confidence = "HIGH"
    elif best_score >= 6 and margin >= 1.5:
        confidence = "MEDIUM"
    elif best_score >= 3:
        confidence = "LOW"
    else:
        confidence = "LOW"

    return {
        "document_category": best_category,
        "document_label": DOCUMENT_LABELS.get(best_category, "Unknown Document"),
        "confidence": confidence,
        "score": round(best_score, 2),
        "margin": round(margin, 2),
        "all_scores": scores,
        "evidence": evidence.get(best_category, {}),
    }


# ============================================================
# OCR-TOLERANT IDENTIFIER RECOVERY
# ============================================================

def _normalize_digit_ocr(value):
    replacements = {"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"}
    return "".join(replacements.get(c, c) for c in str(value or "").upper())


def extract_aadhaar_number(text):
    upper = str(text or "").upper()
    patterns = [
        r"(?<!\d)(\d{4}[ -]?\d{4}[ -]?\d{4})(?!\d)",
        r"(?<!\d)([0-9OQILZSB]{4}[ -]?[0-9OQILZSB]{4}[ -]?[0-9OQILZSB]{4})(?!\d)",
    ]

    for pattern in patterns:
        try:
            matches = re.findall(pattern, upper)
        except Exception:
            matches = []
        for value in matches:
            digits = re.sub(r"\D", "", _normalize_digit_ocr(value))
            if len(digits) == 12:
                return f"{digits[:4]} {digits[4:8]} {digits[8:]}"

    if any(k in upper for k in ("AADHAAR", "AADHAR", "UIDAI", "आधार")):
        compact = re.sub(r"\s+", "", upper)
        for run in re.findall(r"[0-9OQILZSB]{12,16}", compact):
            digits = re.sub(r"\D", "", _normalize_digit_ocr(run))
            if len(digits) == 12:
                return f"{digits[:4]} {digits[4:8]} {digits[8:]}"

    return None


def extract_pan_number(text):
    upper = str(text or "").upper()
    match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", upper)
    return match.group(0) if match else None


# ============================================================
# UNIVERSAL STRUCTURED EXTRACTION
# ============================================================

def _extract_generic_address(text):
    lines = split_clean_lines(text)
    for index, line in enumerate(lines):
        if re.match(r"^(?:ADDRESS|RESIDENTIAL ADDRESS|PERMANENT ADDRESS)\b", line, re.I):
            value = re.sub(r"^(?:ADDRESS|RESIDENTIAL ADDRESS|PERMANENT ADDRESS)\s*[:.-]?\s*", "", line, flags=re.I).strip()
            if value and len(value) >= 6:
                return value[:250]
            parts = []
            for next_line in lines[index + 1:index + 4]:
                if re.search(r"\b(?:DOB|DATE|GENDER|SEX|NAME|SIGNATURE|MOBILE|PHONE)\b", next_line, re.I):
                    break
                parts.append(next_line)
            if parts:
                return ", ".join(parts)[:250]
    return None


def _extract_parent_name(text):
    for pattern in (
        r"(?:FATHER(?:'S)? NAME|FATHER NAME|S/O|D/O|W/O|C/O)\s*[:.-]?\s*([A-Za-z][A-Za-z .'-]{2,70})",
        r"(?:MOTHER(?:'S)? NAME|MOTHER NAME)\s*[:.-]?\s*([A-Za-z][A-Za-z .'-]{2,70})",
    ):
        match = re.search(pattern, str(text or ""), re.I)
        if match:
            value = clean_field_value(match.group(1))
            if is_valid_name(value):
                return value
    return None


def _generic_document_number(text, category):
    patterns = {
        "PASSPORT": r"(?:PASSPORT\s*(?:NO|NUMBER)|PERSONAL\s*(?:NO|NUMBER))\s*[:#.-]?\s*([A-Z0-9]{6,12})",
        "DRIVING_LICENCE": r"(?:LICEN[CS]E\s*(?:NO|NUMBER)|DL\s*(?:NO|NUMBER))\s*[:#.-]?\s*([A-Z0-9-]{8,24})",
        "VOTER_ID": r"(?:EPIC|VOTER\s*(?:ID|NO|NUMBER))\s*[:#.-]?\s*([A-Z0-9]{6,16})",
        "VISA": r"VISA\s*(?:NO|NUMBER)\s*[:#.-]?\s*([A-Z0-9-]{5,20})",
        "PERMIT": r"PERMIT\s*(?:NO|NUMBER)\s*[:#.-]?\s*([A-Z0-9-]{5,24})",
    }
    pattern = patterns.get(category)
    if not pattern:
        return None
    match = re.search(pattern, str(text or "").upper(), re.I)
    return clean_field_value(match.group(1)) if match else None


def _safe_extract_name(text):
    """Extract a person name without accepting document labels as names."""
    lines = split_clean_lines(text)
    blocked = {
        "name", "permanent account number", "income tax department",
        "government of india", "aadhaar", "aadhar", "passport",
        "identity card", "driving licence", "driving license",
        "date of birth", "date of issue", "date of expiry",
        "nationality", "address", "signature", "gender", "male", "female",
    }

    def acceptable(value):
        value = clean_field_value(value)
        if not value:
            return None
        value = re.sub(r"^(?:NAME|FULL NAME|CARDHOLDER|CARD HOLDER)\s*[:.-]?\s*", "", value, flags=re.I)
        value = re.split(r"\b(?:DOB|DATE OF BIRTH|DATE OF ISSUE|DATE OF EXPIRY|GENDER|SEX|ADDRESS|SIGNATURE)\b", value, flags=re.I)[0].strip(" :-|,")
        if not value or value.lower() in blocked:
            return None
        if not is_valid_name(value):
            return None
        if any(phrase in value.lower() for phrase in blocked if len(phrase) > 3):
            return None
        return normalize_name(value)

    explicit = re.compile(r"^(?:NAME|FULL NAME|CARDHOLDER|CARD HOLDER)\s*[:.-]?\s*(.*)$", re.I)
    for index, line in enumerate(lines):
        match = explicit.match(line)
        if not match:
            continue
        candidate = acceptable(match.group(1))
        if candidate:
            return candidate
        if index + 1 < len(lines):
            candidate = acceptable(lines[index + 1])
            if candidate:
                return candidate

    surname = None
    given = None
    for index, line in enumerate(lines):
        if re.match(r"^(?:SURNAME|NOM)\b", line, re.I):
            value = re.sub(r"^(?:SURNAME|NOM)\s*[:.-]?\s*", "", line, flags=re.I)
            surname = acceptable(value) or (acceptable(lines[index + 1]) if index + 1 < len(lines) else None)
        if re.match(r"^(?:GIVEN NAMES?|FORENAMES?)\b", line, re.I):
            value = re.sub(r"^(?:GIVEN NAMES?|FORENAMES?)\s*[:.-]?\s*", "", line, flags=re.I)
            given = acceptable(value) or (acceptable(lines[index + 1]) if index + 1 < len(lines) else None)
    if given and surname:
        return acceptable(f"{given} {surname}")
    if given:
        return given
    if surname:
        return surname

    # Positional fallback: use only a short, plausible line near a date or gender.
    for index, line in enumerate(lines):
        upper = line.upper()
        if re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", line) or upper in {"MALE", "FEMALE", "पुरुष", "महिला"}:
            for previous in range(max(0, index - 3), index):
                candidate = acceptable(lines[previous])
                if candidate:
                    return candidate
    return None


def _generic_document_number(text, category):
    patterns = {
        "PASSPORT": r"(?:PASSPORT\s*(?:NO|NUMBER)|PERSONAL\s*(?:NO|NUMBER))\s*[:#.-]?\s*([A-Z][A-Z0-9]{5,11})",
        "DRIVING_LICENCE": r"(?:LICEN[CS]E\s*(?:NO|NUMBER)|DL\s*(?:NO|NUMBER))\s*[:#.-]?\s*([A-Z0-9-]{8,24})",
        "VOTER_ID": r"(?:EPIC|VOTER\s*(?:ID|NO|NUMBER))\s*[:#.-]?\s*([A-Z0-9]{6,16})",
        "VISA": r"(?:VISA\s*(?:NO|NUMBER))\s*[:#.-]?\s*([A-Z0-9-]{5,20})",
        "PERMIT": r"(?:PERMIT\s*(?:NO|NUMBER))\s*[:#.-]?\s*([A-Z0-9-]{5,24})",
    }
    pattern = patterns.get(category)
    if pattern:
        match = re.search(pattern, str(text or "").upper(), re.I)
        if match:
            return clean_field_value(match.group(1))
    return None


def extract_fields_from_text(text, detection):
    category = detection.get("document_category", "UNKNOWN")
    text = normalize_text(text)

    name = _safe_extract_name(text)
    try: dob = extract_date_of_birth(text)
    except Exception: dob = None
    try: issue = extract_date_of_issue(text)
    except Exception: issue = None
    try: expiry = extract_date_of_expiry(text)
    except Exception: expiry = None
    try: gender = extract_gender(text)
    except Exception: gender = None
    try: nationality = extract_nationality(text) if category == "PASSPORT" else None
    except Exception: nationality = None

    aadhaar = extract_aadhaar_number(text) if category == "AADHAAR_CARD" or re.search(r"AADHAAR|AADHAR|UIDAI", text, re.I) else None
    pan = extract_pan_number(text) if category == "PAN_CARD" or re.search(r"PERMANENT ACCOUNT|INCOME TAX", text, re.I) else None

    dl = None
    passport = None
    voter = None
    gstin = None
    visa = None
    visa_type = None
    stay = None
    permit = None

    if category == "DRIVING_LICENCE":
        try: dl = extract_driving_licence_number(text)
        except Exception: dl = None
        if not dl: dl = _generic_document_number(text, category)

    elif category == "PASSPORT":
        try: passport = extract_passport_number(text)
        except Exception: passport = None
        if not passport: passport = _generic_document_number(text, category)
        if not passport:
            direct = re.search(r"\b[A-Z][0-9]{7}\b", text.upper())
            passport = direct.group(0) if direct else None

    elif category == "VOTER_ID":
        try: voter = extract_voter_id_number(text)
        except Exception: voter = None
        if not voter: voter = _generic_document_number(text, category)
        if not voter:
            direct = re.search(r"\b[A-Z]{3}[0-9]{7}\b", text.upper())
            voter = direct.group(0) if direct else None

    elif category == "GST_DOCUMENT":
        try: gstin = extract_gstin(text)
        except Exception: gstin = None

    elif category == "VISA":
        try: visa = extract_visa_number(text)
        except Exception: visa = None
        if not visa: visa = _generic_document_number(text, category)
        try: visa_type = extract_visa_type(text)
        except Exception: visa_type = None
        try: stay = extract_stay_duration(text)
        except Exception: stay = None

    elif category == "PERMIT":
        permit = _generic_document_number(text, category)

    return {
        "name": name,
        "document": detection.get("document_label", "Unknown Document"),
        "document_category": category,
        "aadhaar_number": aadhaar,
        "pan_number": pan,
        "driving_licence_number": dl,
        "passport_number": passport,
        "voter_id_number": voter,
        "gstin": gstin,
        "visa_number": visa,
        "visa_type": visa_type,
        "stay_duration": stay,
        "permit_number": permit,
        "nationality": nationality,
        "date_of_birth": dob,
        "gender": gender,
        "date_of_issue": issue,
        "date_of_expiry": expiry,
        "validity_status": extract_validity_status(expiry),
        "address": _extract_generic_address(text),
        "parent_name": _extract_parent_name(text),
    }


# ============================================================
# FIELD + DOCUMENT VALIDATION
# ============================================================

def _validate_field(field, value):
    if value is None or not str(value).strip():
        return {"status": "NOT_PRESENT", "valid": None, "reason": "Field was not extracted"}

    value = str(value).strip()

    if field == "name":
        valid = is_valid_name(value)
        return {"status": "PASSED" if valid else "FAILED", "valid": valid, "reason": "Name is plausible" if valid else "Name is not plausible"}

    if field == "aadhaar_number":
        digits = re.sub(r"\D", "", value)
        if len(digits) != 12:
            return {"status": "FAILED", "valid": False, "reason": "Aadhaar number is not 12 digits"}
        valid = is_valid_aadhaar_number(digits)
        return {"status": "PASSED" if valid else "FAILED", "valid": valid, "reason": "Verhoeff checksum passed" if valid else "Verhoeff checksum failed"}

    if field == "pan_number":
        valid = bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", value.upper()))
        return {"status": "PASSED" if valid else "FAILED", "valid": valid, "reason": "PAN pattern is valid" if valid else "PAN pattern is invalid"}

    if field == "passport_number":
        compact = re.sub(r"[^A-Z0-9]", "", value.upper())
        valid = bool(6 <= len(compact) <= 12 and compact[0].isalpha() and any(c.isdigit() for c in compact))
        return {"status": "PASSED" if valid else "FAILED", "valid": valid, "reason": "Passport number structure is plausible" if valid else "Passport number structure is invalid"}

    if field == "driving_licence_number":
        compact = re.sub(r"[^A-Z0-9]", "", value.upper())
        valid = bool(8 <= len(compact) <= 24 and compact[:2].isalpha() and any(c.isdigit() for c in compact))
        return {"status": "PASSED" if valid else "FAILED", "valid": valid, "reason": "Driving licence number structure is plausible" if valid else "Driving licence number structure is invalid"}

    if field == "voter_id_number":
        compact = re.sub(r"[^A-Z0-9]", "", value.upper())
        valid = bool(re.fullmatch(r"[A-Z]{3}[0-9]{7}", compact))
        return {"status": "PASSED" if valid else "FAILED", "valid": valid, "reason": "Voter ID pattern is valid" if valid else "Voter ID pattern is invalid"}

    if field == "gstin":
        compact = re.sub(r"\s+", "", value.upper())
        valid = bool(re.fullmatch(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]", compact))
        return {"status": "PASSED" if valid else "FAILED", "valid": valid, "reason": "GSTIN pattern is valid" if valid else "GSTIN pattern is invalid"}

    if field in {"date_of_birth", "date_of_issue", "date_of_expiry"}:
        valid = _date_is_valid(value)
        return {"status": "PASSED" if valid else "FAILED", "valid": valid, "reason": "Date is parseable" if valid else "Date is invalid or unreadable"}

    return {"status": "PASSED", "valid": True, "reason": "Value is present"}


def validate_document_specific(category, structured, qr_result=None):
    structured = structured or {}
    qr_result = qr_result or {}
    checks = []
    failures = []
    reviews = []

    def add(name, status, reason, weight=1):
        item = {"name": name, "status": status, "reason": reason, "weight": weight}
        checks.append(item)
        if status == "FAILED": failures.append(item)
        elif status in {"REVIEW", "NOT_PRESENT"}: reviews.append(item)

    for field, label in (
        ("date_of_birth", "Date of birth"),
        ("date_of_issue", "Date of issue"),
        ("date_of_expiry", "Date of expiry"),
    ):
        value = structured.get(field)
        if value:
            v = _validate_field(field, value)
            add(f"{label} format", v["status"], v["reason"])

    dob = _parse_date_value(structured.get("date_of_birth"))
    issue = _parse_date_value(structured.get("date_of_issue"))
    expiry = _parse_date_value(structured.get("date_of_expiry"))
    today = date.today()

    if dob:
        add("DOB chronology", "FAILED" if dob > today else "PASSED", "DOB is in the future" if dob > today else "DOB is not in the future", 2)

    if issue and expiry:
        add("Issue/expiry chronology", "FAILED" if expiry < issue else "PASSED", "Expiry date is earlier than issue date" if expiry < issue else "Issue date is not later than expiry date", 2)

    if issue and issue > today:
        add("Issue date chronology", "REVIEW", "Issue date is in the future")

    if category == "AADHAAR_CARD":
        value = structured.get("aadhaar_number")
        if value:
            v = _validate_field("aadhaar_number", value)
            add("Aadhaar checksum", v["status"], v["reason"], 3)
        else:
            add("Aadhaar number extraction", "REVIEW", "Aadhaar was detected but a complete 12-digit number was not extracted", 2)

        if qr_result.get("decoded"):
            if qr_result.get("data_consistent") is True:
                add("QR/document data consistency", "PASSED", "Decoded QR data matches an extracted document identifier", 3)
            elif qr_result.get("data_consistent") is False:
                add("QR/document data consistency", "FAILED", "Decoded QR data conflicts with extracted document data", 4)
            else:
                add("QR/document data consistency", "REVIEW", "QR decoded but no comparable identifier was available")
        else:
            add("QR verification", "NOT_PRESENT", "No readable QR code was available")

    elif category == "PAN_CARD":
        value = structured.get("pan_number")
        if value:
            v = _validate_field("pan_number", value)
            add("PAN number format", v["status"], v["reason"], 3)
        else:
            add("PAN number extraction", "REVIEW", "PAN detected but PAN number was not extracted", 2)

    elif category == "PASSPORT":
        value = structured.get("passport_number")
        if value:
            v = _validate_field("passport_number", value)
            add("Passport number format", v["status"], v["reason"], 3)
        else:
            add("Passport number extraction", "REVIEW", "Passport detected but passport number was not extracted", 2)

    elif category == "DRIVING_LICENCE":
        value = structured.get("driving_licence_number")
        if value:
            v = _validate_field("driving_licence_number", value)
            add("Driving licence number format", v["status"], v["reason"], 3)
        else:
            add("Driving licence number extraction", "REVIEW", "Driving licence detected but licence number was not extracted", 2)

    elif category == "VOTER_ID":
        value = structured.get("voter_id_number")
        if value:
            v = _validate_field("voter_id_number", value)
            add("Voter ID format", v["status"], v["reason"], 3)
        else:
            add("Voter ID extraction", "REVIEW", "Voter ID detected but EPIC number was not extracted", 2)

    elif category == "GST_DOCUMENT":
        value = structured.get("gstin")
        if value:
            v = _validate_field("gstin", value)
            add("GSTIN format", v["status"], v["reason"], 3)
        else:
            add("GSTIN extraction", "REVIEW", "GST document detected but GSTIN was not extracted", 2)

    elif category == "VISA":
        if structured.get("visa_number"):
            add("Visa number extraction", "PASSED", "Visa number was extracted", 2)
        else:
            add("Visa number extraction", "REVIEW", "Visa detected but visa number was not extracted", 1)

    elif category == "PERMIT":
        if structured.get("permit_number"):
            add("Permit number extraction", "PASSED", "Permit number was extracted", 2)
        else:
            add("Permit number extraction", "REVIEW", "Permit detected but permit number was not extracted", 1)

    else:
        meaningful = [k for k, v in structured.items() if v and k not in {"document", "document_category", "validity_status"}]
        if len(meaningful) >= 2:
            add("Structured field extraction", "PASSED", f"{len(meaningful)} useful fields were extracted")
        elif len(meaningful) == 1:
            add("Structured field extraction", "REVIEW", "Only one useful field was extracted")
        else:
            add("Structured field extraction", "REVIEW", "No reliable structured fields were extracted", 2)

    return {
        "overall_status": "FAILED" if failures else "REVIEW" if reviews else "PASSED",
        "checks": checks,
        "failures": failures,
        "warnings": reviews,
        "passed_count": sum(1 for x in checks if x["status"] == "PASSED"),
        "failed_count": len(failures),
        "review_count": len(reviews),
    }


def analyze_cross_field_consistency(structured):
    structured = structured or {}
    checks = []
    conflicts = []
    warnings = []

    def add(name, status, reason):
        item = {"name": name, "status": status, "reason": reason}
        checks.append(item)
        if status == "FAILED": conflicts.append(item)
        elif status == "REVIEW": warnings.append(item)

    dob = _parse_date_value(structured.get("date_of_birth"))
    issue = _parse_date_value(structured.get("date_of_issue"))
    expiry = _parse_date_value(structured.get("date_of_expiry"))
    today = date.today()

    if dob:
        add("DOB chronology", "FAILED" if dob > today else "PASSED", "DOB is in the future" if dob > today else "DOB is chronologically plausible")
    if dob and issue:
        add("DOB vs issue date", "FAILED" if issue < dob else "PASSED", "Issue date is earlier than DOB" if issue < dob else "Issue date follows DOB")
    if issue and expiry:
        add("Issue vs expiry", "FAILED" if expiry < issue else "PASSED", "Expiry date is earlier than issue date" if expiry < issue else "Expiry date follows issue date")
    if expiry:
        add("Expiry status", "REVIEW" if expiry < today else "PASSED", "Extracted document appears expired" if expiry < today else "Extracted expiry date has not passed")

    if structured.get("name"):
        v = _validate_field("name", structured["name"])
        add("Name plausibility", "PASSED" if v["valid"] else "FAILED", v["reason"])

    if structured.get("gender"):
        g = str(structured["gender"]).upper()
        add("Gender value", "PASSED" if g in {"MALE", "FEMALE", "OTHER", "M", "F"} else "REVIEW", "Recognized gender value" if g in {"MALE", "FEMALE", "OTHER", "M", "F"} else "Unusual gender value")

    if not checks:
        add("Cross-field consistency", "REVIEW", "Not enough structured data was available")

    return {
        "status": "FAILED" if conflicts else "REVIEW" if warnings else "PASSED",
        "checks": checks,
        "conflicts": conflicts,
        "warnings": warnings,
        "conflict_count": len(conflicts),
        "warning_count": len(warnings),
    }

# ============================================================
# FORENSIC SIGNALS
# ============================================================

def analyze_metadata_tampering(metadata):
    metadata = metadata or {}
    keywords = [
        "photoshop", "adobe", "gimp", "canva", "pixlr",
        "lightroom", "coreldraw", "illustrator", "affinity",
    ]
    matches = []
    for key, value in metadata.items():
        combined = f"{key} {value}".lower()
        for keyword in keywords:
            if keyword in combined and keyword not in matches:
                matches.append(keyword)
    return {
        "metadata_present": bool(metadata),
        "editing_software_signals": matches,
        "status": "REVIEW" if matches else "NO_STRONG_SIGNAL",
    }


def perform_error_level_analysis(image):
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
        if str(image.format or "").upper() not in {"JPEG", "JPG"}:
            result["status"] = "LIMITED_FOR_NON_JPEG"
            return result
        rgb = image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")
        a = np.asarray(rgb, dtype=np.uint8)
        b = np.asarray(recompressed, dtype=np.uint8)
        diff = cv2.absdiff(a, b)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        mean_difference = float(np.mean(gray_diff))
        max_difference = float(np.max(gray_diff))
        score = min(100.0, mean_difference * 3.5)
        safe_close(recompressed)
        safe_close(rgb)
        buffer.close()
        return {
            "available": True,
            "status": "REVIEW" if score >= 45 else "NORMAL",
            "score": round(score, 1),
            "mean_difference": round(mean_difference, 2),
            "max_difference": round(max_difference, 2),
        }
    except Exception as error:
        result["status"] = "ERROR"
        result["error"] = str(error)
        return result


def _forensic_metrics(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = cv2.absdiff(gray, blur)
    edges = cv2.Canny(gray, 70, 170)
    return {
        "noise": float(np.std(residual)),
        "edge": float(np.mean(edges > 0)),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean": float(np.mean(gray)),
    }


def analyze_image_region_consistency(image):
    result = {
        "available": False,
        "noise_score": 0.0,
        "edge_score": 0.0,
        "sharpness_score": 0.0,
        "suspicious_regions": [],
        "regions_analyzed": 0,
    }
    if not CV2_AVAILABLE:
        return result
    try:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        if width < 80 or height < 80:
            return result

        rows, columns = 4, 4
        bh = max(1, height // rows)
        bw = max(1, width // columns)
        regions = []

        for row in range(rows):
            for column in range(columns):
                y1, x1 = row * bh, column * bw
                y2 = height if row == rows - 1 else (row + 1) * bh
                x2 = width if column == columns - 1 else (column + 1) * bw
                block = gray[y1:y2, x1:x2]
                if block.size < 100:
                    continue
                metrics = _forensic_metrics(block)
                regions.append({
                    "row": row, "column": column,
                    "x": int(x1), "y": int(y1),
                    "width": int(x2 - x1), "height": int(y2 - y1),
                    **metrics,
                })

        if len(regions) < 4:
            return result

        values = {
            key: np.asarray([r[key] for r in regions], dtype=float)
            for key in ("noise", "edge", "sharpness")
        }
        medians = {key: float(np.median(value)) for key, value in values.items()}
        mads = {
            "noise": max(float(np.median(np.abs(values["noise"] - medians["noise"]))), 0.8),
            "edge": max(float(np.median(np.abs(values["edge"] - medians["edge"]))), 0.002),
            "sharpness": max(float(np.median(np.abs(values["sharpness"] - medians["sharpness"]))), 20.0),
        }

        suspicious = []
        noise_zs, edge_zs, sharp_zs = [], [], []

        for region in regions:
            nz = abs(region["noise"] - medians["noise"]) / mads["noise"]
            ez = abs(region["edge"] - medians["edge"]) / mads["edge"]
            sz = abs(region["sharpness"] - medians["sharpness"]) / mads["sharpness"]
            region["noise_deviation"] = round(nz, 2)
            region["edge_deviation"] = round(ez, 2)
            region["sharpness_deviation"] = round(sz, 2)
            noise_zs.append(min(nz, 8))
            edge_zs.append(min(ez, 8))
            sharp_zs.append(min(sz, 8))

            strong = [nz >= 4.0, ez >= 4.0, sz >= 4.0]
            if sum(strong) >= 2:
                suspicious.append({
                    "row": region["row"], "column": region["column"],
                    "x": region["x"], "y": region["y"],
                    "width": region["width"], "height": region["height"],
                    "reason": "Multiple local image statistics deviate from the document baseline",
                    "noise_deviation": region["noise_deviation"],
                    "edge_deviation": region["edge_deviation"],
                    "sharpness_deviation": region["sharpness_deviation"],
                })

        safe_close(rgb)
        return {
            "available": True,
            "noise_score": round(min(100.0, float(np.mean(noise_zs)) * 12), 1),
            "edge_score": round(min(100.0, float(np.mean(edge_zs)) * 12), 1),
            "sharpness_score": round(min(100.0, float(np.mean(sharp_zs)) * 10), 1),
            "suspicious_regions": suspicious[:12],
            "regions_analyzed": len(regions),
            "baseline": {k: round(v, 4) for k, v in medians.items()},
        }
    except Exception as error:
        result["error"] = str(error)
        return result


def analyze_text_region_for_tampering(image, x, y, w, h):
    if not CV2_AVAILABLE or image is None or w <= 2 or h <= 2:
        return {"suspicious": False, "score": 0.0, "reason": "Invalid or unavailable text region"}
    try:
        if hasattr(image, "convert"):
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            array = np.asarray(image)
            gray = array if array.ndim == 2 else cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape[:2]
        x, y, w, h = int(x), int(y), int(w), int(h)
        x1 = max(0, x - max(8, int(w * 0.8)))
        y1 = max(0, y - max(8, int(h * 1.2)))
        x2 = min(width, x + w + max(8, int(w * 0.8)))
        y2 = min(height, y + h + max(8, int(h * 1.2)))
        context = gray[y1:y2, x1:x2]
        if context.size < 100:
            return {"suspicious": False, "score": 0.0, "reason": "Insufficient local context"}

        rx1, ry1 = max(0, x - x1), max(0, y - y1)
        rx2, ry2 = min(context.shape[1], rx1 + w), min(context.shape[0], ry1 + h)
        roi = context[ry1:ry2, rx1:rx2]
        if roi.size < 50:
            return {"suspicious": False, "score": 0.0, "reason": "Insufficient text ROI"}

        mask = np.ones_like(context, dtype=np.uint8)
        mask[ry1:ry2, rx1:rx2] = 0
        ring_pixels = context[mask == 1]
        if ring_pixels.size < 50:
            return {"suspicious": False, "score": 0.0, "reason": "Insufficient surrounding context"}

        roi_metrics = _forensic_metrics(roi)
        ring_metrics = _forensic_metrics(ring_pixels.reshape(-1, 1))

        def ratio(a, b):
            return abs(a - b) / max(abs(b), 1e-6)

        noise_ratio = ratio(roi_metrics["noise"], ring_metrics["noise"])
        edge_ratio = ratio(roi_metrics["edge"], ring_metrics["edge"])
        sharp_ratio = ratio(roi_metrics["sharpness"], ring_metrics["sharpness"])
        brightness_difference = abs(roi_metrics["mean"] - ring_metrics["mean"])

        signals = []
        if noise_ratio >= 1.8:
            signals.append("local noise differs strongly from surrounding context")
        if edge_ratio >= 1.8:
            signals.append("local edge density differs strongly from surrounding context")
        if sharp_ratio >= 2.2:
            signals.append("local sharpness differs strongly from surrounding context")
        if brightness_difference >= 45:
            signals.append("local brightness differs strongly from surrounding context")

        suspicious = len(signals) >= 2
        raw_score = (
            min(noise_ratio / 3, 1) * .25
            + min(edge_ratio / 3, 1) * .25
            + min(sharp_ratio / 4, 1) * .25
            + min(brightness_difference / 80, 1) * .25
        )
        score = .55 + raw_score * .45 if suspicious else raw_score * .35

        return {
            "suspicious": suspicious,
            "score": round(min(max(score, 0), 1), 3),
            "reason": "; ".join(signals) if signals else "No strong local forensic signal",
            "region": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
            "signals": signals,
            "metrics": {
                "noise_ratio": round(noise_ratio, 3),
                "edge_ratio": round(edge_ratio, 3),
                "sharpness_ratio": round(sharp_ratio, 3),
                "brightness_difference": round(brightness_difference, 2),
            },
        }
    except Exception as error:
        return {"suspicious": False, "score": 0.0, "reason": f"Local analysis unavailable: {error}"}


def analyze_targeted_text_regions(image, ocr_tokens, structured):
    result = {"available": False, "suspicious": False, "suspicious_fields": [], "regions": [], "region_count": 0}
    if not CV2_AVAILABLE or image is None:
        return result

    tokens = [
        t for t in (ocr_tokens or [])
        if isinstance(t, dict) and t.get("text")
        and int(t.get("width", 0) or 0) > 2
        and int(t.get("height", 0) or 0) > 2
    ]
    if not tokens:
        return result

    target_fields = [
        "name", "date_of_birth", "aadhaar_number", "pan_number",
        "passport_number", "driving_licence_number", "voter_id_number",
        "gstin", "date_of_issue", "date_of_expiry",
    ]

    def norm(value):
        return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

    for field in target_fields:
        value = structured.get(field)
        value_norm = norm(value)
        if not value_norm:
            continue

        matching = []
        for token in tokens:
            token_norm = norm(token.get("text", ""))
            if not token_norm:
                continue
            if token_norm in value_norm or value_norm in token_norm:
                matching.append(token)
            elif any(
                token_norm == part
                for part in re.findall(r"[a-z0-9]+", str(value).lower())
                if len(part) >= 3
            ):
                matching.append(token)

        for token in matching[:4]:
            analysis = analyze_text_region_for_tampering(
                image,
                token.get("left", 0), token.get("top", 0),
                token.get("width", 0), token.get("height", 0),
            )
            result["regions"].append({
                "field": field,
                "text": token.get("text", ""),
                "confidence": token.get("confidence", 0),
                **analysis,
            })
            if analysis.get("suspicious"):
                result["suspicious"] = True
                if field not in result["suspicious_fields"]:
                    result["suspicious_fields"].append(field)

    result["available"] = bool(result["regions"])
    result["region_count"] = len(result["regions"])
    return result


def analyze_document_tampering(image, metadata, ocr_tokens=None, structured=None):
    metadata_analysis = analyze_metadata_tampering(metadata)
    ela = perform_error_level_analysis(image)
    consistency = analyze_image_region_consistency(image)
    targeted = analyze_targeted_text_regions(image, ocr_tokens or [], structured or {})

    signals = []
    evidence = []
    score = 0.0

    if metadata_analysis.get("editing_software_signals"):
        score += 18
        signals.append("Editing-software indicators found in image metadata")
        evidence.append({"source": "metadata", "strength": "MEDIUM", "details": metadata_analysis["editing_software_signals"]})

    if ela.get("available"):
        score += min(18, float(ela.get("score", 0) or 0) * .22)
        if ela.get("score", 0) >= 45:
            signals.append("JPEG recompression differences require review")
            evidence.append({"source": "ela", "strength": "LOW_TO_MEDIUM", "score": ela.get("score")})

    global_score = (
        float(consistency.get("noise_score", 0) or 0) * .30
        + float(consistency.get("edge_score", 0) or 0) * .30
        + float(consistency.get("sharpness_score", 0) or 0) * .20
    )
    score += min(16, global_score * .18)

    suspicious_regions = consistency.get("suspicious_regions", []) or []
    if suspicious_regions:
        score += min(18, len(suspicious_regions) * 4)
        signals.append(f"{len(suspicious_regions)} image region(s) show multiple local statistical deviations")
        evidence.append({"source": "region_consistency", "strength": "MEDIUM", "regions": suspicious_regions[:8]})

    suspicious_fields = targeted.get("suspicious_fields", []) or []
    if suspicious_fields:
        score += min(30, 12 + len(suspicious_fields) * 6)
        labels = {
            "name": "name", "date_of_birth": "date of birth", "aadhaar_number": "Aadhaar number",
            "pan_number": "PAN number", "passport_number": "passport number", "driving_licence_number": "driving licence number",
            "voter_id_number": "voter ID number", "gstin": "GSTIN", "date_of_issue": "issue date", "date_of_expiry": "expiry date",
        }
        readable = [labels.get(f, f) for f in suspicious_fields]
        signals.append("Local forensic variation around " + ", ".join(readable))
        evidence.append({"source": "targeted_text_regions", "strength": "MEDIUM_TO_HIGH", "fields": readable, "regions": targeted.get("regions", [])[:10]})

    probability = round(min(100, max(0, score)), 1)
    if probability >= 70 and len(evidence) >= 2:
        status, level = "HIGH_REVIEW_REQUIRED", "HIGH"
    elif probability >= 40 and evidence:
        status, level = "REVIEW_RECOMMENDED", "SUSPICIOUS"
    else:
        status, level = "NO_STRONG_TAMPERING_SIGNAL", "LOW"

    return {
        "status": status,
        "tampering_probability": probability,
        "risk_level": level,
        "signals": signals,
        "evidence": evidence,
        "metadata": metadata_analysis,
        "ela": ela,
        "region_consistency": consistency,
        "targeted_text_regions": targeted,
        "noise_analysis": {"score": consistency.get("noise_score", 0)},
        "edge_analysis": {"score": consistency.get("edge_score", 0)},
        "sharpness_analysis": {"score": consistency.get("sharpness_score", 0)},
        "suspicious_regions": suspicious_regions,
        "suspicious_region_count": len(suspicious_regions),
    }

# ============================================================
# FINAL OCR OVERRIDE
# ============================================================

def _ocr_candidate_score(candidate):
    text = normalize_text(candidate.get("text", ""))
    if not text:
        return -999
    confidence = float(candidate.get("confidence", 0) or 0)
    detection = candidate.get("detection", {}) or {}
    structured = candidate.get("structured", {}) or {}
    score = confidence * .55
    score += min(len(text), 1800) * .025
    score += float(detection.get("score", 0) or 0) * 2.5

    important = [
        "name", "aadhaar_number", "pan_number", "passport_number",
        "driving_licence_number", "voter_id_number", "gstin",
        "date_of_birth", "date_of_issue", "date_of_expiry", "gender",
    ]
    score += sum(5 for field in important if structured.get(field))

    useful = sum(
        1 for char in text
        if char.isalnum() or char.isspace() or char in ".,:/-#()&'"
    )
    score += useful / max(len(text), 1) * 18
    return round(score, 3)


def _make_candidate(result, variant, coordinate_variant, language):
    text = normalize_text(result.get("text", ""))
    detection = detect_document_type(text)
    structured = extract_fields_from_text(text, detection)
    candidate = {
        "text": text,
        "confidence": float(result.get("confidence", 0) or 0),
        "lines": result.get("lines", []),
        "line_objects": result.get("line_objects", []),
        "tokens": result.get("tokens", []),
        "variant": variant,
        "image_variant": coordinate_variant,
        "engine": "tesseract",
        "language": language,
        "detection": detection,
        "structured": structured,
    }
    candidate["score"] = _ocr_candidate_score(candidate)
    return candidate


def extract_ocr_data(image):
    candidates = []
    language = get_ocr_language()
    original = None
    enhanced = None
    temporary_images = []

    try:
        original = resize_for_ocr(
            fix_orientation(image).convert("RGB")
        )

        def add(result, variant, coordinate="original"):
            candidate = _make_candidate(
                result, variant, coordinate, language
            )
            if candidate.get("text"):
                candidates.append(candidate)

        # Always run two layouts: block text and sparse text.
        add(
            run_ocr_pass(original, "--oem 3 --psm 6", language),
            "original_psm6",
        )
        add(
            run_ocr_pass(original, "--oem 3 --psm 11", language),
            "original_psm11",
        )

        best = max(
            candidates,
            key=lambda x: x.get("score", -999),
            default=None,
        )

        weak = (
            best is None
            or float(best.get("confidence", 0) or 0) < 50
            or best.get("detection", {}).get("document_category") == "UNKNOWN"
            or len(normalize_text(best.get("text", ""))) < 35
        )

        if weak:
            enhanced = create_enhanced_gray(original)
            temporary_images.append(enhanced)
            add(
                run_ocr_pass(enhanced, "--oem 3 --psm 6", language),
                "enhanced_psm6",
                "enhanced",
            )
            add(
                run_ocr_pass(enhanced, "--oem 3 --psm 11", language),
                "enhanced_psm11",
                "enhanced",
            )

        best = max(
            candidates,
            key=lambda x: x.get("score", -999),
            default=None,
        )

        difficult = (
            best is None
            or float(best.get("confidence", 0) or 0) < 42
            or best.get("detection", {}).get("document_category") == "UNKNOWN"
        )

        if difficult and CV2_AVAILABLE:
            if enhanced is None:
                enhanced = create_enhanced_gray(original)
                temporary_images.append(enhanced)

            arr = np.asarray(enhanced, dtype=np.uint8)

            threshold_images = []
            try:
                otsu = cv2.threshold(
                    arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]
                threshold_images.append(("otsu", Image.fromarray(otsu)))
            except Exception:
                pass

            try:
                adaptive = cv2.adaptiveThreshold(
                    arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 8
                )
                threshold_images.append(("adaptive", Image.fromarray(adaptive)))
            except Exception:
                pass

            for name, threshold_image in threshold_images:
                temporary_images.append(threshold_image)
                add(
                    run_ocr_pass(threshold_image, "--oem 3 --psm 6", language),
                    f"{name}_psm6",
                    name,
                )
                add(
                    run_ocr_pass(threshold_image, "--oem 3 --psm 11", language),
                    f"{name}_psm11",
                    name,
                )

        # Paddle is independent evidence, not the coordinate source.
        if PADDLE_AVAILABLE and (
            not candidates
            or max(x.get("score", -999) for x in candidates) < 60
        ):
            try:
                paddle = run_paddle_ocr_pass(original)
                ptext = normalize_text(paddle.get("text", ""))
                if ptext:
                    pdet = detect_document_type(ptext)
                    pstructured = extract_fields_from_text(ptext, pdet)
                    pc = {
                        "text": ptext,
                        "confidence": float(paddle.get("confidence", 0) or 0),
                        "lines": paddle.get("lines", []),
                        "line_objects": paddle.get("lines", []),
                        "tokens": [],
                        "variant": "paddleocr",
                        "image_variant": "original",
                        "engine": "paddleocr",
                        "language": "en",
                        "detection": pdet,
                        "structured": pstructured,
                    }
                    pc["score"] = _ocr_candidate_score(pc)
                    candidates.append(pc)
            except Exception as error:
                print("PADDLE OCR CANDIDATE ERROR:", str(error))

        if not candidates:
            return {
                "extracted_text": "", "raw_ocr_text": "",
                "ocr_confidence": 0.0, "ocr_status": "NO_TEXT_DETECTED",
                "ocr_language": language, "ocr_engine": "NONE",
                "ocr_variant": None, "ocr_coordinate_variant": None,
                "ocr_candidates_tested": 0,
                "document_detection": detect_document_type(""),
                "structured_data": {}, "field_confidence": {},
                "candidate_summary": [], "ocr_tokens": [],
            }

        candidates.sort(
            key=lambda x: (x.get("score", -999), x.get("confidence", 0)),
            reverse=True,
        )

        # Re-score against the best detected document class.
        best = candidates[0]
        final_text = normalize_text(best.get("text", ""))
        final_detection = detect_document_type(final_text)

        for candidate in candidates:
            candidate["detection"] = detect_document_type(candidate.get("text", ""))
            candidate["structured"] = extract_fields_from_text(
                candidate.get("text", ""),
                candidate["detection"],
            )
            candidate["score"] = _ocr_candidate_score(candidate)

        candidates.sort(
            key=lambda x: (x.get("score", -999), x.get("confidence", 0)),
            reverse=True,
        )
        best = candidates[0]
        final_text = normalize_text(best.get("text", ""))
        final_detection = detect_document_type(final_text)

        structured, field_confidence = build_field_consensus(
            candidates,
            final_detection,
        )

        # Identifier recovery from ANY OCR candidate.
        for field in (
            "aadhaar_number", "pan_number", "passport_number",
            "driving_licence_number", "voter_id_number", "gstin",
        ):
            if structured.get(field):
                continue
            for candidate in candidates:
                value = candidate.get("structured", {}).get(field)
                if value and _validate_field(field, value).get("valid"):
                    structured[field] = value
                    field_confidence[field] = float(candidate.get("confidence", 0) or 0)
                    break

        # If OCR found a unique identifier but the header was missed, classify it.
        if final_detection.get("document_category") == "UNKNOWN":
            if structured.get("aadhaar_number"):
                final_detection = detect_document_type(final_text + "\nAADHAAR UIDAI")
            elif structured.get("pan_number"):
                final_detection = detect_document_type(final_text + "\nPERMANENT ACCOUNT NUMBER")
            elif structured.get("passport_number"):
                final_detection = detect_document_type(final_text + "\nPASSPORT")
            elif structured.get("voter_id_number"):
                final_detection = detect_document_type(final_text + "\nELECTION COMMISSION")

        structured["document"] = final_detection.get("document_label", "Unknown Document")
        structured["document_category"] = final_detection.get("document_category", "UNKNOWN")

        summary = [
            {
                "variant": c.get("variant"),
                "engine": c.get("engine"),
                "confidence": round(float(c.get("confidence", 0) or 0), 1),
                "score": round(float(c.get("score", 0) or 0), 2),
                "document": c.get("detection", {}).get("document_label", "Unknown Document"),
            }
            for c in candidates[:10]
        ]

        return {
            "extracted_text": final_text[:MAX_OCR_TEXT_LENGTH],
            "raw_ocr_text": final_text[:MAX_OCR_TEXT_LENGTH],
            "ocr_confidence": float(best.get("confidence", 0) or 0),
            "ocr_status": "TEXT_DETECTED" if final_text else "NO_TEXT_DETECTED",
            "ocr_language": language,
            "ocr_engine": best.get("engine", "tesseract"),
            "ocr_variant": best.get("variant"),
            "ocr_coordinate_variant": best.get("image_variant", "original"),
            "ocr_candidates_tested": len(candidates),
            "document_detection": final_detection,
            "structured_data": structured,
            "field_confidence": field_confidence,
            "candidate_summary": summary,
            "ocr_tokens": best.get("tokens", []),
        }

    except Exception as error:
        print("FINAL OCR ERROR:", str(error))
        return {
            "extracted_text": "", "raw_ocr_text": "",
            "ocr_confidence": 0.0, "ocr_status": "OCR_ERROR",
            "ocr_language": language, "ocr_engine": "tesseract",
            "ocr_variant": None, "ocr_coordinate_variant": None,
            "ocr_candidates_tested": len(candidates),
            "document_detection": detect_document_type(""),
            "structured_data": {}, "field_confidence": {},
            "candidate_summary": [], "ocr_tokens": [],
            "ocr_error": str(error),
        }
    finally:
        safe_close(original)
        for temp in temporary_images:
            safe_close(temp)


# ============================================================
# FINAL IMAGE ANALYSIS
# ============================================================

def _extract_image_metadata(image):
    metadata = {}
    try:
        for tag_id, value in image.getexif().items():
            metadata[ExifTags.TAGS.get(tag_id, str(tag_id))] = str(value)
    except Exception:
        pass
    return metadata


def _document_object(detection):
    return {
        "document_type": detection.get("document_label", "Unknown Document"),
        "document_category": detection.get("document_category", "UNKNOWN"),
        "confidence": detection.get("confidence", "LOW"),
        "score": detection.get("score", 0),
    }


def analyze_image(file_content):
    image = None
    try:
        check = Image.open(io.BytesIO(file_content))
        check.verify()
        safe_close(check)

        image = Image.open(io.BytesIO(file_content))
        image.load()
        image = fix_orientation(image)

        width, height = image.size
        image_format = image.format or "UNKNOWN"
        metadata = _extract_image_metadata(image)
        quality = analyze_image_quality(image)
        ocr = extract_ocr_data(image)

        raw_text = normalize_text(ocr.get("raw_ocr_text", ""))
        detection = ocr.get("document_detection") or detect_document_type(raw_text)
        structured = ocr.get("structured_data") or extract_fields_from_text(raw_text, detection)

        # Final classification fallback from extracted identifiers.
        if detection.get("document_category") == "UNKNOWN":
            if structured.get("aadhaar_number"):
                detection = detect_document_type(raw_text + "\nAADHAAR UIDAI")
            elif structured.get("pan_number"):
                detection = detect_document_type(raw_text + "\nPERMANENT ACCOUNT NUMBER")
            elif structured.get("passport_number"):
                detection = detect_document_type(raw_text + "\nPASSPORT")
            elif structured.get("voter_id_number"):
                detection = detect_document_type(raw_text + "\nELECTION COMMISSION")
            structured = extract_fields_from_text(raw_text, detection)

        display_text = build_display_text(structured)
        face = detect_faces_in_document(image)
        qr = analyze_qr_signal(image, structured)
        doc_validation = validate_document_specific(
            detection.get("document_category", "UNKNOWN"),
            structured,
            qr,
        )
        consistency = analyze_cross_field_consistency(structured)

        # Use OCR coordinate space for targeted analysis.
        forensic_image = resize_for_ocr(image.convert("RGB"))
        targeted = analyze_targeted_text_regions(
            forensic_image,
            ocr.get("ocr_tokens", []),
            structured,
        )
        safe_close(forensic_image)

        tampering = analyze_document_tampering(
            image,
            metadata,
            ocr.get("ocr_tokens", []),
            structured,
        )
        tampering["targeted_text_regions"] = targeted

        text_analysis = analyze_extracted_text(
            raw_text,
            structured,
        )

        return {
            "valid": True,
            "file_category": "IMAGE",
            "document_type": "IMAGE",
            "width": width,
            "height": height,
            "format": image_format,
            "mode": image.mode,
            "metadata_found": bool(metadata),
            "metadata_count": len(metadata),
            "metadata": metadata,
            "image_quality": quality,
            "face_detection": face,
            "qr_analysis": qr,
            "tampering_analysis": tampering,
            "metadata_analysis": tampering.get("metadata", {}),
            "ela": tampering.get("ela", {}),
            "region_consistency": tampering.get("region_consistency", {}),
            "text_region_tampering": targeted,
            "extracted_text": display_text or raw_text,
            "raw_ocr_text": raw_text,
            "ocr_confidence": float(ocr.get("ocr_confidence", 0) or 0),
            "ocr_status": ocr.get("ocr_status", "NO_TEXT_DETECTED"),
            "ocr_language": ocr.get("ocr_language"),
            "ocr_engine": ocr.get("ocr_engine", "tesseract"),
            "ocr_variant": ocr.get("ocr_variant"),
            "ocr_coordinate_variant": ocr.get("ocr_coordinate_variant", "original"),
            "ocr_candidates_tested": ocr.get("ocr_candidates_tested", 0),
            "candidate_summary": ocr.get("candidate_summary", []),
            "ocr_tokens": ocr.get("ocr_tokens", []),
            "extracted_characters": len(raw_text),
            "display_characters": len(display_text),
            "document_category": detection.get("document_category", "UNKNOWN"),
            "document_label": detection.get("document_label", "Unknown Document"),
            "document_detection_confidence": detection.get("confidence", "LOW"),
            "document_detection_score": detection.get("score", 0),
            "document_detection_evidence": detection.get("evidence", {}),
            "structured_data": structured,
            "field_confidence": ocr.get("field_confidence", {}),
            "document_validation": doc_validation,
            "cross_field_consistency": consistency,
            "extracted_data": structured,
            "extracted": structured,
            "document": _document_object(detection),
            **text_analysis,
        }
    except Exception as error:
        return {
            "valid": False,
            "file_category": "IMAGE",
            "document_type": "IMAGE",
            "error": str(error),
        }
    finally:
        safe_close(image)

# ============================================================
# FINAL PDF ANALYSIS
# ============================================================

def analyze_pdf(file_content):
    pdf = None
    try:
        pdf = fitz.open(stream=file_content, filetype="pdf")
        page_count = pdf.page_count
        if page_count <= 0:
            return {"valid": False, "file_category": "PDF", "document_type": "PDF", "error": "PDF contains no pages"}

        metadata = pdf.metadata or {}
        encrypted = bool(pdf.is_encrypted)
        pages_to_scan = min(page_count, MAX_PDF_OCR_PAGES)
        parts = []
        methods = []
        page_summaries = []

        for page_number in range(pages_to_scan):
            page = pdf.load_page(page_number)
            native = normalize_text(page.get_text("text"))

            if len(native) >= 25:
                parts.append(native)
                methods.append("native_pdf_text")
                page_summaries.append({
                    "page": page_number + 1,
                    "method": "native_pdf_text",
                    "ocr_confidence": 100.0,
                })
                continue

            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            page_image = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr = extract_ocr_data(page_image)
            page_text = normalize_text(ocr.get("raw_ocr_text", ""))
            if page_text:
                parts.append(page_text)
            methods.append("rendered_page_ocr")
            page_summaries.append({
                "page": page_number + 1,
                "method": "rendered_page_ocr",
                "ocr_confidence": ocr.get("ocr_confidence", 0),
                "document_detection": ocr.get("document_detection", {}),
            })
            safe_close(page_image)

        combined = normalize_text("\n\n".join(parts))[:MAX_OCR_TEXT_LENGTH]
        detection = detect_document_type(combined)
        structured = extract_fields_from_text(combined, detection)

        if detection.get("document_category") == "UNKNOWN":
            if structured.get("aadhaar_number"):
                detection = detect_document_type(combined + "\nAADHAAR UIDAI")
            elif structured.get("pan_number"):
                detection = detect_document_type(combined + "\nPERMANENT ACCOUNT NUMBER")
            structured = extract_fields_from_text(combined, detection)

        display_text = build_display_text(structured)
        validation = validate_document_specific(detection.get("document_category", "UNKNOWN"), structured, {})
        consistency = analyze_cross_field_consistency(structured)
        ocr_confidences = [
            float(x.get("ocr_confidence", 0) or 0)
            for x in page_summaries
            if x.get("method") == "rendered_page_ocr"
        ]
        ocr_confidence = round(sum(ocr_confidences) / len(ocr_confidences), 1) if ocr_confidences else (95.0 if combined else 0.0)

        return {
            "valid": True,
            "file_category": "PDF",
            "document_type": "PDF",
            "page_count": page_count,
            "pages_scanned": pages_to_scan,
            "encrypted": encrypted,
            "metadata_found": bool(metadata),
            "metadata": metadata,
            "image_quality": None,
            "face_detection": {"available": False, "face_count": 0, "status": "PDF_PAGE_ANALYSIS_NOT_RUN", "faces": []},
            "qr_analysis": {"available": False, "decoded": False, "status": "PDF_PAGE_ANALYSIS_NOT_RUN", "data_consistent": None},
            "tampering_analysis": {"status": "PDF_VISUAL_FORENSICS_LIMITED", "tampering_probability": 0.0, "risk_level": "LOW", "signals": [], "evidence": []},
            "metadata_analysis": {"status": "NOT_APPLICABLE", "editing_software_signals": []},
            "ela": {"available": False, "status": "PDF_LEVEL_NOT_APPLIED", "score": 0.0},
            "region_consistency": {"available": False, "noise_score": 0.0, "edge_score": 0.0, "suspicious_regions": []},
            "text_region_tampering": {"available": False, "suspicious": False, "suspicious_fields": [], "regions": [], "region_count": 0},
            "extracted_text": display_text or combined,
            "raw_ocr_text": combined,
            "ocr_confidence": ocr_confidence,
            "ocr_status": "TEXT_DETECTED" if combined else "NO_TEXT_DETECTED",
            "ocr_language": get_ocr_language(),
            "ocr_engine": "pdf_native_text" if not ocr_confidences else "tesseract",
            "ocr_variant": None,
            "ocr_candidates_tested": len(page_summaries),
            "candidate_summary": page_summaries,
            "ocr_tokens": [],
            "extracted_characters": len(combined),
            "display_characters": len(display_text),
            "document_category": detection.get("document_category", "UNKNOWN"),
            "document_label": detection.get("document_label", "Unknown Document"),
            "document_detection_confidence": detection.get("confidence", "LOW"),
            "document_detection_score": detection.get("score", 0),
            "document_detection_evidence": detection.get("evidence", {}),
            "structured_data": structured,
            "field_confidence": {},
            "document_validation": validation,
            "cross_field_consistency": consistency,
            "extracted_data": structured,
            "extracted": structured,
            "document": _document_object(detection),
            "extraction_method": sorted(set(methods)),
            **analyze_extracted_text(combined, structured),
        }
    except Exception as error:
        return {"valid": False, "file_category": "PDF", "document_type": "PDF", "error": str(error)}
    finally:
        try:
            if pdf is not None:
                pdf.close()
        except Exception:
            pass


# ============================================================
# FINAL RISK ENGINE
# ============================================================

def _unique_append(items, value):
    if value and value not in items:
        items.append(value)


def calculate_risk(file_format_valid, structure_valid, file_size, analysis):
    analysis = analysis or {}
    score = 0
    signals = []
    evidence = []

    if not file_format_valid:
        score += 35
        _unique_append(signals, "File signature does not match declared content type")
        evidence.append({"category": "file_integrity", "weight": 35})

    if not structure_valid:
        score += 35
        _unique_append(signals, "Document could not be parsed successfully")
        evidence.append({"category": "structure", "weight": 35})

    if file_size > 9 * 1024 * 1024:
        score += 4
        _unique_append(signals, "Large upload size requires review")

    ocr_status = analysis.get("ocr_status", "NO_TEXT_DETECTED")
    ocr_conf = float(analysis.get("ocr_confidence", 0) or 0)
    if ocr_status in {"NO_TEXT_DETECTED", "OCR_ERROR"}:
        score += 12
        _unique_append(signals, "No reliable OCR text was extracted")
    elif ocr_conf < 35:
        score += 10
        _unique_append(signals, "Low OCR readability confidence")
    elif ocr_conf < 55:
        score += 5
        _unique_append(signals, "Moderate OCR readability confidence")

    category = analysis.get("document_category", "UNKNOWN")
    detection_conf = str(analysis.get("document_detection_confidence", "LOW")).upper()
    structured = analysis.get("structured_data", {}) or {}

    if category == "UNKNOWN":
        score += 8
        _unique_append(signals, "Document type could not be identified with sufficient confidence")
        evidence.append({"category": "document_detection", "weight": 8})
    elif detection_conf == "LOW":
        score += 3
        _unique_append(signals, "Document type detection confidence is low")

    validation = analysis.get("document_validation", {}) or {}
    for failure in validation.get("failures", [])[:5]:
        weight = float(failure.get("weight", 1) or 1)
        score += min(28, int(round(8 * weight)))
        _unique_append(signals, failure.get("reason") or failure.get("name"))
        evidence.append({"category": "document_validation", "check": failure.get("name"), "weight": weight})

    consistency = analysis.get("cross_field_consistency", {}) or {}
    conflict_count = int(consistency.get("conflict_count", 0) or 0)
    warning_count = int(consistency.get("warning_count", 0) or 0)
    if conflict_count:
        score += min(25, conflict_count * 10)
        for item in consistency.get("conflicts", [])[:5]:
            _unique_append(signals, item.get("reason"))
        evidence.append({"category": "cross_field_consistency", "weight": min(25, conflict_count * 10)})
    elif warning_count:
        score += min(5, warning_count * 2)

    quality = analysis.get("image_quality", {}) or {}
    for issue in quality.get("issues", []):
        _unique_append(signals, issue)
        score += 3 if "resolution" in issue.lower() or "blurry" in issue.lower() else 2

    qr = analysis.get("qr_analysis", {}) or {}
    if qr.get("status") == "DATA_MISMATCH":
        score += 25
        _unique_append(signals, "Decoded QR data conflicts with extracted document data")
        evidence.append({"category": "qr_consistency", "weight": 25})

    tampering = analysis.get("tampering_analysis", {}) or {}
    tamper_status = tampering.get("status", "")
    tamper_probability = float(tampering.get("tampering_probability", 0) or 0)
    tamper_evidence = tampering.get("evidence", []) or []
    if tamper_status == "HIGH_REVIEW_REQUIRED":
        score += min(32, int(round(18 + tamper_probability * .18)))
        _unique_append(signals, "Multiple forensic signals require high-priority manual review")
        evidence.append({"category": "tampering", "weight": 32})
    elif tamper_status == "REVIEW_RECOMMENDED":
        score += min(20, int(round(8 + tamper_probability * .15)))
        _unique_append(signals, "Forensic image signals require manual review")
        evidence.append({"category": "tampering", "weight": 20})

    face = analysis.get("face_detection", {}) or {}
    if int(face.get("face_count", 0) or 0) > 1:
        score += 5
        _unique_append(signals, "Multiple faces detected in a single document image")

    if analysis.get("encrypted"):
        score += 3
        _unique_append(signals, "PDF is encrypted or password protected")

    score = min(100, max(0, int(round(score))))
    if score <= 20:
        level, decision = "LOW RISK", "LOW"
    elif score <= 50:
        level, decision = "MEDIUM RISK", "REVIEW"
    elif score <= 75:
        level, decision = "HIGH RISK", "HIGH_REVIEW"
    else:
        level, decision = "CRITICAL RISK", "CRITICAL_REVIEW"

    if not signals:
        signals.append("No strong technical risk signal detected")

    return score, level, signals, {
        "decision": decision,
        "evidence": evidence,
        "tampering_probability": tamper_probability,
        "tampering_evidence_count": len(tamper_evidence),
    }

# ============================================================
# FINAL VALIDATION RESPONSE
# ============================================================

def build_validation_results(file_format_valid, structure_valid, analysis):
    analysis = analysis or {}
    results = [
        {"name": "File format check", "status": "PASSED" if file_format_valid else "FAILED"},
        {"name": "Document structure check", "status": "PASSED" if structure_valid else "FAILED"},
        {"name": "OCR readability check", "status": "PASSED" if analysis.get("ocr_status") == "TEXT_DETECTED" else "REVIEW REQUIRED"},
    ]

    category = analysis.get("document_category", "UNKNOWN")
    conf = str(analysis.get("document_detection_confidence", "LOW")).upper()
    results.append({
        "name": "Document type detection",
        "status": "REVIEW REQUIRED" if category == "UNKNOWN" else "PASSED" if conf == "HIGH" else "REVIEW",
        "value": analysis.get("document_label", "Unknown Document"),
    })

    doc_validation = analysis.get("document_validation", {}) or {}
    doc_status = doc_validation.get("overall_status", "REVIEW")
    results.append({
        "name": "Document-specific validation",
        "status": "PASSED" if doc_status == "PASSED" else "FAILED" if doc_status == "FAILED" else "REVIEW REQUIRED",
    })

    consistency = analysis.get("cross_field_consistency", {}) or {}
    cstatus = consistency.get("status", "REVIEW")
    results.append({
        "name": "Cross-field consistency check",
        "status": "PASSED" if cstatus == "PASSED" else "FAILED" if cstatus == "FAILED" else "REVIEW REQUIRED",
    })

    quality = analysis.get("image_quality", {}) or {}
    results.append({
        "name": "Document image quality",
        "status": quality.get("status", "NOT APPLICABLE"),
    })

    qr_status = (analysis.get("qr_analysis", {}) or {}).get("status", "NOT_AVAILABLE")
    if qr_status == "DATA_MATCH":
        qr_display = "PASSED"
    elif qr_status == "DATA_MISMATCH":
        qr_display = "FAILED"
    elif qr_status in {"NOT_DECODED", "DETECTED_NOT_DECODED", "NOT_AVAILABLE", "PDF_PAGE_ANALYSIS_NOT_RUN"}:
        qr_display = "NOT PRESENT / NOT DECODED"
    else:
        qr_display = "REVIEW"
    results.append({"name": "QR / encoded-data consistency", "status": qr_display})

    tampering = analysis.get("tampering_analysis", {}) or {}
    tstatus = tampering.get("status", "NO_STRONG_TAMPERING_SIGNAL")
    results.append({
        "name": "Tampering / anomaly signal analysis",
        "status": "HIGH REVIEW REQUIRED" if tstatus == "HIGH_REVIEW_REQUIRED" else "REVIEW REQUIRED" if tstatus == "REVIEW_RECOMMENDED" else "NO STRONG SIGNAL",
    })

    targeted = analysis.get("text_region_tampering", {}) or {}
    results.append({
        "name": "Targeted text-region forensic screening",
        "status": "REVIEW REQUIRED" if targeted.get("suspicious_fields") else "NO STRONG SIGNAL" if targeted.get("available") else "NOT AVAILABLE",
    })
    return results


# ============================================================
# FINAL DISPLAY TEXT
# ============================================================

def build_display_text(structured_data):
    data = structured_data or {}
    fields = [
        ("Name", "name"),
        ("Document", "document"),
        ("Aadhaar Number", "aadhaar_number"),
        ("PAN Number", "pan_number"),
        ("Passport Number", "passport_number"),
        ("Driving Licence Number", "driving_licence_number"),
        ("Voter ID Number", "voter_id_number"),
        ("GSTIN", "gstin"),
        ("Visa Number", "visa_number"),
        ("Permit Number", "permit_number"),
        ("Date of Birth", "date_of_birth"),
        ("Gender", "gender"),
        ("Nationality", "nationality"),
        ("Date of Issue", "date_of_issue"),
        ("Date of Expiry", "date_of_expiry"),
        ("Validity", "validity_status"),
        ("Address", "address"),
        ("Parent Name", "parent_name"),
    ]
    return "\n".join(
        f"{label}: {data[key]}"
        for label, key in fields
        if data.get(key)
    )


# ============================================================
# FINAL ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "SecureDoc AI Backend is Running!",
        "status": "online",
        "version": "6.0.0",
        "phase": "Universal Document Screening + Multi-Layer Forensics",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "SecureDoc AI",
        "version": "6.0.0",
        "tesseract_configured": bool(get_ocr_language()),
        "opencv_available": CV2_AVAILABLE,
        "paddleocr_available": PADDLE_AVAILABLE,
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    declared_type = normalize_content_type(file.content_type)
    allowed = {normalize_content_type(x) for x in ALLOWED_CONTENT_TYPES}
    if declared_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload JPG, PNG, WEBP or PDF.",
        )

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
        analysis = analyze_image(file_content)
    elif detected_type == "application/pdf":
        analysis = analyze_pdf(file_content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported detected file type.")

    structure_valid = bool(analysis.get("valid", False))
    validation_results = build_validation_results(file_format_valid, structure_valid, analysis)
    risk_score, risk_level, risk_signals, risk_meta = calculate_risk(
        file_format_valid,
        structure_valid,
        file_size,
        analysis,
    )

    if risk_score >= 76:
        decision = "CRITICAL_REVIEW"
    elif risk_score >= 51:
        decision = "HIGH_REVIEW"
    elif risk_score >= 21:
        decision = "REVIEW"
    else:
        decision = "LOW_RISK_SCREENING"

    tampering = analysis.get("tampering_analysis", {}) or {}
    anomaly = bool(
        risk_score >= 21
        or tampering.get("status") in {"REVIEW_RECOMMENDED", "HIGH_REVIEW_REQUIRED"}
    )

    if tampering.get("status") == "HIGH_REVIEW_REQUIRED":
        anomaly_title = "Multiple Forensic Signals Require Review"
        anomaly_description = "Multiple technical signals warrant manual forensic review; this is not proof of forgery."
    elif tampering.get("status") == "REVIEW_RECOMMENDED":
        anomaly_title = "Forensic Signals Require Review"
        anomaly_description = "One or more local or visual signals require manual review."
    elif anomaly:
        anomaly_title = "Technical Review Recommended"
        anomaly_description = "One or more validation, OCR, quality or consistency signals require review."
    else:
        anomaly_title = "No Strong Technical Anomaly Detected"
        anomaly_description = "Available technical checks completed without a strong combined anomaly signal."

    ocr_conf = float(analysis.get("ocr_confidence", 0) or 0)
    tamper_probability = float(tampering.get("tampering_probability", 0) or 0)
    anomaly_confidence = round(min(99, max(50, 60 + ocr_conf * .30 + min(20, tamper_probability * .20))))

    validation_status = (
        "PASSED"
        if file_format_valid
        and structure_valid
        and (analysis.get("document_validation", {}) or {}).get("overall_status") == "PASSED"
        and (analysis.get("cross_field_consistency", {}) or {}).get("status") == "PASSED"
        else "REVIEW"
    )

    face = analysis.get("face_detection", {}) or {}
    qr = analysis.get("qr_analysis", {}) or {}
    structured = analysis.get("structured_data", {}) or {}

    risk_breakdown = {
        "file_integrity": "PASS" if file_format_valid else "FAIL",
        "ocr_quality": "GOOD" if ocr_conf >= 70 else "MODERATE" if ocr_conf >= 45 else "LOW",
        "document_structure": "VALID" if structure_valid else "INVALID",
        "document_type": analysis.get("document_label", "Unknown Document"),
        "document_validation": (analysis.get("document_validation", {}) or {}).get("overall_status", "REVIEW"),
        "cross_field_consistency": (analysis.get("cross_field_consistency", {}) or {}).get("status", "REVIEW"),
        "qr_consistency": qr.get("status", "N/A"),
        "tampering_signals": tampering.get("risk_level", "LOW"),
        "face_detection": f"{int(face.get('face_count', 0) or 0)} FACE",
    }

    return {
        "success": True,
        "message": "Document uploaded and analyzed successfully",
        "document": {
            "filename": file.filename,
            "content_type": declared_type,
            "detected_type": detected_type,
            "file_size": file_size,
            "sha256": hashlib.sha256(file_content).hexdigest(),
        },
        "validation": {
            "status": validation_status,
            "results": validation_results,
        },
        "analysis_data": analysis,
        "anomaly": {
            "detected": anomaly,
            "title": anomaly_title,
            "description": anomaly_description,
            "confidence": anomaly_confidence,
        },
        "face_verification": face,
        "qr_verification": qr,
        "risk_breakdown": risk_breakdown,
        "risk_assessment": {
            "score": risk_score,
            "level": risk_level,
            "decision": decision,
            "signals": risk_signals,
            "evidence": risk_meta.get("evidence", []),
            "description": (
                "Risk is a technical screening score combining file integrity, OCR, "
                "document-specific validation, cross-field consistency, QR evidence, "
                "image quality and forensic signals. It is not a legal authenticity verdict."
            ),
        },
        "screening_summary": {
            "document_type": analysis.get("document_label", "Unknown Document"),
            "document_type_confidence": analysis.get("document_detection_confidence", "LOW"),
            "ocr_confidence": ocr_conf,
            "tampering_probability": tamper_probability,
            "final_decision": decision,
            "risk_level": risk_level,
            "extracted_fields": [
                key for key, value in structured.items()
                if value and key not in {"document", "document_category"}
            ],
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
