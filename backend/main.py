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

OCR_TARGET_WIDTH = 2000
OCR_MAX_DIMENSION = 3000

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

        # English-only is the fastest and usually the most accurate
        # mode for identity/document fields. Mixed eng+hin can reduce
        # recognition quality on English names and numbers.
        if "eng" in languages:
            return "eng"

        if (
            "hin" in languages
        ):
            return "hin"

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
    if not lines:
        return None

    # --------------------------------------------------------
    # Strong name cleaner. OCR often adds punctuation or field
    # labels, so clean only the value and reject document text.
    # --------------------------------------------------------
    def clean_name_candidate(value):
        if not value:
            return None

        value = str(value).strip()
        value = re.sub(
            r"(?i)^(?:name|full\s*name|name\s*of\s*(?:holder|applicant|person)|card\s*holder|cardholder|surname|nom|given\s*names?|given\s*name|forenames?|pr[eé]noms?)\s*[:.\-]?\s*",
            "",
            value,
        )
        value = re.sub(
            r"(?i)\b(?:s/o|d/o|w/o|c/o|son\s+of|daughter\s+of|wife\s+of)\b.*",
            "",
            value,
        )
        value = value.strip(" :-|,.;")
        value = normalize_name(value)
        if not value or not is_valid_name(value):
            return None

        lower = value.lower()
        blocked = {
            "passport", "issuing country", "issuing authority", "country",
            "nationality", "authority", "government", "department",
            "transport", "licence to drive", "license to drive",
            "authorisation", "authorization", "date of birth", "date of issue",
            "date of expiry", "date of expiration", "place of birth", "sex",
            "gender", "signature", "holder", "address", "validity", "valid till",
            "valid until", "passport no", "licence no", "license no", "document",
            "identity card", "permanent account number", "income tax department",
            "election commission", "elector", "blood group", "date of issue",
        }
        if any(phrase in lower for phrase in blocked):
            return None
        if re.search(r"\d", value):
            return None
        if len(value.split()) > 6:
            return None
        return value

    # --------------------------------------------------------
    # 1. PASSPORT / INTERNATIONAL DOCUMENTS
    # OCR can return bilingual labels. Always combine Given Names
    # + Surname when both are available.
    # --------------------------------------------------------
    surname = None
    given_name = None

    surname_re = re.compile(
        r"(?i)^(?:surname|nom)\s*[:.\-]?\s*(.*)$"
    )
    given_re = re.compile(
        r"(?i)^(?:given\s*names?|given\s*name|forenames?|pr[eé]noms?)\s*[:.\-]?\s*(.*)$"
    )

    for i, line in enumerate(lines):
        m = surname_re.search(line.strip())
        if m:
            value = clean_name_candidate(m.group(1))
            if not value and i + 1 < len(lines):
                value = clean_name_candidate(lines[i + 1])
            if value:
                surname = value

        m = given_re.search(line.strip())
        if m:
            value = clean_name_candidate(m.group(1))
            if not value and i + 1 < len(lines):
                value = clean_name_candidate(lines[i + 1])
            if value:
                given_name = value

    if given_name and surname:
        full = clean_name_candidate(f"{given_name} {surname}")
        if full:
            return full
    if given_name:
        return given_name
    if surname:
        return surname

    # --------------------------------------------------------
    # 2. EXPLICIT NAME LABEL — strongest universal signal
    # Supports same-line and next-line layouts, including OCR
    # variants such as NAME:, NAME -, FULL NAME, etc.
    # --------------------------------------------------------
    explicit_re = re.compile(
        r"(?i)^(?:name|full\s*name|name\s*of\s*(?:holder|applicant|person)|cardholder|card\s*holder)\s*[:.\-]?\s*(.*)$"
    )

    for i, line in enumerate(lines):
        m = explicit_re.search(line.strip())
        if not m:
            continue

        same_line = clean_name_candidate(m.group(1))
        if same_line:
            return same_line

        if i + 1 < len(lines):
            next_line = clean_name_candidate(lines[i + 1])
            if next_line:
                return next_line

    # --------------------------------------------------------
    # 3. LABEL + VALUE may be separated by OCR noise.
    # Look within the next two lines, but stop at another field label.
    # --------------------------------------------------------
    label_words = re.compile(
        r"(?i)^(?:name|full\s*name|cardholder|card\s*holder)\b"
    )
    stop_labels = re.compile(
        r"(?i)^(?:father|father's|dob|date|gender|sex|address|blood|pan|aadhaar|aadhar|passport|licence|license|document|nationality|valid|expiry|issue)\b"
    )

    for i, line in enumerate(lines):
        if not label_words.search(line.strip()):
            continue
        for j in (i + 1, i + 2):
            if j >= len(lines):
                break
            if stop_labels.search(lines[j].strip()):
                break
            candidate = clean_name_candidate(lines[j])
            if candidate:
                return candidate

    # --------------------------------------------------------
    # 4. DOCUMENT-SPECIFIC fallbacks.
    # PAN/Aadhaar/DL commonly have a bare "Name" followed by the
    # value; father's name and other nearby fields are explicitly
    # rejected before accepting anything.
    # --------------------------------------------------------
    bad_context = re.compile(
        r"(?i)(father|father's|date\s+of\s+birth|dob|permanent\s+account|income\s+tax|government|address|blood\s+group)"
    )

    candidates = []
    for i, line in enumerate(lines):
        candidate = clean_name_candidate(line)
        if not candidate:
            continue

        score = score_name_candidate(candidate)
        words = candidate.split()
        if 2 <= len(words) <= 4:
            score += 18
        elif len(words) == 1:
            score += 2

        previous = lines[i - 1].strip().lower() if i > 0 else ""
        next_line = lines[i + 1].strip().lower() if i + 1 < len(lines) else ""

        if previous in {"name", "name:", "name -", "full name", "full name:"}:
            score += 120
        if "name" in previous and not bad_context.search(previous):
            score += 70
        if bad_context.search(previous) or bad_context.search(next_line):
            score -= 80

        # Avoid accepting common headings / authority text as names.
        if any(x in candidate.lower() for x in (
            "india", "government", "department", "transport", "passport",
            "licence", "license", "authority", "nationality", "address",
            "signature", "validity", "identity", "card"
        )):
            score -= 1000

        candidates.append((score, candidate))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_candidate = candidates[0]
        if best_score >= 45:
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
        current_structured = (
            best.get("structured", {})
            if best
            else {}
        )

        # A second sparse-text pass is useful when Tesseract found text
        # but missed the actual identity fields. This keeps clear scans
        # fast while recovering names/numbers on difficult layouts.
        needs_second_pass = (
            best is None
            or confidence < 42
            or not normalize_text(best.get("text", ""))
            or text_length < 35
            or not current_structured.get("name")
            or (
                category in {
                    "AADHAAR_CARD", "PAN_CARD",
                    "DRIVING_LICENCE", "PASSPORT",
                    "VOTER_ID", "VISA"
                }
                and not any(
                    current_structured.get(field)
                    for field in important_fields
                    if field in current_structured
                )
            )
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
            or confidence < 25
            or not normalize_text(best.get("text", ""))
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
# FORENSIC UPGRADE PATCH — DOCUMENT-SPECIFIC SIGNALS
# ============================================================

_base_analyze_image = analyze_image
_base_build_validation_results = build_validation_results

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
        checksum = _VERHOEFF_D[
            checksum
        ][
            _VERHOEFF_P[index % 8][int(char)]
        ]

    return checksum == 0


def detect_faces_in_document(image):
    result = {
        "available": False,
        "face_count": 0,
        "status": "NOT_AVAILABLE",
        "description": "Face detection is not available.",
        "faces": [],
    }
    if not CV2_AVAILABLE or cv2 is None:
        result["status"] = "OPENCV_NOT_AVAILABLE"
        result["description"] = "OpenCV is not available in this backend."
        return result
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        result["status"] = "OPENCV_INCOMPATIBLE"
        result["description"] = (
            "Installed OpenCV build does not provide the Haar face detector API. "
            "Install a stable OpenCV build and restart the backend."
        )
        return result
    try:
        rgb = image.convert("RGB")
        array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        cascade_path = os.path.join(
            cv2.data.haarcascades,
            "haarcascade_frontalface_default.xml",
        )
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            result["status"] = "MODEL_LOAD_ERROR"
            result["description"] = "Face detection model could not be loaded."
            return result
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(30, 30),
        )
        detected = [
            {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
            for x, y, w, h in faces
        ]
        count = len(detected)
        if count == 0:
            status, description = "NO_FACE_DETECTED", "No face detected in the document image."
        elif count == 1:
            status, description = "ONE_FACE_DETECTED", "One face detected in the document image."
        else:
            status = "MULTIPLE_FACES_DETECTED"
            description = f"{count} faces detected. Manual review is recommended."
        return {
            "available": True,
            "face_count": count,
            "status": status,
            "description": description,
            "faces": detected,
        }
    except Exception as error:
        print("FACE DETECTION ERROR:", repr(error))
        result["status"] = "PROCESSING_FAILED"
        result["description"] = f"Face detection failed: {str(error)}"
        result["error"] = str(error)
        return result


def analyze_qr_signal(image, structured):
    result = {
        "available": False,
        "status": "NOT_AVAILABLE",
        "decoded": False,
        "data_consistent": None,
    }
    if not CV2_AVAILABLE or cv2 is None or not hasattr(cv2, "QRCodeDetector"):
        return result
    try:
        array = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(array)
        if not data:
            result["available"] = True
            result["status"] = "NOT_DECODED"
            return result
        result["available"] = True
        result["decoded"] = True
        result["status"] = "DECODED"
        result["data_length"] = len(data)
        aadhaar = re.sub(r"\D", "", str(structured.get("aadhaar_number") or ""))
        qr_digits = re.sub(r"\D", "", data)
        if len(aadhaar) == 12 and len(qr_digits) >= 12:
            result["data_consistent"] = aadhaar in qr_digits
            if not result["data_consistent"]:
                result["status"] = "DATA_MISMATCH"
        return result
    except Exception as error:
        result["status"] = "ERROR"
        result["error"] = str(error)
        return result


def analyze_image(file_content):
    data = _base_analyze_image(file_content)
    if not data.get("valid"):
        return data
    try:
        image = Image.open(io.BytesIO(file_content))
        image.load()
        image = fix_orientation(image)
        # Face detection already runs in the base image analysis.
        # Reuse that result here to avoid a duplicate pass.
        data["metadata_analysis"] = analyze_metadata_tampering(data.get("metadata") or {})
        data["ela"] = perform_error_level_analysis(image)
        data["region_consistency"] = analyze_image_region_consistency(image)
        data["qr_analysis"] = analyze_qr_signal(
            image,
            data.get("structured_data") or {},
        )
        safe_close(image)
    except Exception as error:
        data["forensic_error"] = str(error)
    return data


def calculate_risk(file_format_valid, structure_valid, file_size, analysis):
    score = 0
    signals = []

    if not file_format_valid:
        score += 45
        signals.append("File signature does not match declared content type")
    if not structure_valid:
        score += 35
        signals.append("Document could not be parsed successfully")
    if file_size > 8 * 1024 * 1024:
        score += 5
        signals.append("Large file size")

    ocr_status = analysis.get("ocr_status", "NO_TEXT_DETECTED")
    ocr_confidence = float(analysis.get("ocr_confidence", 0) or 0)
    if ocr_status == "NO_TEXT_DETECTED":
        score += 8
        signals.append("No readable text detected")
    elif 0 < ocr_confidence < 40:
        score += 12
        signals.append("Low OCR readability confidence")
    elif 0 < ocr_confidence < 60:
        score += 6
        signals.append("Moderate OCR readability confidence")

    quality = analysis.get("image_quality") or {}
    for issue in quality.get("issues", []):
        score += 4
        signals.append(issue)

    if analysis.get("encrypted"):
        score += 8
        signals.append("PDF is encrypted")

    category = analysis.get("document_category", "UNKNOWN")
    structured = analysis.get("structured_data") or {}
    if category == "AADHAAR_CARD":
        aadhaar = re.sub(r"\D", "", str(structured.get("aadhaar_number") or ""))
        if len(aadhaar) == 12:
            if not is_valid_aadhaar_number(aadhaar):
                score += 40
                signals.append("Extracted Aadhaar number fails checksum validation")
        else:
            score += 8
            signals.append("Aadhaar document detected but no complete 12-digit number was extracted")

    metadata_analysis = analysis.get("metadata_analysis") or {}
    if metadata_analysis.get("status") == "REVIEW":
        score += 18
        signals.append("Editing-software metadata signal requires review")

    ela = analysis.get("ela") or {}
    if ela.get("available") and ela.get("status") == "REVIEW":
        ela_score = float(ela.get("score", 0) or 0)
        score += min(25, max(10, int(ela_score * 0.30)))
        signals.append("Error-level analysis shows unusual recompression differences")

    consistency = analysis.get("region_consistency") or {}
    suspicious_count = len(consistency.get("suspicious_regions") or [])
    if suspicious_count:
        score += min(25, 8 + suspicious_count * 3)
        signals.append(f"{suspicious_count} locally inconsistent image region(s) detected")

    qr = analysis.get("qr_analysis") or {}
    if qr.get("status") == "DATA_MISMATCH":
        score += 35
        signals.append("Decoded QR data is inconsistent with extracted document data")

    score = min(max(int(round(score)), 0), 100)
    if score <= 25:
        level = "LOW RISK"
    elif score <= 60:
        level = "MEDIUM RISK"
    else:
        level = "HIGH RISK"
    return score, level, signals


def build_validation_results(file_format_valid, structure_valid, analysis):
    results = _base_build_validation_results(file_format_valid, structure_valid, analysis)
    structured = analysis.get("structured_data") or {}
    if analysis.get("document_category") == "AADHAAR_CARD":
        aadhaar = re.sub(r"\D", "", str(structured.get("aadhaar_number") or ""))
        if len(aadhaar) == 12:
            results.append({
                "name": "Aadhaar checksum validation",
                "status": "PASSED" if is_valid_aadhaar_number(aadhaar) else "FAILED",
            })
    face = analysis.get("face_detection") or {}
    if face.get("status") == "OPENCV_INCOMPATIBLE":
        results.append({
            "name": "Face detector runtime",
            "status": "REVIEW REQUIRED",
        })
    return results

# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": (
            "SecureDoc AI Backend is Running!"
        ),
        "status": "online",
        "version": "5.1.0",
        "phase": (
            "Layout-aware multi-pass OCR "
            "with field-wise consensus extraction"
        ),
        "opencv_available": (
            CV2_AVAILABLE
        ),
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    try:

        version = (
            pytesseract
            .get_tesseract_version()
        )

        tesseract_ready = True

    except Exception:

        version = None

        tesseract_ready = False

    return {
        "status": "online",
        "tesseract_ready": (
            tesseract_ready
        ),
        "tesseract_version": (
            str(version)
            if version
            else None
        ),
        "opencv_available": (
            CV2_AVAILABLE
        ),
        "paddleocr_available": (
            PADDLE_AVAILABLE
        ),
        "paddleocr_error": (
            _PADDLE_ENGINE_ERROR
        ),
        "backend_version": "5.1.0",
    }


# ============================================================
# UPLOAD AND ANALYSIS
# ============================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    declared_type = normalize_content_type(
        file.content_type
    )

    if (
        declared_type
        not in ALLOWED_CONTENT_TYPES
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Upload JPG, PNG, WEBP or PDF."
            ),
        )

    file_content = await file.read()

    file_size = len(
        file_content
    )

    if file_size == 0:

        raise HTTPException(
            status_code=400,
            detail="File is empty.",
        )

    if (
        file_size
        > MAX_FILE_SIZE
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "File is too large. "
                "Maximum allowed size is 10 MB."
            ),
        )

    detected_type = (
        detect_file_signature(
            file_content
        )
    )

    if (
        detected_type
        == "unknown"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "File signature could not be verified."
            ),
        )

    file_format_valid = (
        content_type_matches(
            declared_type,
            detected_type,
        )
    )

    if detected_type.startswith(
        "image/"
    ):

        analysis_data = analyze_image(
            file_content
        )

    elif (
        detected_type
        == "application/pdf"
    ):

        analysis_data = analyze_pdf(
            file_content
        )

    else:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported detected file type."
            ),
        )

    structure_valid = bool(
        analysis_data.get(
            "valid",
            False,
        )
    )

    validation_results = (
        build_validation_results(
            file_format_valid,
            structure_valid,
            analysis_data,
        )
    )

    risk_score, risk_level, risk_signals = (
        calculate_risk(
            file_format_valid,
            structure_valid,
            file_size,
            analysis_data,
        )
    )

    anomaly_detected = (
        risk_score > 25
    )

    if anomaly_detected:

        anomaly_title = (
            "Potential Technical Anomalies Detected"
        )

        anomaly_description = (
            "One or more OCR, quality or "
            "file-integrity signals require review. "
            "This is not proof of document tampering."
        )

    else:

        anomaly_title = (
            "No Critical Technical Anomalies Detected"
        )

        anomaly_description = (
            "Available file, OCR and "
            "technical quality checks completed "
            "without critical signals."
        )

    ocr_confidence = float(
        analysis_data.get(
            "ocr_confidence",
            0,
        )
        or 0
    )

    anomaly_confidence = round(
        min(
            99,
            max(
                50,
                70
                + ocr_confidence
                * 0.25,
            ),
        ),
        0,
    )

    file_hash = hashlib.sha256(
        file_content
    ).hexdigest()

    validation_status = (
        "PASSED"
        if (
            file_format_valid
            and structure_valid
        )
        else "REVIEW"
    )

        # -----------------------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------------------

    return {
        "success": True,

        "message": (
            "Document uploaded and analyzed successfully"
        ),

        "document": {
            "filename": file.filename,
            "content_type": declared_type,
            "detected_type": detected_type,
            "file_size": file_size,
            "sha256": file_hash,
        },

        "validation": {
            "status": validation_status,
            "results": validation_results,
        },

        "analysis_data": analysis_data,

        "anomaly": {
            "detected": anomaly_detected,
            "title": anomaly_title,
            "description": anomaly_description,
            "confidence": anomaly_confidence,
        },

        "face_verification": (
    analysis_data.get(
        "face_detection",
        {
            "available": False,
            "face_count": 0,
            "status": "NOT_AVAILABLE",
            "description": (
                "Face detection is not available "
                "for this document."
            ),
            "faces": [],
        },
    )
),

        "risk_assessment": {
            "score": risk_score,
            "level": risk_level,
            "signals": risk_signals,
            "description": (
                "Risk score is based on file integrity, "
                "parse success, OCR readability and "
                "available technical quality signals. "
                "It is not a legal authenticity verdict."
            ),
        },
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
