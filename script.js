/* =========================================================
   SECUREDOC AI
   FINAL FRONTEND SCRIPT
   Backend-Driven Analysis Version
   ========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    /* =====================================================
       CONFIG
    ====================================================== */

    const API_BASE = "https://securedoc-ai-t5m0.onrender.com";
    const UPLOAD_ENDPOINT =
        `${API_BASE}/upload`;

    const allowedFileTypes = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "application/pdf"
    ];

    const maxFileSize =
        10 * 1024 * 1024;


    /* =====================================================
       DOM ELEMENTS
    ====================================================== */

    const startAnalysisButton =
        document.getElementById("startAnalysis");

    const workspace =
        document.getElementById("workspace") ||
        document.getElementById("uploadSection");

    const uploadBox =
        document.getElementById("uploadBox");

    const documentInput =
        document.getElementById("documentInput");

    const analyzeButton =
        document.getElementById("analyzeButton");

    const filePreview =
        document.getElementById("filePreview");

    const fileName =
        document.getElementById("fileName");

    const fileInfo =
        document.getElementById("fileInfo");

    const selectedFileIcon =
        document.getElementById("selectedFileIcon");

    const removeFile =
        document.getElementById("removeFile");

    const fileNameFallback =
        document.getElementById("fileNameFallback");

    const processingSection =
        document.getElementById("processingSection");

    const processingText =
        document.getElementById("processingText");

    const progress =
        document.getElementById("progress");

    const percentage =
        document.getElementById("percentage");

    const processingSteps =
        [...document.querySelectorAll(".processing-step")];

    const resultsSection =
        document.getElementById("resultsSection");

    const newScanButton =
        document.getElementById("newScanButton");

    const resultSummary =
        document.getElementById("resultSummary");

    const resultFileName =
        document.getElementById("resultFileName");

    const resultFileType =
        document.getElementById("resultFileType");

    const resultFileSize =
        document.getElementById("resultFileSize");

    const documentTypeBadge =
        document.getElementById("documentTypeBadge");

    const validationBadge =
        document.getElementById("validationBadge");

    const validationList =
        document.getElementById("validationList");

    const tamperingBadge =
        document.getElementById("tamperingBadge");

    const tamperingTitle =
        document.getElementById("tamperingTitle");

    const tamperingDescription =
        document.getElementById("tamperingDescription");

    const tamperingConfidence =
        document.getElementById("tamperingConfidence");

    const faceBadge =
        document.getElementById("faceBadge");

    const faceResult =
        document.getElementById("faceResult");

    const faceDescription =
        document.getElementById("faceDescription");

    const riskLevel =
        document.getElementById("riskLevel");

    const riskDescription =
        document.getElementById("riskDescription");

    const riskScore =
        document.getElementById("riskScore");


    /* =====================================================
       APP STATE
    ====================================================== */

    let selectedFile = null;

    let analysisRunning = false;

    let animationFrameId = null;


    /* =====================================================
       BASIC UTILITIES
    ====================================================== */

    function setText(element, value) {

        if (!element) {
            return;
        }

        element.textContent =
            value === null ||
            value === undefined ||
            value === ""
                ? "Not available"
                : String(value);
    }


    function wait(ms) {

        return new Promise(resolve => {

            setTimeout(resolve, ms);

        });
    }


    function cleanValue(value) {

        if (
            value === null ||
            value === undefined
        ) {
            return "";
        }

        const text =
            String(value).trim();

        const invalidValues = [
            "",
            "null",
            "undefined",
            "none",
            "unknown",
            "not available",
            "n/a"
        ];

        if (
            invalidValues.includes(
                text.toLowerCase()
            )
        ) {
            return "";
        }

        return text;
    }


    function formatFileSize(bytes) {

        if (
            !bytes ||
            bytes <= 0
        ) {
            return "0 Bytes";
        }

        const units = [
            "Bytes",
            "KB",
            "MB",
            "GB"
        ];

        const index =
            Math.min(
                Math.floor(
                    Math.log(bytes) /
                    Math.log(1024)
                ),
                units.length - 1
            );

        const size =
            bytes /
            Math.pow(
                1024,
                index
            );

        return `${size.toFixed(
            index === 0 ? 0 : 2
        )} ${units[index]}`;
    }


    function getFileFormat(file) {

        if (!file) {
            return "Unknown";
        }

        const formats = {

            "application/pdf":
                "PDF",

            "image/jpeg":
                "JPG / JPEG",

            "image/png":
                "PNG",

            "image/webp":
                "WEBP"

        };

        return (
            formats[file.type] ||
            "Unknown"
        );
    }


    function getFileIcon(file) {

        if (!file) {
            return "📁";
        }

        if (
            file.type ===
            "application/pdf"
        ) {
            return "📕";
        }

        if (
            file.type.startsWith(
                "image/"
            )
        ) {
            return "🖼️";
        }

        return "📄";
    }


    /* =====================================================
       STATUS NORMALIZATION
       IMPORTANT:
       INVALID MUST BE CHECKED BEFORE VALID
    ====================================================== */

    function normalizeStatus(value, fallback = "REVIEW REQUIRED") {
    const text = cleanValue(value).toUpperCase();

    if (!text) {
        return fallback;
    }

    // IMPORTANT:
    // "NO_CRITICAL_SIGNAL" contains the word "CRITICAL",
    // so check NO CRITICAL cases BEFORE CRITICAL.

    if (
        text.includes("NO_CRITICAL") ||
        text.includes("NO CRITICAL") ||
        text.includes("NO ANOMALY") ||
        text.includes("NO_TAMPERING") ||
        text.includes("NO TAMPERING")
    ) {
        return "NO CRITICAL SIGNAL";
    }

    if (
        text.includes("INVALID") ||
        text.includes("FAILED") ||
        text.includes("FAIL") ||
        text.includes("ERROR")
    ) {
        return "INVALID";
    }

    if (text.includes("CRITICAL")) {
        return "CRITICAL";
    }

    if (text.includes("HIGH_RISK") || text.includes("HIGH RISK")) {
        return "HIGH RISK";
    }

    if (
        text.includes("MEDIUM_RISK") ||
        text.includes("MEDIUM RISK") ||
        text.includes("MODERATE_RISK") ||
        text.includes("MODERATE RISK")
    ) {
        return "MODERATE RISK";
    }

    if (text.includes("LOW_RISK") || text.includes("LOW RISK")) {
        return "LOW RISK";
    }

    if (
        text.includes("REVIEW") ||
        text.includes("INCONSISTENT")
    ) {
        return "REVIEW REQUIRED";
    }

    if (
        text.includes("PASSED") ||
        text.includes("VALID") ||
        text.includes("SUCCESS")
    ) {
        return "PASSED";
    }

    return text.replace(/_/g, " ");
}


    function setStatusStyle(
        element,
        status
    ) {

        if (!element) {
            return;
        }

        element.classList.remove(
            "success",
            "warning",
            "danger",
            "info"
        );

        const value =
            String(status || "")
                .toUpperCase();


        if (
            value.includes("INVALID") ||
            value.includes("FAILED") ||
            value.includes("CRITICAL") ||
            value.includes("HIGH RISK")
        ) {

            element.classList.add(
                "danger"
            );

        } else if (

            value.includes("REVIEW") ||
            value.includes("MODERATE") ||
            value.includes("MEDIUM")

        ) {

            element.classList.add(
                "warning"
            );

        } else if (

            value.includes("PASSED") ||
            value.includes("VALID") ||
            value.includes("SUCCESS") ||
            value.includes("LOW RISK") ||
            value.includes("NO CRITICAL")

        ) {

            element.classList.add(
                "success"
            );

        } else {

            element.classList.add(
                "info"
            );
        }
    }


    /* =====================================================
       FILE VALIDATION
    ====================================================== */

    function validateFile(file) {

        if (!file) {
            return false;
        }


        if (
            !allowedFileTypes.includes(
                file.type
            )
        ) {

            alert(
                "Unsupported file format. Please select JPG, PNG, WEBP or PDF."
            );

            return false;
        }


        if (
            file.size >
            maxFileSize
        ) {

            alert(
                "File is too large. Maximum allowed size is 10 MB."
            );

            return false;
        }


        return true;
    }


    /* =====================================================
       FILE SELECTION
    ====================================================== */

    function selectFile(file) {

        if (
            !validateFile(file)
        ) {
            return;
        }


        selectedFile =
            file;


        setText(
            fileName,
            file.name
        );


        setText(
            fileInfo,
            `${getFileFormat(file)} · ${formatFileSize(file.size)}`
        );


        if (selectedFileIcon) {

            selectedFileIcon.textContent =
                getFileIcon(file);
        }


        if (filePreview) {

            filePreview.classList.remove(
                "hidden"
            );
        }


        if (analyzeButton) {

            analyzeButton.classList.remove(
                "hidden"
            );

            analyzeButton.disabled =
                false;
        }


        setText(
            fileNameFallback,
            "File selected successfully. Ready for AI analysis."
        );
    }


    function resetSelectedFile() {

        selectedFile =
            null;


        if (documentInput) {

            documentInput.value =
                "";
        }


        if (filePreview) {

            filePreview.classList.add(
                "hidden"
            );
        }


        if (analyzeButton) {

            analyzeButton.classList.add(
                "hidden"
            );

            analyzeButton.disabled =
                false;
        }


        setText(
            fileName,
            "No document selected"
        );


        setText(
            fileInfo,
            "Waiting for file"
        );


        if (selectedFileIcon) {

            selectedFileIcon.textContent =
                "📁";
        }


        setText(
            fileNameFallback,
            "Supported: JPG, PNG, WEBP and PDF documents"
        );
    }


    /* =====================================================
       PROGRESS SYSTEM
    ====================================================== */

    function updateProgress(value) {

        const safeValue =
            Math.max(
                0,
                Math.min(
                    100,
                    Math.round(value)
                )
            );


        if (progress) {

            progress.style.width =
                `${safeValue}%`;
        }


        if (percentage) {

            percentage.textContent =
                `${safeValue}%`;
        }
    }


    function animateProgress(
        from,
        to,
        duration
    ) {

        return new Promise(resolve => {

            const startTime =
                performance.now();


            function animate(currentTime) {

                const ratio =
                    Math.min(
                        (
                            currentTime -
                            startTime
                        ) /
                        duration,
                        1
                    );


                updateProgress(
                    from +
                    (
                        (to - from) *
                        ratio
                    )
                );


                if (ratio < 1) {

                    animationFrameId =
                        requestAnimationFrame(
                            animate
                        );

                } else {

                    updateProgress(to);

                    resolve();
                }
            }


            animationFrameId =
                requestAnimationFrame(
                    animate
                );
        });
    }


    /* =====================================================
       PROCESSING STEPS
    ====================================================== */

    function resetProcessingSteps() {

        processingSteps.forEach(step => {

            step.classList.remove(
                "active",
                "completed"
            );


            const status =
                step.querySelector(
                    ".step-status"
                );


            if (status) {

                status.textContent =
                    "WAITING";
            }
        });
    }


    function updateProcessingStep(
        activeIndex,
        completedIndexes = []
    ) {

        processingSteps.forEach(
            (step, index) => {

                step.classList.remove(
                    "active",
                    "completed"
                );


                const status =
                    step.querySelector(
                        ".step-status"
                    );


                if (
                    completedIndexes.includes(
                        index
                    )
                ) {

                    step.classList.add(
                        "completed"
                    );

                    if (status) {

                        status.textContent =
                            "COMPLETED";
                    }

                } else if (
                    index === activeIndex
                ) {

                    step.classList.add(
                        "active"
                    );

                    if (status) {

                        status.textContent =
                            "PROCESSING";
                    }

                } else if (status) {

                    status.textContent =
                        "WAITING";
                }
            }
        );
    }


    function completeAllSteps() {

        processingSteps.forEach(step => {

            step.classList.remove(
                "active"
            );

            step.classList.add(
                "completed"
            );


            const status =
                step.querySelector(
                    ".step-status"
                );


            if (status) {

                status.textContent =
                    "COMPLETED";
            }
        });
    }


    /* =====================================================
       OCR RESULT CARD
    ====================================================== */

    function ensureOcrCard() {

        let card =
            document.getElementById(
                "ocrResultCard"
            );


        if (card) {
            return card;
        }


        if (!resultsSection) {
            return null;
        }


        card =
            document.createElement(
                "article"
            );


        card.id =
            "ocrResultCard";


        card.className =
            "result-card glass-card";


        card.innerHTML = `

            <div class="card-title">

                <div class="title-icon">
                    ⌕
                </div>

                <div>

                    <h3>
                        OCR & Extracted Information
                    </h3>

                    <p>
                        Text extracted from uploaded document
                    </p>

                </div>

                <span
                    class="badge info"
                    id="ocrBadge"
                >
                    WAITING
                </span>

            </div>

            <div
                class="tampering-box"
                style="margin-top:16px"
            >

                <div style="width:100%">

                    <h4>
                        Extracted Text
                    </h4>

                    <pre
                        id="extractedText"
                        style="
                            margin:12px 0 0;
                            white-space:pre-wrap;
                            word-break:break-word;
                            max-height:320px;
                            overflow:auto;
                            font-family:inherit;
                            line-height:1.6;
                        "
                    >
Waiting for OCR analysis...</pre>

                </div>

            </div>

        `;


        resultsSection.prepend(
            card
        );


        return card;
    }


    /* =====================================================
       BACKEND DATA NORMALIZATION

       BACKEND STRUCTURE:

       root
       ├── analysis_data
       ├── validation
       │   └── results[]
       ├── anomaly
       │   └── tampering_analysis
       └── risk_assessment

    ====================================================== */

    function normalizeBackendData(
        backendData
    ) {

        const root =
            backendData || {};


        const analysis =
            root.analysis_data ||
            root.analysis ||
            {};


        const documentData =
            root.document ||
            analysis.document ||
            {};


        const validation =
            root.validation ||
            analysis.validation ||
            {};


        const validationResults =
            Array.isArray(
                validation.results
            )
                ? validation.results
                : [];


        const ocr = {

            extracted_text:

                analysis.extracted_text ||

                analysis.raw_ocr_text ||

                root.extracted_text ||

                "",


            confidence:

                analysis.ocr_confidence ||

                root.ocr_confidence ||

                0,


            status:

                analysis.ocr_status ||

                root.ocr_status ||

                "ANALYZED"

        };


        const anomaly =
            root.anomaly ||
            analysis.anomaly ||
            {};


        const tamperingAnalysis =
            anomaly.tampering_analysis ||
            analysis.tampering_analysis ||
            {};


        const tampering = {

            ...anomaly,

            ...tamperingAnalysis,


            status:

                tamperingAnalysis.status ||

                anomaly.status ||

                tamperingAnalysis.result ||

                anomaly.result ||

                "",


            title:

                anomaly.title ||

                tamperingAnalysis.title ||

                "",


            description:

                anomaly.description ||

                tamperingAnalysis.description ||

                tamperingAnalysis.message ||

                anomaly.message ||

                "",


            confidence:

                anomaly.confidence ||

                tamperingAnalysis.confidence ||

                anomaly.confidence_score ||

                tamperingAnalysis.confidence_score ||

                0

        };


        const face =
            root.face_verification ||
            analysis.face_verification ||
            {};


        const risk =
            root.risk_assessment ||
            analysis.risk_assessment ||
            root.risk ||
            analysis.risk ||
            {};
       const qr =
    analysis.qr_analysis ||
    root.qr_analysis ||
    {};



        return {

            root,

            analysis,

            documentData,

            validation,

            validationResults,

            ocr,

            anomaly,

            tampering,

            face,

            risk,

           qr

        };
    }


    /* =====================================================
       DOCUMENT OVERVIEW
    ====================================================== */

    function renderDocumentOverview(
        data
    ) {

        const {
            analysis,
            documentData
        } = data;


        const documentType =
            cleanValue(

                analysis.document_label ||

                analysis.document ||

                analysis.document_type ||

                documentData.document_type ||

                documentData.document_label

            ) ||
            "Unknown Document";


        const backendFileName =
            cleanValue(

                documentData.filename ||

                analysis.filename

            );


        const backendFileSize =
            Number(

                documentData.file_size ||

                analysis.file_size ||

                0

            );


        setText(
            resultFileName,

            backendFileName ||

            selectedFile?.name ||

            "Not available"
        );


        setText(
            resultFileType,
            documentType
        );


        setText(
            resultFileSize,

            backendFileSize > 0

                ? formatFileSize(
                    backendFileSize
                )

                : formatFileSize(
                    selectedFile?.size || 0
                )

        );


        if (documentTypeBadge) {

            documentTypeBadge.textContent =
                documentType;

            setStatusStyle(
                documentTypeBadge,
                "INFO"
            );
        }
    }


    /* =====================================================
       OCR DATA
    ====================================================== */

    function buildExtractedText(
        data
    ) {

        const {
            analysis,
            ocr
        } = data;


        const directText =
            cleanValue(

                ocr.extracted_text ||

                analysis.extracted_text ||

                analysis.raw_ocr_text

            );


        if (directText) {
            return directText;
        }


        const structuredData =
            analysis.structured_data ||
            {};


        const lines =
            [];


        Object.entries(
            structuredData
        ).forEach(
            ([key, value]) => {

                const cleaned =
                    cleanValue(value);


                if (cleaned) {

                    const label =
                        key
                            .replace(
                                /_/g,
                                " "
                            )
                            .replace(
                                /\b\w/g,
                                char =>
                                    char.toUpperCase()
                            );


                    lines.push(
                        `${label}: ${cleaned}`
                    );
                }
            }
        );


        return lines.length > 0

            ? lines.join("\n")

            : "No readable OCR text was detected.";
    }


    function renderOcrData(
        data
    ) {

        const card =
            ensureOcrCard();


        if (!card) {
            return;
        }


        const {
            ocr
        } = data;


        const extractedText =
            buildExtractedText(
                data
            );


        const extractedTextElement =
            document.getElementById(
                "extractedText"
            );


        const ocrBadge =
            document.getElementById(
                "ocrBadge"
            );


        setText(
            extractedTextElement,
            extractedText
        );


        if (ocrBadge) {

            const status =
                cleanValue(
                    ocr.status
                ) ||
                "ANALYZED";


            ocrBadge.textContent =
                status;


            setStatusStyle(
                ocrBadge,
                status
            );
        }
    }


    /* =====================================================
       VALIDATION RESULTS

       BACKEND:
       validation.results = [
           {
               name: "...",
               status: "...",
               ...
           }
       ]

    ====================================================== */

    function getValidationItems(
        data
    ) {

        const {
            validationResults
        } = data;


        if (
            !Array.isArray(
                validationResults
            ) ||

            validationResults.length === 0
        ) {

            return [

                {
                    label:
                        "Validation status",

                    status:
                        "REVIEW REQUIRED"
                }

            ];
        }


        return validationResults.map(
    item => {

        const label =
            cleanValue(
                item.name ||
                item.label ||
                item.check ||
                item.field
            ) ||
            "Validation check";

        let status =
            normalizeStatus(
                item.status ||
                item.result ||
                item.value,

                "REVIEW REQUIRED"
            );

        if (
            label
                .toLowerCase()
                .includes("tampering")
        ) {

            const tampering =
                data.tampering || {};

            const tamperingStatus =
                normalizeStatus(
                    tampering.status ||
                    tampering.result ||
                    "",
                    "LOW"
                );

            if (
                tamperingStatus ===
                "CRITICAL"
            ) {
                status = "CRITICAL";
            }
        }

        return {
            label,
            status
        };
    }
);
    }


    function renderValidation(
        data
    ) {

        const {
            validation
        } = data;


        const items =
            getValidationItems(
                data
            );


        if (validationList) {

            validationList.innerHTML =
                "";


            items.forEach(item => {

                const row =
                    document.createElement(
                        "div"
                    );


                row.className =
                    "validation-item";


                const label =
                    document.createElement(
                        "span"
                    );


                const status =
                    document.createElement(
                        "strong"
                    );


                label.textContent =
                    item.label;


                status.textContent =
                    item.status;


                setStatusStyle(
                    status,
                    item.status
                );


                row.appendChild(
                    label
                );


                row.appendChild(
                    status
                );


                validationList.appendChild(
                    row
                );
            });
        }


        /*
           BACKEND OVERALL STATUS
           IS PREFERRED
        */

        let finalStatus =
            normalizeStatus(

                validation.status ||

                validation.overall_status,

                ""

            );


        /*
           IF BACKEND DID NOT SEND
           OVERALL STATUS
        */

        if (!finalStatus) {

            const hasCritical =
                items.some(
                    item =>
                        item.status ===
                        "CRITICAL"
                );


            const hasInvalid =
                items.some(
                    item =>
                        item.status ===
                        "INVALID"
                );


            const hasReview =
                items.some(
                    item =>
                        item.status ===
                        "REVIEW REQUIRED"
                );


            if (hasCritical) {

                finalStatus =
                    "CRITICAL";

            } else if (hasInvalid) {

                finalStatus =
                    "INVALID";

            } else if (hasReview) {

                finalStatus =
                    "REVIEW REQUIRED";

            } else {

                finalStatus =
                    "VALID";
            }

        } else if (
            finalStatus === "PASSED"
        ) {

            finalStatus =
                "VALID";
        }


        if (validationBadge) {

            validationBadge.textContent =
                finalStatus;


            setStatusStyle(
                validationBadge,
                finalStatus
            );
        }
    }


    /* =====================================================
   TAMPERING / ANOMALY
====================================================== */

function renderTampering(
    data
) {

    const {
        tampering
    } = data;


    const rawStatus =
        cleanValue(

            tampering.status ||

            tampering.result

        );


    const detected =
        tampering.detected === true;


    /*
       FINAL STATUS LOGIC

       CRITICAL
       -> Actual critical status/result

       REVIEW REQUIRED
       -> Non-critical technical anomaly
          / warning / low-level issue

       NO CRITICAL SIGNAL
       -> No critical anomaly detected
    */

    const normalizedStatus =
        rawStatus
            ? normalizeStatus(
                rawStatus,
                "NO CRITICAL SIGNAL"
            )
            : "NO CRITICAL SIGNAL";


    /*
       TITLE
    */

    const title =
        (
            normalizedStatus ===
            "CRITICAL"

                ? "Potential Critical Anomaly Detected"

                : normalizedStatus ===
                  "REVIEW REQUIRED"

                    ? "Potential Technical Anomalies Detected"

                    : "No Critical Anomalies Detected"
        );


    /*
       DESCRIPTION
    */

    const description =
        (
            normalizedStatus ===
            "CRITICAL"

                ? "Critical document anomaly signals were detected and require manual review."

                : normalizedStatus ===
                  "REVIEW REQUIRED"

                    ? "One or more OCR, quality or file-integrity signals require review."

                    : "No critical tampering signals were detected."
        );


    /*
       CONFIDENCE
    */

    const confidence =
        Number(
            tampering.confidence || 0
        );


    /*
       STATUS BADGE
    */

    if (tamperingBadge) {

        tamperingBadge.textContent =
            normalizedStatus;


        setStatusStyle(
            tamperingBadge,
            normalizedStatus
        );
    }


    /*
       TITLE
    */

    setText(
        tamperingTitle,
        title
    );


    /*
       DESCRIPTION
    */

    setText(
        tamperingDescription,
        description
    );


    /*
       CONFIDENCE
    */

    if (tamperingConfidence) {

        tamperingConfidence.textContent =
            confidence > 0

                ? `${Math.round(confidence)}%`

                : "—";
    }
}

/* =====================================================
   QR VERIFICATION
====================================================== */

function renderQRVerification(data) {

    const qr = data.qr || {};

    const status =
        String(
            qr.status ||
            "NOT_AVAILABLE"
        ).toUpperCase();

    let qrStatus = "Not Available";
    let consistency = "Not Available";
    let description = "QR analysis is not available for this document.";

    if (status === "DECODED") {

        qrStatus = "Decoded";

        if (qr.data_consistent === true) {
            consistency = "Matched";
            description =
                "QR data is consistent with the extracted document information.";
        }
        else if (qr.data_consistent === false) {
            consistency = "Mismatch";
            description =
                "QR data does not match the extracted document information.";
        }
        else {
            consistency = "Not Compared";
            description =
                "QR code was decoded, but no comparable document field was available.";
        }

    }
    else if (status === "DATA_MISMATCH") {

        qrStatus = "Decoded";
        consistency = "Mismatch";

        description =
            "Decoded QR data is inconsistent with the extracted document information.";

    }
    else if (status === "NOT_DECODED") {

        qrStatus = "Not Decoded";
        consistency = "Not Available";

        description =
            "No readable QR code was detected in the document.";

    }
    else if (status === "ERROR") {

        qrStatus = "Error";
        consistency = "Not Available";

        description =
            "QR analysis could not be completed.";

    }

    if (!resultsSection) {
        return;
    }

    let card =
        document.getElementById(
            "qrVerificationCard"
        );

    if (!card) {

        card =
            document.createElement("article");

        card.id =
            "qrVerificationCard";

        card.className =
            "result-card glass-card";

        resultsSection.appendChild(card);
    }

    const statusClass =
        (
            status === "DECODED" &&
            qr.data_consistent === true
        )
            ? "success"
            : (
                status === "DATA_MISMATCH"
                ? "danger"
                : "info"
            );

    card.innerHTML = `

        <div class="card-title">

            <div class="title-icon">
                🔳
            </div>

            <div>
                <h3>
                    QR Verification
                </h3>

                <p>
                    QR code decoding and data consistency check
                </p>
            </div>

            <span
                class="badge ${statusClass}"
            >
                ${qrStatus}
            </span>

        </div>


        <div
            class="tampering-box"
            style="
                margin-top:16px;
                display:grid;
                grid-template-columns:
                    repeat(auto-fit,minmax(200px,1fr));
                gap:12px;
            "
        >

            <div>
                <strong>
                    QR Status
                </strong>

                <div
                    style="
                        margin-top:6px;
                        font-weight:600;
                    "
                >
                    ${qrStatus}
                </div>
            </div>


            <div>
                <strong>
                    Data Consistency
                </strong>

                <div
                    style="
                        margin-top:6px;
                        font-weight:600;
                    "
                >
                    ${consistency}
                </div>
            </div>


            ${
                qr.data_length
                    ? `
                        <div>
                            <strong>
                                QR Data Length
                            </strong>

                            <div
                                style="
                                    margin-top:6px;
                                    font-weight:600;
                                "
                            >
                                ${qr.data_length}
                            </div>
                        </div>
                    `
                    : ""
            }

        </div>


        <div
            style="
                margin-top:14px;
                line-height:1.6;
            "
        >
            ${description}
        </div>

    `;
}
   
   
        /* =====================================================
       FACE VERIFICATION
    ====================================================== */

    function renderFaceVerification(
        data
    ) {

        const face =
            data.face || {};

        const faceCount =
            Number(
                face.face_count ?? 0
            );

        const rawStatus =
            cleanValue(
                face.status ||
                face.result
            ) ||
            "NOT_AVAILABLE";

        const normalizedStatus =
            rawStatus
                .toUpperCase()
                .replace(/\s+/g, "_");

        let result =
            "Not Available";

        let description =
            cleanValue(
                face.description ||
                face.message
            );

        if (
            normalizedStatus ===
            "ONE_FACE_DETECTED"
        ) {

            result =
                "One Face Detected";

            if (!description) {

                description =
                    "One face was detected in the uploaded document image.";
            }

        } else if (
            normalizedStatus ===
            "NO_FACE_DETECTED"
        ) {

            result =
                "No Face Detected";

            if (!description) {

                description =
                    "No face was detected in the uploaded document image.";
            }

        } else if (
            normalizedStatus ===
            "MULTIPLE_FACES_DETECTED"
        ) {

            result =
                faceCount > 1
                    ? `${faceCount} Faces Detected`
                    : "Multiple Faces Detected";

            if (!description) {

                description =
                    "Multiple faces were detected. Manual review is recommended.";
            }

        } else if (
            normalizedStatus ===
            "MODEL_LOAD_ERROR"
        ) {

            result =
                "Detection Model Error";

            if (!description) {

                description =
                    "The face detection model could not be loaded.";
            }

        } else if (
            normalizedStatus ===
            "ERROR"
        ) {

            result =
                "Face Detection Failed";

            if (!description) {

                description =
                    "Face detection could not be completed.";
            }

        } else if (
            normalizedStatus ===
            "NOT_AVAILABLE"
        ) {

            result =
                "Not Available";

            if (!description) {

                description =
                    "Face detection is not available for this document.";
            }

        } else {

            result =
                rawStatus
                    .replace(/_/g, " ")
                    .toLowerCase()
                    .replace(
                        /\b\w/g,
                        char =>
                            char.toUpperCase()
                    );

            if (!description) {

                description =
                    "Face detection analysis completed.";
            }
        }


        if (faceBadge) {

            faceBadge.textContent =
                rawStatus;

            setStatusStyle(
                faceBadge,
                rawStatus
            );
        }


        setText(
            faceResult,
            result
        );


        setText(
            faceDescription,
            description
        );
    }

    /* =====================================================
       FALLBACK RISK

       USED ONLY IF BACKEND DOES NOT
       RETURN risk_assessment.score
    ====================================================== */

    function calculateFallbackRisk(
        data
    ) {

        const items =
            getValidationItems(
                data
            );


        let score =
            0;


        items.forEach(
            item => {

                if (
                    item.status ===
                    "CRITICAL"
                ) {

                    score += 45;

                } else if (
                    item.status ===
                    "INVALID"
                ) {

                    score += 30;

                } else if (
                    item.status ===
                    "REVIEW REQUIRED"
                ) {

                    score += 10;
                }
            }
        );


        const tamperingStatus =
            cleanValue(
                data.tampering.status
            ).toUpperCase();


        if (
            tamperingStatus.includes(
                "CRITICAL"
            )
        ) {

            score += 25;
        }


        return Math.min(
            score,
            100
        );
    }


   /* =====================================================
   RISK ASSESSMENT + RISK BREAKDOWN
====================================================== */

function renderRisk(data) {

    const risk =
        data.risk || {};

   let score =
    Number(
        risk.score ||
        risk.risk_score
    );

/*
   FALLBACK ONLY IF BACKEND
   SCORE IS NOT AVAILABLE
*/

if (!Number.isFinite(score)) {

    score =
        calculateFallbackRisk(
            data
        );
}


/*
   CRITICAL TAMPERING + CHECKSUM
*/

const tamperingRawStatus =
    String(
        data.tampering?.status ||
        data.tampering?.result ||
        ""
    );

const tamperingCritical =
    data.tampering?.detected === true ||
    normalizeStatus(
        tamperingRawStatus,
        data.tampering?.detected === true
            ? "CRITICAL"
            : "NO CRITICAL SIGNAL"
    ) === "CRITICAL";

const validationItems =
    getValidationItems(
        data
    );

const checksumFailed =
    validationItems.some(
        item =>
            String(item.label)
                .toLowerCase()
                .includes("checksum") &&
            String(item.status)
                .toUpperCase()
                === "INVALID"
    );


/*
   MINIMUM RISK RULE
*/

if (
    tamperingCritical &&
    checksumFailed
) {

    score =
        Math.max(
            score,
            95
        );

}
else if (
    tamperingCritical
) {

    score =
        Math.max(
            score,
            92
        );
}


/*
   FINAL SCORE LIMIT
*/

score =
    Math.max(
        0,
        Math.min(
            100,
            Math.round(score)
        )
    );
    let level =
        cleanValue(
            risk.level ||
            risk.risk_level
        );


   if (
    tamperingCritical
) {

    level =
        "HIGH RISK";

}
else if (!level) {

    if (score <= 25) {
        level = "LOW RISK";
    }
    else if (score <= 60) {
        level = "MEDIUM RISK";
    }
    else {
        level = "HIGH RISK";
    }
}


    const description =
        cleanValue(
            risk.description ||
            risk.message
        ) ||
        "Final assessment is based on backend document analysis signals.";


    /*
       EXISTING RISK UI
    */

    setText(
        riskLevel,
        level
    );

    setText(
        riskDescription,
        description
    );

    if (riskScore) {

        riskScore.textContent =
            score;
    }

    setStatusStyle(
        riskLevel,
        level
    );


    /*
       RISK BREAKDOWN
    */

    if (!resultsSection) {
        return;
    }


    let card =
        document.getElementById(
            "riskBreakdownCard"
        );


    if (!card) {

        card =
            document.createElement("article");

        card.id =
            "riskBreakdownCard";

        card.className =
            "result-card glass-card";

        resultsSection.appendChild(card);
    }


    const signals =
        Array.isArray(
            risk.signals
        )
            ? risk.signals
            : [];


    const analysis =
        data.analysis || {};


    /*
       Helper
    */

    function signalContains(text) {

        return signals.some(
            signal =>
                String(signal)
                    .toLowerCase()
                    .includes(
                        text.toLowerCase()
                    )
        );
    }


    /*
       FILE INTEGRITY
    */

    const fileIntegrity =
        signalContains("signature")
            ? "REVIEW"
            : "PASS";


    /*
       OCR QUALITY
    */

    const ocrConfidence =
        Number(
            data.ocr?.confidence || 0
        );

    let ocrQuality =
        "GOOD";

    if (
        data.ocr?.status ===
        "NO_TEXT_DETECTED"
    ) {
        ocrQuality =
            "NO TEXT";
    }
    else if (
        ocrConfidence > 0 &&
        ocrConfidence < 40
    ) {
        ocrQuality =
            "LOW";
    }
    else if (
        ocrConfidence > 0 &&
        ocrConfidence < 60
    ) {
        ocrQuality =
            "MODERATE";
    }


    /*
       DOCUMENT STRUCTURE
    */

    const structure =
        signalContains(
            "could not be parsed"
        )
            ? "INVALID"
            : "VALID";


    /*
       AADHAAR CHECKSUM
    */

    const aadhaarChecksum =
        signalContains(
            "fails checksum"
        )
            ? "FAILED"
            : (
                signalContains(
                    "aadhaar"
                )
                    ? "VALID"
                    : "N/A"
            );


    /*
       QR CONSISTENCY
    */

    const qr =
        data.qr || {};

    let qrConsistency =
        "N/A";

    if (
        qr.status ===
        "DATA_MISMATCH"
    ) {
        qrConsistency =
            "MISMATCH";
    }
    else if (
        qr.data_consistent === true
    ) {
        qrConsistency =
            "MATCHED";
    }
    else if (
        qr.status ===
        "DECODED"
    ) {
        qrConsistency =
            "NOT COMPARED";
    }


    /*
       TAMPERING
    */

    const tampering =
        data.tampering || {};

    let tamperingStatus =
    "LOW";

const normalizedTamperingStatus =
    normalizeStatus(
        tampering.status ||
        tampering.result ||
        "",
        tampering.detected === true
            ? "CRITICAL"
            : "LOW"
    );

if (
    normalizedTamperingStatus === "CRITICAL" ||
    tampering.detected === true ||
    signalContains("tampering") ||
    signalContains("editing-software") ||
    signalContains("recompression") ||
    signalContains("inconsistent image")
) {
    tamperingStatus =
        "CRITICAL";
}


    /*
       FACE
    */

    const face =
        data.face || {};

    let faceStatus =
        "N/A";

    const faceCount =
        Number(
            face.face_count || 0
        );

    if (faceCount === 1) {
        faceStatus =
            "1 FACE";
    }
    else if (faceCount > 1) {
        faceStatus =
            `${faceCount} FACES`;
    }
    else if (
        String(face.status || "")
            .toUpperCase() ===
        "NO_FACE_DETECTED"
    ) {
        faceStatus =
            "NO FACE";
    }


    /*
       RENDER
    */

    const rows = [
        [
            "File Integrity",
            fileIntegrity
        ],
        [
            "OCR Quality",
            ocrQuality
        ],
        [
            "Document Structure",
            structure
        ],
        [
            "Aadhaar Checksum",
            aadhaarChecksum
        ],
        [
            "QR Consistency",
            qrConsistency
        ],
        [
            "Tampering Signals",
            tamperingStatus
        ],
        [
            "Face Detection",
            faceStatus
        ]
    ];


    card.innerHTML = `

        <div class="card-title">

            <div class="title-icon">
                🛡
            </div>

            <div>
                <h3>
                    Risk Breakdown
                </h3>

                <p>
                    Signals contributing to the final risk assessment
                </p>
            </div>

            <span
                class="badge info"
            >
                ${level}
            </span>

        </div>


        <div
            style="
                margin-top:16px;
                display:grid;
                gap:10px;
            "
        >

            ${
                rows.map(
                    row => `

                        <div
                            style="
                                display:flex;
                                justify-content:space-between;
                                align-items:center;
                                gap:16px;
                                padding:12px 14px;
                                border-radius:10px;
                                background:rgba(255,255,255,0.04);
                            "
                        >

                            <span>
                                ${row[0]}
                            </span>

                            <strong>
                                ${row[1]}
                            </strong>

                        </div>

                    `
                ).join("")
            }

        </div>


        <div
            style="
                margin-top:18px;
                padding:16px;
                border-radius:12px;
                background:rgba(255,255,255,0.05);
                display:flex;
                justify-content:space-between;
                align-items:center;
                gap:16px;
            "
        >

            <div>

                <div
                    style="
                        font-size:13px;
                        opacity:.75;
                    "
                >
                    FINAL RISK SCORE
                </div>

                <strong
                    style="
                        font-size:28px;
                    "
                >
                    ${score}/100
                </strong>

            </div>


            <strong>
                ${level}
            </strong>

        </div>

    `;
}

    /* =====================================================
       RENDER COMPLETE ANALYSIS RESULT
    ====================================================== */

    function renderAnalysisResult(
        backendData
    ) {

        console.log(
            "SecureDoc AI Backend Response:",
            backendData
        );


        const data =
            normalizeBackendData(
                backendData
            );


        console.log(
            "Normalized Backend Data:",
            data
        );


        renderDocumentOverview(
            data
        );


        renderOcrData(
            data
        );


        renderValidation(
            data
        );


        renderTampering(
            data
        );

       renderQRVerification(data);


        renderFaceVerification(
            data
        );


        renderRisk(
            data
        );


        if (resultSummary) {

            resultSummary.textContent =

                cleanValue(
                    backendData.message
                ) ||

                "Document analysis completed successfully.";
        }
    }


    /* =====================================================
       START AI ANALYSIS
    ====================================================== */

    async function startAIAnalysis() {

        if (!selectedFile) {

            alert(
                "Please select a document first."
            );

            return;
        }


        if (analysisRunning) {
            return;
        }


        analysisRunning =
    true;

// Lock file selection while analysis is running
if (documentInput) {
    documentInput.disabled = true;
}

if (uploadBox) {
    uploadBox.classList.add("upload-locked");
}

if (removeFile) {
    removeFile.disabled = true;
}

if (analyzeButton) {
            analyzeButton.disabled =
                true;

            analyzeButton.innerHTML =
                "Analyzing...";
        }


        if (resultsSection) {

            resultsSection.classList.add(
                "hidden"
            );
        }


        updateProgress(0);

        resetProcessingSteps();


        if (processingSection) {

            processingSection.classList.remove(
                "hidden"
            );
        }


        try {

            /* ==========================
               STEP 1
            ========================== */

            updateProcessingStep(
                0,
                []
            );


            setText(
                processingText,
                "Preparing secure document upload..."
            );


            await animateProgress(
                0,
                15,
                350
            );


            /* ==========================
               STEP 2
               UPLOAD TO BACKEND
            ========================== */

            updateProcessingStep(
                1,
                [0]
            );


            setText(
                processingText,
                "Uploading document to SecureDoc AI backend..."
            );


            const formData =
                new FormData();


            formData.append(
                "file",
                selectedFile
            );


            const response =
                await fetch(
                    UPLOAD_ENDPOINT,
                    {
                        method: "POST",
                        body: formData
                    }
                );


            if (!response.ok) {

                let errorMessage =
                    `Upload failed with status ${response.status}`;


                try {

                    const errorData =
                        await response.json();


                    errorMessage =

                        errorData.detail ||

                        errorData.message ||

                        errorMessage;

                } catch (error) {

                    console.error(
                        "Could not parse backend error:",
                        error
                    );
                }


                throw new Error(
                    errorMessage
                );
            }


            const backendData =
                await response.json();


            console.log(
                "Raw backend response:",
                backendData
            );


            await animateProgress(
                15,
                55,
                400
            );


            /* ==========================
               STEP 3
            ========================== */

            updateProcessingStep(
                2,
                [0, 1]
            );


            setText(
                processingText,
                "Processing OCR and validation results..."
            );


            await animateProgress(
                55,
                75,
                350
            );


            /*
               RENDER REAL BACKEND DATA
            */

            renderAnalysisResult(
                backendData
            );


            /* ==========================
               STEP 4
            ========================== */

            updateProcessingStep(
                3,
                [0, 1, 2]
            );


            setText(
                processingText,
                "Analyzing anomaly and security signals..."
            );


            await animateProgress(
                75,
                92,
                300
            );


            /* ==========================
               STEP 5
            ========================== */

            updateProcessingStep(
                4,
                [0, 1, 2, 3]
            );


            setText(
                processingText,
                "Finalizing security assessment..."
            );


            await animateProgress(
                92,
                100,
                250
            );


            completeAllSteps();


            setText(
                processingText,

                cleanValue(
                    backendData.message
                ) ||

                "Document analysis completed successfully."
            );


            await wait(400);


            if (resultsSection) {

                resultsSection.classList.remove(
                    "hidden"
                );


                resultsSection.scrollIntoView({
                    behavior:
                        "smooth",

                    block:
                        "start"
                });
            }

        } catch (error) {

            console.error(
                "SecureDoc AI Error:",
                error
            );


            setText(
                processingText,
                `Analysis failed: ${error.message}`
            );


            alert(
                `Document analysis failed.

${error.message}

Please check:
1. FastAPI backend is running
2. Backend is running on port 8000
3. Browser console for errors`
            );

        } finally {

    analysisRunning =
        false;

    // Unlock file selection after analysis finishes
    if (documentInput) {
        documentInput.disabled = false;
    }

    if (uploadBox) {
        uploadBox.classList.remove("upload-locked");
    }

    if (removeFile) {
        removeFile.disabled = false;
    }

    if (analyzeButton) { 
                analyzeButton.disabled =
                    false;

                analyzeButton.innerHTML =
                    "<span>✦</span> Start AI Analysis";
            }
        }
    }


    /* =====================================================
       EVENT LISTENERS
    ====================================================== */

    if (startAnalysisButton) {

        startAnalysisButton.addEventListener(
            "click",
            () => {

                if (workspace) {

                    workspace.scrollIntoView({
                        behavior:
                            "smooth"
                    });
                }


                setTimeout(() => {

    if (analysisRunning) {
        return;
    }

    if (documentInput) {

        documentInput.click();
    }

}, 400);
            }
        );
    }


    /* =====================================================
       UPLOAD BOX CLICK
    ====================================================== */

    if (uploadBox) {

        uploadBox.addEventListener(
    "click",
    () => {

        if (analysisRunning) {
            return;
        }

        if (documentInput) {

            documentInput.click();
        }
    }
);

        /* DRAG ENTER + OVER */

        [
            "dragenter",
            "dragover"
        ].forEach(
            eventName => {

                uploadBox.addEventListener(
                    eventName,
                    event => {

                        event.preventDefault();

                        uploadBox.classList.add(
                            "drag-active"
                        );
                    }
                );
            }
        );


        /* DRAG LEAVE + DROP */

        [
            "dragleave",
            "drop"
        ].forEach(
            eventName => {

                uploadBox.addEventListener(
                    eventName,
                    event => {

                        event.preventDefault();

                        uploadBox.classList.remove(
                            "drag-active"
                        );
                    }
                );
            }
        );


        /* DROP FILE */

        uploadBox.addEventListener(
            "drop",
            event => {

                const file =
                    event.dataTransfer.files[0];


                if (!file) {
                    return;
                }


                if (
                    documentInput &&
                    window.DataTransfer
                ) {

                    const dataTransfer =
                        new DataTransfer();


                    dataTransfer.items.add(
                        file
                    );


                    documentInput.files =
                        dataTransfer.files;
                }


                selectFile(
                    file
                );
            }
        );
    }


    /* =====================================================
       FILE INPUT
    ====================================================== */

    if (documentInput) {

        documentInput.addEventListener(
            "change",
            event => {

                const file =
                    event.target.files[0];


                if (file) {

                    selectFile(
                        file
                    );
                }
            }
        );
    }


    /* =====================================================
       REMOVE FILE
    ====================================================== */

    if (removeFile) {

        removeFile.addEventListener(
            "click",
            event => {

                event.stopPropagation();

                resetSelectedFile();
            }
        );
    }


    /* =====================================================
       ANALYZE BUTTON
    ====================================================== */

    if (analyzeButton) {

        analyzeButton.addEventListener(
            "click",
            startAIAnalysis
        );
    }


    /* =====================================================
       NEW SCAN
    ====================================================== */

    if (newScanButton) {

        newScanButton.addEventListener(
            "click",
            () => {

                if (animationFrameId) {

                    cancelAnimationFrame(
                        animationFrameId
                    );
                }


                analysisRunning =
                    false;


                if (processingSection) {

                    processingSection.classList.add(
                        "hidden"
                    );
                }


                if (resultsSection) {

                    resultsSection.classList.add(
                        "hidden"
                    );
                }


                updateProgress(
                    0
                );


                resetProcessingSteps();


                resetSelectedFile();


                if (workspace) {

                    workspace.scrollIntoView({

                        behavior:
                            "smooth",

                        block:
                            "start"

                    });
                }
            }
        );
    }


    /* =====================================================
       INITIAL STATE
    ====================================================== */

    resetProcessingSteps();


    updateProgress(
        0
    );


    if (resultsSection) {

        resultsSection.classList.add(
            "hidden"
        );
    }


    console.log(
        "SecureDoc AI frontend initialized successfully."
    );

});
