
# 🛡️ SecureDoc AI
### AI-Powered Document Screening & Fraud Risk Analysis System

<p align="center">
  <img src="https://img.shields.io/badge/Domain-AI%20%26%20Cybersecurity-blue" />
  <img src="https://img.shields.io/badge/Project-SIH26188-purple" />
  <img src="https://img.shields.io/badge/Frontend-HTML%20%7C%20CSS%20%7C%20JavaScript-orange" />
  <img src="https://img.shields.io/badge/Backend-Python%20%7C%20FastAPI-green" />
</p>

<p align="center">
  An intelligent document screening solution designed to assist
  identity document verification through OCR, validation,
  image analysis, and risk-based assessment.
</p>

<p align="center">
  <a href="https://ankitprajapati29.github.io/SecureDoc--AI/">
    <strong>🚀 Live </strong>
  </a>
  |
  <a href="https://github.com/ankitprajapati29">
    <strong>👨‍💻 Developer Profile</strong>
  </a>
</p>

---

## 📌 1. Project Overview

**SecureDoc AI** is an AI-assisted document screening and fraud-risk analysis project developed for the Smart India Hackathon problem reference **SIH26188**.

The project explores how document processing, Optical Character Recognition (OCR), validation rules, and image analysis can assist in screening identity documents and identifying potentially suspicious characteristics.

Traditional document verification can require significant manual effort. SecureDoc AI aims to organize relevant document information and screening signals into a structured assessment that can support further human verification.

The system is designed around multiple screening layers rather than relying on a single OCR result or validation check.

> **Important:** SecureDoc AI is a screening and decision-support prototype. Its risk assessment should not be treated as conclusive proof that a document is genuine or fraudulent.

---

## 🎯 2. Problem Statement

Identity document verification can be challenging when documents contain poor-quality images, inconsistent information, unusual visual characteristics, or potentially manipulated content.

Manual screening may also be time-consuming when multiple document types need to be examined.

A document screening solution should help users:

- Extract readable information from document images and PDFs.
- Identify the likely document type.
- Validate extracted information using relevant rules.
- Examine image quality and suspicious inconsistencies.
- Combine available signals into an understandable risk assessment.
- Present findings in a structured security report.

SecureDoc AI explores a layered, technology-assisted approach to these challenges.

---

## 💡 3. Proposed Solution

SecureDoc AI combines document text extraction, document-specific validation, image processing, and risk analysis into a unified screening workflow.

The proposed system follows these principles:

- **Multi-pass OCR:** Use alternative OCR configurations when necessary to improve text extraction.
- **Document-specific validation:** Apply appropriate checks depending on the detected document type.
- **Multi-layer screening:** Consider text, validation, image quality, and anomaly signals together.
- **Risk-based assessment:** Organize findings into Low, Medium, or High risk categories.
- **Explainable results:** Present the screening signals that contribute to the assessment.
- **Human review:** Keep the final verification decision with an authorized reviewer.

---

## ✨ 4. Key Features

### 📄 Document Processing
- Support for document images and PDF processing, depending on the implemented workflow.
- Text extraction using OCR.
- PDF text extraction and page rendering.
- Image preprocessing for document analysis.

### 🔍 Document Information Extraction
- Extract readable text from supported documents.
- Identify relevant information fields.
- Organize extracted information for subsequent validation.

### ✅ Document Validation
- Apply document-specific validation rules.
- Check extracted fields for inconsistencies.
- Use checksum validation where applicable, such as the Verhoeff algorithm for Aadhaar-number format validation.

*A valid checksum does not establish the authenticity or ownership of an identity document.*

### 🖼️ Image Analysis
- Inspect document image quality.
- Support image preprocessing and visual analysis.
- Examine potential anomalies where the implemented analysis supports them.

### 🛡️ Fraud-Risk Assessment
- Combine available validation and analysis signals.
- Classify the overall screening result into Low, Medium, or High risk categories.
- Highlight suspicious characteristics for further review.

### 📊 Security Assessment Report
- Present document information and screening findings in a structured format.
- Help reviewers understand why a document may require additional inspection.
- Support a consistent review process.

*Actual feature availability depends on the implementation and the backend services connected to the deployed frontend.*

---

## 🔄 5. System Workflow

The intended screening workflow is:

1. **Document Upload**  
   The user submits a supported identity document.

2. **File Processing**  
   The document is prepared for text extraction and analysis.

3. **OCR Processing**  
   Tesseract OCR extracts readable text. Alternative page segmentation settings may be used when needed.

4. **Document Type Identification**  
   The system determines the likely document category using available document characteristics and extracted information.

5. **Information Validation**  
   Relevant extracted fields are checked using document-specific validation rules.

6. **Image and Anomaly Analysis**  
   Image quality and available visual or structural indicators are examined.

7. **Risk Assessment**  
   Available screening signals contribute to the risk classification.

8. **Security Report**  
   The results are organized for review and follow-up verification.

---

## 🏗️ 6. Technology Stack

### Frontend

| Technology | Purpose |
|---|---|
| HTML5 | Web page structure |
| CSS3 | Styling and responsive layout |
| JavaScript | User interaction and frontend logic |

### Backend and API

| Technology | Purpose |
|---|---|
| Python | Backend processing and validation logic |
| FastAPI | HTTP API framework |
| Uvicorn | ASGI server for running the API |

### OCR and Document Processing

| Technology | Purpose |
|---|---|
| Tesseract OCR | Optical Character Recognition |
| PyTesseract | Python interface for Tesseract |
| PyMuPDF | PDF text extraction and page rendering |
| OpenCV | Image processing and analysis |
| Pillow | Image loading and manipulation |

### Validation and File Handling

| Technology | Purpose |
|---|---|
| Pydantic | Data validation and structured data models |
| Python Multipart | Handling file uploads in supported API workflows |
| Verhoeff Algorithm | Checksum validation for applicable numeric identifiers |

### Deployment

| Platform | Intended Role |
|---|---|
| GitHub Pages | Static frontend hosting |
| Render | Backend deployment |

The technologies above describe the project stack and intended architecture. A component should be considered operational only when its implementation and integration have been verified.

---

## 🧠 7. Risk Assessment Approach

The system is designed to organize screening findings into three risk categories.

| Risk Level | Interpretation |
|---|---|
| 🟢 Low | Available screening checks indicate relatively few concerns |
| 🟡 Medium | Some inconsistencies or uncertain signals require review |
| 🔴 High | Multiple or significant suspicious signals require further investigation |

Risk levels represent screening outcomes, not guaranteed fraud probabilities.

OCR errors, poor image quality, document variations, and incomplete information can affect results. A high-risk result does not independently prove fraud, and a low-risk result does not guarantee authenticity.

---

## 🔐 8. Security and Privacy Considerations

Identity documents contain sensitive personal information. A document screening application should handle such information carefully.

Important considerations include:

- Validate uploaded file types and file sizes.
- Reject unsupported or malformed files safely.
- Avoid exposing uploaded documents or extracted personal information.
- Use HTTPS for communication with deployed APIs.
- Apply appropriate access controls and secure file handling.
- Avoid storing identity documents longer than necessary.
- Remove temporary files when processing is complete.
- Avoid logging sensitive document contents.
- Clearly communicate the limitations of automated screening.

These measures should be verified against the actual deployment before the system is used with real identity documents.

---

## ⚠️ 10. Current Limitations

Document screening systems have several technical limitations:

- OCR quality depends on image clarity, language, fonts, and document layout.
- Poor-quality scans may result in incorrect or missing extracted information.
- Validation rules may not cover every legitimate document variation.
- Image anomalies alone cannot establish document tampering.
- Risk assessments can produce false positives and false negatives.
- Authenticity cannot be guaranteed without appropriate authoritative verification.
- Backend-dependent features will not function when their required services are unavailable.

SecureDoc AI should therefore be treated as an assistive screening prototype rather than a replacement for official verification procedures.

---

## 🔮 11. Future Enhancements

Potential improvements include:

- More robust document type classification.
- Better image preprocessing and OCR accuracy.
- Expanded validation rules for supported document types.
- Improved explainability of risk assessments.
- More reliable tampering and anomaly analysis.
- Secure integration between frontend and backend.
- Automated testing and API error handling.
- Performance optimization for multi-page PDF documents.
- Stronger privacy, access control, and audit capabilities.
- Evaluation using representative test datasets and documented accuracy metrics.

---

## 🎓 12. Project Information

| Field | Details |
|---|---|
| Project Name | SecureDoc AI |
| Project Reference | SIH26188 |
| Domain | AI and Cybersecurity |
| Application Area | Identity Document Screening |
| Project Type | Web-Based Screening Prototype |
| Frontend Hosting | GitHub Pages |
| Backend Platform | Render, where configured |

---

## 👨‍💻 13. Developer

**Ankit Prajapati**

B.Tech Computer Science and Engineering Student

- GitHub: https://github.com/ankitprajapati29
- Live Demo: https://ankitprajapati29.github.io/SecureDoc--AI/

---

## 📜 14. Disclaimer

SecureDoc AI is an educational and prototype project intended to explore document screening and fraud-risk analysis.

It is not an official government verification service and does not independently establish the authenticity of identity documents. Results should be reviewed by an appropriately authorized person before consequential decisions are made.

---

<p align="center">
  <strong>Built to explore safer, smarter document screening.</strong>
</p>
