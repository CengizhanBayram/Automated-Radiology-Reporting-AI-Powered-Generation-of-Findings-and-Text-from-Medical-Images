# Dataset Card

## Dataset Card for Multi-Source Radiology Chest Image Dataset

A unified chest X-ray (CXR) dataset definition that allows mixing large public radiology sources (e.g. MIMIC-CXR) with local/experimental image folders under one common label vocabulary. It is designed so you can train a multi-label image model (e.g. DenseNet121) and also feed the same labels to a radiology/RAG/LLM pipeline (e.g. `Cosmobillian/samsung_innovation_campus_radiology_model` or your own multilingual radiology model).

---

## Dataset Details

### Dataset Description
This dataset groups chest radiograph images and their associated radiology findings into a single, consistent schema. It is meant to bridge two typical pipelines:

1. **Vision pipeline** – train/evaluate a multi-label CXR classifier on a fixed set of findings (Atelectasis, Effusion, Cardiomegaly, Normal, …).
2. **Text/RAG pipeline** – take those predicted findings and pass them to a retrieval or radiology LLM that speaks in MIMIC-like style (English or multilingual).

The dataset itself does **not** ship the raw MIMIC-CXR images; it documents how to organize images and labels you download under original licenses.

- **Tasks:** multi-label chest X-ray classification, report conditioning, RAG query building
- **Modalities:** frontal/posterior chest X-ray (CXR) images, optionally other modalities later

**Curated by:** user / project team  
**License:** original sources’ licenses (e.g. PhysioNet for MIMIC-CXR; medmnist license for ChestMNIST; Kaggle dataset licenses for Kaggle CXRs)

---

## Dataset Sources (optional)

- **Repository:** MIMIC-CXR mirror on Hugging Face; Kaggle chest X-ray datasets; local Google Drive folders
- **Paper (optional):** Johnson AEW et al., “MIMIC-CXR, a large publicly available database of labeled chest radiographs.”
- **Demo (optional):** Hugging Face model page (e.g. `Cosmobillian/samsung_innovation_campus_radiology_model`) or the project’s Colab notebook

---

## Uses

### Direct Use
- Train a **multi-label** chest X-ray classifier (e.g. DenseNet121 → 14 findings).
- Feed model outputs into a **RAG** layer that contains short radiology explanations per finding.
- Generate **radiology-style reports** by giving the findings + retrieved notes to an LLM (Gemini / HF radiology model).
- Benchmark or re-train HF models that expect MIMIC-style finding names.

### Out-of-Scope / Caution
- Direct clinical decision making.
- Redistributing licensed MIMIC-CXR images/reports outside allowed terms.
- Changing the medical meaning of labels without documenting it.

---

## Dataset Structure
A minimal structure looks like this:

- **image_id**: unique identifier
- **image_path**: relative/absolute path to the CXR image (PNG/JPG/converted DICOM)
- **labels**: one or more chest findings from a shared vocabulary  
  - e.g. `Atelectasis|Effusion` for multi-label  
  - or a JSON list like `["Atelectasis", "Effusion"]`
- **split**: `train`, `val`, or `test`
- **source** (optional): `mimic_cxr`, `chestmnist`, `local`, …

**Example:**

| image_id         | image_path                  | labels                | split | source      |
|------------------|-----------------------------|-----------------------|-------|-------------|
| p10/p1000001.png | images/p10/p1000001.png     | Atelectasis\|Effusion | train | mimic_cxr   |
| p10/p1000002.png | images/p10/p1000002.png     | Normal                | train | mimic_cxr   |
| local_001.png    | drive/cxr/local_001.png     | Cardiomegaly          | val   | local_drive |

This structure is compatible with both:
- vision dataloaders (PyTorch, torchvision)
- text/RAG layers (because labels are clean strings)

---

## Dataset Creation

### Source Data
You can combine multiple sources:

1. **Public / credentialed**: MIMIC-CXR (HF mirror or PhysioNet), NIH ChestX-ray14, CheXpert.
2. **Small / educational**: ChestMNIST (medmnist) – easy to download in Colab.
3. **Local**: folders on Google Drive or local disk, already organized by patient/case.

Each source keeps its own license.

### Data Collection and Processing
1. **Download** each source under its own terms.
2. **Filter** to frontal/PA views if needed.
3. **Normalize label names** to a single vocabulary, e.g.:
   - “Pleural Effusion” → “Effusion”
   - “No Finding” → “Normal”
4. **Create splits**: 80% train, 10% validation, 10% test (or to match original splits).
5. **Save** one central file such as `dataset.csv` that lists `image_id, image_path, labels, split, source`, and keep the images in their folders.

This single CSV can then be consumed by your vision model and also by a RAG builder that needs the label strings.

---

## Features and the Target
- **Features:**
  - image: chest radiograph (PNG/JPG/DICOM-converted)
  - optional metadata: source, view position, patient sex/age (if allowed)
- **Target:**
  - multi-label list of chest findings (strings)

This matches what typical CXR models output (multi-label sigmoid over N findings).

---

## Annotations (optional)

### Annotation Process
This dataset does **not** introduce new medical annotations. It **reuses** the labels provided by the original datasets (MIMIC-CXR, NIH CXR, etc.) or the folder structure of local data (e.g. `/Normal`, `/Pneumonia`, …).

If further annotation is required (e.g. adding “COVID” or “Post-op change”), new rows can be added to the CSV following the same format.

### Who are the Annotators?
- For public medical datasets: clinical experts/annotators from the original dataset.
- For local folders: the user/project team who organized the images.
- No extra crowdsourcing is assumed here.

---

## Bias, Risks, and Limitations
- **Source bias:** MIMIC-CXR is single-center; demographics and imaging protocols may not generalize.
- **Class imbalance:** rare findings (Hernia, Pneumothorax) will be under-represented; training may need class weights or focal loss.
- **Chained errors in RAG:** if the image model predicts a wrong label, the RAG step will retrieve a wrong passage, and the LLM may produce a plausible but incorrect report.
- **Licensing:** some sources (MIMIC-CXR) must not be redistributed; only the schema/card should be shared.
- **Non-clinical use:** outputs should be treated as research/assistant material, not as diagnostic reports.

