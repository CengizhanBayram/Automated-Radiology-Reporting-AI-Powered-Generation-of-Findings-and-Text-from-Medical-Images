---
# MODEL CARD

# Model Card for Multi-Source Radiology Vision+RAG / Multilingual Report Model

This model setup describes a radiology workflow that can run in two modes:

1. **Vision mode**: a multi-label chest X-ray classifier (e.g. DenseNet121) trained on a unified chest-finding vocabulary.
2. **Text/RAG mode**: the vision output (findings) is sent to a small RAG layer and then to a multilingual / MIMIC-style radiology LLM to generate a human-readable report.

It is intended for research, prototyping, and educational radiology assistants — not for clinical deployment.

---

## Model Details

### Model Description

This model/card represents a composite radiology assistant:

- A **vision backbone** (e.g. `torchvision.models.densenet121`) finetuned to predict multiple chest findings (Atelectasis, Cardiomegaly, Effusion, Consolidation/Pneumonia, Normal, etc.) from frontal CXR.
- A **retrieval component (RAG)** that, given those findings, pulls short explanatory passages (e.g. “Pleural effusion: blunting of costophrenic angle…”).
- A **multilingual / MIMIC-style radiology LLM** (for example the style of `Cosmobillian/samsung_innovation_campus_radiology_model`, or your own model) that turns `findings + retrieved passages` into a structured report in Turkish or English.

This card documents the whole stack so the same dataset (the one we defined in the dataset card) can feed both vision and text.

- **Developed by:** project / user  
- **Model date:** [More Information Needed]  
- **Model type:** multi-label CXR classifier + retrieval-augmented generation for radiology reports  
- **Language(s):** English, Turkish (via multilingual radiology model / downstream LLM)  
- **Finetuned from model (optional):** `torchvision.models.densenet121` (ImageNet weights)

### Model Sources (optional)

- **Repository:** [More Information Needed]  
- **Paper (optional):** MIMIC-CXR original paper (for style and labels)  
- **Demo (optional):** Gradio/Streamlit UI that takes an image and outputs: model findings, RAG notes, final prompt

---

## Uses

### Direct Use

- Upload a frontal chest X-ray → get model-predicted findings (multi-label).
- Show the top retrieved radiology notes for these findings (RAG).
- Build a ready-to-send prompt for Gemini / HF radiology model to produce a full report.
- Educational/assistant use: show “why” a finding matters by attaching a short description.

### Downstream Use (optional)

- Fine-tune the vision part on a larger hospital-specific CXR dataset.
- Replace the RAG knowledge base with a bigger, hospital-specific set of radiology guidelines.
- Plug the final prompt into production LLM endpoints (Gemini, OpenAI, HF Inference) to standardize reporting style.

### Out-of-Scope Use

- Real-time clinical diagnosis or treatment decisions.
- Use on non-chest modalities (CT, MRI, US) without retraining.
- Use on patient populations that differ greatly from the training data, without validation.
- Automatic distribution of MIMIC-CXR data if the source is under restricted license.

---

## Bias, Risks, and Limitations

- Vision model may **overfit** to the small source (e.g. ChestMNIST-like images) and not generalize to full-size hospital CXRs.
- Labels are **multi-label** and often weakly supervised; some findings (Hernia, Pneumothorax) are rare.
- RAG is **downstream** of vision — if vision is wrong, RAG will confidently fetch the wrong explanation.
- Generated reports can **hallucinate** extra findings because LLMs are generative.
- Source datasets (e.g. MIMIC-CXR) come from a single center → demographic and device bias.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases, and limitations of the model. Always review the generated report by a human radiologist in any setting beyond research. Validate on your own data distribution before deployment.

---

## How to Get Started with the Model

```python
import torch
from torchvision import models, transforms
from PIL import Image

# 1) load vision model
num_labels = 22  # example unified vocabulary size
model = models.densenet121(weights=None)
model.classifier = torch.nn.Linear(model.classifier.in_features, num_labels)
state = torch.load("/path/to/best_model.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

# 2) preproc
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

img = Image.open("cxr.png").convert("RGB")
x = tfm(img).unsqueeze(0)
with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits).squeeze(0)

# 3) pick findings above threshold and send to your RAG/LLM
