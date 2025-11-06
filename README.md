# Automated-Radiology-Reporting-AI-Powered-Generation-of-Findings-and-Text-from-Medical-Images 

To develop an integrated AI system that enhances radiologists' workflow by automatically analyzing text-based radiology reports (using NLP) and localizing the described clinical findings on corresponding medical images (using Computer Vision).HF model link: https://huggingface.co/Cosmobillian/samsung_innovation_campus_radiology_model
<img width="1882" height="940" alt="image" src="https://github.com/user-attachments/assets/06b61eb7-1864-470c-8aa5-c648c695f5e8" />
<img width="1807" height="766" alt="image" src="https://github.com/user-attachments/assets/1b4eb9ae-9c40-42c3-a2f2-c21803fdf8fa" />
````markdown
# Automated Radiology Reporting – AI-Powered Generation of Findings and Text from Medical Images

This repository shows **two complementary approaches** to assist radiology workflows:

1. **Image → Findings → RAG → LLM path**  
   A Streamlit app (`rag+densenet121/app.py`) that takes a medical image (CXR-like), runs a DenseNet-121 multi-label classifier on it, retrieves short explanatory radiology snippets (mini-RAG), and then builds a prompt you can send to Gemini (via LangChain) to get a nicely formatted radiology-style report.

2. **Text / multimodal NLP path**  
   A notebook (`multimodel_llm/radiologist_llama.ipynb`) that experiments with text-based / multilingual radiology generation and can be combined with the Hugging Face model:  
   **HF model:** https://huggingface.co/Cosmobillian/samsung_innovation_campus_radiology_model

Both paths can be used in the same project: one starts from the **image**, the other from **text**, but both end up producing a radiology-style output.

---

## 1. Repository Structure

```text
.
├── multimodel_llm/
│   └── radiologist_llama.ipynb        # text / LLM experiments
├── rag+densenet121/
│   ├── app.py                         # Streamlit UI: image → DenseNet → RAG → Gemini
│   └── densenet_model/
│       └── chestmnist_best.pt         # trained DenseNet-121 weights (14-label, ChestMNIST-style)
├── datasetcard.md                     # dataset card for multi-source CXR idea
├── modelcard.md                       # model card describing vision+LLM approach
├── LICENSE
└── README.md                          # this file
````

> `rag+densenet121/densenet_model/chestmnist_best.pt`


---

## 2. Image → Findings → RAG → LLM

This is the main “automated reporting” flow in `rag+densenet121/app.py`.

**Steps:**

1. **Upload image** (PNG/JPG).

2. **DenseNet-121 classifier** runs on the image.
   The classifier is set up for **multi-label chest findings** (ChestMNIST-like 14 labels):

   * Atelectasis
   * Cardiomegaly
   * Effusion
   * Infiltration
   * Mass
   * Nodule
   * Pneumonia
   * Pneumothorax
   * Consolidation
   * Edema
   * Emphysema
   * Fibrosis
   * Pleural_Thickening
   * Hernia

3. The model outputs probabilities, e.g.:

   ```text
   - Infiltration (p=0.07)
   - Atelectasis (p=0.05)
   - Effusion (p=0.02)
   ```

4. These predicted labels are turned into a **query** for a tiny in-memory RAG (TF-IDF over 4–5 short radiology paragraphs such as “Pleural effusion on PA CXR…”).

5. The app **builds a prompt** that contains:

   * model findings (with probabilities),
   * retrieved radiology notes,
   * and an instruction to “write a Turkish radiology report with Findings / Impression / Recommendation”.

6. The prompt is sent to **Gemini via LangChain** (you enter your API key in the sidebar).

7. The report is shown in Streamlit.

So the UI shows **three outputs**:

* Model Findings
* RAG Explanations
* LLM / Gemini Prompt (and the generated report if you click the button)

---

## 3. Text / HF Model Path

The repo also mentions and links to:

* **`Cosmobillian/samsung_innovation_campus_radiology_model`** on Hugging Face

This model can be used to **refine or generate radiology text**. You can do:

* Image model → labels → HF radiology model → nicer sentences
* or
* Raw clinical text → HF model → radiology wording → (optional) RAG → Gemini

That’s why there is a separate folder: `multimodel_llm/`. It represents the **“I also tried the LLM-based route”** part of your project.

---

## 4. Running the App

### 4.1 Install dependencies

```bash
pip install streamlit torch torchvision scikit-learn pillow \
            langchain langchain-google-genai
```

### 4.2 Run Streamlit

From inside the `rag+densenet121` folder:

```bash
cd rag+densenet121
streamlit run app.py
```

Then open the URL Streamlit prints (usually [http://localhost:8501](http://localhost:8501)).

### 4.3 Model path

In `app.py` you’ll see:

```python
MODEL_PATH = r".\densenet_model\chestmnist_best.pt"
```

Leave it like that **as long as** your file is at:

```text
rag+densenet121/densenet_model/chestmnist_best.pt
```

If you rename the file or move it, update `MODEL_PATH`.

---

## 5. What the Model Actually Is

* It is a **DenseNet-121** from `torchvision.models`.
* The classifier head is replaced with `nn.Linear(in_features, num_labels)` where `num_labels = 14`.
* It was trained to behave like a ChestMNIST-style multi-label chest classifier.
* The Streamlit app loads this `.pt` weight and applies **sigmoid** on the logits.

Because it is multi-label, the app:

* keeps all labels above a threshold (default 0.4),
* or, if none passes threshold, picks the top-3.

This makes it easy to always generate *some* findings, which you need for RAG/LLM.

---

## 6. Dataset and Model Cards

You already have:

* **`datasetcard.md`** – describes a “multi-source CXR dataset” idea (MIMIC-CXR + open CXR + local images unified under one label schema).
* **`modelcard.md`** – describes the model, usage, risks, and limitations.

Those files are meant to be pushed to HF along with the model/dataset, or just to document the repo.

---

## 7. Limitations

* This is **not for clinical use**. It’s an educational / research-style demo.
* The RAG here is **very small** (hardcoded TF-IDF over a few texts). For real use, switch to FAISS/Chroma and feed it with real radiology references.
* The model currently expects chest-like images and 14 labels. If you train on MRI brain tumor or 22-class mixed medical dataset, you must also change the label list in the app.
* LLM output must be reviewed by a human radiologist.

---

## 8. References

* HF text model: [https://huggingface.co/Cosmobillian/samsung_innovation_campus_radiology_model](https://huggingface.co/Cosmobillian/samsung_innovation_campus_radiology_model)
* Quick CXR dataset (for prototyping): [https://huggingface.co/datasets/medmnist/medmnist](https://huggingface.co/datasets/medmnist/medmnist)
* DenseNet-121 (torchvision): [https://pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html](https://pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html)
* LangChain for Gemini: [https://python.langchain.com](https://python.langchain.com)
* Google Gemini (API): [https://ai.google.dev](https://ai.google.dev)

---

::contentReference[oaicite:0]{index=0}
```
