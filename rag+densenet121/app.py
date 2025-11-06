import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

# 🔗 LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI


# ============ AYARLAR ============
MODEL_PATH = r".\densenet_model\chestmnist_best.pt"  # kendi modelin
# ================================


# ---------- CONFIG ----------
@dataclass
class Config:
    model_path: str = MODEL_PATH
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_size: int = 224
    threshold: float = 0.4
    topk_docs: int = 3
    # ChestMNIST 14 label
    labels: List[str] = field(
        default_factory=lambda: [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ]
    )


# ---------- MODEL TARAFI ----------
class MedModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = self._load_model()
        self.tf = transforms.Compose(
            [
                transforms.Resize((cfg.img_size, cfg.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_model(self) -> nn.Module:
        num_classes = len(self.cfg.labels)
        model = models.densenet121(weights=None)
        in_feats = model.classifier.in_features
        model.classifier = nn.Linear(in_feats, num_classes)

        if os.path.isfile(self.cfg.model_path):
            state = torch.load(self.cfg.model_path, map_location=self.cfg.device)
            # bazı checkpointler dict içinde geliyor
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]

            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                st.sidebar.warning(
                    f"Model yüklendi ama bazı katmanlar eşleşmedi.\n"
                    f"missing: {missing}\nunexpected: {unexpected}"
                )
            else:
                st.sidebar.success(f"Model yüklendi: {self.cfg.model_path}")
        else:
            st.sidebar.warning(
                f"Model bulunamadı, boş model kullanılıyor: {self.cfg.model_path}"
            )

        model.to(self.cfg.device)
        model.eval()
        return model

    def predict(self, pil_img: Image.Image) -> List[Dict[str, Any]]:
        x = self.tf(pil_img).unsqueeze(0).to(self.cfg.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().squeeze(0)

        findings = []
        for i, p in enumerate(probs):
            score = float(p)
            if score >= self.cfg.threshold:
                findings.append({"label": self.cfg.labels[i], "score": score})

        # hiçbiri eşiği geçmezse en iyi 3'ü ver
        if not findings:
            top_vals, top_idxs = torch.topk(probs, k=3)
            for v, idx in zip(top_vals, top_idxs):
                findings.append(
                    {"label": self.cfg.labels[int(idx)], "score": float(v)}
                )

        return findings


# ---------- MINI RAG ----------
class MiniRAG:
    def __init__(self):
        self.docs = [
            {
                "title": "Pleural Effusion (CXR)",
                "text": "Pleural effusion: costophrenic angle blunting, meniscus sign. Side and amount should be reported.",
            },
            {
                "title": "Cardiomegaly",
                "text": "Cardiomegaly is suspected when cardiothoracic ratio > 0.5 on PA view.",
            },
            {
                "title": "Lobar Pneumonia",
                "text": "Lobar pneumonia presents with segmental/lobar consolidation and air bronchograms.",
            },
            {
                "title": "Normal Chest X-ray",
                "text": "Clear lung fields, normal cardiac silhouette, sharp costophrenic angles.",
            },
            {
                "title": "Pneumothorax",
                "text": "Pneumothorax shows visceral pleural line with absent lung markings peripheral to it.",
            },
        ]
        texts = [d["text"] for d in self.docs]
        self.vectorizer = TfidfVectorizer()
        self.tfidf = self.vectorizer.fit_transform(texts)

    def get(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.tfidf).flatten()
        idxs = sims.argsort()[::-1][:k]
        return [
            {
                "title": self.docs[i]["title"],
                "text": self.docs[i]["text"],
                "score": float(sims[i]),
            }
            for i in idxs
        ]


# ---------- PROMPT İNŞA ----------
def build_query(findings: List[Dict[str, Any]]) -> str:
    if not findings:
        return "Normal chest X-ray. Write a standard chest radiograph report in Turkish."
    labels = [f["label"] for f in findings]
    return (
        "The chest X-ray shows: " + ", ".join(labels) + ". Write a Turkish radiology report."
    )


def build_prompt(findings: List[Dict[str, Any]], docs: List[Dict[str, Any]]) -> str:
    ftxt = (
        "\n".join([f"- {f['label']} (p={f['score']:.2f})" for f in findings])
        or "No high-confidence findings."
    )
    dtxt = "\n\n".join([f"### {d['title']}\n{d['text']}" for d in docs])
    prompt = f"""
Sen bir radyoloji asistanısın. Aşağıdaki model bulguları ve açıklamaları kullanarak **Türkçe** ve **kısa** bir PA akciğer grafisi raporu üret.

MODEL BULGULARI:
{ftxt}

AÇIKLAYICI NOTLAR:
{dtxt}

Format:
- Bulgular:
- Değerlendirme:
- Öneri (varsa):
"""
    return prompt.strip()


# ---------- GEMINI ÇAĞRISI ----------
def call_gemini(prompt: str, api_key: str, model_name: str = "gemini-1.5-flash") -> str:
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0.4,
    )
    resp = llm.invoke(prompt)
    # langchain objesini stringe çevir
    return resp.content if hasattr(resp, "content") else str(resp)


# ---------- STREAMLIT APP ----------
def main():
    st.set_page_config(page_title="Radyoloji + RAG + Gemini", layout="wide")
    st.title("🩻 Radyoloji Yardımcı Demo (DenseNet + mini-RAG + Gemini)")

    cfg = Config()

    # session state
    if "med_model" not in st.session_state:
        st.session_state["med_model"] = MedModel(cfg)
    if "mini_rag" not in st.session_state:
        st.session_state["mini_rag"] = MiniRAG()

    with st.sidebar:
        st.markdown("### Gemini Ayarları")
        default_key = os.getenv("GEMINI_API_KEY", "")
        gemini_key = st.text_input(
            "Gemini API Key", value=default_key, type="password"
        )
        model_name = st.text_input("Gemini Model", value="gemini-1.5-flash")

    col1, col2 = st.columns(2)

    with col1:
        img_file = st.file_uploader("Görüntü yükle", type=["png", "jpg", "jpeg"])
        if img_file is not None:
            pil_img = Image.open(img_file).convert("RGB")
            st.image(pil_img, caption="Yüklenen Görüntü", use_column_width=True)

            med_model: MedModel = st.session_state["med_model"]
            mini_rag: MiniRAG = st.session_state["mini_rag"]

            findings = med_model.predict(pil_img)
            query = build_query(findings)
            docs = mini_rag.get(query, k=cfg.topk_docs)
            prompt_txt = build_prompt(findings, docs)

            # sağ tarafa ver
            with col2:
                st.subheader("Model Bulguları")
                findings_txt = "\n".join(
                    [f"- {f['label']} (p={f['score']:.2f})" for f in findings]
                )
                st.text_area("Bulgular", value=findings_txt, height=140)

                st.subheader("RAG Açıklamaları")
                docs_txt = "\n\n".join(
                    [f"{d['title']} (score={d['score']:.2f})\n{d['text']}" for d in docs]
                )
                st.text_area("Açıklamalar", value=docs_txt, height=160)

            st.subheader("LLM / Gemini Prompt")
            st.text_area("Oluşan Prompt", value=prompt_txt, height=180, key="prompt_box")

            st.markdown("---")
            if st.button("✨ Gemini'den rapor üret"):
                if not gemini_key:
                    st.error("Önce Gemini API key gir.")
                else:
                    try:
                        report = call_gemini(prompt_txt, gemini_key, model_name=model_name)
                        st.success("Gemini raporu:")
                        st.text_area("Rapor", value=report, height=220)
                    except Exception as e:
                        st.error(f"Gemini çağrısı hata verdi: {e}")
        else:
            with col2:
                st.info("Önce bir görüntü yükle.")


if __name__ == "__main__":
    main()
