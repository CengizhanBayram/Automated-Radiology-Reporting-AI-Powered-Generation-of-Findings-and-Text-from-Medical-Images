---
# MODEL CARD

# Model Card for Multi-Radiology Classification Stack (HF MIMIC-CXR model + DenseNet121 Demo)

Bu model kartı, aynı projede kullanılan **iki** görüntü sınıflandırma modelini birlikte dokümante eder:

1. **`Cosmobillian/samsung_innovation_campus_radiology_model`**  
   Hugging Face üzerinden çekilen ve MIMIC-CXR benzeri veriyle eğitilmiş, gerçek radyoloji bulgularını tahmin etmeye yönelik çok etiketli model.

2. **`densenet121-multisource-radiology-demo`**  
   Colab üzerinde, ortak 22’lik etiket uzayını ve RAG/LLM hattını test etmek için eğitilmiş, örnek/demonstratif model.

Bu iki model birlikte kullanıldığında:
- HF modeli **daha gerçekçi** bulgular üretmek için kullanılır,
- Demo DenseNet modeli ise **ortak format** (etiket → skor → RAG → prompt) akışının çalıştığını göstermek için kullanılır.

---

## Model Details

### Model Description

Bu stack’in amacı, radyoloji görüntüsünden otomatik bulgu çıkarıp bunu sonraki aşamalarda (RAG, Gemini, rapor üretimi) kullanabilmektir. Her iki model de göğüs radyografisi senaryosunu hedefler, ancak veri kaynağı ve olgunluk seviyesi farklıdır:

- **HF MIMIC-CXR modeli**: Büyük, klinik kaynaklı ve pratikte kullanılabilecek seviyede.
- **DenseNet121 demo**: Yerel/deneysel görüntü klasörleriyle bile çalışacak kadar esnek ama klinik doğruluk garantisi yok.

- **Developed by:** kullanıcı (Colab) + HF repo sahibi (`Cosmobillian`)  
- **Model date:** 2025  
- **Model type:** Vision, multi-label image classification  
- **Language(s):** Görüntü girişi; çıktı etiketleri İngilizce  
- **Finetuned from model (HF):** `AutoModelForImageClassification` tabanlı, ImageNet gövdesi  
- **Finetuned from model (Demo):** `torchvision.models.densenet121(pretrained=ImageNet)`

### Model Sources

- **Repository (HF):** https://huggingface.co/Cosmobillian/samsung_innovation_campus_radiology_model
- **Repository (Demo):** yerel Colab kaydı `/content/drive/MyDrive/radiologyst/best_model.pt`
- **Paper [optional]:** MIMIC-CXR (Johnson et al.), DenseNet (Huang et al.)
- **Demo [optional]:** Colab defteri içinde eğitim + inference hücreleri

---

## Uses

### Direct Use

- Göğüs röntgeninden otomatik **bulgu listesi** çıkarma
- “Normal vs. patoloji var” tarzı ön eleme
- LLM/RAG sistemlerine medikal bağlam sağlama: model → etiket → eğitim dokümanı → rapor
- Benzer CXR veri setlerinde hızlı karşılaştırma

### Downstream Use

- Klinik eğitim amaçlı rapor örneği üretme
- Radyoloji rapor kalıbına otomatik doldurma
- Daha büyük veriyle yeniden eğitilerek kurumsal modele dönüştürme
- Çok kaynaklı (NIH, CheXpert, MIMIC) eğitim pipeline’larına entegre etme

### Out-of-Scope Use

- Tek başına klinik tanı koyma
- Modalite dışı (MR, BT) görüntüleri açıklama (HF modeli CXR odaklıdır)
- Demografisi çok farklı popülasyonlara doğrulama olmadan uygulama
- Düşük kaliteli, projeksiyonu farklı (AP/lat vs.) görüntülerde rapor yazdırıp klinik karar vermek

---

## Bias, Risks, and Limitations

- MIMIC-CXR veri kaynağı tek merkezlidir; başka kurumlarda domain shift görülebilir.
- Metinden türetilen etiketler gürültülü olabilir (false positive / false negative).
- Demo DenseNet modeli gerçek medikal veriyle değil, proje sırasında mevcut olan görüntü klasörüyle eğitildiği için **sadece format testine** uygundur.
- RAG aşamasında hatalı etiket → hatalı pasaj → hatalı rapor zincirlemesi olabilir.

### Recommendations

Kullanıcılar:
1. HF modelinden gelen skorları bir **eşik** (0.3–0.5) ile filtrelemeli,
2. RAG’e sadece bu etiketleri göndermeli,
3. LLM’nin ürettiği raporu **insan radyolog** gözünden geçirmelidir.

---

## How to Get Started with the Model

Aşağıda her iki model için örnek kod verilmektedir.

### 1) HF MIMIC-CXR modeli (ana model)

```python
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

model_id = "Cosmobillian/samsung_innovation_campus_radiology_model"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)
model.eval()

img = Image.open("example_cxr.png").convert("RGB")
inputs = processor(images=img, return_tensors="pt")

with torch.no_grad():
    out = model(**inputs)
    logits = out.logits.squeeze(0)
    probs = torch.sigmoid(logits)  # multi-label
    id2label = model.config.id2label
    predictions = [(id2label[i], float(p)) for i, p in enumerate(probs) if p > 0.4]

print(predictions)
2) DenseNet121 demo modeli (ortak 22’lik çıkış)
python
Kodu kopyala
import torch
from torchvision import models, transforms
from PIL import Image

labels_22 = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia",
    "COVID-19","Viral_Pneumonia","Tuberculosis","Normal",
    "Glioma","Meningioma","Pituitary","No_Tumor",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.densenet121(weights=None)
in_feats = model.classifier.in_features
model.classifier = torch.nn.Linear(in_feats, len(labels_22))
state = torch.load("/content/drive/MyDrive/radiologyst/best_model.pt", map_location=device)
model.load_state_dict(state)
model.to(device).eval()

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

img = Image.open("example.png").convert("RGB")
x = tf(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits).cpu().squeeze(0)

preds = [(labels_22[i], float(p)) for i, p in enumerate(probs) if p > 0.4]
print(preds)
Training Details
Training Data
HF modeli: MIMIC-CXR (HF üzerinden çekilen sürüm), büyük ve klinik raporlu.
→ Daha fazla bilgi için ilgili Dataset Card’a bakılmalı.

Demo modeli: Yerel Colab’da bulunan görüntü klasörü (EuRoC örnekleri) “Normal” olarak işaretlenerek eğitilmiş oyuncak veri.

Training Procedure
Her iki modelde de görüntüler sabit boyuta resize edilip ImageNet istatistikleriyle normalize edilir.

Multi-label sınıflandırma için BCEWithLogitsLoss kullanılır.

HF modelinde Hugging Face Trainer veya benzeri training loop; demo modelde manuel PyTorch training loop + early stopping kullanılmıştır.

Preprocessing
Resize (224×224)

ToTensor

Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

Gerektiğinde random horizontal flip

Training Hyperparameters
Training regime: fp16 veya fp32 (Colab/GPU’ya bağlı)

lr: 1e-4 civarı

optimizer: AdamW

epochs: 5–30 arası, early stopping ile

Speeds, Sizes, Times
Checkpoint boyutu: DenseNet121 için ~30–35 MB civarı

HF model boyutu: base modele göre değişmekle birlikte ~100–300 MB

Evaluation
Testing Data, Factors & Metrics
Testing Data
HF modeli: MIMIC-CXR validation/test bölümleri

Demo modeli: aynı klasörden ayrılmış küçük validation bölümü

Factors
Görüntü projeksiyonu (PA vs AP)

Patoloji sıklığı

Veri kaynağı (MIMIC vs diğer CXR setleri)

Metrics
AUROC per label (CXR alanında standart)

mAUC (tüm etiketlerin ortalaması)

Opsiyonel: F1@0.5

Results
HF modelinde beklenen AUROC: 0.7–0.9 aralığında, etikete göre değişir

Demo modelinde loss çok küçük çıkmış olsa da bu tek sınıflı veri sebebiyledir ve gerçek performansı göstermez

Summary
HF modeli → gerçek kullanıma daha yakın
Demo DenseNet → entegrasyon ve RAG testi için

Model Examination
Class activation map (CAM) / Grad-CAM uygulaması yoluyla modelin hangi alanlara baktığı analiz edilebilir.

Çok etiketli çıktıda belirli patolojilerin birlikte çıkma olasılığı incelenebilir (örn. effusion + cardiomegaly).

Technical Specifications
Model Architecture and Objective
HF modeli: Vision backbone + multi-label classification head

Demo modeli: DenseNet121 + Linear(1024 → 22) + sigmoid

Objective: her etiket için bağımsız olasılık tahmini

Compute Infrastructure
Eğitim: Colab GPU (T4 / L4) yeterli

Inference: CPU’da da çalışır (tek görüntü için)

Hardware
1 × GPU (8–16 GB VRAM) eğitim için rahat

CPU inference için yeterli

Software
Python 3.x

PyTorch

Torchvision

Transformers (HF modeli için)

scikit-learn (RAG benzeri benzerlik için)

Citation
Lütfen kullandığınız veri setlerinin orijinal makalelerini (MIMIC-CXR vb.) ve kullanılan model repo’sunu (HF adresi) ayrı ayrı atıfleyin.

Glossary
CXR: Chest X-ray

Multi-label: Tek görüntüde birden fazla etiketin aynı anda pozitif olabilmesi

RAG: Retrieval-Augmented Generation, model çıktısını metin arama ile zenginleştirme

More Information
Bu kart, aynı projede iki farklı radyoloji modelini birlikte dokümante etmek için yazıldı. Gerçek üretim ortamında HF’de eğittiğiniz MIMIC-CXR modelinin kullanılması, demo DenseNet’in ise sadece akış testi olarak kalması önerilir.


