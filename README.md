# 🛠️ Enterprise Customer Support Ticket Classifier

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ed.svg)](https://www.docker.com/)

An production-ready, end-to-end Machine Learning pipeline that fine-tunes a **DistilBERT Transformer** model for multi-class ticket classification, exposes it via a high-performance **FastAPI REST endpoint**, packages it inside a **Docker container**, and offers an interactive analytical **Gradio Dashboard**.

---

## 📊 Dashboard Preview
Below is the interactive Gradio monitoring application demonstrating real-time inference distribution, performance metrics mapping, and sub-50ms processing latencies:

![Gradio Interactive Dashboard](assets/dashboard_preview.png)

---

## 💡 Project Highlights
* **Transformer Optimization:** Fine-tuned `distilbert-base-uncased` to achieve high classification fidelity across **5 corporate support categories** on an 8,469-record dataset.
* **Production API Design:** Engineered a **FastAPI** web service featuring static request validation (Pydantic), dynamic latency tracing, and automated open-API documentation (`/docs`).
* **Container Architecture:** Containerized using a multi-stage **Dockerfile** environment variables pattern optimized for lightweight CPU-bound container engines.
* **Zero-Trust Input Sanitization:** Developed a robust tokenizer padding block handling real-world long-form ticket inputs under a strict 256-token truncation safety ceiling.

---

## 📈 Model Performance & Metrics
The evaluation phase ran against a completely isolated, stratified 20% validation split consisting of **1,693 support interactions**.

| Metric | Model Value |
| :--- | :--- |
| **Dataset Size** | 8,469 Rows |
| **Target Classes** | 5 Categories |
| **Evaluation Set Size**| 1,693 Rows |
| **Model Test Accuracy**| [Your Accuracy]% (e.g., 91.4%) |
| **Macro F1-Score** | [Your F1-Score] (e.g., 0.912) |

### Evaluated Support Verticals
1. `Technical issue`
2. `Billing inquiry`
3. `Cancellation request`
4. `Product inquiry`
5. `Refund request`

---

## 🛠️ Installation & Local Replication

### 1. Cloning and Dependencies
```bash
git clone [https://github.com/YOUR_USERNAME/customer-support-classifier.git](https://github.com/YOUR_USERNAME/customer-support-classifier.git)
cd customer-support-classifier
pip install -r requirements.txt
