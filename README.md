# FashionSense-AI: Multimodal Fashion Classification using DistilBERT + MobileNetV2

## Overview

FashionSense-AI is an end-to-end multimodal deep learning system that combines **textual product descriptions** and **image data** for fashion item classification. Traditional models often rely on either text or image information independently, which can miss contextual relationships between product appearance and descriptions.

This project integrates:

- **DistilBERT** → for contextual text embeddings
- **MobileNetV2** → for image feature extraction
- **Multimodal feature fusion** → combining text and image representations
- **TensorFlow data pipelines** → for efficient training and preprocessing

The model learns richer product representations by leveraging both modalities simultaneously.

---

## Dataset

The dataset contains:

- Product Images
- Product Descriptions
- Product Display Names
- Product Categories

### Dataset Processing

The original dataset contained **142 categories** with heavy class imbalance.

Preprocessing steps:

- Selected categories with more than **500 samples**
- Reduced dataset to **22 major categories**
- Applied downsampling to create balanced classes
- Shuffled dataset
- Split into:
  - Training: **80%**
  - Validation: **10%**
  - Testing: **10%**

Final balanced dataset:

- Total samples: **11,660**
- Categories: **22**

---

## Project Pipeline

### Step 1: Data Preprocessing

- Loaded dataset using Pandas
- Combined:
  - Product Display Name
  - Product Description
- Handled missing values
- Encoded category labels

---

### Step 2: Text Processing

Used **DistilBERT** from Hugging Face Transformers.

Text preprocessing:

- Tokenization
- Padding
- Truncation
- Batch embedding generation

Generated:

- 768-dimensional contextual embeddings

---

### Step 3: Image Processing

Image preprocessing includes:

- Image resizing → 224 × 224
- Pixel normalization
- Data augmentation:

  - Random horizontal flip
  - Random rotation
  - Random zoom

---

### Step 4: Feature Extraction

#### Text Encoder

Model:

- DistilBERT

Output:

- 768-dimensional text embeddings

#### Image Encoder

Model:

- MobileNetV2 (pretrained on ImageNet)

Configuration:

- Transfer learning
- Frozen pretrained layers

---

### Step 5: Multimodal Fusion

Combined:

- Text embeddings
- Image embeddings

Fusion pipeline:

- Feature concatenation
- Dense layers
- Classification head

---

## Architecture

```

Input Text ──► DistilBERT ──► Text Features ─┐
│
├──► Feature Fusion ──► Dense Layers ──► Output
│
Input Image ─► MobileNetV2 ─► Image Features ┘

```

---

## Tech Stack

### Languages

- Python

### Libraries

- TensorFlow
- PyTorch
- Transformers
- Scikit-learn
- NumPy
- Pandas

### Models

- DistilBERT
- MobileNetV2

---

## Features

✅ Multimodal learning using text + image data

✅ Transfer learning with MobileNetV2

✅ Context-aware embeddings with DistilBERT

✅ Data balancing and preprocessing

✅ GPU optimization using mixed precision

✅ Efficient TensorFlow pipelines

---

## Installation

Clone repository:

```bash
git clone https://github.com/yourusername/FashionSense-AI.git

cd FashionSense-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Project

Run notebook/script:

```bash
python train.py
```

or

```bash
jupyter notebook
```

---

## Future Improvements

- Fine-tune DistilBERT and MobileNetV2
- Add attention-based multimodal fusion
- Deploy using Streamlit
- Add recommendation functionality
- Experiment with larger vision-language models

---

## Results

The multimodal approach improves understanding of products by leveraging both visual and textual information simultaneously, enabling more accurate fashion classification compared to single-modality approaches.

---

## Author

**Shashank Shekhar**

AI/ML | LLMs | Computer Vision | RAG Pipelines

LinkedIn: [Your LinkedIn URL]

GitHub: [Your GitHub URL]
