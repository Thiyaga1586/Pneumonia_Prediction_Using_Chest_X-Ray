# Pneumonia Detection with Grad-CAM (Chest X-ray Classification)

This repository showcases a deep learning–powered medical imaging application that detects **Pneumonia vs Normal** from chest X-ray images using a custom-designed CNN model with **accuracy ~97%**. The model is visualized using Grad-CAM and deployed using **Gradio on Hugging Face Spaces**.

---

## Live Demo

🔗 **[Try the Web App Here]([https://your-hf-space-link.hf.space](https://thiyaga158-pneumonia-detector.hf.space/))**  

---

## Problem Statement

Pneumonia is a serious lung infection that can be fatal if not detected early. Chest X-rays provide a non-invasive and accessible way to diagnose it. The goal of this project is to automatically classify X-rays as **Pneumonia** or **Normal**, and to **visually explain** the model's decision using **Grad-CAM heatmaps**.

---

## Project Evolution (What We Learned & Improved)

This project went through several thoughtful stages, each driven by experimentation and learning:

### Stage 1: Simple CNN (Baseline)
- Started with a simple 3–layer CNN.
- Achieved moderate accuracy (~75%), indicating underfitting.

### Stage 2: Deep Custom CNN (8 Layers)
- Increased convolutional depth.
- Accuracy improved (~81%), but model quickly overfit due to limited dataset size.

### Stage 3: Transfer Learning
- Tried **EfficientNetB0** and **ResNet120** fine-tuned on the dataset.
- These models yielded higher AUC but were inconsistent on real-world generalization.

### Stage 4: Custom DeepResNet Replica
- Built a deeper, ResNet-inspired model with residual connections and SE/CBAM blocks.
- This gave us more control and interpretability, with performance comparable to pretrained models.

### Realization: **Data Size Bottleneck**
- Original dataset had **just 5,216 X-rays**, leading to overfitting and poor generalization.

### Stage 5: Dataset Augmentation & Expansion
- Collected and cleaned multiple pneumonia-related datasets.
- Merged them to build a robust **22,000+ image dataset** (balanced and clean).
- Resized using a custom function with **black padding** to maintain aspect ratio.

### Final Model (This Repo)
- Enhanced ImprovedPneumonia with:
  - **CBAM attention modules**
  - **Dropout regularization**
  - **SiLU activations**
  - **Grad-CAM integration**
The ensemble model (built with these custom models and EfficientNetB0):
- Achieved:
  - AUC: **0.9973**
  - F1 Score: **0.9724**
  - Accuracy: **97.18%**

---

## Grad-CAM Integration

We used **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize the model’s decision-making process by highlighting important regions in the input image.

### Preprocessing:
We included:
- Image resizing with **black padding** to 224x224
- Grayscale conversion
- Normalization (mean=0.5, std=0.5)

---

## Deployment (Gradio + Hugging Face Spaces)

We deployed this using **Gradio** and hosted it on **Hugging Face Spaces**.

### Steps:
1. Created a `gradio-deploy/` folder with:
   - `app.py`
   - `requirements.txt`
   - `ImprovedPneumoniaCNN.pth`

2. Included:
   - A resize and padding function to ensure deployment images match training preprocessing.

3. Uploaded everything to [Hugging Face Spaces](https://huggingface.co/spaces) using the **Blank template**.

4. Fixed CORS/localhost issues with:
```python
interface.launch(share=True)
```
---

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for more info.

---

