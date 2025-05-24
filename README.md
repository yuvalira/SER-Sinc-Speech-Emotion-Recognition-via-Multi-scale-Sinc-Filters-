# SER-Sinc-Speech-Emotion-Recognition-via-Multi-scale-Sinc-Filters
# SER-Sinc: Speech Emotion Recognition via Multi-scale Sinc Filters

**Final Project - BGU Software Engineering Department**  
By: Yuval Ratzabi, Gal Zohar, Tomer Baziza, Tomer Abram

---

## Overview

This project investigates the use of **MS-SincResNet**, an architecture originally designed for music genre classification, in the domain of **Speech Emotion Recognition (SER)**. Instead of relying on traditional handcrafted features like MFCCs, the model uses **multi-scale SincNet filters** to directly learn frequency-selective features from raw audio waveforms. We adapt and evaluate this architecture on the **RAVDESS dataset**.

---

## Goals

- Adapt the MS-SincResNet model for SER using raw audio signals.
- Train and evaluate the model on the RAVDESS dataset.
- Compare the results to the state-of-the-art MFCC-based CNN-LSTM benchmark (Ouyang, 2025).
- Analyze confusion between similar emotions (e.g., calm vs. neutral).
- Demonstrate the viability of raw waveform learning in low-resource SER settings.

---

## Architecture

### Preprocessing
- **Input**: Raw audio waveform  
- **Sample Rate**: 16 kHz  
- **Clip Length**: 3 seconds (48,000 samples)  
- **Normalization**: Zero-mean, unit-variance  
- **Augmentation (offline)**: Optional (e.g., time shift, pitch shift, noise injection)

### Model
- **Front-end**: Multi-scale SincNet layer (learnable band-pass filters)  
- **Feature Extractor**: ResNet-18 (2D CNN)  
- **Pooling**: Spatial Pyramid Pooling (SPP)  
- **Classifier**: Fully-connected layer with Softmax (8 emotion classes)

### Loss
- **Cross-Entropy Loss**, optionally weighted for class imbalance

---

## Dataset

- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song  
  [Kaggle Link](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)  
- **Classes**: Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised, Neutral  
- **Total Samples**: 1,440 (Balanced gender and actor split)

---

## Evaluation

- **Metrics**: Accuracy, F1-score (macro), Confusion Matrix  
- **Target Accuracy**: ≥ 60% on test set  
- **Baseline**: Ouyang (2025), CNN-LSTM with MFCC, 61.07% accuracy

---

## Inference

- Input: Raw 3-second waveform at 16 kHz  
- Output: One-hot class prediction of emotion  
- Can be extended to real-time with a sliding window approach

---

## Novelty

Unlike traditional models that rely on handcrafted features (e.g., MFCC + LSTM), this project proposes a raw waveform-based solution using multi-scale Sinc filters and deep convolutional layers. This approach provides **end-to-end learning**, improved interpretability, and avoids manual feature engineering.

---

## References

1. **MS-SincResNet** – Chang et al., 2021  
   [🔗 Paper](https://arxiv.org/pdf/2109.08910)

2. **Raw Waveforms for SER** – Kilimci et al., 2023  
   [🔗 Paper](https://arxiv.org/abs/2307.02820)

3. **MFCC + CNN-LSTM Benchmark** – Ouyang, 2025  
   [🔗 Paper](https://arxiv.org/abs/2501.10666)

4. **RAVDESS Dataset** – Livingstone & Russo, 2018  
   [🔗 Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

---

## ⚠Known Challenges

- Small dataset size → Risk of overfitting  
- Emotion confusion (e.g., Fear vs. Surprise)  
- Class imbalance → Consider using weighted loss or oversampling

---

## Future Work

- Real-time inference pipeline  
- Fine-tuning augmentation techniques  
- Experimenting with SPP variants or attention mechanisms

---

## Contact

For questions or collaborations:

- Yuval Ratzabi — ratzabiy@post.bgu.ac.il  
- Gal Zohar — galzoh@post.bgu.ac.il  
- Tomer Baziza — baziza@post.bgu.ac.il  
- Tomer Abram — abram@post.bgu.ac.il

---
