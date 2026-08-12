# AI-Driven Pharmaceutical Defect Detection System

An automated computer vision and machine learning pipeline designed to modernize pharmaceutical Quality Control (QC). This system leverages Deep Learning and Convolutional Neural Networks (CNNs) to identify and classify manufacturing defects in pill production, ensuring stringent safety compliance and increasing high-throughput manufacturing yield.

---

## 🚀 Project Overview & Business Value

Manual visual inspection on pharmaceutical production lines is susceptible to human error, fatigue, and throughput bottlenecks. This AI system replaces manual checks with an automated image processing pipeline.

**Core Capabilities:**
* Rapidly identifies structural anomalies (e.g., chipped, broken, cracked) and contamination.
* Generates comprehensive performance metrics (Accuracy Graphs & Confusion Matrices) for QA auditing.
* Deploys a lightweight, highly efficient architecture utilizing Transfer Learning, suitable for edge-device deployment on factory floors.

---

## 🛠 Technical Architecture & Features

| Component | Description |
| :--- | :--- |
| **Model Architecture** | Convolutional Neural Network (CNN) utilizing **MobileNetV2** for feature extraction. |
| **Optimization Method** | **Transfer Learning** pre-trained on ImageNet to achieve high accuracy with smaller datasets. |
| **Model Serialization** | Saves in the modern, native Keras format (`medicine_defect_model.keras`) rather than legacy `.h5`. |
| **Evaluation Metrics** | Automated generation of classification reports, accuracy curves, and confusion matrices. |
| **Inference Mode** | Supports single-image prediction for real-time defect routing. |

---

## 📂 Dataset Structure

The pipeline requires a strict directory structure. The `train`, `validation`, and `test` splits must contain the exact same set of class subfolders to ensure reliable cross-validation. 

Run the initialization script to build this structure automatically:
```bash
python setup_dataset.py
