# Dual-Membrane Semantic Segmentation of Bacteria using U-Net

## Overview
This project implements a customized U-Net architecture designed to perform high-fidelity semantic segmentation of bacterial membranes from Electron Microscopy (EM) imagery. The model simultaneously segments the **Outer Membrane (OM)** and **Inner Membrane (IM)** as distinct labels, achieving a Dice Coefficient of **0.97**.

This repository focuses on the architectural implementation and training optimization required to resolve nanometer-scale biological structures.

## Technical Implementation
* **Architecture:** 5-level deep U-Net with skip connections to preserve spatial gradients during the decoding phase.
* **Input/Output:** Grayscale input ($1 \times 640 \times 640$) with a multi-label output ($2 \times 640 \times 640$).
* **Optimization:** Implemented a hybrid **Dice-BCE Loss** function to address class imbalance (membranes occupy <5% of the total pixel area).
* **Training Logic:** Includes Early Stopping and Best-Model checkpointing based on Validation Dice scores.

## Performance Metrics
The model was evaluated on a held-out test set using pixel-level metrics:

| Metric | Combined Mean | Outer Membrane (OM) | Inner Membrane (IM) |
| :--- | :--- | :--- | :--- |
| **Dice Coefficient** | 0.972 | 0.978 | 0.966 |
| **Precision** | 0.975 | 0.981 | 0.969 |
| **Recall** | 0.969 | 0.974 | 0.964 |
| **IoU** | 0.958 | 0.964 | 0.952 |

## Visualization
The model generates high-fidelity traces using OpenCV contour detection:
* **Red Trace:** Predicted Outer Membrane
* **Blue Trace:** Predicted Inner Membrane

> **Note:** These high-accuracy masks provide the foundation for downstream morphometric analysis, such as membrane thickness quantification (Proprietary/Patent-Pending).

## Usage
### 1. Installation
Install all required dependencies using the following command:
```bash
pip install -r requirements.txt


**### 2. Installation Run benchmarking & Analysis**
To process the images and verify the model against ground truth, run:
```bash
python main.py --input data --coco_json data/_annotations.coco.json

**3. How to run on New Images (Inference Only)**
To run the analysis on your own images without providing a ground truth JSON file:
python main.py --input data




