# CrowdStampedeAnalysis

**Real-Time Computer Vision System for Crowd Density Estimation and Stampede Risk Detection**

---

## Project Overview

This repository presents an **end-to-end computer vision pipeline** for analyzing crowded scenes in video footage to estimate crowd density and detect potential stampede risks. Developed as a Proof of Work for AI/ML roles, it combines object detection, multi-object tracking, and regression-based counting to provide actionable insights for public safety and event management.

The system processes videos from standard crowd datasets (UMN, UCSD) and custom footage, outputting per-frame density maps, motion analytics, and risk alerts.

---

## Key Features

- **Automated Data Pipeline**: Video ingestion, metadata extraction, and frame-level preprocessing.
- **Multi-Model Detection**: YOLOv8 for accurate person detection with confidence thresholding.
- **Robust Tracking**: DeepSORT-based multi-object tracking for trajectory analysis and velocity computation.
- **Crowd Counting & Fusion**: Combines detection + regression models (CSRNet, MCNN) with metadata scaling for accurate counts.
- **Risk Analytics**: Motion vectors, density heatmaps, and anomaly detection for stampede early warning.
- **Comprehensive Evaluation**: MAE, RMSE, and per-subset metrics with rich visualization.

---

## Tech Stack

- **Language**: Python 3.12
- **Core CV**: OpenCV, Ultralytics (YOLOv8)
- **Tracking**: DeepSORT, Supervision
- **Models**: CSRNet, MCNN (pretrained)
- **Data**: pandas, tqdm, pathlib
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Google Colab / Jupyter

---

## Technical Architecture

### High-Level Flow

```mermaid
graph TD
    A[Raw Videos] --> B[Metadata Extraction]
    B --> C[Frame Extraction]
    C --> D[YOLOv8 Detection]
    D --> E[DeepSORT Tracking]
    E --> F[Motion & Velocity Analysis]
    F --> G[Density Regression Fusion]
    G --> H[Risk Scoring & Heatmaps]
    H --> I[Evaluation & Reports]
``````

Key Algorithms

YOLOv8 Detection: Real-time bounding box prediction with post-processing for person class only.
DeepSORT Tracking: Kalman filter + Re-ID for robust ID assignment across frames.
Crowd Counting Fusion: Combines detection counts with density regression (CSRNet) and metadata scaling for occlusion handling.
Motion Analysis: Computes average speed and direction per tracked object to detect abnormal flows.


Installation & Usage
1. Clone Repository
   git clone (https://github.com/AbhishekGitBot/Crowd-Stampede-Analysis.git)
   cd CrowdStampedeAnalysis
2. Setup Environment
   pip install ultralytics opencv-python pandas matplotlib seaborn tqdm deep_sort_realtime supervision filterpy
3. Project Structure Setup
    Place videos in datasets/videos/
    Run Step1_DataPreparation.ipynb to extract frames and metadata.
4. Run Pipeline
    # Run detection + tracking
    python step2_detection.py
    python step3_tracking.py


Future Research Roadmap

Integrate transformer-based density models (e.g., DM-Count) for improved accuracy.
Add real-time streaming support using RTSP/FFmpeg.
Develop multi-camera fusion and 3D crowd reconstruction.


Contact & Portfolio
Built by Abhishek — AI Research Engineer specializing in Computer Vision

LinkedIn:(https://www.linkedin.com/in/abhiisheksharrma/)
Email: sharrmaabhishek1@gmail.com
