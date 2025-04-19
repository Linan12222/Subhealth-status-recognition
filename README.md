The dataset for this project was obtained from https://github.com/cathysiyu/Mechanical-datasets/tree/master/dataset.
This project implements a sub-health state recognition algorithm for rotating machinery (e.g., gearboxes and bearings) using publicly available vibration datasets.
We simulate the process of constructing a health state space from normal condition data via PCA, then apply Mahalanobis distance-based anomaly detection to identify potential sub-health conditions. A multi-level classification visualization helps interpret the results under different working conditions (RPMs).
🔍 Key Features
Dataset: Based on the open-source Southeast University Gearbox Dataset
Signal Preprocessing: Raw vibration signals are segmented into 5120-point samples
Feature Extraction: Time-domain statistical features (mean, std, RMS, peak, kurtosis, skewness)
Health Space Construction: Normal samples are reduced via PCA to 3D
Anomaly Detection: Mahalanobis distance from PCA space centroid, with adaptive thresholds
Multi-tier Visualization:
Yellow: Healthy (Center)
Red: Healthy (Inner Margin)
Blue: Healthy (Outer Margin)
Green: Abnormal (Sub-health or fault)
