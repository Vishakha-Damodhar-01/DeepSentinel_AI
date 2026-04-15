# 🛡️ DeepSentinel AI  
### An Uncertainty-Aware Deepfake Detection System using MobileNetV2  


## 📌 Overview  
DeepSentinel AI is a deep learning-based system designed to detect deepfake content in **images, videos, and real-time webcam streams**.  
The system leverages **MobileNetV2** with **transfer learning** and introduces an **uncertainty-aware multi-inference mechanism** to improve prediction reliability.

Unlike traditional deepfake detection systems, DeepSentinel AI not only classifies content as *real* or *fake* but also identifies **uncertain cases**, making it more robust for real-world cybersecurity applications.

---

## 🚀 Features  

- 🔍 Deepfake detection for:
  - Images  
  - Videos  
  - Real-time webcam  

- 🧠 Lightweight Deep Learning Model (MobileNetV2)  
- 🔁 Multi-Inference Prediction for stability  
- ⚠️ Uncertainty Estimation (Real / Fake / Uncertain)  
- 📊 Interactive Dashboard for visualization  
- ⚡ Real-time performance with optimized computation  

---

---

## 🧠 Methodology  

- Transfer Learning using pre-trained MobileNetV2  
- Face Detection to focus on manipulated regions  
- Frame Extraction for video analysis  
- Multi-Pass Inference for stable predictions  
- Threshold-Based Classification  
- Uncertainty Handling using prediction variance  

---

## 📂 Dataset  

The model is trained using a combination of:

- FaceForensics++ (FF++)  
- DeepFake Detection Challenge (DFDC) Dataset  
- 140K Real & Fake Faces Dataset (Kaggle)  

This hybrid dataset improves:
- Generalization  
- Robustness  
- Real-world performance  

---

## 📊 Results  

| Metric    | Value |
|----------|------|
| Accuracy | 0.89 |
| Precision | 0.50 |
| Recall | 0.50 |
| F1-Score | 0.50 |

⚠️ Current model performs at baseline level. Improvements are ongoing.

---

## 🛠️ Tech Stack  

- Programming Language: Python  
- Frameworks: TensorFlow / Keras  
- Libraries: OpenCV, NumPy, Pandas, Matplotlib  
- Model: MobileNetV2  
- Deployment: Streamlit / Flask  

---

## how to run it on local system  

Open terminal and type>> streamlit app.py
