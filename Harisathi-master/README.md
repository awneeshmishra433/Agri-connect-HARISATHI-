# 🌿 HARISATHI — Smart Agriculture Assistant

**Harisathi** is a machine learning and deep learning-based web application designed to assist farmers with **crop recommendation, fertilizer recommendation, and plant disease detection**.

The project demonstrates how AI/ML can be applied to agriculture and precision farming to provide data-driven assistance for common agricultural decisions.

> ⚠️ **Disclaimer:** Harisathi is a proof-of-concept project. The datasets used in this project are not guaranteed to be accurate, complete, or suitable for real-world agricultural decisions. This application should **not** be used as a substitute for professional agricultural advice.

---

## 🚀 Features

Harisathi provides three core agricultural assistance modules:

### 🌾 1. Crop Recommendation

Recommends a suitable crop based on soil and environmental parameters provided by the user.

**Input parameters include:**

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* pH
* Rainfall

The trained machine learning model analyzes the provided parameters and recommends a suitable crop.

---

### 🧪 2. Fertilizer Recommendation

Helps identify potential nutrient deficiencies or excesses in the soil and recommends suitable fertilizer improvements based on:

* Soil nutrient values
* Soil conditions
* Selected crop
* Nutrient requirements

This module is intended to demonstrate how machine learning can assist with fertilizer-related recommendations.

---

### 🍃 3. Plant Disease Detection

Uses **Deep Learning** and image classification to identify diseases affecting plant leaves.

Users can upload an image of a plant leaf, and the trained model predicts the possible disease.

The application also provides:

* Disease identification
* Basic information about the detected disease
* Suggested treatment/prevention information

---

## 🧠 Machine Learning & Deep Learning

The project uses different machine learning approaches for its agricultural applications.

| Module                       | Technology       |
| ---------------------------- | ---------------- |
| Crop Recommendation          | Machine Learning |
| Fertilizer Recommendation    | Machine Learning |
| Plant Disease Detection      | Deep Learning    |
| Disease Classification Model | ResNet           |
| Web Application              | Flask            |
| Image Processing             | Python / OpenCV  |
| Data Processing              | NumPy / Pandas   |
| ML Utilities                 | Scikit-learn     |
| Deep Learning Framework      | PyTorch          |

---

## 🏗️ Project Architecture

The application follows a simple machine-learning-powered web architecture:

```text
User
  │
  ▼
Web Interface
  │
  ├───────────────┬──────────────────┐
  ▼               ▼                  ▼
Crop           Fertilizer        Disease
Recommendation Recommendation     Detection
  │               │                  │
  ▼               ▼                  ▼
ML Model        ML Model         Deep Learning
                                  Model (ResNet)
  │               │                  │
  └───────────────┴──────────────────┘
                  │
                  ▼
             Prediction
                  │
                  ▼
            User Interface
```

---

## 📊 Datasets

The project uses publicly available datasets for experimentation and proof-of-concept development.

### Crop Recommendation Dataset

Source: Kaggle — Crop Recommendation Dataset

[Crop Recommendation Dataset](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset?utm_source=chatgpt.com)

### Fertilizer Recommendation Dataset

Source: Harvestify dataset

[Fertilizer Recommendation Dataset](https://github.com/Gladiator07/Harvestify/blob/master/Data-processed/fertilizer.csv?utm_source=chatgpt.com)

### Plant Disease Detection Dataset

Source: Kaggle — New Plant Diseases Dataset

[New Plant Diseases Dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset?utm_source=chatgpt.com)

> **Note:** Dataset quality and suitability for real-world agricultural deployment have not been independently validated.

---

## 📓 Machine Learning Notebooks

The corresponding model development and experimentation notebooks are available on Kaggle.

### 🌾 Crop Recommendation

[Crop Recommendation Notebook](https://www.kaggle.com/atharvaingle/what-crop-to-grow?utm_source=chatgpt.com)

### 🍃 Plant Disease Classification

[Plant Disease Classification — ResNet](https://www.kaggle.com/atharvaingle/plant-disease-classification-resnet-99-2?utm_source=chatgpt.com)

---

## 🛠️ Technologies Used

* **Python**
* **Flask**
* **PyTorch**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **OpenCV**
* **HTML**
* **CSS**
* **Machine Learning**
* **Deep Learning**
* **ResNet**

---

## 📁 Project Structure

```text
Harisathi/
│
├── models/
│   ├── crop_recommendation/
│   ├── fertilizer_recommendation/
│   └── disease_detection/
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   └── *.html
│
├── app.py
├── requirements.txt
└── README.md
```

> The exact structure may vary depending on the current project files.

---

## 🎯 Motivation

Agriculture is an important part of the Indian economy, and modern technologies such as **Machine Learning, Deep Learning, and Computer Vision** can potentially help address several challenges faced in agriculture.

Harisathi was developed as a demonstration of how AI-powered systems can be integrated into agriculture to assist with:

* 🌱 Crop selection
* 🧪 Soil nutrient management
* 🍃 Plant disease identification
* 📊 Data-driven agricultural assistance

The long-term potential of such systems depends heavily on **high-quality, verified datasets, field validation, domain expertise, and real-world testing**.

---

## ⚠️ Disclaimer

Harisathi is a **Proof of Concept (POC)** developed for educational and experimental purposes.

The predictions and recommendations generated by this application should **not be treated as professional agricultural advice**.

The project uses publicly available datasets that may contain limitations, biases, or inaccuracies. Real-world deployment would require:

* Verified agricultural datasets
* Extensive field testing
* Regional crop and soil information
* Validation by agricultural experts
* Continuous model evaluation
* Proper monitoring of model performance

---

## 👨‍💻 Developer

**Awneesh Mishra**

GitHub:
[github.com/awneeshmishra433](https://github.com/awneeshmishra433?utm_source=chatgpt.com)

LinkedIn:
[linkedin.com/in/awneesh-mishra-530bab377](https://www.linkedin.com/in/awneesh-mishra-530bab377?utm_source=chatgpt.com)

Email:
[awneeshmsihra433@gmail.com](mailto:awneeshmsihra433@gmail.com)

---

## 🤝 Contributions

Suggestions, improvements, and contributions are welcome.

If you find an issue or have an idea for improving the project, feel free to open an **Issue** or submit a **Pull Request**.

---

## 📜 License

Please refer to the repository's license file for information regarding the use, modification, and distribution of this project.

---

### 🌿 Harisathi

**Exploring the use of Machine Learning and Deep Learning for smarter agriculture.**
