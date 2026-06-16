# 🧠 Teen Depression Prediction using Machine Learning

An end-to-end data science and predictive analytics web application built using **Scikit-Learn**, **Streamlit**, and **Imbalanced-Learn**. The project targets a critical mental health business problem: identifying teenagers (aged 13–19) who are potentially at risk of depression based on lifestyle habits, daily social media patterns, sleep hygiene, and physiological stress metrics.

This repository includes a full exploratory analysis notebook handling class imbalance (SMOTE/Oversampling), a serialized fine-tuned Random Forest pipeline, and an interactive frontend dashboard for real-time inference.

---

## 🚀 Key Features

* **Advanced Risk Modeling:** Powered by a tuned **Random Forest Classifier (`TeenDepressionAnalysisModel.pkl`)** optimized specifically for high *Recall* to ensure vulnerable individuals are not missed.
* **Class Imbalance Rectification:** The core training workflow handles heavily skewed distributions using synthetic minority oversampling strategies.
* **Comprehensive Demographics & Lifestyle Tracking:** Processes multiple operational parameters including screen time, platform usage (Instagram/TikTok), sleep hours, anxiety indexes, and academic performance levels.
* **Interactive Risk Engine:** Streamlit UI allows school counselors, parents, or health organizations to input specific teenager metrics and instantly obtain safe risk assessments.

---

## 📊 Dataset Profile & Attributes

The predictive engine processes the custom **`Teen_Mental_Health_Dataset.csv`** comprising the following behavioral variables:
* **Demographics:** `age` (13–19), `gender` (Encoded)
* **Digital Footprint:** `daily_social_media_hours`, `platform_usage` (Instagram, TikTok, Both), `screen_time_before_sleep`
* **Health & Wellness:** `sleep_hours`, `physical_activity`
* **Academic & Social Indices:** `academic_performance`, `social_interaction_level`
* **Psychological Scales:** `stress_level`, `anxiety_level`, `addiction_level`
* **Target Categorization:** `depression_label` (`1` for High Risk / `0` for Low Risk)

---

## 🛠️ Project Structure

```text
├── Assignment_ Teen Depression Prediction...pdf   # Project requirements guidelines
├── TeenDepressionAnalysis.ipynb                  # Notebook containing EDA, SMOTE tuning, & validation
├── app.py                                        # Production script for the Streamlit web layout
├── TeenDepressionAnalysisModel.pkl               # Trained serialized Random Forest Classifier
├── Teen_Mental_Health_Dataset.csv                # Baseline structural training records
└── requirements.txt                              # Complete project dependencies list
