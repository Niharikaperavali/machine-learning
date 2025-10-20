
# ❤️ Heart Disease Prediction using Machine Learning

### 🔍 Overview

This project aims to **predict the likelihood of heart disease** in patients based on clinical data such as age, cholesterol, blood pressure, and ECG results.  
It applies multiple **Machine Learning algorithms** — Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, and KNN — to build, compare, and select the best model for prediction.  
Finally, the project deploys the best-performing model using **Streamlit** to make real-time predictions.

---

## 🎯 Objective

- To develop an **accurate and reliable model** for predicting heart disease.  
- To identify the **most influential medical features** affecting heart disease likelihood.  
- To compare and tune different ML algorithms for best performance.  
- To deploy the model as an **interactive Streamlit web app**.

---

## 🧠 Dataset Information

- **Source:** UCI Machine Learning Repository (Cleveland Heart Disease Dataset)  
- **Rows:** ~1000  
- **Columns:** 14 (13 input features + 1 target label)  
- **Target Variable:** `target` → (0 = No Disease, 1 = Heart Disease)

### 📋 **Feature Description**

| Feature | Description | Type | Example |
|----------|-------------|------|----------|
| **age** | Age of the patient | Numeric | 52 |
| **sex** | Gender (1 = Male, 0 = Female) | Categorical | 1 |
| **cp (chest pain type)** | 0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic | Categorical | 2 |
| **trestbps (resting blood pressure)** | Resting blood pressure (in mm Hg) | Numeric | 130 |
| **chol (serum cholesterol)** | Cholesterol level in mg/dl | Numeric | 230 |
| **fbs (fasting blood sugar)** | Fasting blood sugar >120 mg/dl (1 = True, 0 = False) | Binary | 0 |
| **restecg (resting ECG results)** | 0 = Normal, 1 = ST-T abnormality, 2 = Left ventricular hypertrophy | Categorical | 1 |
| **thalach (maximum heart rate achieved)** | Max heart rate reached during exercise | Numeric | 150 |
| **exang (exercise induced angina)** | 1 = Yes, 0 = No | Binary | 0 |
| **oldpeak** | ST depression induced by exercise relative to rest | Numeric (float) | 2.3 |
| **slope** | Slope of the peak exercise ST segment (0 = Upsloping, 1 = Flat, 2 = Downsloping) | Categorical | 1 |
| **ca** | Number of major vessels (0–4) colored by fluoroscopy | Numeric | 0 |
| **thal** | Thalassemia (1 = Normal, 2 = Fixed defect, 3 = Reversible defect) | Categorical | 2 |
| **target** | 0 = No Heart Disease, 1 = Heart Disease | Target | 1 |

---

## ⚙️ Workflow

### 🧹 **1️⃣ Data Cleaning & Preprocessing**
- Removed duplicate records.  
- Replaced missing values using **median imputation** to handle skewness.  
- Detected and handled outliers using **IQR (Interquartile Range)** method — capped extreme values instead of removing them to preserve data.  

### 📊 **2️⃣ Exploratory Data Analysis (EDA)**
- Visualized class balance with bar plots.  
- Used **histograms** for feature distribution visualization.  
- Applied **boxplots** to identify potential outliers.  
- Generated **correlation heatmap** to study relationships between attributes.

### 🧩 **3️⃣ Feature Engineering**
- Scaled numeric features using **StandardScaler** (inside ML pipelines).  
- Split dataset into **train (80%)** and **test (20%)** sets using stratified sampling.

### 🧠 **4️⃣ Model Building**
Implemented the following ML models with Scikit-learn Pipelines:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)

Each pipeline included:
- A StandardScaler (for normalization)  
- The ML model (for training & prediction)

### ⚡ **5️⃣ Hyperparameter Tuning**
Optimized key models using:
- **GridSearchCV** for Decision Tree, Random Forest, and SVM  
- **RandomizedSearchCV** for XGBoost  

This improved performance and reduced overfitting.

### 📈 **6️⃣ Model Evaluation**
Metrics used:
- **Accuracy** — overall correctness  
- **F1 Score** — balance between precision and recall  
- **ROC AUC Score** — model’s ability to distinguish between classes  

📌 *Primary selection criteria:*  
**ROC AUC** (to ensure separation power) and **F1 Score** (to balance sensitivity and precision), since in medical prediction, missing a positive case is more critical than a false alarm.

### 🌟 **7️⃣ Feature Importance**
Using **XGBoost’s feature importance**, we identified the most influential medical parameters:
- **cp (chest pain type)** → highest impact  
- **thalach (max heart rate)** → strong negative correlation with disease  
- **oldpeak** → exercise-induced ECG depression, major predictor  
- **ca, thal** → moderate influence  
- **chol, trestbps, age** → smaller but relevant roles

### 🌐 **8️⃣ Deployment**
- The best model (tuned **XGBoost Pipeline**) saved as `best_model.joblib` using **Joblib**.  
- A **Streamlit web app** built to allow users to input parameters and predict heart disease risk interactively.

---

## 🧩 Project Structure

```

heart-disease-prediction/
│
├── data/
│   └── heart.csv
│
├── models/
│   └── best_model.joblib
│
├── notebook/
│   └── heart_disease_prediction.ipynb
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md

````

---

## 🧰 Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python |
| **Libraries** | NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn |
| **Deployment** | Streamlit |
| **Model Saving** | Joblib |
| **IDE** | VS Code + Jupyter Notebook |

---

## 📊 Results Summary

| Model | Accuracy | F1 Score | ROC AUC |
|--------|-----------|-----------|-----------|
| **XGBoost (Tuned)** | **1.00** | **1.00** | **1.00** |
| Random Forest (Tuned) | 1.00 | 1.00 | 1.00 |
| SVM (Tuned) | 0.99 | 0.99 | 0.99 |
| Decision Tree (Tuned) | 0.98 | 0.98 | 0.98 |

✅ **Final Model Selected:** Tuned **XGBoost Pipeline** — highest ROC AUC and F1 performance.

---

## 💡 Key Insights from Feature Importance

| Rank | Feature | Medical Significance |
|-------|----------|----------------------|
| 1️⃣ | **cp (Chest Pain Type)** | Type and severity of chest pain strongly correlate with heart disease. |
| 2️⃣ | **thalach (Max Heart Rate)** | Lower max heart rate during exercise often indicates poor cardiovascular health. |
| 3️⃣ | **oldpeak** | ST depression during exercise indicates reduced blood flow to heart muscles. |
| 4️⃣ | **ca (Major Vessels)** | Number of blocked vessels — directly linked to heart disease presence. |
| 5️⃣ | **thal (Thalassemia)** | Abnormal thalassemia test indicates risk of heart defects. |
| 6️⃣ | **chol, trestbps, age** | Supportive factors that influence risk but not the sole determinants. |

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Jupyter Notebook

```bash
jupyter notebook notebook/heart_disease_prediction.ipynb
```

### 4️⃣ Train and Save Best Model

Run all notebook cells — the best model will be saved automatically in the `/models` folder.

### 5️⃣ Run Streamlit App

```bash
cd app
streamlit run streamlit_app.py
```

### 6️⃣ Access the Web App

* Streamlit will open automatically in your browser.
* Enter patient details → click **Predict** → view heart disease probability.

---

## 🏁 Conclusion

* Built a reliable, interpretable, and high-performing **ML model for heart disease prediction**.
* **XGBoost** achieved the best overall performance across all evaluation metrics.
* Implemented **data preprocessing, outlier capping, EDA, model comparison, and tuning** systematically.
* Successfully deployed an **interactive prediction web app** using Streamlit.


---

## 👨‍💻 Author

**Harshavardhan Nadiveedi**
🎓 B.Tech — Artificial Intelligence & Machine Learning
💡 Interests: Data Science, Machine Learning, Artificial Intelligence


