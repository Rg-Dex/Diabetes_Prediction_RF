to run the gui open anaconda > then > MachineLearning > Terminal and paste this :
streamlit run "C:\Users\abhay\OneDrive\Desktop (1)\Diabetes prediction\Diabetes Pred Gui.py"

# 🩺 Diabetes Prediction using Machine Learning

This project uses machine learning models to predict the likelihood of diabetes in patients based on medical diagnostic data. It leverages supervised learning techniques and a streamlined data science workflow to build, compare, and deploy classification models.

---

## 📊 Overview

- **Dataset**: Pima Indians Diabetes Dataset
- **Tools Used**: Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, pickle, **Streamlit**
- **Techniques**: Data Cleaning, Imputation, Feature Scaling, Model Building, Evaluation, Serialization, GUI Development

---

## 🔍 Problem Statement

Diabetes is a chronic medical condition that requires early diagnosis to manage effectively. The goal of this project is to predict whether a patient has diabetes using health indicators such as glucose levels, BMI, insulin levels, etc.

---

## 🧹 Data Preprocessing

- Loaded dataset using `pandas`
- Identified and replaced biologically invalid `0` values in key columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with the column means
- Created feature matrix `X` and target vector `y` from the dataset
- Split the data into **80% training** and **20% testing** sets

---

## ⚙️ Model Building

Used scikit-learn `Pipeline` to streamline preprocessing and modeling. Models built and evaluated:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Classifier (SVC)**
- **Decision Tree**
- **Random Forest** (with `max_depth=3`)
- **Gradient Boosting Classifier**

Each model pipeline includes:
- Feature scaling using `StandardScaler` (where needed)
- Model training and evaluation using `model.score()` on the test set

---

## 📈 Model Evaluation

Test accuracy comparison across models:

| Model         | Test Accuracy (%) |
|---------------|-------------------|
| Logistic Regression | ~78% |
| KNN               | ~75% |
| SVC               | ~77% |
| Decision Tree     | ~74% |
| Random Forest     | ~81% |
| Gradient Boosting | **~82%** ✅ |

*Gradient Boosting achieved the highest accuracy.*

---

## 🖥️ Streamlit App

Built a dynamic **Graphical User Interface (GUI)** using **Streamlit** to interact with the trained model in real time:

- Accepts user input for independent features:
  - `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
- Returns a prediction output: **"Diabetic"** or **"Non-Diabetic"**
- Provides a fast and accessible way for non-technical users to interact with the model

Run it locally:
```bash
streamlit run app.py
````

---

## 💾 Model Serialization

The best model was saved using `pickle`:

```python
import pickle
filename = 'trained_model.sav'
pickle.dump(best_model, open(filename, 'wb'))
```

This allows for reloading and using the model in production without retraining.

---


## 🚀 Future Enhancements

* Add **input validation** and **probability sliders** to Streamlit app
* Include **ROC curves**, **confusion matrices**, and **feature importance plots**
* Extend with **Power BI integration** for visual summary dashboards

---

## 🙌 Acknowledgements

* [UCI Machine Learning Repository – Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* scikit-learn, pandas, and the open-source Python community

