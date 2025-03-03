# 🏥 **Predicting Medical Expenses using Linear Regression**  

**Skills:** Linear Regression, Feature Engineering, Scikit-Learn, Data Visualization  

---

## 🚀 **Project Overview**  
This project builds a **Linear Regression model** to predict **medical insurance charges for customers** based on factors like:  
- **Age**  
- **Sex**  
- **Body Mass Index (BMI)**  
- **Number of children**  
- **Smoking habits**  
- **Region of residence**  

This project is part of **Jovian’s Machine Learning with Python course**, where we develop an **interpretable model** that helps insurance companies estimate **medical costs** for new customers.  

📌 **Reference:** [Jovian ML Course](https://jovian.com/learn/machine-learning-with-python-zero-to-gbms/lesson/linear-regression-with-scikit-learn)  

---

## 🎯 **Key Objectives**  
✔ **Understand the relationship between patient features & medical expenses**  
✔ **Build an interpretable Linear Regression model using Scikit-Learn**  
✔ **Perform Exploratory Data Analysis (EDA) to visualize key trends**  
✔ **Evaluate model performance & optimize feature selection**  

---

## 📊 **Dataset Overview**  
The dataset contains **verified historical medical charges** for over **1,300 customers**.  

📌 **Dataset Source:** [Medical Charges Dataset](https://github.com/stedy/Machine-Learning-with-R-datasets)  

✅ **Example: Loading the Dataset**  
```python
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv")
df.head()
```
💡 **Key Features:**  
- **age:** Patient's age (numeric)  
- **sex:** Gender (categorical: male/female)  
- **bmi:** Body Mass Index (numeric)  
- **children:** Number of dependent children (numeric)  
- **smoker:** Smoker status (binary: yes/no)  
- **region:** Geographical region (categorical: northeast, northwest, southeast, southwest)  
- **charges:** Actual medical expenses (target variable)  

---

## 📈 **Exploratory Data Analysis (EDA)**  
We perform **data visualization & statistical analysis** to understand trends in medical expenses.  

✅ **Example: Distribution of Medical Charges**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df["charges"], bins=30, kde=True)
plt.title("Distribution of Medical Charges")
plt.show()
```
💡 **Observation:**  
- Most medical charges **fall below $20,000**, but **some outliers exceed $60,000**.  
- Smoking status significantly impacts medical costs.  

✅ **Example: Correlation Analysis**  
```python
df.corr()["charges"].sort_values(ascending=False)
```
💡 **Key Findings:**  
- **Smoking status has the highest correlation with charges.**  
- **BMI & age also influence medical costs.**  

✅ **Example: Box Plot of Charges by Smoking Status**  
```python
sns.boxplot(x="smoker", y="charges", data=df)
plt.title("Medical Charges for Smokers vs Non-Smokers")
```
💡 **Key Insight:**  
Smokers **pay significantly higher medical expenses** compared to non-smokers.  

---

## 🏗 **Feature Engineering & Data Preprocessing**  
Before training our model, we **encode categorical variables** and **scale numerical features**.  

✅ **Encoding Categorical Features**  
```python
df = pd.get_dummies(df, columns=["sex", "region", "smoker"], drop_first=True)
```
💡 **Why?** – Converts **categorical data into numerical form** for regression.  

✅ **Feature Scaling using StandardScaler**  
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[["age", "bmi", "children"]] = scaler.fit_transform(df[["age", "bmi", "children"]])
```
💡 **Why?** – Linear regression performs better when features are scaled.  

---

## 🤖 **Training the Linear Regression Model**  
We split the data into **training & testing sets** and fit a **linear regression model**.  

✅ **Train-Test Split**  
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["charges"])
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

✅ **Fitting the Model**  
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

✅ **Model Predictions**  
```python
y_pred = model.predict(X_test)
```

---

## 📊 **Model Evaluation & Performance Metrics**  
We evaluate the model using **R² score, Mean Squared Error (MSE), and Mean Absolute Error (MAE)**.  

✅ **Calculate RMSE & R² Score**  
```python
from sklearn.metrics import mean_squared_error, r2_score

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")
```
💡 **Interpretation:**  
- A **high R² score (closer to 1)** means the model explains most of the variance.  
- **Lower RMSE = Better model accuracy.**  

✅ **Feature Importance Analysis**  
```python
feature_importance = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
print(feature_importance)
```
💡 **Key Insights:**  
- **Smoking status is the strongest predictor of medical expenses.**  
- **Age and BMI also have significant impact.**  

---

## 🔮 **Future Enhancements**  
🔹 **Try Polynomial Regression** – Captures non-linear relationships  
🔹 **Apply Regularization (Ridge/Lasso)** – Prevents overfitting  
🔹 **Use Decision Trees or Random Forests** – Compare with non-linear models  

---

## 🎯 **Why This Project Stands Out for ML & Data Science Roles**  
✔ **Explains the entire regression pipeline** – EDA, feature engineering, model evaluation  
✔ **Interpretable ML model** – Important for real-world business applications  
✔ **Applies best practices in Scikit-Learn** – StandardScaler, feature selection, R² score  
✔ **Practical use case** – Insurance companies use similar models for premium estimation  

---

## 🛠 **How to Run This Project**  
1️⃣ Clone the repo:  
   ```bash
   git clone https://github.com/shrunalisalian/medical-expenses-prediction.git
   ```
2️⃣ Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ Run the Jupyter Notebook:  
   ```bash
   jupyter notebook "Predicting Medical Expenses using Linear Regression.ipynb"
   ```

---

## 📌 **Connect with Me**  
- **LinkedIn:** [Shrunali Salian](https://www.linkedin.com/in/shrunali-salian/)  
- **Portfolio:** [Your Portfolio Link](#)  
- **Email:** [Your Email](#)  
