# 📊 Customer Segmentation Dashboard

A Machine Learning web application that segments wholesale customers using the **K-Means Clustering** algorithm. The dashboard provides interactive data exploration, customer cluster prediction, PCA visualization, and business insights through a modern Streamlit interface.

---

## 🌐 Live Demo

🔗 https://customer-segmentation-dashboard-tcgjinsqc395963eeqyca4.streamlit.app/

---

## 📂 GitHub Repository

🔗 https://github.com/Mohamed-ELazab22/Customer-Segmentation-Dashboard

---

# 🚀 Features

- 📊 Interactive Dashboard
- 🤖 Customer Cluster Prediction
- 📈 PCA Cluster Visualization
- 📋 Dataset Exploration
- 📉 Correlation Matrix
- 📌 Customer Spending Analysis
- 🎨 Modern Responsive UI
- 🌙 Dark Theme Design

---

# 📁 Dataset

**Wholesale Customers Dataset**

The dataset contains annual spending for wholesale customers across different product categories.

### Features

- Channel
- Region
- Fresh
- Milk
- Grocery
- Frozen
- Detergents_Paper
- Delicassen

Target:

- Customer Cluster (Generated using K-Means)

---

# 🧠 Machine Learning Pipeline

The project follows this workflow:

```
Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Log Transformation (np.log1p)
      │
      ▼
StandardScaler
      │
      ▼
K-Means Clustering
      │
      ▼
PCA (2D Visualization)
      │
      ▼
Prediction Dashboard
```

---

# 📊 Dashboard Pages

## 🏠 Dashboard

- Dataset Preview
- Dataset Statistics
- Customer Distribution
- Region Distribution
- KPI Cards

---

## 🎯 Prediction

Predicts the cluster for a new customer using:

- Fresh
- Milk
- Grocery
- Frozen
- Detergents Paper
- Delicassen

Outputs:

- Predicted Cluster
- Distance from Cluster Center
- Confidence Score
- Customer vs Cluster Center Comparison

---

## 📈 Visualization

Interactive PCA visualization including:

- Customer Distribution
- Cluster Centers
- Predicted Customer Position

---

## 📂 Dataset

Provides:

- Dataset Preview
- Summary Statistics
- Missing Values
- Correlation Heatmap

---

## ℹ️ About

Project overview including:

- Dataset
- Data Preprocessing
- K-Means Algorithm
- Technology Stack

---

# 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Plotly
- Joblib

---

# 📦 Project Structure

```
Customer-Segmentation-Dashboard
│
├── app.py
├── style.css
├── requirements.txt
├── Wholesale customers data.csv
├── scaler.pkl
├── pca.pkl
├── kmeans_model.pkl
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Mohamed-ELazab22/Customer-Segmentation-Dashboard.git
```

Go to the project

```bash
cd Customer-Segmentation-Dashboard
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📸 Dashboard Preview

You can add screenshots here.

Example:

```
images/dashboard.png
images/prediction.png
images/visualization.png
```

---

# 🎯 Future Improvements

- Automatic Cluster Description
- Feature Importance Analysis
- Export Prediction Report
- Customer Recommendation System
- Download Results as PDF
- Advanced Business Insights

---

# 👨‍💻 Developed By

**Mohamed Elazab**

Faculty of Computers & Artificial Intelligence

---

# ⭐ If you like this project

Give it a ⭐ on GitHub.