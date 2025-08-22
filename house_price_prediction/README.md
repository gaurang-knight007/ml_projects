# 🏡 House Price Prediction

A beginner-friendly **Machine Learning project** that predicts median house prices using the **California Housing dataset**.  
This project demonstrates **data preprocessing, feature engineering, regression model training, evaluation, and predictions**.  

---

## 📂 Project Structure
```bash
house-price-prediction/
│── data/ # datasets (if needed)
│── models/ # trained model + scaler
│ ├── house_model.pkl
│ └── scaler.pkl
│── notebooks/
│ └── house_price.ipynb # Jupyter exploration + visualization
│── src/
│ ├── preprocess.py # preprocessing script
│ ├── train.py # training script
│ └── predict.py # prediction script
│── requirements.txt # dependencies
│── README.md # documentation
│── venv/ # virtual environment (not uploaded)
```

---

## 📊 Dataset

We use the **California Housing Dataset**, available directly from `scikit-learn`.  
- Features: `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude`  
- Target: `MedHouseVal` (Median House Value in $100,000s)  

📎 Dataset Reference: [Scikit-learn California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

### 2️⃣ Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate   # Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
## ⚙️ Usage
### 🔹 1. Explore Dataset
```bash
jupyter notebook notebooks/house_price.ipynb
```
Visualizations
Correlation Heatmap
Data Overview

### 🔹 2. Train Model
```bash
python src/train.py
```
This will:

Load and preprocess data
Train a Linear Regression model
Print evaluation metrics (MSE, R² Score)
Save the model to models/house_model.pkl

### 🔹 3. Make Predictions
```bash
python src/predict.py
```
output:
Predicted House Price: 2.13
💡 Note: The target variable MedHouseVal is in $100,000s.
So 2.13 → $213,000

## 📈 Model Performance
Example metrics from Linear Regression:

MSE (Mean Squared Error): ~0.55
R² Score: ~0.57
You can improve performance by:
Trying other models (Random Forest, Gradient Boosting, XGBoost)
Feature engineering
Hyperparameter tuning

## 📌 Features
✅ Data preprocessing with StandardScaler
✅ Linear Regression model training
✅ Model persistence using joblib
✅ Predict house prices for custom input
✅ Clean modular project structure

## 🔮 Future Improvements
Deploy with Flask/FastAPI (Web App)
Try Neural Networks (TensorFlow / PyTorch)
Use Kaggle House Price dataset for advanced regression
Add hyperparameter tuning (GridSearchCV, RandomizedSearchCV)

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📜 License
This project is licensed under the MIT License.

## 👨‍💻 Author
```bash
GAURANG GAUTAM
📧 gaurangbdb@gmail.com
```