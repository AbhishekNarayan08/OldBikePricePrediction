# 🏍️ Old Bike Price Prediction

This repository implements a neural network model to predict the prices of old bikes using historical listings data. It includes data processing, model training, and a Flask web app for inference.

---

## 📁 Repository Structure

```
.
├── templates/           # HTML templates for the Flask web app
├── used_bikes/          # Static assets (if any) like images/CSS/JS
├── Bike_Data.csv        # Raw dataset of used bike listings
├── app.py               # Flask application for serving predictions
├── model.py             # Defines and trains the neural network model
├── requirements.txt     # Python dependencies
├── Procfile             # Heroku process file for deployment
└── README.md            # This file
```

---

## 🧰 Environment Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/AbhishekNarayan08/OldBikePricePrediction.git
   cd OldBikePricePrediction
   ```

2. **Create a virtual environment** (optional but recommended)  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## 📝 Data Preparation

- **Dataset:** `Bike_Data.csv` contains columns such as model, year, mileage, engine capacity, and price.
- **Preprocessing:** Implemented in `model.py`:
  - Handling missing values
  - Encoding categorical features (e.g., model names)
  - Feature scaling (StandardScaler for numerical fields)

---

## 🤖 Model Training

Defined in `model.py`:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output: predicted price
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
model.save('bike_price_model.h5')
```

- **Loss function:** Mean Squared Error  
- **Metrics:** Mean Absolute Error

---

## 🚀 Running the Web App

1. **Ensure model file exists** (`bike_price_model.h5`). If not, train the model by running:  
   ```bash
   python model.py
   ```

2. **Start Flask server**  
   ```bash
   python app.py
   ```

3. **Access** the app at [http://localhost:5000](http://localhost:5000).

---

## 📦 Deployment

- **Procfile** included for Heroku deployment:

```
web: gunicorn app:app
```

---

## 📈 Results

- **Example Prediction:**  
  | Year | Mileage | Engine CC | Predicted Price |
  |------|---------|-----------|-----------------|
  | 2015 | 25000   | 150       | ₹45,000         |

- **Model Performance:**  
  - MSE: 120000000  
  - MAE: 8000  

*(Adjust metrics after retraining)*

---

## 📚 References

- [Keras Documentation](https://keras.io/)  
- [Flask Documentation](https://flask.palletsprojects.com/)

---

## 👤 Author

**Abhishek Narayan**  
IIT Delhi  
