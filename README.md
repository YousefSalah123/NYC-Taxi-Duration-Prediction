
# 🚖 NYC Taxi Trip Duration Prediction

Predicting the duration of NYC taxi rides using Linear Regression, based on historical trip data. This project demonstrates an end-to-end machine learning pipeline from raw data to an interactive prediction app.

---

## 🎯 Project Goal

Build a full ML pipeline to:
- Clean and explore the dataset
- Engineer effective features
- Train and evaluate a regression model
- Deploy a real-time prediction app using Streamlit

---

## 📊 Dataset

- **Source**: NYC Taxi Trip data (`https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data`)
- **Content**:  
  Includes:
  - Pickup and dropoff timestamps
  - Latitude and longitude coordinates
  - Passenger count
  - Vendor ID and flags
  - Target: `trip_duration` in seconds

---

## 🛠️ Steps Followed

### 1. 🧹 Data Cleaning
- Removed extreme outliers (unrealistic durations, distances)
- Filtered missing or invalid entries

### 2. 📈 Exploratory Data Analysis (EDA)
- Visualizations:
  - Histograms
  - Boxplots
  - Scatter plots
  - Correlation heatmap
- Added expert observations after each plot

## 📊 Exploratory Data Analysis (Visuals)

To better understand the data, several visualizations were performed:

### 1. 📦 Trip Duration Boxplot

This boxplot shows the distribution of `trip_duration` with many visible outliers. Most trips fall under 2000 seconds, while a long tail of outliers exists above that.

![Trip Duration Boxplot](images/eda_trip_duration_boxplot.png)

---

### 2. 📍 Trip Duration vs. Distance (Scatter Plot)

This scatter plot indicates a strong positive correlation between `distance_km` and `trip_duration`. Most trips are short-distance and take under 2000 seconds.

![Trip Duration vs Distance](images/eda_trip_duration_vs_distance.png)

---

### 3. 🧮 Histogram of Log-Transformed Trip Duration

The original trip duration was right-skewed. After applying a log transformation, the distribution became more Gaussian—ideal for regression modeling.

![Log Trip Duration Distribution](images/log_trip_duration_hist.png)

---

### 4. 👥 Trip Duration by Passenger Count (Boxplot)

This boxplot shows how trip duration varies with the number of passengers. There is no significant variation, implying passenger count has limited predictive power.

![Trip Duration by Passenger Count](images/eda_boxplot_passenger_count.png)

---

### 5. 🔗 Correlation Heatmap

This heatmap shows the correlation between different numerical features. The strongest positive correlation is between `distance_km` and `trip_duration` (~0.77).

![Correlation Heatmap](images/eda_correlation_heatmap.png)

---

📘 *For the full exploratory data analysis with more plots and detailed explanations, please refer to the full [Jupyter Notebook](trip_duration_prediction_reviewed_final.ipynb).*


### 3. 🧬 Feature Engineering
- Extracted time-based features: `pickup_hour`, `pickup_weekday`, `pickup_month`
- Calculated trip `distance_km` using Haversine formula
- One-hot encoded categorical variables
- Applied **cyclic encoding** to time-based features (e.g., hour, weekday)

### 4. 🤖 Model Training
- Used `LinearRegression` from scikit-learn
- Applied log transformation to target (`trip_duration`) to reduce skewness

### 5. 📊 Model Evaluation
- Metrics reported:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- Included prediction error visualization

### 6. 💾 Model Saving
- Trained model saved as: `model.pkl`
- Feature list saved as: `features_used.pkl`

### 7. 🚀 Deployment with Streamlit
- Built an interactive web app (`streamlit_app.py`)
- Accepts user inputs for trip parameters
- Returns predicted trip duration (in minutes/seconds)
- Includes comparison plot between actual and predicted

---

## 🧠 Model

| Item              | Details                     |
|-------------------|-----------------------------|
| Algorithm         | Linear Regression           |
| Target Variable   | `log(trip_duration)`        |
| Engineered Features | Distance, Time, One-hot, Cyclic |
| Libraries Used    | scikit-learn, pandas, numpy, seaborn, matplotlib |

---

## 💻 Streamlit App

### ⚙️ How to Run Locally

1. 📦 Install dependencies:

```bash
pip install -r requirements.txt
```

2. ▶️ Run the Streamlit app:

```bash
streamlit run Streamlit_app.py
```

3. 🌐 Open your browser and access:
```
http://localhost:8501
```

---

## 🖥️ Streamlit App Preview

### 🔹 App Interface
Shows the input form and overall layout of the web app.

![Streamlit App](images/streamlit_app.png.jpg)

### 🔹 Output Example
Displays predicted duration vs. NYC average and user input breakdown.

![Streamlit Output](images/streamlit_output_summary.jpg.jpg)
---

## 📁 Project Files

| File Name                 | Description                                             |
|------------------------|----------------------------------------------|
| `trip_duration_prediction_reviewed_final.ipynb`  | Full pipeline: cleaning, EDA,modeling  |
| `Streamlit_app.py`   | Streamlit app for real-time predictions   |
| `model.pkl`               | Saved regression model                            |
| `features_used.pkl` | List of selected features used in the model  |
| `requirements.txt`   | Python dependencies                                      |

---

## 🚧 Future Improvements

- Try ensemble models (Random Forest, XGBoost) for better accuracy
- Incorporate external data (e.g., weather, traffic)
- Add map-based pickup/dropoff clustering (zone encoding)
- Deploy on cloud (Streamlit Cloud, Heroku, AWS, etc.)
- Add unit tests and CI/CD pipeline

---

## 🙋‍♂️ Author 
**Yousef Salah Nage**  
AI & Machine Learning Enthusiast  

## 🔗 Connect with Me
- [LinkedIn](https://www.linkedin.com/in/yousef-salah-nage-a3583636b)
- [GitHub](https://github.com/YousefSalah123)


---

## 📄 License

This project is licensed under the [MIT License](LICENSE).


---

## ⭐ Show Support

If you found this project helpful, feel free to:
- ⭐ Star the repo
- 🧠 Fork and experiment
- 📣 Share it with others
