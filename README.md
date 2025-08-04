# Smart Irrigation System using AI

This project was developed as part of the **Shell-Edunet Skills4Future Internship in collaboration with AICTE**, aimed at building sustainable, AI-powered solutions. The goal is to automate irrigation decisions using sensor data and machine learning for optimal water usage in agriculture.

---

## Project Summary

An AI-powered Smart Irrigation System that predicts sprinkler activity (ON/OFF) for different farm parcels based on real-time environmental and soil sensor data. The system is trained using a MultiOutput Random Forest Classifier and deployed via an interactive Streamlit dashboard.

---

## Features

- **MultiParcel Prediction**: Classifies sprinkler status (ON/OFF) for 3 different land parcels.
- **Real-Time Dashboard**: Interactive sliders to input scaled sensor values and view predictions instantly.
- **Random Input Generator**: Auto-fill with random scaled values for quick simulations.
- **Clean Visualizations**: Step plots showcasing irrigation patterns over time.
- **Deployment Ready**: App built using Streamlit, model exported via `joblib`.

---

## Tech Stack

- **Language**: Python  
- **Libraries**: 
  - `scikit-learn` for model training  
  - `pandas`, `numpy` for data processing  
  - `matplotlib`, `seaborn` for data visualization  
  - `streamlit` for deployment  
  - `joblib` for model serialization

---


---

## Sensor Inputs Used

- Soil Moisture (2 sensors)  
- Soil Temperature  
- Air Temperature  
- Humidity  
- Light Intensity  
- Wind Speed  
- Rain Detection  
- pH Level  
- EC Value (Electrical Conductivity)  
- Nitrogen, Phosphorus, Potassium Levels  
- Crop Type Index  
- Leaf Wetness  
- Canopy Temperature  
- Solar Radiation  
- Water Pressure  
- Valve Feedback

*All inputs were scaled between 0 and 1 using MinMaxScaler.*

---

## Model Overview

- **Model**: MultiOutputClassifier (Random Forest Base)  
- **Training Accuracy**: High F1-scores on test set for all parcels  
- **Split**: 80% training, 20% testing  
- **Evaluation**: `classification_report()` per parcel

---

Developed Under
Program: Shell-Edunet Skills4Future AICTE Internship (July–Aug 2025)
Domain: Green Skills using Artificial Intelligence
Mentorship: Provided by Edunet Foundation, AICTE & Shell

--- 

## Author
Somya Das
https://www.linkedin.com/in/somya-das-30715b262/
somyadas.2005@gmail.com

----
