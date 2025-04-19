import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_bmi(weight, height):
    return weight / (height ** 2)

def calculate_daily_calories(weight, height, age, gender, activity_level):
    if gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height * 100 + 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height * 100 + 5 * age - 161

    activity_multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very active': 1.9
    }

    return bmr * activity_multipliers.get(activity_level.lower(), 1.2)

# Load dataset
data = pd.read_csv("recipes.csv")

nutrition_features = ["Calories", "Protein", "Carbohydrates", "FatContent", "CholesterolContent", "FiberContent", "SodiumContent", "SugarContent"]

# Use selected features for prediction
selected_features = ["Calories", "Protein", "Carbohydrates"]
X = data[selected_features]
y = data["recipe_id"]

# Handle missing values
for feature in selected_features:
    X[feature] = pd.to_numeric(X[feature], errors='coerce')
    X[feature].fillna(X[feature].median(), inplace=True)

data.fillna(data.median(numeric_only=True), inplace=True)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train KNN model
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X_train)

# Compute MAE & MSE
distances, indices = knn.kneighbors(X_test)
y_pred = X_train[indices].mean(axis=1)
mae = mean_absolute_error(X_test, y_pred)
mse = mean_squared_error(X_test, y_pred)

# Streamlit UI
st.title("🍽️ Recipe Recommendation System")
option = st.sidebar.selectbox("Choose an Option:", ["Model Accuracy & Data Analysis", "BMI & Calorie Calculator", "Food Prediction"])

if option == "Model Accuracy & Data Analysis":
    st.subheader("📊 Model Accuracy")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")

elif option == "BMI & Calorie Calculator":
    st.subheader("⚖️ BMI & Calorie Calculator")
    weight = st.number_input("Enter your weight (kg):", min_value=0.0, value=70.0)
    height = st.number_input("Enter your height (m):", min_value=0.0, value=1.75)
    age = st.number_input("Enter your age:", min_value=1, value=25)
    gender = st.selectbox("Select your gender:", ["Male", "Female"])
    activity_level = st.selectbox("Select your activity level:", ["Sedentary", "Light", "Moderate", "Active", "Very Active"])

    if st.button("Calculate"):
        bmi = calculate_bmi(weight, height)
        daily_calories = calculate_daily_calories(weight, height, age, gender, activity_level)
        st.write(f"**Your BMI:** {bmi:.2f}")
        st.write(f"**Daily Caloric Need:** {daily_calories:.2f} kcal")

        st.subheader("🍽️ Recommended Foods")
        input_features = scaler.transform([[daily_calories, weight * 1.2, weight * 2]])
        distances, indices = knn.kneighbors(input_features)
        recommendations = y_train.iloc[indices[0]].tolist()

        for i, recipe_id in enumerate(recommendations, start=1):
            recipe_name = data.loc[data['recipe_id'] == recipe_id, 'Name'].iloc[0]
            instructions = data.loc[data['recipe_id'] == recipe_id, 'RecipeInstructions'].iloc[0]
            st.write(f"### {i}. {recipe_name}")
            st.write(f"📜 **Instructions:** {instructions}")
            st.write("---")

        for idx in range(len(recommendations)):
            recipe_name = data.loc[data['recipe_id'] == recommendations[idx], 'Name'].iloc[0]
            st.subheader(f"Nutrient Breakdown for {recipe_name}")
            fig, ax = plt.subplots()
            nutrient_values = data.loc[data['recipe_id'] == recommendations[idx], nutrition_features].iloc[0]
            nutrient_values.plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")
            ax.set_title(f"Nutrient Breakdown for {recipe_name}")
            st.pyplot(fig)


elif option == "Food Prediction":
    st.subheader("🔍 Predict Food Based on Nutrition")
    calories = st.number_input("Enter desired Calories:", min_value=0.0, value=2000.0)
    proteins = st.number_input("Enter desired Proteins (g):", min_value=0.0, value=50.0)
    carbohydrates = st.number_input("Enter desired Carbohydrates (g):", min_value=0.0, value=250.0)

    if st.button("Get Recommendations"):
        input_features = scaler.transform([[calories, proteins, carbohydrates]])
        distances, indices = knn.kneighbors(input_features)
        recommendations = y_train.iloc[indices[0]].tolist()

        st.subheader("🍽️ Recommended Foods")
        for i, recipe_id in enumerate(recommendations, start=1):
            recipe_name = data.loc[data['recipe_id'] == recipe_id, 'Name'].iloc[0]
            instructions = data.loc[data['recipe_id'] == recipe_id, 'RecipeInstructions'].iloc[0]
            st.write(f"### {i}. {recipe_name}")
            st.write(f"📜 **Instructions:** {instructions}")
            st.write("---")

        for idx in range(len(recommendations)):
            recipe_name = data.loc[data['recipe_id'] == recommendations[idx], 'Name'].iloc[0]
            st.subheader(f"Nutrient Breakdown for {recipe_name}")
            fig, ax = plt.subplots()
            nutrient_values = data.loc[data['recipe_id'] == recommendations[idx], nutrition_features].iloc[0]
            nutrient_values.plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")
            ax.set_title(f"Nutrient Breakdown for {recipe_name}")
            st.pyplot(fig)
