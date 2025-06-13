import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Load dataset
def load_dataset():
    csv_path = os.path.join('frgapp', 'recipes.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.dropna(subset=['title', 'ingredients', 'instructions'], inplace=True)
        return df
    else:
        print("CSV not found!")
        return None

# Train KNN model using TF-IDF on ingredients
def train_knn_model(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['ingredients'])  # Features
    y = df['title']                                  # Target label (title of recipe)

    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X.toarray())

    model = KNeighborsClassifier(n_neighbors=1,metric='cosine')
    model.fit(X_scaled, y)

    return vectorizer, model  # Return only vectorizer and model, not df


# Function to predict multiple recipes based on input ingredients
def predict_recipe(ingredients, vectorizer, model, df, top_n=5):
    # Vectorize the input ingredients
    input_features = vectorizer.transform([ingredients])
    
    # Scale the input features (as the model was trained with scaled features)
    scaler = StandardScaler(with_mean=False)
    input_scaled = scaler.fit_transform(input_features.toarray())
    
    # Predict using KNN model (finding nearest neighbors)
    distances, indices = model.kneighbors(input_scaled, n_neighbors=top_n)  # Get top N closest recipes
    
    # Extract the predicted recipe titles
    predicted_recipes = []
    for index in indices[0]:
        recipe = df.iloc[index]
        predicted_recipes.append({
            'title': recipe['title'],
            'ingredients': recipe['ingredients'],  # Assuming 'ingredients' is a column in df
            'instructions': recipe['instructions']  # Assuming 'instructions' is a column in df
        })
    
    return predicted_recipes
# Example test (if you run this file directly)
if __name__ == "__main__":
    df = load_dataset()
    if df is not None:
        vectorizer, scaler, knn, df = train_knn_model(df)

        # Sample user input
        user_input = "chicken, garlic, onion"
        result = predict_recipe(user_input, vectorizer, scaler, knn, df)

        print("\n🔍 Predicted Recipe:")
        print(f"🍽️ Title: {result['title']}")
        print(f"📋 Ingredients: {result['ingredients']}")
        print(f"📝 Instructions: {result['instructions'][:150]}...")
