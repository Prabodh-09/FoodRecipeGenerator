from django.shortcuts import render
from .ml_logic import load_dataset, train_knn_model, predict_recipe

# Load dataset and train model once at startup
df = load_dataset()
vectorizer, model = train_knn_model(df)  # Expecting only two values: vectorizer and model

def home(request):
    return render(request, 'home.html')

def result(request):
    if request.method == 'POST':
        # Get the input ingredients from the user
        ingredients = request.POST.get('ingredients')
        
        if ingredients:
            # Predict multiple recipes using the KNN model
            top_n = 5  # Adjust this value for the number of recipes you want to return
            predictions = predict_recipe(ingredients, vectorizer, model, df, top_n=top_n)
            
            return render(request, 'result.html', {
                'ingredients': ingredients,
                'recipes': predictions,  # Pass multiple recipes to the template
            })
        else:
            return render(request, 'result.html', {
                'error': "Please enter ingredients."  # Handle case where no ingredients are provided
            })
    
    return render(request, 'home.html')


