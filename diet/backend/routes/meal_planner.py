from flask import Blueprint, request, jsonify, render_template
import pandas as pd
import os
import logging
import numpy as np
import random

meal_planner_blueprint = Blueprint('meal_planner', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the food data and models
DATASET_PATH = os.path.join(os.path.dirname(__file__), '../1000_plus_nutritional_food_data_full.csv')
try:
    # Load the CSV file directly
    food_df = pd.read_csv(DATASET_PATH)
    
    # Clean column names - remove spaces and convert to lowercase
    food_df.columns = food_df.columns.str.strip().str.lower()
    
    # Extract relevant columns and rename them
    if 'food item' in food_df.columns:
        food_df = food_df[['food item', 'calories', 'protein (g)', 'carbs (g)', 'fat (g)']].copy()
        food_df.columns = ['food', 'calories', 'protein', 'carbs', 'fat']
        
        # Add food category based on food name
        def categorize_food(food_name):
            food_name = str(food_name).lower()
            if any(term in food_name for term in ['beef', 'steak', 'hamburger', 'veal', 'lamb']):
                return 'beef'
            elif any(term in food_name for term in ['chicken', 'poultry', 'turkey', 'duck']):
                return 'poultry'
            elif any(term in food_name for term in ['fish', 'salmon', 'tuna', 'cod', 'seafood', 'shrimp', 'prawn']):
                return 'seafood'
            elif any(term in food_name for term in ['vegetable', 'veggie', 'broccoli', 'spinach', 'carrot', 'tomato', 'cabbage', 'lettuce']):
                return 'vegetables'
            elif any(term in food_name for term in ['fruit', 'apple', 'banana', 'orange', 'berry', 'grape', 'melon']):
                return 'fruits'
            elif any(term in food_name for term in ['grain', 'rice', 'pasta', 'bread', 'cereal', 'oat']):
                return 'grains'
            elif any(term in food_name for term in ['dairy', 'milk', 'cheese', 'yogurt', 'cream']):
                return 'dairy'
            elif any(term in food_name for term in ['snack', 'chip', 'crisp', 'chocolate', 'candy', 'sweet']):
                return 'snacks'
            elif any(term in food_name for term in ['nut', 'peanut', 'almond', 'cashew']):
                return 'nuts'
            elif any(term in food_name for term in ['bean', 'lentil', 'legume', 'pea']):
                return 'legumes'
            elif any(term in food_name for term in ['egg']):
                return 'eggs'
            else:
                return 'other'
                
        food_df['category'] = food_df['food'].apply(categorize_food)
        
        logging.info(f"Food dataset loaded successfully with {len(food_df)} items")
        logging.info(f"Categories identified: {food_df['category'].unique()}")
    else:
        logging.error("CSV file has unexpected column format")
        # Create simple placeholder data if the file doesn't have expected columns
        food_df = pd.DataFrame({
            'food': ['Chicken Breast', 'Salmon', 'Brown Rice', 'Broccoli', 'Egg'],
            'calories': [165, 206, 216, 55, 78],
            'protein': [31, 22, 5, 3.7, 6.3],
            'carbs': [0, 0, 45, 11.2, 0.6],
            'fat': [3.6, 12, 1.8, 0.6, 5.3],
            'category': ['poultry', 'seafood', 'grains', 'vegetables', 'eggs']
        })
except Exception as e:
    logging.error(f"Error loading food dataset: {e}")
    # Create sample data as a fallback
    food_df = pd.DataFrame({
        'food': ['Chicken Breast', 'Salmon', 'Brown Rice', 'Broccoli', 'Egg'],
        'calories': [165, 206, 216, 55, 78],
        'protein': [31, 22, 5, 3.7, 6.3],
        'carbs': [0, 0, 45, 11.2, 0.6],
        'fat': [3.6, 12, 1.8, 0.6, 5.3],
        'category': ['poultry', 'seafood', 'grains', 'vegetables', 'eggs']
    })

@meal_planner_blueprint.route('/meal-planner')
def meal_planner_page():
    """Render the meal planner page"""
    return render_template('meal_planner.html')

@meal_planner_blueprint.route('/api/get-food-categories')
def get_food_categories():
    """Get all available food categories"""
    try:
        categories = sorted(food_df['category'].unique().tolist())
        return jsonify({"categories": categories})
    except Exception as e:
        logging.error(f"Error getting food categories: {e}")
        return jsonify({"error": str(e)}), 500

@meal_planner_blueprint.route('/api/calculate-nutrition', methods=['POST'])
def calculate_nutrition():
    """Calculate nutritional requirements based on user data"""
    try:
        # Get user input data
        data = request.json
        age = int(data.get('age', 30))
        weight = float(data.get('weight', 70))  # in kg
        height = float(data.get('height', 170))  # in cm
        gender = data.get('gender', 'male').lower()
        activity = data.get('activity', 'moderate').lower()
        goal = data.get('goal', 'maintain').lower()
        meals = int(data.get('meals', 3))
        
        # Calculate BMR (Basal Metabolic Rate)
        bmr = calculate_bmr(weight, height, age, gender)
        
        # Calculate TDEE (Total Daily Energy Expenditure)
        tdee = calculate_tdee(bmr, activity)
        
        # Adjust calories based on goal
        adjusted_calories = adjust_goal(tdee, goal)
        
        # Calculate macros
        protein_target, carb_target, fat_target = calculate_macros(adjusted_calories, goal)
        
        # Calculate per meal targets
        cal_per_meal = adjusted_calories / meals
        protein_per_meal = protein_target / meals
        carb_per_meal = carb_target / meals
        fat_per_meal = fat_target / meals
        
        # Create response
        response = {
            'daily': {
                'calories': round(adjusted_calories),
                'protein': round(protein_target, 1),
                'carbs': round(carb_target, 1),
                'fat': round(fat_target, 1)
            },
            'per_meal': {
                'meals': meals,
                'calories': round(cal_per_meal),
                'protein': round(protein_per_meal, 1),
                'carbs': round(carb_per_meal, 1),
                'fat': round(fat_per_meal, 1)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error calculating nutrition: {e}")
        return jsonify({"error": str(e)}), 400

@meal_planner_blueprint.route('/api/get-food-suggestions', methods=['POST'])
def get_food_suggestions():
    """Get food suggestions based on user's nutritional needs"""
    try:
        # Get target nutrition values from request
        data = request.json
        target_cal = float(data.get('calories', 500))
        target_protein = float(data.get('protein', 30))
        target_carbs = float(data.get('carbs', 50))
        target_fat = float(data.get('fat', 15))
        excluded_foods = data.get('excluded', [])
        food_types = data.get('food_types', [])
        meal_number = int(data.get('meal_number', 1))
        limit = int(data.get('limit', 15))
        
        # Log request details
        logging.info(f"Food suggestion request: meal {meal_number}, calories {target_cal}, food types {food_types}")
        
        # Get food suggestions
        suggestions = suggest_food_options(
            food_df, 
            target_cal, 
            target_protein, 
            target_carbs, 
            target_fat, 
            limit=limit, 
            excluded_foods=excluded_foods,
            food_types=food_types
        )
        
        # Convert to list of dicts for JSON response
        result = []
        for i, row in suggestions.iterrows():
            result.append({
                'id': int(i),
                'food': row['food'],
                'calories': float(row['calories']),
                'protein': float(row['protein']),
                'carbs': float(row['carbs']),
                'fat': float(row['fat']),
                'category': row['category']
            })
        
        # Log the number of suggestions
        logging.info(f"Returning {len(result)} food suggestions for meal {meal_number}")
        
        return jsonify({"meal_number": meal_number, "suggestions": result})
        
    except Exception as e:
        logging.error(f"Error getting food suggestions: {e}")
        return jsonify({"error": str(e)}), 400

@meal_planner_blueprint.route('/api/get-all-foods', methods=['GET'])
def get_all_foods():
    """Returns the full list of food items"""
    try:
        all_foods = food_df.to_dict(orient='records')
        return jsonify({"foods": all_foods})
    except Exception as e:
        logging.error(f"Error getting all foods: {e}")
        return jsonify({"error": str(e)}), 500

@meal_planner_blueprint.route('/api/generate-custom-meal-plan', methods=['POST'])
def generate_custom_meal_plan():
    """Generates a meal plan based on user's favorite foods and nutrition"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request"}), 400

        nutrition_reqs = data.get('nutrition_reqs')
        if not nutrition_reqs:
            return jsonify({"error": "Nutrition requirements not provided"}), 400

        favorite_foods = data.get('favorite_foods', [])
        
        per_meal_targets = nutrition_reqs.get('per_meal', {})
        target_cal = per_meal_targets.get('calories', 500)
        target_protein = per_meal_targets.get('protein', 30)
        target_carbs = per_meal_targets.get('carbs', 50)
        target_fat = per_meal_targets.get('fat', 15)
        num_meals = per_meal_targets.get('meals', 3)

        # Always use predefined meal suggestions for 3 meals (user's specific meals)
        if num_meals == 3:
            meal_plan = generate_three_meal_suggestions()
        else:
            # Use the original logic for other meal counts
            meal_plan = {}
            used_foods = set()

            for i in range(1, num_meals + 1):
                suggestions = suggest_food_options(
                    food_df,
                    target_cal,
                    target_protein,
                    target_carbs,
                    target_fat,
                    limit=3,
                    favorite_foods=favorite_foods,
                    excluded_foods=list(used_foods)
                )
                
                for food_item in suggestions['food']:
                    used_foods.add(food_item)

                meal_plan[f'Meal {i}'] = suggestions.to_dict(orient='records')
            
        return jsonify({"meal_plan": meal_plan})
        
    except Exception as e:
        logging.error(f"Error generating custom meal plan: {e}")
        return jsonify({"error": str(e)}), 400

def generate_three_meal_suggestions():
    """Generate structured meal suggestions for 3 meals - ALWAYS returns user's specific meals"""
    
    # User's specific meal structures (exactly as provided)
    meal_structures = {
        "Meal 1: Chicken & Oats Power Bowl": [
            {"food": "Chicken breast (150g cooked)", "calories": 248, "protein": 46, "carbs": 0, "fat": 5, "category": "Protein"},
            {"food": "Oats (50g dry)", "calories": 194, "protein": 6.8, "carbs": 33, "fat": 3.5, "category": "Grains"},
            {"food": "Apple (1 medium, ~180g)", "calories": 95, "protein": 0.5, "carbs": 25, "fat": 0.3, "category": "Fruits"},
            {"food": "Almonds (15g, ~12 nuts)", "calories": 87, "protein": 3.2, "carbs": 3, "fat": 7.5, "category": "Nuts"},
            {"food": "Broccoli (100g steamed)", "calories": 34, "protein": 3, "carbs": 7, "fat": 0.4, "category": "Vegetables"}
        ],
        "Meal 2: Greek Yogurt Egg Mix": [
            {"food": "Greek yogurt (200g, plain low-fat)", "calories": 130, "protein": 20, "carbs": 7, "fat": 3, "category": "Dairy"},
            {"food": "Eggs (2 whole)", "calories": 140, "protein": 12, "carbs": 1, "fat": 10, "category": "Protein"},
            {"food": "Oats (30g)", "calories": 116, "protein": 4, "carbs": 20, "fat": 2, "category": "Grains"},
            {"food": "Orange (1 medium)", "calories": 62, "protein": 1.2, "carbs": 15, "fat": 0.2, "category": "Fruits"},
            {"food": "Carrots (100g raw)", "calories": 41, "protein": 1, "carbs": 10, "fat": 0.2, "category": "Vegetables"},
            {"food": "Almonds (10g)", "calories": 58, "protein": 2, "carbs": 2, "fat": 5, "category": "Nuts"},
            {"food": "Cucumbers (100g raw)", "calories": 16, "protein": 0.7, "carbs": 4, "fat": 0.1, "category": "Vegetables"}
        ],
        "Meal 3: Beef & Veggie Cottage Combo": [
            {"food": "Beef (100g lean cooked)", "calories": 250, "protein": 30, "carbs": 0, "fat": 17, "category": "Protein"},
            {"food": "Cottage cheese (150g low-fat)", "calories": 120, "protein": 18, "carbs": 4, "fat": 4, "category": "Dairy"},
            {"food": "Broccoli (100g)", "calories": 34, "protein": 3, "carbs": 7, "fat": 0.4, "category": "Vegetables"},
            {"food": "Carrots (100g)", "calories": 41, "protein": 1, "carbs": 10, "fat": 0.2, "category": "Vegetables"},
            {"food": "Mushrooms (100g sautéed)", "calories": 28, "protein": 3, "carbs": 4, "fat": 1, "category": "Vegetables"},
            {"food": "Oats (30g)", "calories": 116, "protein": 4, "carbs": 20, "fat": 2, "category": "Grains"},
            {"food": "Apple (1 small, 130g)", "calories": 68, "protein": 0.3, "carbs": 18, "fat": 0.2, "category": "Fruits"}
        ]
    }
    
    # Add nutritional totals for each meal
    meal_plan = {}
    for meal_name, foods in meal_structures.items():
        # Calculate totals
        total_calories = sum(food["calories"] for food in foods)
        total_protein = sum(food["protein"] for food in foods)
        total_carbs = sum(food["carbs"] for food in foods)
        total_fat = sum(food["fat"] for food in foods)
        
        # Add foods plus totals
        meal_data = foods.copy()
        meal_data.append({
            "food": "🔹 TOTAL",
            "calories": total_calories,
            "protein": total_protein,
            "carbs": total_carbs,
            "fat": total_fat,
            "category": "Total"
        })
        
        meal_plan[meal_name] = meal_data
    
    return meal_plan

# Utility functions
def calculate_bmr(weight, height, age, gender='male'):
    """Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor Equation"""
    if gender == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

def calculate_tdee(bmr, activity_level):
    """Calculate Total Daily Energy Expenditure (TDEE) based on activity level"""
    factors = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very active': 1.9
    }
    return bmr * factors.get(activity_level, 1.2)

def adjust_goal(tdee, goal):
    """Adjust TDEE based on the goal (bulk, cut, maintain)"""
    if goal == 'cut':
        return tdee - 500  # Subtract 500 calories to lose weight
    elif goal == 'bulk':
        return tdee + 500  # Add 500 calories to gain weight
    return tdee  # Maintain weight

def calculate_macros(calories, goal):
    """Calculate macros based on the goal"""
    if goal == 'cut':
        p, c, f = 0.4, 0.3, 0.3  # Higher protein for cutting
    elif goal == 'bulk':
        p, c, f = 0.3, 0.5, 0.2  # Higher carbs for bulking
    else:
        p, c, f = 0.3, 0.4, 0.3  # Balanced for maintenance
        
    protein = (calories * p) / 4  # Protein has 4 calories per gram
    carbs = (calories * c) / 4    # Carbs have 4 calories per gram
    fat = (calories * f) / 9      # Fat has 9 calories per gram
    
    return protein, carbs, fat

def score_food(row, target_protein, target_carbs, target_fat, target_cal):
    """Score how well a food matches the target macros"""
    protein_score = 1 - min(abs(row['protein'] - target_protein) / max(target_protein, 1), 1)
    carbs_score = 1 - min(abs(row['carbs'] - target_carbs) / max(target_carbs, 1), 1)
    fat_score = 1 - min(abs(row['fat'] - target_fat) / max(target_fat, 1), 1)
    calorie_score = 1 - min(abs(row['calories'] - target_cal) / max(target_cal, 1), 1)
    return (protein_score + carbs_score + fat_score + calorie_score) / 4

def suggest_food_options(food_df, target_cal, target_protein, target_carbs, target_fat, limit=15, excluded_foods=None, food_types=None, favorite_foods=None):
    """
    Suggests food options prioritizing favorite foods while meeting nutritional targets.
    """
    if favorite_foods is None:
        favorite_foods = []
    
    options = food_df.copy()
    if excluded_foods:
        options = options[~options['food'].isin(excluded_foods)]
    
    # Separate favorite foods from the main list
    favorite_options = options[options['food'].isin(favorite_foods)]
    other_options = options[~options['food'].isin(favorite_foods)]
    
    # Score all options
    favorite_options['score'] = favorite_options.apply(lambda row: score_food(row, target_protein, target_carbs, target_fat, target_cal), axis=1)
    other_options['score'] = other_options.apply(lambda row: score_food(row, target_protein, target_carbs, target_fat, target_cal), axis=1)
    
    # Sort both lists by score
    favorite_options = favorite_options.sort_values('score', ascending=False)
    other_options = other_options.sort_values('score', ascending=False)
    
    # Combine lists, with favorites first
    final_selection = pd.concat([favorite_options, other_options]).reset_index(drop=True)
    
    # Ensure variety by taking a sample from the top candidates
    top_n = min(len(final_selection), limit * 3)
    top_options = final_selection.head(top_n)
    
    # Shuffle and sample 'limit' foods
    diverse_selection = top_options.sample(frac=1).reset_index(drop=True).head(limit)
    
    return diverse_selection
