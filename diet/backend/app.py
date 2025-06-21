from flask import Flask, render_template, redirect
from routes.predict import predict_blueprint
from routes.meal_planner import meal_planner_blueprint

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')

# Register Blueprints
app.register_blueprint(predict_blueprint)
app.register_blueprint(meal_planner_blueprint)

@app.route('/')
def home():
    """Render the home page"""
    return render_template('home.html')

@app.route('/obesity-prediction')
def obesity_prediction():
    """Render the obesity prediction page"""
    return render_template('index.html')

@app.route('/meal-plan-result')
def meal_plan_result():
    """Render the meal plan result page"""
    return render_template('meal_plan_result.html')

if __name__ == '__main__':
    app.run(debug=True)