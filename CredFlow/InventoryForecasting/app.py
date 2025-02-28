import gradio as gr
import pandas as pd
import datetime
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt

# Load the trained XGBoost model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Available products
products = ["Milk", "Bread", "Eggs", "Rice", "Sugar"]

# Mapping for days of the week
day_mapping = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
    "Friday": 4, "Saturday": 5, "Sunday": 6
}

# Inventory Prediction Function
def predict_inventory(product, day, month, year, past_day_sales, past_week_sales, avg_week_sales):
    # Convert input to datetime
    date = datetime.datetime(year, month, list(day_mapping.keys()).index(day) + 1)

    # Compute additional required features
    hour = 12  # Assuming midday sales
    quarter = (month - 1) // 3 + 1
    dayofyear = date.timetuple().tm_yday
    dayofmonth = date.day
    weekofyear = date.isocalendar()[1]
    rolling_std_7 = past_week_sales * 0.1  # Placeholder value
    lag_2 = (past_day_sales + past_week_sales) / 2  # Placeholder value

    # Create DataFrame with correct feature columns
    input_data = pd.DataFrame([[
        hour, day_mapping[day], quarter, month, year, dayofyear, dayofmonth, weekofyear,
        past_day_sales, lag_2, past_week_sales, avg_week_sales, rolling_std_7
    ]], columns=[
        "hour", "dayofweek", "quarter", "month", "year", "dayofyear", "dayofmonth", "weekofyear",
        "lag_1", "lag_2", "lag_7", "rolling_mean_7", "rolling_std_7"
    ])

    # Predict inventory
    prediction = model.predict(input_data)
    predicted_inventory = round(float(prediction[0]))

    # Generate a single graph
    plt.figure(figsize=(8, 4))
    
    # Past sales data (last 7 days)
    past_dates = pd.date_range(start=date - pd.Timedelta(days=6), periods=7, freq='D')
    past_sales = [past_week_sales / 7] * 6 + [past_day_sales]  # Simulating past sales trend
    plt.plot(past_dates, past_sales, marker='o', linestyle='-', color='blue', label="Past Sales")

    # Future forecast data (next 7 days)
    future_dates = pd.date_range(start=date, periods=7, freq='D')
    forecasted_sales = [predicted_inventory] * 7  # Simulating forecasted inventory
    plt.plot(future_dates, forecasted_sales, marker='o', linestyle='--', color='red', label="Forecasted Inventory")

    plt.xlabel("Date")
    plt.ylabel("Sales / Inventory")
    plt.title(f"Sales & Inventory Forecast for {product}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()

    graph_path = "sales_forecast_graph.png"
    plt.savefig(graph_path)
    plt.close()

    return predicted_inventory, graph_path

# Gradio UI Layout
with gr.Blocks() as app:
    gr.Markdown("# 🛒 Inventory Forecasting App for Shopkeepers")

    with gr.Row():
        product = gr.Dropdown(choices=products, label="Select Product")
    
    with gr.Row():
        day = gr.Dropdown(choices=list(day_mapping.keys()), label="Day of the Week")
        month = gr.Slider(1, 12, step=1, label="Month")
        year = gr.Slider(2020, 2030, step=1, label="Year")

    with gr.Row():
        past_day_sales = gr.Number(label="Past Day Sales")
        past_week_sales = gr.Number(label="Past Week Sales")
        avg_week_sales = gr.Number(label="7-day Average Sales")

    predict_button = gr.Button("Predict Inventory")
    output_text = gr.Textbox(label="Predicted Required Inventory")
    
    with gr.Row():
        graph_output = gr.Image(label="Sales & Forecast Graph")

    predict_button.click(
        predict_inventory, 
        inputs=[product, day, month, year, past_day_sales, past_week_sales, avg_week_sales], 
        outputs=[output_text, graph_output]
    )

# Launch the Gradio app
app.launch()