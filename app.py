import gradio as gr
import pandas as pd
import joblib
import numpy as np

model = joblib.load("model/model.joblib")
feature_columns = joblib.load("model/feature_columns.joblib")

def predict_price(overall_qual, gr_liv_area, total_bsmt_sf, garage_cars, year_built):
    
    # Create a row of zeros with all feature columns
    input_df = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)
    
    # Fill in the values the user provided
    input_df["OverallQual"] = overall_qual
    input_df["GrLivArea"] = gr_liv_area
    input_df["TotalBsmtSF"] = total_bsmt_sf
    input_df["GarageCars"] = garage_cars
    input_df["YearBuilt"] = year_built
    
    prediction = model.predict(input_df)[0]
    return f"Estimated Price: ${prediction:,.0f}"

demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(1, 10, value=5, label="Overall Quality (1-10)"),
        gr.Number(value=1500, label="Above Ground Living Area (sq ft)"),
        gr.Number(value=800, label="Total Basement Area (sq ft)"),
        gr.Slider(0, 4, value=2, step=1, label="Garage Capacity (cars)"),
        gr.Number(value=2000, label="Year Built"),
    ],
    outputs=gr.Text(label="Predicted Price"),
    title="House Price Predictor",
    description="Enter house details to get an estimated sale price."
)

demo.launch()