import joblib
import pandas as pd

model = joblib.load("artifacts/best_model.pkl")

sample = pd.DataFrame([{
    "Product_Weight": 100,
    "Product_Sugar_Content": 1,
    "Product_Allocated_Area": 0.1,
    "Product_Type": 3,
    "Product_MRP": 150,
    "Store_Id": 1,
    "Store_Establishment_Year": 2010,
    "Store_Size": 2,
    "Store_Location_City_Type": 1,
    "Store_Type": 0
}])

prediction = model.predict(sample)
print("Predicted Sales:", prediction[0])
