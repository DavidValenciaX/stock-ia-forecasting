"""Stock shortage prediction API for inventory management and forecasting."""

import math
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
import pytz
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from prophet import Prophet
from scheduler import start_scheduler

async def lifespan(app):
    """Handle application startup and shutdown events."""
    # Start the scheduler on startup
    start_scheduler(app)
    
    yield  # Wait for shutdown
    
    # Shutdown the scheduler on shutdown
    scheduler = getattr(app.state, "scheduler", None)
    if scheduler and scheduler.running:
        scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

# Modelos Pydantic
class SaleItem(BaseModel):
    date: str
    quantity: float

class ProductRequest(BaseModel):
    product_id: int
    sales: List[SaleItem]
    stock: int

# --- Helper Functions for predict_shortage ---

def _prepare_sales_data(sales: List[SaleItem], current_date: date) -> pd.DataFrame:
    """Prepares the historical sales data DataFrame."""
    df = pd.DataFrame([{"ds": s.date, "y": s.quantity} for s in sales])
    df['ds'] = pd.to_datetime(df['ds']).dt.date

    if not df.empty:
        # Ensure all dates from first sale to current date are present
        full_dates = pd.DataFrame({'ds': pd.date_range(df['ds'].min(), current_date)})
        full_dates['ds'] = full_dates['ds'].dt.date
        # Merge and fill missing daily sales with 0
        df_full = full_dates.merge(df, on='ds', how='left').fillna(0)
    else:
        # Handle case with no sales data
        df_full = pd.DataFrame({'ds': [current_date], 'y': [0]})
    return df_full

def _get_forecast(df_full: pd.DataFrame, current_date: date, periods: int = 14) -> pd.DataFrame:
    """Trains Prophet model and returns future forecast."""
    model = Prophet()
    model.fit(pd.DataFrame({
        'ds': pd.to_datetime(df_full['ds']),
        'y': df_full['y']
    }))

    # Create future dataframe starting from current date
    future = model.make_future_dataframe(periods=periods)
    future = future[future['ds'].dt.date >= current_date]
    forecast = model.predict(future)

    # Process forecast: select relevant columns, round, ensure non-negative
    forecast_future = forecast[['ds', 'yhat']].tail(periods).copy()
    forecast_future['yhat'] = forecast_future['yhat'].apply(lambda x: max(0, round(x)))
    forecast_future.rename(columns={'ds': 'date', 'yhat': 'quantity'}, inplace=True)
    forecast_future['date'] = pd.to_datetime(forecast_future['date']) # Keep as datetime for potential time operations
    return forecast_future

def _find_shortage_date(stock: int, forecast_future: pd.DataFrame) -> Optional[date]:
    """Finds the date when stock is predicted to run out."""
    current_stock = stock
    for _, row in forecast_future.iterrows():
        current_stock -= row['quantity']
        if current_stock <= 0:
            return row['date'].date()
    return None

def _handle_zero_stock(request: ProductRequest, forecast_future: pd.DataFrame, current_date: date) -> Dict[str, Any]:
    """Handles the scenario where initial stock is zero."""
    shortage_date = current_date
    recommended_units = 0
    replenishment_plan = ""

    if len(request.sales) == 1:
        q = request.sales[0].quantity
        recommended_units = q
        replenishment_plan = f"Comprar {q} unidades hoy para cubrir la última venta."
    elif len(request.sales) > 1:
        s7 = forecast_future.head(7)['quantity'].sum()
        recommended_units = math.ceil(s7 * 1.1)
        replenishment_plan = f"Comprar {recommended_units} unidades hoy para cubrir demanda 7 días."
    else: # No stock, no sales history
        recommended_units = 10
        replenishment_plan = "Agregar al menos 10 unidades para disponibilidad."

    message = "⚠️ El producto ya está en escasez (stock = 0)"
    if not request.sales:
        message = "⚠️ Sin stock ni historial de ventas."


    return {
        "recommended_units": recommended_units,
        "message": message,
        "replenishment_plan": replenishment_plan,
        "shortage_date": shortage_date,
        "days_until": 0,
        "order_date": current_date
    }

def _handle_future_shortage(shortage_date: date, forecast_future: pd.DataFrame, current_date: date, df_full: pd.DataFrame) -> Dict[str, Any]:
    """Handles the scenario where a future shortage is predicted."""
    days_until = (shortage_date - current_date).days
    message = f"Escaseará en {days_until} días ({shortage_date.isoformat()})."

    # Calculate demand for 7 days after shortage date
    ts = datetime.combine(shortage_date, datetime.min.time())
    s_after = forecast_future[
        (forecast_future['date'] > ts) &
        (forecast_future['date'] <= ts + timedelta(days=7))
    ]['quantity'].sum()
    recommended_units = math.ceil(s_after * 1.1)

    # If forecast for next 7 days after shortage is zero, use past 7 days actuals
    if recommended_units == 0:
        ventas_prev = df_full[
            (df_full['ds'] >= current_date - timedelta(days=7)) &
            (df_full['ds'] < current_date)
        ]['y'].sum()
        recommended_units = max(1, math.ceil(ventas_prev * 1.1)) # Ensure at least 1 unit recommended

    # Determine order date (3 days before shortage, but not before today)
    order_date = shortage_date - timedelta(days=3)
    if order_date < current_date:
        order_date = current_date

    replenishment_plan = (
        f"Ordenar {recommended_units} unidades el {order_date.isoformat()} "
        "para 7 días tras escasez."
    )

    return {
        "recommended_units": recommended_units,
        "message": message,
        "replenishment_plan": replenishment_plan,
        "days_until": days_until,
        "order_date": order_date
    }

def _calculate_recommendation_no_shortage(df_full: pd.DataFrame, current_date: date) -> int:
    """Calculates recommended units based on recent sales when no shortage is predicted."""
    ventas_prev = df_full[
        (df_full['ds'] >= current_date - timedelta(days=7)) &
        (df_full['ds'] < current_date)
    ]['y'].sum()
    # Recommend 110% of last 7 days sales, minimum 1
    return max(1, math.ceil(ventas_prev * 1.1))

# --- API Endpoints ---

@app.post("/predict-shortage")
def predict_shortage(request: ProductRequest):
    """Predict stock shortage for a single product."""
    colombia_tz = pytz.timezone('America/Bogota')
    current_date = datetime.now(colombia_tz).date()

    # 1. Prepare Data & Get Forecast
    df_full = _prepare_sales_data(request.sales, current_date)
    forecast_future = _get_forecast(df_full, current_date)

    # 2. Initialize response variables
    response_data: Dict[str, Any] = {
        "recommended_units": 0,
        "shortage_date": None,
        "message": "Sin riesgo de escasez en próximos 14 días",
        "replenishment_plan": None,
        "days_until": None, # Added for consistency
        "order_date": None   # Added for consistency
    }

    # 3. Handle Scenarios
    if request.stock == 0:
        response_data.update(_handle_zero_stock(request, forecast_future, current_date))
    else:
        calculated_shortage_date = _find_shortage_date(request.stock, forecast_future)
        if calculated_shortage_date:
            response_data.update(
                _handle_future_shortage(calculated_shortage_date, forecast_future, current_date, df_full)
            )
            response_data["shortage_date"] = calculated_shortage_date # Ensure correct date is set
        else:
            # No future shortage
             response_data["recommended_units"] = _calculate_recommendation_no_shortage(df_full, current_date)
             response_data["replenishment_plan"] = (
                 f"Se recomienda tener al menos {response_data['recommended_units']} unidades disponibles "
                 "basado en ventas recientes."
             )
             # message remains default "Sin riesgo..."

    # 4. Format forecast for output
    forecast_list = [
        {'date': d.date().isoformat(), 'quantity': int(q)}
        for d, q in zip(forecast_future['date'], forecast_future['quantity'])
    ]

    # 5. Construct final response
    return {
        "product_id": request.product_id,
        "recommended_units": response_data["recommended_units"],
        "shortage_date": response_data["shortage_date"].isoformat() if response_data["shortage_date"] else None,
        "message": response_data["message"],
        "forecast": forecast_list,
        "replenishment_plan": response_data["replenishment_plan"]
        # Include days_until and order_date if needed in response, currently internal use
    }

# Múltiples productos
class MultipleProductRequest(BaseModel):
    status: str
    message: str
    data: List[ProductRequest]

@app.post("/predict-multiple-shortages")
async def predict_multiple_shortages(request: MultipleProductRequest):
    """Predict stock shortages for multiple products."""
    return {
        "status": "success",
        "message": "Predicciones generadas",
        "data": [predict_shortage(p) for p in request.data]
    }
