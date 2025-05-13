"""Stock shortage prediction API for inventory management and forecasting."""

import math
from datetime import datetime, timedelta
from typing import List
import pytz
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from prophet import Prophet
from scheduler import start_scheduler

app = FastAPI()

@app.lifespan("startup")
def on_startup():
    """Start the scheduler on application startup."""
    start_scheduler(app)

@app.lifespan("shutdown")
def on_shutdown():
    """Shutdown the scheduler on application shutdown."""
    scheduler = getattr(app.state, "scheduler", None)
    if scheduler and scheduler.running:
        scheduler.shutdown()

# Modelos Pydantic
class SaleItem(BaseModel):
    date: str
    quantity: float

class ProductRequest(BaseModel):
    product_id: int
    sales: List[SaleItem]
    stock: int

@app.post("/predict-shortage")
def predict_shortage(request: ProductRequest):
    """Predict stock shortage for a single product."""
    colombia_tz = pytz.timezone('America/Bogota')
    current_date = datetime.now(colombia_tz).date()

    # Ventas históricas
    df = pd.DataFrame([{"ds": s.date, "y": s.quantity} for s in request.sales])
    df['ds'] = pd.to_datetime(df['ds']).dt.date

    if not df.empty:
        full_dates = pd.DataFrame({'ds': pd.date_range(df['ds'].min(), current_date)})
        full_dates['ds'] = full_dates['ds'].dt.date
        df_full = full_dates.merge(df, on='ds', how='left').fillna(0)
    else:
        df_full = pd.DataFrame({'ds': [current_date], 'y': [0]})

    # Ajuste Prophet
    model = Prophet()
    model.fit(pd.DataFrame({
        'ds': pd.to_datetime(df_full['ds']),
        'y': df_full['y']
    }))

    # Pronóstico a 14 días
    future = model.make_future_dataframe(periods=14)
    future = future[future['ds'].dt.date >= current_date]
    forecast = model.predict(future)

    forecast_future = forecast[['ds', 'yhat']].tail(14).copy()
    forecast_future['yhat'] = forecast_future['yhat'].apply(lambda x: max(0, round(x)))
    forecast_future.rename(columns={'ds': 'date', 'yhat': 'quantity'}, inplace=True)
    forecast_future['date'] = pd.to_datetime(forecast_future['date'])

    # Detectar escasez
    stock = request.stock
    shortage_date = None
    for _, row in forecast_future.iterrows():
        stock -= row['quantity']
        if stock <= 0:
            shortage_date = row['date'].date()
            break

    # Salida inicial
    recommended_units = 0
    message = "Sin riesgo de escasez en próximos 14 días"
    replenishment_plan = None
    days_until = None
    order_date = None

    # Stock cero
    if request.stock == 0:
        shortage_date = current_date
        if len(request.sales) == 1:
            q = request.sales[0].quantity
            recommended_units = q
            message = "⚠️ El producto ya está en escasez (stock = 0)"
            replenishment_plan = f"Comprar {q} unidades hoy para cubrir la última venta."
        elif len(request.sales) > 1:
            s7 = forecast_future.head(7)['quantity'].sum()
            recommended_units = math.ceil(s7 * 1.1)
            message = "⚠️ El producto ya está en escasez (stock = 0)"
            replenishment_plan = f"Comprar {recommended_units} unidades hoy para cubrir demanda 7 días."
        else:
            recommended_units = 10
            message = "⚠️ Sin stock ni historial de ventas."
            replenishment_plan = "Agregar al menos 10 unidades para disponibilidad."

    # Escasez futura
    elif shortage_date:
        days_until = (shortage_date - current_date).days
        message = f"Escaseará en {days_until} días ({shortage_date.isoformat() if shortage_date else 'fecha desconocida'})."        
        ts = datetime.combine(shortage_date, datetime.min.time())
        s_after = forecast_future[
            (forecast_future['date'] > ts) &
            (forecast_future['date'] <= ts + timedelta(days=7))
        ]['quantity'].sum()
        recommended_units = math.ceil(s_after * 1.1)
        order_date = shortage_date - timedelta(days=3)
        if order_date < current_date:
            order_date = current_date
        replenishment_plan = (
            f"Ordenar {recommended_units} unidades el {order_date.isoformat() if order_date else 'fecha desconocida'} "
            "para 7 días tras escasez."
        )

    # Ajuste final cuando recommended_units == 0
    if recommended_units == 0:
        ventas_prev = df_full[
            (df_full['ds'] >= current_date - timedelta(days=7)) &
            (df_full['ds'] < current_date)
        ]['y'].sum()
        recommended_units = max(1, math.ceil(ventas_prev * 1.1))
        message = f"Escaseará en {days_until} días ({shortage_date.isoformat() if shortage_date else 'fecha desconocida'})."      
        replenishment_plan = (
            f"Ordenar {recommended_units} unidades el {order_date.isoformat() if order_date else 'fecha desconocida'} "
            "para 7 días tras escasez."
        )

    # Serializar forecast
    forecast_list = [
        {'date': d.date().isoformat(), 'quantity': int(q)}
        for d, q in zip(forecast_future['date'], forecast_future['quantity'])
    ]

    return {
        "product_id": request.product_id,
        "recommended_units": recommended_units,
        "shortage_date": shortage_date.isoformat() if shortage_date else None,
        "message": message,
        "forecast": forecast_list,
        "replenishment_plan": replenishment_plan
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
