"""Scheduler for product inventory monitoring and shortage forecasting."""
import requests
import json
from apscheduler.schedulers.background import BackgroundScheduler

external_endpoint = "http://173.212.224.226:3000/api/notifications"

def scheduled_task():
    print("⏰ Ejecutando tarea programada...")

    response = requests.get(
        "http://173.212.224.226:3000/inventory-transactions/confirmed-sales"
    )
    if response.status_code != 200:
        print("❌ Error al obtener datos de ventas")
        return

    sales_data = response.json().get("data", [])
    print(f"📦 Productos recibidos: {len(sales_data)}")

    for product in sales_data:
        try:
            predict_response = requests.post(
                "http://localhost:8080/predict-shortage",
                json={
                    "product_id": product["product_id"],
                    "sales": product["sales"],
                    "stock": product["stock"]
                }
            )
            if predict_response.status_code != 200:
                print(f"❌ Error para producto {product['product_id']}")
                continue

            result = predict_response.json()
            if result.get("shortage_date"):
                print(f"⚠️ Producto {product['product_id']} en riesgo de escasez.")

                # Reconstruir forecast limpio
                clean_forecast = [
                    {"date": entry["date"], "quantity": entry["quantity"]}
                    for entry in result["forecast"]
                    if entry.get("date") is not None and entry.get("quantity") is not None
                ]

                filtered_result = {
                    "product_id": result["product_id"],
                    "message": result["message"].strip(),
                    "forecast": clean_forecast,
                    "shortage_date": result["shortage_date"],
                    "replenishment_plan": result["replenishment_plan"]
                }

                payload = json.dumps(filtered_result, ensure_ascii=False)
                print("📤 JSON enviado al endpoint externo:")
                print(payload)

                external_response = requests.post(
                    external_endpoint,
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                if external_response.status_code == 201:
                    print(f"✅ Producto {product['product_id']} enviado correctamente.")
                else:
                    print(f"❌ Error al enviar producto {product['product_id']}: {external_response.status_code}")
            else:
                print(f"✅ Producto {product['product_id']} sin riesgo de escasez.")

        except Exception as e:
            print(f"❌ Error procesando producto {product['product_id']}: {e}")

def start_scheduler(app):
    scheduler = BackgroundScheduler(timezone='America/Bogota')
    scheduler.add_job(scheduled_task, 'cron', hour=0, minute=0)
    scheduler.start()
    app.state.scheduler = scheduler
