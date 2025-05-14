# Stock Forecasting AI Service

A FastAPI-based application that predicts stock shortages for inventory management using Prophet forecasting.

## Project Overview

This service provides APIs to:

- Predict when products will be out of stock based on historical sales data
- Generate recommended replenishment plans
- Automate inventory monitoring via scheduled tasks

This application uses the new uv package manager.

## Installing Dependencies

To install dependencies, run:

```bash
uv sync
```

## Running the Service (Development)

To run the service in development mode:

```bash
uv run fastapi dev --port 8080
```

## Running the Service (Production)

For production environments, use:

```bash
uv run fastapi run --port 8080
```

## API Endpoints

### Predict Shortage for a Single Product

```bash
POST /predict-shortage
```

Request body:

```json
{
  "product_id": 123,
  "sales": [
    {"date": "2023-01-01", "quantity": 5},
    {"date": "2023-01-02", "quantity": 3}
  ],
  "stock": 10
}
```

### Predict Shortages for Multiple Products

```bash
POST /predict-multiple-shortages
```

## Automated Monitoring

The service includes a scheduler that:

- Runs daily at midnight (Bogotá timezone)
- Fetches confirmed sales data from an external API
- Predicts potential shortages
- Sends notifications for at-risk products

## Requirements

- Python >= 3.13
- Dependencies:
  - FastAPI
  - Pandas
  - Prophet
  - APScheduler
  - Requests
  - Uvicorn
  - PyTZ
  - Pydantic

## Development

This project uses `uv` for dependency management and `FastAPI` for the web framework.
