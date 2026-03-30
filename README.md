# Sales Order Value Forecasting

A full-stack demo application that predicts sales order value (EUR) using a machine learning model, a LangChain AI chat agent, and a Vite frontend — all orchestrated with Docker Compose.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Frontend   │────▶│   Chat Agent     │────▶│  Forecast API │
│  (Vite)     │     │  (LangChain +    │     │  (FastAPI +   │
│  :3000      │     │   Claude)  :8001 │     │   sklearn)    │
└─────────────┘     └──────────────────┘     │   :8000       │
                                             └───────────────┘
```

- **Frontend** — Chat UI served on port 3000
- **Chat Agent** — LangChain agent backed by Claude (Sonnet) that interprets natural-language queries and calls the forecast API; served on port 8001
- **Forecast API** — FastAPI service wrapping a scikit-learn linear regression model; served on port 8000

## Features

- Predict order value by **date**, **country**, **product category**, and **sales channel (device type)**
- Single and batch prediction endpoints
- Conversational AI agent that extracts forecast parameters from natural language
- Per-session conversation memory (in-memory checkpointing)

## Prediction Inputs

| Parameter | Options |
|-----------|---------|
| Country | Sweden, Finland, Portugal, Spain, UK, France, Netherlands, Belgium, Bulgaria, Luxembourg, Italy, Ireland, Germany, Denmark, Austria |
| Category | Books, Games, Clothing, Beauty, Accessories, Appliances, Smartphones, Outdoors, Electronics, Other |
| Device | Mobile, PC, Tablet |
| Date | Any date in `YYYY-MM-DD` format |

> The model is a linear regression trained on synthetic data for demonstration purposes.

## Quick Start (Docker)

```bash
cp .env.example .env   # add your ANTHROPIC_API_KEY
docker compose up --build
```

Then open `http://localhost:3000`.

## Local Development

**Prerequisites:** Python 3.14+, Poetry, Node.js

```bash
# Install Python dependencies
poetry install

# Train the model (generates model/*.pkl files)
poetry run python model/train_model.py

# Start the forecast API
poetry run uvicorn api.main:app --reload --port 8000

# Start the chat agent (in a new terminal)
API_URL=http://localhost:8000 poetry run python -m chat_agent.agent

# Start the frontend (in a new terminal)
cd frontend && npm install && npm run dev
```

## API Reference

Base URL: `http://localhost:8000`

### `POST /predict`
Single prediction.
```json
{
  "date": "2025-06-15",
  "country": "Sweden",
  "category": "Books",
  "device_type": "Mobile"
}
```
All fields are optional — omit any you don't want to filter on.

### `POST /predict/batch`
Batch predictions. Body: `{ "requests": [ ...array of predict objects... ] }`
Returns predictions plus `total_forecasted_value` and `average_forecasted_value`.

### `GET /options`
Returns valid values for `country`, `category`, and `device_type`.
Filter by type: `/options?type=country`

### `GET /model-info`
Returns model type, feature list, and label mappings.

### `GET /health`
Returns `{ "status": "healthy" }` when the model is loaded.

---

Chat Agent base URL: `http://localhost:8001`

### `POST /chat`
```json
{
  "message": "What are the expected sales in Germany for electronics next month?",
  "thread_id": "user-123"
}
```
Use `thread_id` to maintain conversation context per session.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Required for the chat agent (Claude) |
| `API_URL` | Forecast API URL used by the agent (default: `http://api:8000` in Docker) |
