# freshcart-pipelines

**Data Engineering Assignment: From Git to Gold**

A complete real-world data engineering project demonstrating:

- **Part A** – Git version control & GitHub Actions CI/CD
- **Part B** – FreshCart ELT pipeline on Snowflake (Bronze → Silver → Gold)
- **Part C** – Iris ML pipeline with Mage orchestration
- **Part D** – Food-order event streaming with RabbitMQ

## Repository Structure

```
freshcart-pipelines/
├── README.md
├── .gitignore
├── requirements.txt
├── elt/                        ← FreshCart ELT notebook (Part B)
│   └── freshcart_elt.ipynb
├── iris_project/               ← Mage ML project (Part C)
│   ├── metadata.yaml
│   ├── data_loaders/
│   │   └── load_iris.py
│   ├── transformers/
│   │   └── prepare_data.py
│   ├── custom/
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   └── validate_data.py
│   └── data_exporters/
│       └── save_results.py
├── stream/                     ← RabbitMQ dispatcher & driver (Part D)
│   ├── dispatcher.py
│   └── driver.py
├── tests/
│   └── test_prepare_data.py
└── .github/
    └── workflows/
        └── ci.yml
```

## Quick Start

```bash
pip install -r requirements.txt
# Part C – run the ML pipeline
mage run iris_project iris_classifier --test
# Part D – run the stream (requires RabbitMQ)
python stream/dispatcher.py &
python stream/driver.py Ravi "bangalore.koramangala.*"
```

## Architecture Diagram

```
GitHub (code) ──► GitHub Actions CI ──► pytest (green/red)
                                           │
                                           ▼
FreshCart API ──► Bronze ──► Silver ──► Gold (Snowflake)
                                           │
                                           ▼
                              Mage ML Pipeline (Iris)
                                           │
                                           ▼
RabbitMQ Stream ──► Dispatcher ──► Drivers (ACK/NACK)
                         │
                         └──► Consumer ──► Snowflake Gold (Bonus)
```

