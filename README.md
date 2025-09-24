# 🛒 BigMart Sales Forecasting Pipeline

This project showcases a complete **Data Engineering + Machine Learning pipeline** using BigMart retail sales data. It includes automated data ingestion, MySQL database setup, model training, and deployment via a Streamlit app.

👉 **Live Demo:**[(https://bigmartsaleprediction.streamlit.app/)](https://bigmart-sales-prediction-app.streamlit.app/)

---

## 🧱 Architecture Overview

```mermaid
graph TD
    A[Start] --> B{Load ML Model};
    B --> C{Model Found?};
    C -- No --> D[Display Error & Stop];
    C -- Yes --> E[Render UI: Title & Navigation];
    
    E --> F{User Selects Page};
    F -- Home --> G[Display Prediction Form];
    F -- Data Insights --> H[Display Sales Chart];
    F -- Model Details --> I[Display Model Info];
    F -- About/Contact --> J[Display Static Info];

    G --> K{User Fills Form?};
    K -- Yes --> L[User Clicks Predict];
    L --> M[Process Form Data];
    M --> N[Predict Sales using Model];
    N --> O[Display Prediction Result];
    O --> F;
    
    H --> F;
    I --> F;
    J --> F;
```
