# 🚕 Predicting NYC Taxi Fares: Can we accurately estimate trip costs?

## Project Overview

This project predicts the exact fare of a taxi ride in New York City using historical trip characteristics, geographic coordinates, and time-based features.

- **Target variable:** `fare_amount` (Continuous variable in USD).
- **Goal:** Provide accurate upfront fare estimations for both passengers and drivers based on straight-line distance and time of day.

## Model Performance (Random Forest with Hyperparameter Tuning)

| Metric | Score |
| :--- | :--- |
| **Test R-Squared (R²)** | **76.5%** |
| **Test RMSE** | **$4.84** |
| **Test MAE** | **$2.31** |
| **Optimal numTrees** | **150** |
| **Optimal maxDepth** | **15** |

The model was evaluated on a **20% hold-out test set** (unseen data) with a fixed seed to prevent data leakage. The evaluation metric used for optimization is Root Mean Squared Error (RMSE), representing the average dollar amount the prediction deviates from the actual fare.

## Getting Started

### Prerequisites
- Python 3.7+
- pandas, numpy, scikit-learn, matplotlib, seaborn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/NYC-Taxi-Fare-Prediction.git
   cd "NYC-Taxi-Fare-Prediction"
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
.
├── train.csv          # Training dataset
├── README.md          # This file
├── requirements.txt   # Python dependencies
└── notebooks/         # Jupyter notebooks for analysis
```

## Collaboration Guidelines

- Create a new branch for each feature: `git checkout -b feature/your-feature-name`
- Commit frequently with descriptive messages
- Push to your branch and create a Pull Request
- Code review before merging to main

## Contributors

- [Your Name]
- [Partner Name]

## License

MIT License

## Contact

For questions or issues, please open a GitHub Issue.
