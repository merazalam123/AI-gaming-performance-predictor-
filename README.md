# AI Gaming Performance Predictor

An interactive Streamlit application that predicts a player's gaming performance category using a trained machine learning model.

## Overview

This project uses player statistics such as matches played, win rate, average kills, deaths, assists, accuracy, headshot rate, average damage, playtime, and recent win rate to classify performance as:

- HIGH
- MEDIUM
- LOW

The prediction is powered by a machine learning model trained on a synthetic gaming dataset and bundled with a scaler for preprocessing.

## Project Structure

```text
AI-gaming-performance-predictor-
├── app.py
├── requirements.txt
├── dataset/
├── model/
│   ├── gaming_model.pkl
│   └── scaler.pkl
├── notebooks/
└── README.md
```

## Features

- User-friendly Streamlit UI
- Real-time prediction from gameplay metrics
- Model probability breakdown for each class
- Visual bar chart of prediction confidence
- Scalable and easy-to-run local deployment

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- joblib

## Installation

1. Clone the repository:

```bash
git clone https://github.com/merazalam123/AI-gaming-performance-predictor-.git
cd AI-gaming-performance-predictor-
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

Start the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Model Details

The app loads:

- `model/gaming_model.pkl` — trained classifier
- `model/scaler.pkl` — preprocessing scaler used before inference

The model expects numeric player stats in the same order used during training. The app transforms input values using the stored scaler before performing prediction.

## Input Fields

The app asks for the following player statistics:

- Matches Played
- Win Rate (%)
- Average Kills
- Average Deaths
- Average Assists
- Accuracy (%)
- Headshot Rate (%)
- Average Damage
- Playtime (Hours)
- Recent Win Rate (%)

## Example Use Case

This project is useful for:

- Game analytics and player evaluation
- Esports team performance analysis
- Machine learning demos for gaming data
- UI-based prediction prototypes

## License

This project does not currently include a license file. If you plan to publish or distribute it publicly, consider adding an appropriate open-source license.

## Author

merazalam123

## Notes

The application is designed for demonstration and educational use, and the underlying dataset is synthetically generated.
