# CHURNIQ UI Upgrade Notes

The deployment was redesigned around a compact monochrome interface rather than large dashboard tiles and oversized Matplotlib charts.

## Design changes
- Near-black / white visual system with a restrained blue-gray accent.
- More spacing between sidebar sections and controls.
- System-style typography using DM Sans + Space Grotesk with fallbacks.
- Smaller metric cards and tighter page rhythm.
- Consistent rounded surfaces, borders and hover states.
- Plotly interactive charts with controlled heights instead of oversized static plots.
- Reduced visual noise in legends and chart chrome.

## Functional changes
- Persistent **Model Control** in the left sidebar.
- `Auto — Recall-first` remains the academic default.
- User can explicitly select Logistic Regression, KNN, Decision Tree, Random Forest or Gradient Boosting.
- The selected model is used by the Predictor and displayed throughout the interface.
- Model Lab clearly separates automatic selection from manual model inspection.
- Predictor exposes probability and decision threshold as separate concepts.
- Predictor includes CRITICAL / HIGH / WATCH / LOW risk bands.
- Highest-risk and lowest-risk demo records are chosen from the actual dataset using the active model.
- Added 3D customer landscape in Customer Analytics.
- Added 3D precision/recall/F1 model landscape in Model Lab.

## Page-by-page review

### Overview
Purpose: explain the product quickly. Charts are deliberately compact and the five-step ML workflow is visible immediately.

### Customer Analytics
Purpose: demonstrate EDA without overwhelming the screen. Filters affect the charts and the 3D view.

### Model Lab
Purpose: demonstrate the required algorithms and metrics. The model table, confusion matrix, cross-validation and 3D metric landscape provide enough material for a viva without scrolling through giant graphs.

### Predictor
Purpose: demonstrate deployment. Model selection is visible before entering the form, threshold is explicit, demo modes are reproducible and the output explains the result.

### Methodology
Purpose: answer viva questions from inside the app. It summarizes the exact preprocessing and evaluation logic used by the notebook.
