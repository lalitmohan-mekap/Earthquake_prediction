# VS Code Setup & Presentation Guide

Since you'll be presenting this undergraduate research project, here is exactly how to run it on any machine with VS Code and what to focus on during your presentation.

---

## 1. How to run in VS Code

### Step A: Environment Setup
1. **Open Folder**: Open the `new pj` folder in VS Code.
2. **Open Terminal**: Go to `Terminal -> New Terminal` (ensure it's using PowerShell or Command Prompt).
3. **Create Virtual Env (Recommended)**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install tabulate
   ```

### Step B: Execution Flow
You can run the scripts one by one. I recommend running them in this order for your live demo or to refresh the results:

1. `python execution/fetch_earthquake_data.py` (Downloads USGS data)
2. `python execution/preprocess_data.py` (Cleans and prepares features)
3. `python execution/exploratory_analysis.py` (Generates images in `.tmp/plots`)
4. `python execution/train_models.py` (Trains the ML models)
5. `python execution/evaluate_models.py` (Calculates accuracy/R² and plots results)
6. `python execution/generate_report.py` (Creates the final `research_report.md`)

---

## 2. Presentation Structure (Slides)

Here is a recommended 8-slide structure for your 8th-semester project presentation:

### Slide 1: Title & Group Info
- **Title**: Multi-Model Earthquake Prediction using Machine Learning.
- **Details**: Project Code URP 4301, Group 41010_3.
- **Mentions**: Undergraduate Research Project - 2026.

### Slide 2: Objectives (The "Why")
- Predict magnitude (Regression).
- Categorize seismic risks: Low, Medium, High (Classification).
- Estimate probability of "Big" events (≥6.0 magnitude).
- **Data Source**: Real-time USGS ComCat dataset (2000–2025).

### Slide 3: Methodology & Features
- **3-Layer Architecture**: Explain the separation of Directives (SOPs), Orchestration, and Execution (deterministic scripts).
- **Key Features**: latitude, longitude, depth, and temporal features like `days_since_last_event`.

### Slide 4: Exploratory Data Analysis (Visuals)
- Select 2-3 plots from `.tmp/plots/`:
  - **Geographic Distribution**: Shows the "Ring of Fire" and fault lines.
  - **Magnitude Distribution**: Shows that smaller earthquakes are much more frequent (Gutenberg-Richter law).
  - **Correlation Heatmap**: Shows relationships between depth and magnitude.

### Slide 5: Model 1 - Magnitude Prediction
- Compare **Linear Regression** vs. **Random Forest**.
- **Result**: "Random Forest captured the non-linear nature of seismic waves better, achieving a higher R² score."

### Slide 6: Model 2 & 3 - Risk & Probability
- **Model 2**: Classification accuracy (e.g., ~88%). Mention using "Balanced Class Weights" to handle the rarity of high-magnitude events.
- **Model 3**: Show the **ROC Curve**. Explain that an AUC of 0.70 means the model is 70% better than a random guess at identifying big quakes.

### Slide 7: Live Demo (The "Wow" Factor)
- Run `python execution/predict.py` in the VS Code terminal.
- Ask the audience or panel for a location (Lat/Lon) and depth, then enter them to show a real-time prediction.

### Slide 8: Conclusion & Future Scope
- **Conclusion**: ML can effectively augment seismic risk analysis.
- **Future**: Incorporating tectonic plate boundary data and global strain rate maps.

---

## 3. Demo Tips
- Have the `deliverables/research_report.md` open in VS Code. Press `Ctrl + Shift + V` to show the **Markdown Preview**. This looks very professional as it shows the tables and plots integrated together.
- Keep the `directives/` folder visible to show that your project follows a "Standard Operating Procedure" (Industrial/Research best practice).
