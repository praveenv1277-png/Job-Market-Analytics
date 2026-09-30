# Tech Market Intelligence & AI Salary Predictor

An end-to-end Machine Learning and Data Analytics platform that analyzes tech job postings, extracts high-demand technical skills, benchmarks compensation trends, and predicts salaries using trained regression models.

---

## 🚀 Live Application Link

The Streamlit dashboard is currently running locally:
- **Local URL:** [http://localhost:8501](http://localhost:8501)
- **Network URL:** `http://192.168.0.129:8501`

---

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Run Pipeline & Re-train Models
```bash
python src/seed_data.py
```
This fetches real-time jobs from Remotive API, cleans and standardizes the data, extracts 35+ tech skills, and trains:
- **Salary Regressor:** `GradientBoostingRegressor` (R² = 0.72)
- **Role Classifier:** `RandomForestClassifier` (Accuracy = 79.6%)

### 3. Launch Dashboard
```bash
streamlit run app/app.py
```

---

## 🌟 Dashboard Capabilities

1. **📊 Market Intelligence & EDA:**
   - Distribution of salaries across role categories (Data Scientist, ML Engineer, Data Analyst, etc.).
   - Seniority progression curves (Entry $\rightarrow$ Mid $\rightarrow$ Senior $\rightarrow$ Executive).
   - Top 15 most in-demand technical skills and remote vs metro location breakdown.

2. **🤖 AI Salary Predictor:**
   - Select Target Role, Experience Level, Employment Type, Location, and Technical Skills.
   - Real-time ML salary inference with an 85% confidence interval.
   - Market benchmark positioning and recommended high-ROI skill boosters.

3. **🎯 Role Classifier & Skill Matcher:**
   - Paste job descriptions or profile bios.
   - Auto-extracts technical keywords and uses multi-class classification to predict role categories with probability confidence.

4. **🔎 Live Job Explorer:**
   - Filter, sort, and search job listings with salary badges and skill pills.
   - Live integration with public tech job APIs (Remotive).

5. **📈 ML Model Analytics & Explainability:**
   - Model comparison tables (RandomForest, GradientBoosting, Ridge, LogisticRegression).
   - Global feature importance charts illustrating the biggest drivers of tech compensation.

6. **📥 Data Hub & Export:**
   - Preview cleaned dataset and export as CSV.
