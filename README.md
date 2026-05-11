# DSCI 531 — Fairness in College Admissions

A class project on fairness in machine-learning–based college admissions decisions. We train classifiers on a college admissions dataset, evaluate them under standard fairness criteria (demographic parity and equalized odds), wrap the trained model in an LLM-driven "chance me" chatbot, and run a counterfactual sweep to measure how much a single feature can swing the model's prediction for the same student.

## Repository structure

```
.
├── eda_and_modeling/
│   ├── dsci_531_project_v2.ipynb   # EDA, model training, fairness analysis, model export
│   ├── college-admission-dataset.csv  # Applicant-level admissions data (primary modeling set)
│   ├── Dataset - 2021.csv          # Enrollment data by race/gender across institutions
│   └── Data-Table 1.csv            # IPEDS institution-level data
└── llm/
    ├── gpt_llm.py                  # "Chance me" chatbot wrapping the trained random forest
    ├── run_bias_sweep.py           # Batch script: counterfactual sweep + bias chart
    ├── students.xlsx               # 50 baseline student profiles (input to the sweep)
    ├── rf.joblib                   # Trained random forest exported from the notebook
    ├── encodings.json              # Feature order + label encodings for gender and race
    ├── predictions.csv             # Sweep output: one row per (student, swept value)
    ├── bias_summary.csv            # Per-student probability range across each sweep
    └── bias_chart.png              # Two-panel chart summarizing bias findings
```

## What the project does

### 1. Modeling and fairness analysis (`eda_and_modeling/`)

We train logistic regression and random forest classifiers on a college admissions dataset using GPA, test scores, household income, gender, and race. We then evaluate the random forest under two fairness criteria — demographic parity and equalized odds — and repeat the analysis with demographic variables removed. The notebook also pulls in two supporting datasets (`Dataset - 2021.csv` for enrollment patterns by race and gender, and `Data-Table 1.csv` for institution-level IPEDS data) to contextualize the applicant-level findings.

The headline finding is that the models trained on traditional features systematically produce different outcomes across demographic groups, violating both fairness criteria. Removing explicit demographic variables does not eliminate the disparities, because household income (the single most important feature, at roughly 50% importance in the random forest) acts as a proxy for demographics.

### 2. LLM wrapper (`llm/gpt_llm.py`)

We wrapped the trained random forest in a GPT-4o-mini chat interface that behaves like a "chance me" advisor. The language model conversationally collects five inputs from the student (GPA, ACT, household income, gender, race), passes them to the random forest, and explains the resulting admit probability along with the features that most influenced it. The system prompt instructs the model to flag the project's fairness findings — demographic-parity and equalized-odds violations, and household income as a proxy for demographics — when discussing the result.

The LLM does not change any probabilities. The random forest is the predictor; GPT is the interview layer and the explainer.

### 3. Counterfactual bias sweep (`llm/run_bias_sweep.py`)

To measure how sensitive the model is to a single attribute, we ran 50 student profiles through a counterfactual sweep. For each student we held four features fixed and varied the fifth across a grid, isolating the model's response to that one attribute (190 predictions total). This is the cleanest way to operationalize individual-level bias: same student, change one thing, see how the prediction moves.

GPA produced the largest within-student probability swings (mean range 0.43), which is expected since it's a legitimate academic signal. Race and household income were nearly as influential (0.28 and 0.30), while gender was negligible (0.07). Holding every other feature constant and varying only race, the mean predicted admit probability ranged from 0.21 (other/underrepresented) to 0.34 (asian) — a 13-percentage-point gap driven by race alone. Because the predictions came directly from the random forest rather than the language model, this bias reflects the underlying classifier itself and cannot be attributed to GPT.

## How to run

### Requirements

```
pip install scikit-learn pandas joblib openpyxl matplotlib numpy openai
```

`openai` is only needed to run the interactive chatbot. The bias sweep imports from `gpt_llm.py` but never makes an API call, so the sweep works without an API key.

### Reproduce the notebook

Open `eda_and_modeling/dsci_531_project_v2.ipynb` in Jupyter and run all cells. The three CSV datasets sit next to it in the same folder, so no path changes are needed.

### Run the chatbot

```
cd llm
export OPENAI_API_KEY="sk-..."
python gpt_llm.py
```

### Run the bias sweep

```
cd llm
python run_bias_sweep.py
```

This reads `students.xlsx`, writes `predictions.csv` and `bias_summary.csv`, and saves `bias_chart.png`. To test different profiles, edit `students.xlsx` and rerun.

## Limitations

- Both classifiers have low recall on admitted students. The dataset is missing essays, recommendations, extracurricular quality, and other holistic factors that real admissions offices consider.
- The dataset represents a subset of institutions and may not generalize to elite schools, HBCUs, or regional colleges with different admissions philosophies.
- Results are a single snapshot and do not account for year-to-year shifts in admissions practice.
- The analysis shows correlations, not causation. Unmeasured factors may explain part of the observed disparities.

## Authors

Vyomsarit Singh and Navneet Rajagopal
