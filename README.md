# TikTok Content Moderation: Classifying Claims vs. Opinions

End-to-end analysis and ML pipeline that classifies TikTok videos as factual **claims** or personal **opinions**, triaging user-reported content for the moderation queue.

---

## Problem

TikTok receives a high volume of user reports flagging videos as misleading or policy-violating. Every report is manually reviewed, creating a backlog that delays action on the videos that matter most — those containing factual claims that may spread misinformation.

The business goal is to automate the first stage of triage: predict whether a reported video contains a **claim** (high moderation priority) or an **opinion** (low priority), so human reviewers focus on the content that poses the highest risk.

## Dataset

- ~19,000 labeled TikTok videos with 12 features
- Near-balanced target (50.3% claims / 49.7% opinions)
- Features include engagement counts (views, likes, shares, downloads, comments), author metadata (verified status, ban status), video duration, and full transcription text

## Approach

Four sequential phases following the **PACE framework**:

| Phase | Method | Purpose |
|---|---|---|
| 1 | Exploratory Data Analysis | Profile engagement patterns, missingness, and class balance |
| 2 | Welch's two-sample t-test | Validate that verified vs. unverified accounts differ significantly in mean view count |
| 3 | Logistic regression | Interpretable baseline predicting verified status |
| 4 | Random Forest + XGBoost | Production classifier predicting claim vs. opinion |

### Modeling workflow

- **Stratified 60/20/20 split** (train / validate / test) on `claim_status`
- **CountVectorizer** (bigrams + trigrams, top 15 features) on transcription text, fit on train only to prevent leakage
- **5-fold cross-validated grid search** on both Random Forest and XGBoost
- **Primary metric: recall on the claim class** — the cost of missing a claim (misinformation goes unreviewed) far exceeds the cost of a false positive (opinion gets human-reviewed and cleared)
- Held-out test set touched exactly once, at the end

## Results

**Champion model: Random Forest**

| Metric | Validation | Test |
| --- | --- | --- |
| Recall (claim) | 0.9969 | **0.9974** |
| Precision (claim) | 1.0000 | **1.0000** |
| F1 (claim) | 0.9984 | **0.9987** |
| ROC-AUC | 0.9996 | **0.9999** |
| Accuracy | 0.9984 | **0.9987** |

XGBoost was a close runner-up (test recall 0.9917, precision 0.9995, F1 0.9956). Random Forest was selected as the champion for slightly higher recall on the claim class — the cost-weighted metric for this problem — combined with simpler interpretability for moderation stakeholders.

### Top predictive features

Engagement counts dominate feature importance:

1. `video_view_count` (46.5%)
2. `video_download_count` (20.6%)
3. `video_like_count` (15.3%)
4. `video_share_count` (9.2%)
5. `video_comment_count` (5.5%)

Text n-grams and author metadata contribute secondary signal.

### Key EDA findings

- Claim videos accumulate **~100× more engagement** than opinion videos on every metric
- **~32% of claim videos** come from banned or under-review authors, vs. only ~7% for opinions
- Unverified accounts drive significantly higher mean view counts than verified accounts (t-test p < 0.05)

## Repo structure

​```
.
├── data/
│   └── tiktok_dataset.csv               
├── notebook/
│   └── TikTok_End_to_End_Analysis.ipynb # Full analysis — EDA, statistical testing, modeling
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                 # Reusable data-prep pipeline
│   └── train.py                         # MLflow-tracked training (RF + XGBoost)
├── docs/
│   └── mlflow_screenshots/              # MLflow UI captures
├── MLproject                            # MLflow project definition
├── python_env.yaml                      # Environment spec for `mlflow run`
├── requirements.txt                     # Pinned dependencies
├── .gitignore
└── README.md
​```

## Getting started

​```bash
# Clone and set up environment
git clone https://github.com/Dheerajc15/Tiktok_Claim_Classifier.git
cd Tiktok_Claim_Classifier

python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

pip install -r requirements.txt

# Option 1: Run the full analysis notebook
jupyter notebook notebook/TikTok_End_to_End_Analysis.ipynb

# Option 2: Run the MLflow-tracked training pipeline
python src/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
​```

## Caveats

* The model relies heavily on engagement counts. If TikTok's recommendation algorithm changes how claims get distributed, the engagement-based decision rules may stop holding.
* The dataset is a snapshot — no temporal features. View-velocity in the first 24 hours would likely add signal.
* Feature importance from tree ensembles is correlational, not causal. High views don't *cause* a video to be a claim; they co-occur with it in this data.
* Test-set metrics this high suggest the task itself is largely solvable from engagement signal alone. Real production data with adversarial creators would likely be harder.

## Next steps

* **Threshold tuning.** Move off the default 0.5 cutoff; tune on the precision-recall curve to match moderation team capacity.
* **Velocity features.** Compute view-acceleration in the first 24 hours if timestamp data becomes available.
* **Richer text features.** Count hedging words (*maybe, I think*), assertion words (*proven, fact*), presence of numeric claims.
* **Drift monitoring.** Alert on shifts in the engagement-count distributions and retrain on a schedule.
* **Multi-class extension.** Classify *type* of claim (health, financial, political) to route reports to specialized review queues.
* **Model serving.** Wrap the registered RF model in a FastAPI endpoint for real-time inference.

## Tech stack

Python 3.11 · pandas · NumPy · scikit-learn · XGBoost · SciPy · Matplotlib · Seaborn · **MLflow** (tracking + Model Registry)
