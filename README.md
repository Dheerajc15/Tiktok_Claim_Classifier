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
|---|---|---|
| Recall (claim) | 0.99 | **1.00** |
| Precision (claim) | 1.00 | **1.00** |
| F1 (claim) | 1.00 | **1.00** |
| Accuracy | 1.00 | **1.00** |

XGBoost matched Random Forest's performance (recall 0.99, precision 1.00 on validation). Random Forest was selected as the champion because it trains faster and is easier to interpret for moderation stakeholders.

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

```
.
├── TikTok_End_to_End_Analysis.ipynb   # Full analysis — all four phases
├── tiktok_dataset.csv                 # Input data (not included — see below)
├── requirements.txt                   # Pinned dependencies
└── README.md
```

## Getting started

```bash
# Clone and set up environment
git clone <repo-url>
cd tiktok-claim-classifier

python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

pip install -r requirements.txt

# Place tiktok_dataset.csv in the repo root, then:
jupyter notebook TikTok_End_to_End_Analysis.ipynb
```

**Note on the dataset:** `tiktok_dataset.csv` is the Coursera Google Advanced Data Analytics Professional Certificate TikTok dataset. It is not redistributed here.

## Caveats

- The model relies heavily on engagement counts. If TikTok's recommendation algorithm changes how claims get distributed, the engagement-based decision rules may stop holding.
- The dataset is a snapshot — no temporal features. View-velocity in the first 24 hours would likely add signal.
- Feature importance from tree ensembles is correlational, not causal. High views don't *cause* a video to be a claim; they co-occur with it in this data.

## Next steps

- **Threshold tuning.** Move off the default 0.5 cutoff; tune on the precision-recall curve to match moderation team capacity.
- **Velocity features.** Compute view-acceleration in the first 24 hours if timestamp data becomes available.
- **Richer text features.** Count hedging words (*maybe, I think*), assertion words (*proven, fact*), presence of numeric claims.
- **Drift monitoring.** Alert on shifts in the engagement-count distributions and retrain on a schedule.
- **Multi-class extension.** Classify *type* of claim (health, financial, political) to route reports to specialized review queues.

## Tech stack

Python 3.11 · pandas · NumPy · scikit-learn · XGBoost · SciPy · Matplotlib · Seaborn
