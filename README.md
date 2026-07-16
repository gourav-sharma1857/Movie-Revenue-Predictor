# 🎬 Movie Box Office Revenue Predictor

A machine learning pipeline that predicts a movie's worldwide box office revenue from budget, cast, franchise history, and release strategy — plus a self-contained web app that runs the trained model entirely in the browser, no backend required.

**[Live demo →](#)** *(add your GitHub Pages URL here)*

---

## What this is

Given a movie's budget, genre, cast, director history, and whether it's part of a franchise, the model estimates its total worldwide box office gross. It's trained on 977 real movies pulled from [The Movie Database (TMDB)](https://www.themoviedb.org/) API.

**Important scope note:** every movie in the training set is one of the top 1,000 grossing films of all time. The model has never seen a flop, so it does **not** predict whether a movie will succeed — it predicts *how big a hit will be*, assuming it's already a wide theatrical release with real marketing behind it. It's best suited to franchise entries and blockbuster-scale releases, not indie or original films (see [Limitations](#limitations)).

---

## The pipeline

1. **Data collection** — pulled via TMDB's `/discover/movie`, `/movie/{id}`, `/person/{id}/movie_credits`, and `/collection/{id}` endpoints. Budget, revenue, genres, cast, director, and franchise/collection data collected for 1,000 movies sorted by revenue.
2. **Cleaning** — dropped rows with missing budget data (mostly foreign-market films TMDB lacks financial data for), parsed dates, fixed CSV round-trip issues with list-type columns (genres, cast).
3. **Feature engineering** — built features not directly available from the API:
   - `director_track_record` — average revenue of a director's prior films (careful to only count films released *before* the target movie, avoiding data leakage)
   - `prev_installment_revenue` — the actual revenue of the most recent prior film in the same franchise/collection
   - `primary_genre` (rare genres bucketed into "Other"), `release_season`, `is_franchise`, `is_major_studio`, production footprint (companies/countries/languages/keyword count)
   - Log-transformed budget and revenue-based features (heavily right-skewed distributions)
4. **EDA** — confirmed budget and franchise history as the dominant predictors; genre and release season contribute far less than expected once those are known. Also surfaced a key limitation: the dataset only contains high-grossing films (revenue floor ~$195M), meaning the model can't learn what separates a hit from a flop.
5. **Feature selection** — excluded `popularity`/`vote_average`/`vote_count` (post-release metrics — using them would be data leakage for a pre-release prediction tool).
6. **Modeling** — benchmarked Linear Regression → Random Forest → tuned Random Forest → XGBoost → tuned XGBoost → LightGBM. Final model: **LightGBM**, cross-validated R² ≈ 0.43.
7. **Deployment** — trained model transpiled to plain JavaScript via [`m2cgen`](https://github.com/BayesWitnesses/m2cgen), embedded directly in a static HTML file. No server, no API calls, no dependencies at runtime.

---

## Model performance

| Model | Cross-val R² | Notes |
|---|---|---|
| Linear Regression | 0.33 | Stable baseline |
| Random Forest (default) | 0.31–0.40 | Overfit (train R² 0.80 vs test 0.30) |
| Random Forest (tuned) | 0.44 | Better generalization |
| XGBoost (tuned) | 0.41 | Best raw outlier accuracy, but train R² 0.96 — overfit |
| **LightGBM (final)** | **0.43** | Best balance of generalization and outlier sensitivity |

R² in the 0.3–0.45 range is realistic for box office prediction — real-world outcomes depend heavily on factors not in this dataset (marketing execution quality, word of mouth, competing releases, critical buzz).

---

## The app (`movie_revenue_predictor.html`)

A single-file, no-backend web app:

- **Feature sheet** — manually enter budget, cast popularity, franchise history, genre, release timing, and production details
- **Pick an Upcoming Movie** — a curated, editable list of real not-yet-released titles (`movies_data.json`) with pre-loaded figures; click one to auto-fill and predict instantly
- **Live prediction** — the trained model runs client-side via the embedded, `m2cgen`-transpiled JavaScript function; no API key, no server, no network calls at runtime
- **Honest uncertainty flags** — the app visibly warns when a prediction is for a low-budget or original (non-franchise) film, since those are the model's weakest spots

### Updating the movie list

`movies_data.json` sits next to the HTML file and can be edited directly — no code changes needed. Each entry:

```json
{ "title": "Movie Title", "genre": "Action", "month": 8, "year": 2026,
  "isFranchise": true, "prevRevM": 500, "budgetM": 150, "castPop": 15,
  "dirRevM": 200, "companies": 2, "countries": 1, "langs": 1,
  "isMajorStudio": true, "isEnglish": true, "numKeywords": 15 }
```

`prevRevM` / `budgetM` / `dirRevM` are in **millions of dollars**. `genre` must be one of: Action, Adventure, Animation, Comedy, Crime, Drama, Family, Fantasy, Horror, Mystery, Romance, Science Fiction, Thriller, War, Other.

**Running it:** open `index.html` directly, or for the movie-list JSON to load properly, serve the folder locally:

```bash
python3 -m http.server 8000
```

then visit `http://localhost:8000/`. (Deployed on GitHub Pages, this restriction doesn't apply — the JSON loads normally over HTTPS.)

---

## Limitations

- **No flops in training data.** Every movie made at least ~$195M. The model will not predict low revenue even for a tiny budget — it has never seen failure, so it can't recognize it.
- **Weakest on original/indie films.** Franchise history (`prev_installment_revenue`) is the strongest single predictor. Without it, the model leans on budget and cast alone, which is a meaningfully harder problem — treat predictions for standalone/original films with more skepticism.
- **Director track record is capped.** For speed, only a director's 5 most recent prior films are considered, not their full filmography.
- **Static movie list.** `movies_data.json` reflects a snapshot in time and needs manual updates as release dates pass or new titles are announced.

---

## Tech stack

- **Data collection & modeling:** Python, `requests`, `pandas`, `scikit-learn`, `lightgbm`, `m2cgen`
- **Deployment:** vanilla HTML/CSS/JS (no frameworks, no build step), GitHub Pages
- **Data source:** [TMDB API](https://www.themoviedb.org/documentation/api)

---

## Project structure

```
├── index.html              # the app (model + UI, single file)
├── movies_data.json         # editable list of upcoming movies
└── README.md
```

*(Notebook/training scripts used to produce the model are kept separately and not required to run the app.)*
