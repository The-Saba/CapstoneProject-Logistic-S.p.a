# Capstone Project Documentation

## Project Overview
This project analyses and clusters logistic clients for Logistic S.p.a using Italian company data, monthly sales/activity volumes, and industry codes.

The main goals are:
- Clean and enrich raw client data with geographic and industry information.
- Explore revenue, employees, regional distribution, and monthly activity patterns.
- Build cluster models to segment clients by behavior and value.
- Produce a clustered dataset for downstream business interpretation.

## Main Files and Folders
- `Exploration_analysis.ipynb`: data cleaning and enrichment pipeline.
- `EDA.ipynb`: exploratory data analysis on the cleaned dataset.
- `Clustering.ipynb`: feature engineering, PCA, and clustering with K-Means and DBSCAN.
- `README.md`: repository title and entry point. See `PROJECT_DOCUMENTATION.md` for details.
- `source/`: data files and intermediate outputs.
- `utils/italian_provinces.py`: lookup table for Italian provinces to city/region mapping.
- `Discussion/`: meeting notes and assignment notebooks, not part of the main analysis pipeline.

## Data Files
- `source/dataset.csv`: raw dataset with client information, ATECO codes, revenue, employees, and monthly volumes.
- `source/Ateco-2022-vs-NACE-Rev.-2.xlsx`: industry code reference table used to map ATECO to NACE Rev. 2.
- `source/dataset_eda.csv`: cleaned dataset used by the EDA and clustering notebooks.
- `source/dataset_feature_engineered.csv`: expected feature-engineered dataset (not directly listed in the notebook outputs shown but implied by project flow).
- `source/dataset_clustered.csv`: final output containing cluster labels and business segments.

## Pipeline Steps (What Has Been Done)
1. **Data Loading**
   - Load raw dataset from `source/dataset.csv`.
   - Load ATECO-to-NACE mapping from `source/Ateco-2022-vs-NACE-Rev.-2.xlsx`.

2. **Cleaning & Normalization** (`Exploration_analysis.ipynb`)
   - Rename Italian column names to English.
   - Normalize ATECO descriptions.
   - Enrich geographic data by mapping `Province` codes to `City` and `Region`.

3. **Industry Mapping** (`Exploration_analysis.ipynb`)
   - Normalize ATECO codes.
   - Map ATECO to NACE using hierarchical fallback levels.
   - Report mapping quality and unmatched codes.

4. **Final Cleaning** (`Exploration_analysis.ipynb`)
   - Remove rows with missing key columns.
   - Drop rows with zero activity across all monthly columns.
   - Clean intermediate columns and save the final cleaned dataset.

5. **Exploratory Data Analysis** (`EDA.ipynb`)
   - Analyze revenue and employee distributions.
   - Inspect geographic distribution by region and city.
   - Analyze NACE sectors.
   - Review monthly time-series patterns.
   - Compute correlations between important variables.

6. **Feature Engineering & Clustering** (`Clustering.ipynb`)
   - Create size proxies and activity features from monthly volumes.
   - Standardize the feature matrix.
   - Apply PCA for visualization and DBSCAN input.
   - Run K-Means model selection and fit the final clusters.
   - Search DBSCAN hyperparameters and fit the final model.
   - Profile clusters and compare K-Means vs DBSCAN.
   - Save the clustered dataset to `source/dataset_clustered.csv`.

## What Still Needs To Be Done
- **Document final business cluster labels**
  - The notebook assigns example labels automatically, but those labels should be reviewed and adjusted with business context.

- **Confirm `BEST_K` and DBSCAN final parameters**
  - K selection is currently based on highest silhouette score. It may need manual adjustment based on business goals and cluster interpretability.
  - DBSCAN uses an automatically selected `eps` and `min_samples`; this should be validated for cluster quality and noise tolerance.

- **Validate K-Means with manual checks**
  - Compare `K=5` against the auto-selected `BEST_K` using the same metrics: silhouette score, Davies-Bouldin index, Calinski-Harabasz score, and visual cluster separation.
  - Inspect cluster sizes and labels: too-small clusters or a cluster that is not meaningful may indicate a poor K choice.
  - Use the 2D PCA plot and the cluster profile heatmaps to verify whether `K=5` gives more actionable segments.
  - Current run results: `K=3` yields the best silhouette score at `0.337`, while `K=5` returns `0.285` with cluster sizes `[332, 5973, 3057, 5464, 1958]`.

- **Validate DBSCAN hyperparameters**
  - Review the k-distance plot to confirm whether the chosen `eps` is at the natural knee; test nearby values to see stability.
  - Compare multiple `(eps, min_samples)` pairs from the grid search and choose the combination with:
    - stable cluster count,
    - moderate noise percentage (not too high, not too low),
    - highest silhouette score on core points,
    - interpretable cluster profiles.
  - If noise is too high, increase `eps` or reduce `min_samples`; if clusters are too few, decrease `eps` or reduce `min_samples`.
  - Current DBSCAN study found viable 5D PCA configs such as `eps=0.60, min_samples=9` producing `7` clusters with `5.0%` noise, and `eps=0.70, min_samples=9` producing `5` clusters with `2.8%` noise. The original knee-based `eps` was too large for this reduced PCA space, often collapsing to one cluster.

- **Add descriptive analysis for unsupervised segments**
  - Write specific business interpretations for each cluster.
  - Translate cluster profiles into practical actions (e.g. high-value clients, at-risk accounts, growing customers).

- **Check unmapped industry codes**
  - Review rows with missing or no-match NACE mappings and decide whether to correct or exclude them.

- **Improve pipeline reproducibility**
  - Add a clear execution order in `README.md` or the notebooks.
  - Save intermediate cleaned outputs consistently when running the pipeline.

- **Possible enhancements**
  - Add a separate notebook or section for feature importance and cluster stability.
  - Build a dashboard or summary report for stakeholders.
  - Add tests or validation checks for missing values, duplicate rows, and mapping coverage.

## Recommended Execution Order
1. Run `Exploration_analysis.ipynb` to produce the cleaned dataset.
2. Run `EDA.ipynb` to perform exploratory analysis and understand variable distributions.
3. Run `Clustering.ipynb` to generate cluster labels and save `source/dataset_clustered.csv`.

## Notes
- The main analysis data source is `source/dataset_eda.csv`, which is produced after cleaning.
- `Clustering.ipynb` uses the cleaned dataset and applies PCA for visualization.
- Final output is `source/dataset_clustered.csv`, which contains both K-Means and DBSCAN labels and business segment labels.

## Quick Reference
- `Exploration_analysis.ipynb`: cleaning and industry enrichment
- `EDA.ipynb`: exploratory analysis and distribution checks
- `Clustering.ipynb`: clustering model selection, fit, and profiling
- `source/dataset_clustered.csv`: final labeled dataset
