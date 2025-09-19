# Movie Rating Prediction  

This project explores predicting movie ratings using **supervised** and **semi-supervised** machine learning models with both labelled and unlabelled datasets.  
The work is based on datasets from [TMDB](https://www.themoviedb.org/) and a Kaggle challenge: *“How good is that movie? Predict Movie Ratings”* by Hasti Samadi.  

## Project Overview  

- **Datasets**  
  - *Movie features*: release year, runtime, budget, revenue, language, popularity, genre, country, vote count  
  - *Text features*: title, overview, tagline, production companies (encoded with TF-IDF & Bag-of-Words)  

- **Models Tested**  
  - **Baseline**: Zero-R  
  - **Supervised**: Decision Tree (DT), Logistic Regression (LR), Random Forest (RF)  
  - **Semi-supervised**: Self-training (pseudo-labelling)  

- **Main Research Question**  
  > Does the use of unlabelled data improve the performance of machine learning models in predicting movie ratings?  

## Methods  

1. **Feature Selection**  
   - ANOVA F-test used for non-text features  
   - TF-IDF and Bag-of-Words (BoW) for text features  

2. **Model Training**  
   - Supervised models trained on labelled datasets  
   - Self-training applied to include unlabelled datasets  

3. **Evaluation Metrics**  
   - Precision  
   - Recall  
   - Accuracy  
   - F1-score  

## Results  

- Zero-R baseline performed poorly, as expected.  
- Decision Tree and Random Forest achieved strong performance with labelled features.  
- Logistic Regression suffered from **overfitting** and did not benefit from unlabelled data.  
- Self-training showed **minor improvements**, particularly for Random Forest (precision improved from **0.69 → 0.79** with TF-IDF).  
- Overall, unlabelled datasets had **limited impact** except for boosting Random Forest precision.  

## Key Findings  

- **Random Forest** benefited most from self-training.  
- **Logistic Regression** overfit quickly, limiting improvement.  
- **TF-IDF** outperformed Bag-of-Words by capturing contextual word importance.  
- Balancing datasets slightly improved Kaggle submission accuracy (**0.685 → 0.687**).  

