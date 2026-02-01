# Content Monetization – Ad Revenue Prediction

This project focuses on building machine learning regression models to predict YouTube ad revenue based on video performance metrics and metadata.

**Project Objective**

To demonstrate and compare multiple regression-based machine learning models for estimating potential ad revenue using historical YouTube video data.

**Dataset Overview**

Feature Variables - video_id, views, likes, comments, watch_time_minutes, video_length_minutes, subscribers, category, device, country, Year, Month, Day, engagement rate
Target Variable - ad_revenue_usd

**Machine Learning Models Used**

The following regression algorithms were implemented and compared:

1) Linear Regression
2) Ridge Regression
3) Lasso Regression
4) ElasticNet
5) Random Forest Regressor

Each model was trained and evaluated using consistent data splits to ensure fair comparison.

**📈 Evaluation Metrics**

Model performance was measured using standard regression metrics:

--> R² Score – Measures how well the model explains variance in revenue
--> Root Mean Squared Error (RMSE) – Penalizes larger prediction errors
--> Mean Absolute Error (MAE) – Average absolute prediction error
