#align(center, text(22pt)[
  *Dynamic Demand Prediction for Sustainable Bike Sharing Systems*
])

By Alexander Busch and Kaifeng Lu as part of the course project of ECE1724H: Bio-inspired Algorithms for Smart Mobility. Dr. Alaa Khamis, University of Toronto, 2024.

= Literature Review
== Depth and Breadth of Survey

1. Bike-Sharing Demand Prediction at Community Level under COVID-19 Using Deep Learning:

   This article focuses on short-term bike-sharing demand forecasting in Montreal. It uses deep learning techniques to predict bike pickups 15 minutes ahead in six communities identified within the city's bike-sharing network. The study compares the performance of multiple machine learning models: LSTM, CNN-LSTM hybrids, and ARIMA. Researchers trained the models based on two main feature attributes: historical demand data in 15-minute intervals and weather conditions. 
   
   As a result, the CNN-LSTM hybrid model yields the best performance, achieving the lowest error (MAE of 3.00 and RMSE of 4.77). The LSTM model achieves an MAE of 4.48 and RMSE of 6.86, while the ARIMA model achieves an MAE of 50.95 and RMSE of 61.17.
   
    Pros:
 \   LSTM Models: Captures temporal dependencies effectively with moderate complexity.
 \   CNN-LSTM Hybrid Models: Combines spatial and temporal learning for superior accuracy in complex data.
 \   ARIMA Model: Simple, fast to train, and easy to interpret.
 
    Cons: 
 \   LSTM Models: Lacks spatial awareness, limiting prediction accuracy compared to hybrid models.
 \   CNN-LSTM Hybrid Models: Computationally expensive and harder to interpret.
 \   ARIMA Model: Performs poorly with complex, nonlinear, or disrupted data.

2. Using graph structural information about flows to enhance short-term demand prediction in bike-sharing systems:

   This paper investigates how graph theory can enhance short-term bike-sharing demand forecasting. It focuses on incorporating graph-based features derived from flow interactions such as Out-strength, In-strength, Out-degree, In-degree, and PageRank to improve prediction accuracy. It compares three machine learning models: XGBoost, Multi-Layer Perceptron (MLP), and LSTM. This study found that including graph-based features significantly improves model performance compared to using traditional features like meteorological data alone.

   To conclude the performance of each model, this paper finds that the XGBoost, MLP and LSTM model achieves MAPE values of 27.2%, 27.7%, and 27.0% as well as RMSE values of 6.78, 6.85, and 6.69 correspondingly. The LSTM model turns out to be the most effective model when incorporating complex graph-based features.

   Pros:
 \   XGBoost: Excellent for handling structured data and provides strong performance with less tuning.
 \   MLP: Simple and efficient with moderate performance.
 \   LSTM: Best at capturing temporal dependencies and sequential patterns, especially with time-lagged data.

   Cons:
 \   XGBoost: Less effective at handling complex time dependencies compared to LSTM.
 \   MLP: Struggles with sequential data and lacks the advanced handling of time dependencies.
 \   LSTM: Complex to train and requires more computational resources.

 
3. Modeling bike-sharing demand using a regression model with spatially varying coefficients @8005700:
  
   This article focuses on investigating how various factors such as land use, socio-demographic attributes, and transportation infrastructure influence bike-sharing demand at different stations. The authors propose a spatially varying coefficients (SVC) regression model that accounts for local spatial effects, unlike traditional models that assume the factors are spatially homogeneous. As a result, the SVC model achieves a average RMSE of 0.89 and $R^2$ of 0.557.

   Pros: 
 \   SVC: Captures spatial variability, improving prediction accuracy significantly.
   
   Cons: 
 \   SVC: Requires more computational effort and complexity compared to simpler regression models.
   
4. Modeling Bike Availability in a Bike-Sharing System Using Machine Learning @WANG2021103059:

   This paper explores predicting bike availability at San Francisco Bay Area Bike Share stations using machine learning algorithms. The authors apply three methods: Random Forest (RF) and Least-Squares Boosting (LSBoost) for univariate regression and Partial Least-Squares Regression (PLSR) for multivariate regression. They found that factors like station neighbors, time of prediction, and weather conditions are significant in predicting bike availability.

   The RF, LSBoost and PLSR models achieve a MAE of 0.37, 0.58 and 0.6 bikes/station correspondingly. The Random Forest (RF) model offers the best prediction accuracy.

   Pros:
 \   RF: Highly accurate with low prediction error and robust to overfitting.
 \   LSBoost: Effective for regression tasks with manageable computational complexity.
 \   PLSR: Captures spatial correlations, useful for large networks with interdependent stations.

   Cons:
 \   RF: Requires independent observations. Performance decreases as prediction horizon increases.
 \   LSBoost: Higher prediction error. Requires proper regularization against overfitting.
 \   PLSR: Yields the highest prediction error. Less accurate for smaller networks.
   

== Taxonomic Classification

= References

#bibliography("references.bib", style: "institute-of-electrical-and-electronics-engineers")

// [1] https://www.mdpi.com/1424-8220/22/3/1060

// [2] https://www.sciencedirect.com/science/article/pii/S0198971520302544

// [3] X. Wang, Z. Cheng, M. Trépanier, and L. Sun, “Modeling bike-sharing demand using a regression model with spatially varying coefficients,” 2021. Accessed: Oct. 20, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0966692321001125?ref=pdf_download&fr=RR-2&rr=8d56c29f6c6339ff

// [4] H. I. Ashqar, M. Elhenawy, M. H. Almannaa, A. Ghanem, H. A. Rakha, and L. House, “Modeling bike availability in a bike-sharing system using machine learning,” 2017. Accessed: Oct. 20, 2024. [Online]. Available: https://ieeexplore.ieee.org/abstract/document/8005700


