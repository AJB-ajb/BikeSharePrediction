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

3. Modeling bike-sharing demand using a regression model with spatially varying coefficients @8005700:

4. Modeling Bike Availability in a Bike-Sharing System Using Machine Learning @WANG2021103059:


== Taxonomic Classification

= References

#bibliography("references.bib", style: "institute-of-electrical-and-electronics-engineers")

// [1] https://www.mdpi.com/1424-8220/22/3/1060

// [2] https://www.sciencedirect.com/science/article/pii/S0198971520302544

// [3] X. Wang, Z. Cheng, M. Trépanier, and L. Sun, “Modeling bike-sharing demand using a regression model with spatially varying coefficients,” 2021. Accessed: Oct. 20, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0966692321001125?ref=pdf_download&fr=RR-2&rr=8d56c29f6c6339ff

// [4] H. I. Ashqar, M. Elhenawy, M. H. Almannaa, A. Ghanem, H. A. Rakha, and L. House, “Modeling bike availability in a bike-sharing system using machine learning,” 2017. Accessed: Oct. 20, 2024. [Online]. Available: https://ieeexplore.ieee.org/abstract/document/8005700


