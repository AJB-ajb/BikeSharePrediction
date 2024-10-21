#align(center, text(22pt)[
  *Dynamic Demand Prediction for Sustainable Bike Sharing Systems*
])

By Alexander Busch and Kaifeng Lu as part of the course project of ECE1724H: Bio-inspired Algorithms for Smart Mobility. Dr. Alaa Khamis, University of Toronto, 2024.

= Literature Review
== Depth and Breadth of Survey

1. A review on bike-sharing: The factors affecting bike-sharing demand @EREN2020101882:

2. Data Analysis and Optimization for (Citi)Bike Sharing @OMahony_Shmoys_2015:

3. Modeling bike-sharing demand using a regression model with spatially varying coefficients @8005700:

4. Modeling Bike Availability in a Bike-Sharing System Using Machine Learning @WANG2021103059:


= Taxonomic Table and Tabular Comparison
For the development of accurately prediction approaches the most important factors are:
- The data used:
  Essential is the features incorporated and the size, diversity and quality of the dataset.
  Because all compared approaches use recorded data from bike-sharing systems, the quality of the data is uniformly high, however, the size varies. 
  We propose to classify features:

- The architecture of the model:
  In order to achieve intelligent prediction, usually deeper models perform significantly better. Special modules of neural network architecture, such as Memory Components (RNNs, LSTMs, GRUs), Convolutions (CNN, GNN) or attention components strongly influence what the model is able to learn. 
  // We suggest to classify all neural network approaches in our table based on 
  // Spatial Architecture: CNN / GNN / None
  // Temporal Architecture LSTM / GRU / RNN / None
  // 


#table(columns: 5, table.header([Approach], [Target Variable], [Spatial Features Incorporated], [Spatial Architecture], [Temporal Features], [Temporal Architecture], )) // todo


= References
#bibliography("references.bib", style: "institute-of-electrical-and-electronics-engineers")

// [1] E. Eren and V. Emre Uz, “A review on bike-sharing: The factors affecting bike-sharing demand,” 2020. Accessed: Oct. 20, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2210670719312387

// [2] E. O’Mahony, “Data Analysis and Optimization for (Citi)Bike Sharing,” 2015. Accessed: Oct. 20, 2024. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/9245

// [3] X. Wang, Z. Cheng, M. Trépanier, and L. Sun, “Modeling bike-sharing demand using a regression model with spatially varying coefficients,” 2021. Accessed: Oct. 20, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0966692321001125?ref=pdf_download&fr=RR-2&rr=8d56c29f6c6339ff

// [4] H. I. Ashqar, M. Elhenawy, M. H. Almannaa, A. Ghanem, H. A. Rakha, and L. House, “Modeling bike availability in a bike-sharing system using machine learning,” 2017. Accessed: Oct. 20, 2024. [Online]. Available: https://ieeexplore.ieee.org/abstract/document/8005700


