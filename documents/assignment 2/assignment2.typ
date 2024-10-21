#align(center, text(22pt)[
  *Dynamic Demand Prediction for Sustainable Bike Sharing Systems*
])

By Alexander Busch and Kaifeng Lu as part of the course project of ECE1724H: Bio-inspired Algorithms for Smart Mobility. Dr. Alaa Khamis, University of Toronto, 2024.

= Literature Review
// == Depth and Breadth of Survey  // this seems to be included in the evaluation criteria, but feels like a bad chapter title
Due to the vastly growing number of bike sharing systems, bike sharing demand prediction has been investigated by a number of authors in the last $15$ years. While classical machine learning approaches such as regression and boosting have been used in the early developments, in recent years, deep learning approaches have been found to give significant advantages in the forecasting domain @YANG2020101521.

Based on the literature, we compare the following recent machine learning approaches to bike sharing demand prediction, selected according to publishing date and impact, and identify open problems.


3. Modeling bike-sharing demand using a regression model with spatially varying coefficients @WANG2021103059:

4. Modeling Bike Availability in a Bike-Sharing System Using Machine Learning @WANG2021103059:

// todo: give current state of the art ML approach
// give some directions for future research

= Taxonomic Table and Tabular Comparison
We organize the used models into a tree based on the dominant design features.

#import "@preview/syntree:0.2.0": syntree, tree

#figure(tree([ML],
  tree([Classical ML],
    tree([Regression], 
      [P-LSQ @Ashqar2017], [ARIMA @Dastjerdi2022]),
    tree([Tree-Based], 
      [LSBoost @Ashqar2017], [XGBoost @YANG2020101521], [RF @Ashqar2017])),
  tree([Neural Network Approaches],
    [MLP @YANG2020101521],
    tree([LSTM-based],
      [LSTM-CNN @Dastjerdi2022],
      tree([Graph-based],
        [LSTM @YANG2020101521],
        [DA-MR-GNN @Liang2024]
      ))
  )
), caption: [An overview of the employed ML approaches])


== Directions for Future Research
As noted in @YANG2020101521, there is no single standard benchmarking dataset for the specific objective of bike sharing demand prediction, making it difficult to obtain a precise ranking of the models. Notably, the New York Citibike data has been used in multiple approaches @YANG2020101521, @Liang2024, but the precise data used still changes.
As given in @YANG2020101521, several studies have suggested that XGBoost performing as well most of the state of the art approaches, having also won the 2014 Kaggle competition @cukierski2014bike_kaggle, however the best approach was found to depend strongly on the dataset and modeling. 

Notably @YANG2020101521 suggests, that well-performing deep model architectures in traffic prediction are likely to show good performance in bike-sharing demand prediction. In this line of work, @Liang2024 combines well performing spatio-temporal GNN approach, known from traffic prediction @Yu2017SpatiotemporalGC and @Zhang2022AGT, with a domain-adversarial network. However, little work has been done on the extent that performance of traffic prediction carries over to bike-sharing performance and adaptation of successful models from other adjacent fields, (e.g. models in @Yu2017SpatiotemporalGC, @Zhang2022AGT), appears to be a promising research direction.


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

