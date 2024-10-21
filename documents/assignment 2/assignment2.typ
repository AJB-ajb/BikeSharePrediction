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
We divide the given approaches according to the used models.


#import "@preview/syntree:0.2.0": syntree, tree
#syntree(
  nonterminal: (font: "Linux Biolinum"),
  // terminal: (fill: blue),
  child-spacing: 3em, // default 1em
  layer-spacing: 2em, // default 2.3em
  "[Approach [Optimization [Integer Least Squares Programming ]] 
            [VP [V is] [^NP a wug]]]"
)
#tree([ML],
  tree([Classical ML],
    tree([Regression], 
      [Partial Least Suqares @Ashqar2017], [ARIMA @Dastjerdi2022], []),
    tree([Tree-Based], 
      [LSBoost @Ashqar2017], [XGBoost @YANG2020101521], [Random Forests @Ashqar2017])),
  tree([Neural Network Approaches],
    [MLP @YANG2020101521],
    tree([LSTM-based],
      [LSTM-CNN @Dastjerdi2022],
      tree([Graph-based],
        [LSTM @YANG2020101521],
        [MR-GNN @Liang2024]
      ))
  )
)








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


