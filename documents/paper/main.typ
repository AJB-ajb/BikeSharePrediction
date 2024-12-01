#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
  title: [A Spatio-Temporal Graph Neural Network Approach for Bike-Sharing Demand Prediction],
  abstract: [
    Bicycle sharing plays a significant role in the public transportation system. Accurate dynamic demand prediction is the key to support effective re-balancing and provide reliable service. In this study, we adapt and analyze the model spatio-temporal graph attention network (ST-GAT) architecture known from traffic prediction to predict bike-sharing windows and compare it with several baselines.
    
    // analyze scaling behavior of the model (?)
    // analyze transformer variation (?)
    // The Abstract should be a brief version of the full article. It should give the reader an accurate overview. Be brief, but be specific. 
  ],
  authors: (
    (
      name: "Alexander Julius Busch",
      // department: [],
      organization: [University of Hamburg],
      location: [Hamburg, Germany],
      //email: "alexander.busch@"
    ),
    (
      name: "Kaifeng Lu",
      //department: [],
      organization: [University of Toronto],
      location: [Toronto, Canada],
      //email: ""
    ),
  ),
  index-terms: ("AI", "Smart Mobility", "GNN", "Bike-Sharing"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)
    
= Introduction
// summarize the importance of the problem you are trying to solve and the reason that motivated you to select this project. Explain what was the problem or challenge that you were given? state the purpose of the project and how did you solve it? Enumerate the objectives of the project and describe in brief the structure of the article. 

Stations where bicycles are docked and taken from are a common way of implementing a bicycle sharing system. As bicycle sharing becomes one of the most popular ways to commute, it plays a significant role in the public transportation system. According to the Bike Share Toronto 2023 Business Review, the total number of rides in 2023 is estimated to be about 5.5 million, projected to 2025 to become more than 6.2 million @Hanna_2023. The total number of stations deployed is planned to be more than 1, 000 with more than 10, 000 bikes available. However, the demand varies greatly between location, weekday, season and other factors, which leads to imbalances and congestions in the system. This results in customer dissatisfaction and unreliability of the system, jeopardizing the central role of bicycle in reaching emission-neutral transportation and providing convenience. 

To properly implement dynamic solutions, such as adaptive dynamic pricing and terminal extensions, the demand needs to be reliably predicted. Since the demand fluctuates based on various aspects, we decided to train a machine learning model to investigate the relations between these aspects and the demand. With a sufficiently accurate model, bicycle sharing companies can adopt adjustments to provide more reliable service.


= Literature Review
// conduct a critical survey on similar solutions and explain how your solution extends or differs from these solutions. 

== Scope and Overview

Due to the vastly growing number of bike sharing systems, bike sharing demand prediction has been investigated by a number of authors in the last 15 years. While classical machine learning approaches such as regression and boosting have been used in the early developments, in recent years, deep learning approaches have been found to give significant advantages in the forecasting domain @YANG2020101521.
Based on the literature, we compare the following machine learning approaches to bike sharing demand prediction and identify open problems. 


== Paper Summaries 

=== Bike-Sharing Demand Prediction at Community Level under COVID-19 Using Deep Learning @Dastjerdi2022

This article investigates short-term bike-sharing demand forecasting in Montreal. It compares deep learning techniques for predicting bike pickups 15 minutes ahead in six communities identified within the city’s bike-sharing network. The study compares the performance of LSTMs (Long Short Term Memory, a recurrent network type), CNN (Convolutional Neural Network)-LSTM hybrids, and ARIMA (Auto-Regressive Integrated Moving Average). The authors trained the models based on two
main feature attributes: historical demand data in 15-minute intervals and weather conditions. They found that the CNN-LSTM hybrid model outperforms the other models (MAE of 3.00 and RMSE of 4.77), compared to the LSTM model (MAE of 4.48 and RMSE of 6.86), and the ARIMA (MAE of 50.95 and RMSE of 61.17).

==== Pros

• LSTM Models: These were found to capture temporal dependencies effectively with moderate complexity.

• CNN-LSTM Hybrid Models: These were found to combine spatial and temporal learning for improved accuracy in complex data.

• The ARIMA Model is fast to train and easy to interpret.

==== Cons

• LSTM Models: Lacks spatial awareness, limiting prediction accuracy compared to hybrid models.

• CNN-LSTM Hybrid Models: These are computationally expensive and harder to interpret.

• The ARIMA Model was found to perform poorly for complex, nonlinear and disrupted data.

=== Using graph structural information about flows to enhance short-term demand prediction in bike-sharing systems @YANG2020101521

This paper investigates how graph theory can enhance short-term bike-sharing demand forecasting. It focuses on incorporating graph-based features derived from flow interactions such as Out￾strength, In-strength, Out-degree, In-degree, and PageRank to improve prediction accuracy. It compares three machine learning models: XGBoost, Multi-Layer Perceptron (MLP), and LSTM. This study found that including graph-based features significantly improves model performance compared to using traditional features like meteorological data alone. The authors here find that the XGBoost, MLP and LSTM model achieve MAPE values of 27.2%, 27.7%, and 27.0% as well as RMSE values of 6.78, 6.85, and 6.69 correspondingly. The LSTM model was here found to be the most effective model for incorporating complex graph-based features, while the overall performance of the XGBoost is nearly comparable.

==== Pros

• XGBoost excels at handling structured data and provides good performance with less tuning.

• The MLP architecture is simple and efficient with moderate performance.

• The LSTM is best at capturing temporal dependencies and sequential patterns, especially with time-lagged data.

==== Cons

• The XGBoost is less effective at handling complex time dependencies compared to the LSTM.

• The MLP struggles with sequential data and lacks the advanced handling of time dependencies.

• The LSTM is complex to train and requires significant computational resources.

=== Modeling bike-sharing demand using a regression model with spatially varying coefficients @WANG2021103059

This article focuses on investigating how various factors such as land use, socio-demographic attributes, and transportation infrastructure influence bike-sharing demand at different stations. Notably, they define a graph model, where the data for each bike-sharing cluster is accumulated according to its catchment region. The catchment region is calculated using Thiessen polygons, and ensured to be non-overlapping. The authors propose a spatially varying coefficients (SVC) regression model that accounts for local spatial effects, unlike previous regression models that assume the factors are spatially homogeneous.

==== Pros

• The graph model includes connectivity between station clusters to be used.

• The more sophisticated modeling of station clusters defined by catchment areas instead of single stations notably allows to predict demand in regions where no stations are present and thus allows planning.

==== Cons

• As the model does not directly regress on immediate historical data as e.g. the LSTM models, it fundamentally does not to allow accurate future predictions, but instead better fits general analysis purposes.

=== Modeling Bike Availability in a Bike-Sharing System Using Machine Learning @Ashqar2017

This paper explores predicting bike availability at San Francisco Bay Area Bike Share stations using machine learning algorithms. The authors apply three methods: Random Forest (RF) and Least￾Squares Boosting (LSBoost) for univariate regression and Partial Least-Squares Regression (PLSR) for multivariate regression. They found that factors like station neighbors, time of prediction, and weather conditions are significant in predicting bike availability. The RF, LSBoost and PLSR models achieve MAE of 0.37, 0.58 and 0.6 bikes per station correspondingly, i.e. the Random Forest (RF) model here is found to give the best prediction accuracy.

==== Pros

• The RF model is highly accurate with low prediction error and robust to overfitting.

• The LSBoost model is effective for regression tasks with manageable computational complexity.

• The PLSR captures spatial correlations, useful for large networks with interdependent stations.

==== Cons

• RF: Requires independent observations. Performance decreases as prediction horizon increases.

• LSBoost: Higher prediction error. Requires proper regularization against overfitting.

• PLSR: Yields the highest prediction error. Less accurate for smaller networks.

=== Cross-Mode Knowledge Adaptation for Bike Sharing Demand Prediction Using DomainAdversarial Graph Neural Networks @Liang2024

This recent paper focuses on improving the state of the art in bike sharing demand prediction by integrating additional features from other transport modes, here subway and ride-hailing traffic data, all from New York City. In order to integrate this heterogeneous graph structured data, recurrent CNNs are combined with graph convolutions to yield transport embeddings. The paper also introduces an adversarial training principle to learn these embeddings, such that the embeddings learned are optimized to be indistinguishable. These embeddings are then fed into a multiple GNNs, combined into a single representation, and passed to a final prediction layer.

==== Pros

• The architecture developed seems to give a significant improvement to the compared approaches. However, the dataset itself is one of the most often used in bike-share prediction @YANG2020101521, allowing some degree of comparison of results.

• The architecture allows extension to integrate very heterogeneous other modes of transportation significant to bike-sharing that would otherwise be very difficult to integrate.

==== Cons

• The architecture and total approach is complicated, requiring extensive effort to adapt to different formats of data from other systems.

• The data, especially from ride-hailing, is likely more noisy than the bike-sharing trip data (due to e.g. only several providers contributing to the ride-hailing data, and the different companies providing data using their own standards)

• It is unclear how much improvement the architecture would give without the additional features.

• Due to the station structure used for the graph, the model cannot predict demand for only places themselves.

== Taxonomic Table and Tabular Comparison


We organize the used models into a tree based on the dominant design features for a brief overview of the common investigated models for bikesharing demand prediction. The fundamental division is between two lines of work between “classical” approaches and deep approaches, which split into different lines of works. Further, because the more advanced deep models build on LSTM-approaches to model immediate history, this marks another division in the tree. In order to have concise names for the approaches, we denote every approach by the fundamental architectural characteristic and its paper (e.g. LSTM @Dastjerdi2022 denotes the LSTM approach in @Dastjerdi2022), although this does neglect other important differences in the approaches.

#figure(
  caption: [A classification tree of the employed ML approaches],
  image("MethodTree.png", width: 100%)
)<tab:MethodTree>

We summarize the main advantages and disadvantages in the following table. We classify how well each approach integrates spatial, recent historical and connectivity information. For example, the pure LSTM approach handles and generalizes recent historical information well, while the regression does not have explicit integration of these features, it only incorporates historical information via engineered features.

#figure(
  caption: [Comparison of the Employed ML Approaches],
  table(
    // Table styling is not mandated by the IEEE. Feel free to adjust these
    // settings and potentially move them into a set rule.
    columns: 6,
    align: (left),
    inset: (x: 4pt, y: 3pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0  { rgb("#efefef") },

    [Approach],[Spatial Handling],[Temporal Handling],[Connectivity Modeling],[Model Complexity],[Model Performance],
    
    [PLSR @Ashqar2017],[Decent],[Rudimentary],[Rudimentary],[Rudimentary],[Low],
    
    [LSBoost in @Ashqar2017], [Rudimentary], [Rudimentary], [Rudimentary], [Low], [Decent - Good],

    [RF in @Ashqar2017], [Rudimentary], [Rudimentary], [Rudimentary], [Low], [Good],
    
    [XGBoost in @Dastjerdi2022], [Rudimentary], [Rudimentary], [Rudimentary], [Low], [Good],

    [ARIMA in @Dastjerdi2022], [Rudimentary], [Short Term], [Rudimentary], [Low], [Decent],

    [MLP in @Dastjerdi2022], [Rudimentary], [Short Term], [Rudimentary], [Rudimentary], [Decent],

    [SVR @WANG2021103059], [Good], [Rudimentary], [Decent - Good], [Medium (Complex Feature Engineering)], [Good for spatial modeling],

    [DA-MR-GNN @Liang2024], [Decent], [Very good], [Very Good], [Very high], [Very good]
  )
) <tab:MethodTable>

== Directions for Future Research

As noted in @YANG2020101521, there is no single standard benchmarking dataset for the specific objective of bike sharing demand prediction, making it difficult to obtain a precise ranking of the models. Notably, the New York Citibike data has been used in multiple approaches @YANG2020101521, @Liang2024, but the precise data used still changes. As given in @YANG2020101521, several studies have suggested that XGBoost performing as well most of the state of the art approaches, having also won the 2014 Kaggle competition @cukierski2014bike_kaggle, however the best approach was found to depend strongly on the dataset and modeling. Notably @YANG2020101521 suggests, that well-performing deep model architectures in traffic prediction are likely to show good performance in bike-sharing demand prediction. In this line of work, @Liang2024 combines well performing spatio-temporal GNN approach, known from traffic prediction @Yu2017SpatiotemporalGC and @Zhang2022AGT, with a domain-adversarial network. However, little work has been done on the extent that performance of traffic prediction carries over to bike-sharing performance and adaptation of successful models from other adjacent fields, (e.g. models in @Yu2017SpatiotemporalGC, @Zhang2022AGT), appears to be a promising research direction.

= Problem Formulation and Modeling
// describe the include mathematical formulation of the problem and possible modeling approach.

== Formulation
Our goal is to predict the demand of bicycle sharing using a machine learning approach given historical data and additional features, such as the day of the week and the daytime.
 
We consider a bicycle sharing system consisting of $N_S$ stations with fixed capacities $"Cap"_s$, i.e. a maximum of $"Cap"_s$ bicycles can be docked at station $𝑠 ∈ ℕ$. We denote the current number of bicycles at station $𝑠$ by $𝐵_𝑠 (𝑡)$ We distinguish two types of demands:

• $"OutDem"_𝑠 (𝑡)$: the number of bicycles that people take out the dock at (𝑡, 𝑠) if there are enough bicycles present.

• $"InDem"_𝑠 (𝑡)$ : the number of bicycles that people dock at (𝑡, 𝑠) if there is be enough space present.

In order for the number of bicycles to be conserved, a pair of suitable demand functions has to satisfy

$ sum_(𝑡,𝑠)"InDem"_𝑠 (𝑡) − "OutDem"_𝑠 (𝑡) &= 0, $<eq:equa1>

where the sum runs over all times and states.

These functions encode the psychological, social and logistical aspects of bicycle sharing demand. They define the behavior of the system (i.e. the number of bicycles at station 𝑠) via the difference
equation

$ 𝐵_𝑠 (𝑡 + 1) = 𝐵_𝑠 (𝑡) + min("InDem"_𝑠 (𝑡), "Cap"_𝑠 (𝑡)) \
− min("OutDem"_𝑠 (𝑡), 𝐵_𝑠 (𝑡)). $

In order to formulate a proper machine learning problem, we now approximate the discrete quantities by real numbers, i.e. we introduce

• the hourly rates of In-Demand $"ID"_𝑠 (𝑡)∈ℝ$ and out demand $"OD"_𝑠 (𝑡)∈ℝ$

• the actual hourly rates of bicycles docked $"In" ∈ ℝ$ and bicycles taken out $"Out" ∈ ℝ$.

Our given given data consists of a list of all rides including start, end stations and start and and times in minutes. The discrete behavior previously stated now translates to the partially continuous loss

$ "loss"(𝑡) = cases(
  ("ID"(𝑡) − "In")^2 + ("OD"(𝑡) − "Out")^2 \
  "if" 0 < 𝐵(𝑡) < "Cap"(𝑡),,
  ("OD" − "Out")^2 + max(0,"ID" − "In")^2 \
  "if" 𝐵(𝑡) = "Cap"(𝑡),,
  ("ID" − "In")^2 + max(0, "OD" − "Out")^2 \
  "if" 𝐵(𝑡) = 0,
  
) $

i.e. the demands should normally be identically to the rates of the bicycles taken in or out, only if the station is either full or empty, the demands may be higher, but not lower, than the input or output bicycles.

== Modeling

#set list(indent:  0pt,marker: ([•],[--]))

Modeling factors:

- Controllable Parameters: These include the sharpness of the hourly rate approximation, which depends on the window. Choosing a large window to compute the rates leads to loss of precision in the time domain, while too small window might not capture the continuity of the demand well. Other control parameters of the historic data collection include the accuracy of time resolution which in the given dataset is minutes.

- Signals (i.e. input features): To predict the future hourly demand, we input

 -- the historic input and output rates before the prediction time

 -- the daytime

 -- the day of the week.

- Error States (i.e. failure modes of the model):

 -- Possible failure modes include cases if the future prediction are not accurate enough to be useful, the predictions become unstable or the in-demand does not match the out demand.

- Noise Factors:

 -- Capacity changes of the stations during the month, which appears due to e.g. repositioning of stationing.These changes are not included in the published data.

 -- Population density changes, irregular road and facility closures which are not present in the training data but have a significant influence on bicycle sharing behavior.

= Proposed Solution
//: describe the bio-inspired algorithms selected to solve the project problem.
= Performance Evaluation
//: Establish a set of evaluation metrics and run some experiments with different values of algorithm parameters to quantitativelyand qualitatively assess the performance of the developed solution. Students must identify the pros and cons of each technique and assess the quality of work as well as its fit with project objectives.
= Conclusions & Recommendations
//: summarize the conclusion and future improvement. Explain how did you solve the problem; what problems were met? what did the results show? And how to refine the proposed solution? You may organize ideas using lists or numbered points, if appropriate, but avoid making your article into a check-list or a series of encrypted notes.
= Code <sec:code>
// : provide a GitHub permanent link to the code that implements the proposed solution.





