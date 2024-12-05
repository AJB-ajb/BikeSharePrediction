#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
  title: [Generalized Bike-Sharing Demand Prediction Using A Spatio-Temporal GNN],
  abstract: [
    Bike-sharing usage prediction has been implemented and analyzed using a variety of models, ranging from linear and logistic regression models with extensive spatial features [cit], trees to ARIMA [cit]. 
    However, modeling and prediction of the underlying demand function that drives the usage has not been rigorously attempted to the best of our knowledge, possibly due to the ill-posed nature of the problem.
    We propose several defining properties of a demand function and extend the STGAT model from traffic prediction to jointly predict both the actual usage rate as well as a suitable demand function matching the data, which we take from the City of Toronto's open data portal [cit]. We analyze accuracy and behavior of the model in normal usage prediction and illustrate the behavior of the learned demand function in various scenarios. 
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
      name: "",
      department: [],
      organization: [],
      location: [],
      email: ""
    ),
  ),
  index-terms: ("AI", "Smart Mobility", "GNN", "Bike-Sharing"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction
// summarize the importance of the problem you are trying to solve and the reason that motivated you to select this project. Explain what was the problem or challenge that you were given? state the purpose of the project and how did you solve it? 

// Enumerate the objectives of the project and describe in brief the structure of the article. 

// Problem motivation
For an illustration of the difference between demand and usage, consider a bike-sharing system with docks. Bikes can be taken out at a dock and have to be docked in back at any dock, with the price of the ride being proportional to the time between the dockings. Consider a user that wants to go from A to B at some day, possibly to reach their workplace or possibly to spontaneously take a ride to a park as leisure activity. In the case that there are bikes available, the user takes one and the demand is equal to the number of bikes taken out at that station. However, if the station is empty, the user might either walk to a nearby station, take another transportation medium or cancel the planned ride all together. In this case, the demand can not be inferred from the rate, but probably exhibits other regularities: The station might not be full at similar dates and the local demand is greatly defined by people commuting regularly to work, or other complex regularities. However, manually extracting the components of the demand is infeasible: From the author's own experience with the Toronto bike-sharing system, what people do when a station is full depends on weather, close stations, personal and time considerations, daytime and other factors. Thus we propose to use a deep learning approach to learn a suitable demand function from usage data using proper regularization.

For those familiar with the notion of counterfactual reasoning, the demand can be seen as counterfactual quantity: It is the rate of bikes taken in or out if the station would not have been empty at that point.

= Literature Review
// conduct a critical survey on similar solutions and explain how your solution extends or differs from these solutions. 

= Problem Formulation and Modeling
// describe the include mathematical formulation of the problem and possible modeling approaches.
== Axioms of Demand
From the characteristics of the demand above, we postulate the following properties of an ideal demand function:

- If the station is empty at a time, the out-demand should be greater or equal than the bikes taken out.
- If the station is completely full, the in-demand should be lower than the rate of bikes docked.
- Otherwise, the demand should be equal to the in or out rates.
- Both in-demand and out-demand functions should be smooth, i.e. differentiable three times.

== Regularized Loss Function
From the ideal properties, we derive a loss function …
// Axioms of demand

// Figure: Demand capping at various stations

= Proposed Solution
//: describe the bio-inspired algorithms selected to solve the project problem.
 
// critical choice: smoothing window. We use gaussian, σ 10mins, i.e. ≈63% of the information are from ± 10 minutes, 95% are within 20 minutes (predicting the number of bikes taken out in the next 20 minutes has been identified as the most effective interval)
= Performance Evaluation
//: Establish a set of evaluation metrics and run some experiments with different values of algorithm parameters to quantitativelyand qualitatively assess the performance of the developed solution. Students must identify the pros and cons of each technique and assess the quality of work as well as its fit with project objectives.

// Qualitative Comparison:
// Demand function plotted for trained model, compared to prediction
// Demand function plotted as comparison for different regularizers
// Model and linear model performance over different all time horizons
// Table: Model performance
// 
// Investigation of demand
// Predictions of problematic areas
= Conclusions and Recommendations
//: summarize the conclusion and future improvement. Explain how did you solve the problem; what problems were met? what did the results show? And how to refine the proposed solution? You may organize ideas using lists or numbered points, if appropriate, but avoid making your article into a check-list or a series of encrypted notes.
= Code <sec:code>
// : provide a GitHub permanent link to the code that implements the proposed solution.
