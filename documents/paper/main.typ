#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
  title: [Learning Demand Functions for Bike-Sharing Using Spatio-Temporal Graph Neural Networks],
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
      name: "Kaifeng Lu",
      // department: [],
      organization: [University of Toronto],
      location: [Toronto, Canada],
      email: ""
    ),
  ),
  index-terms: ("Bike-Sharing", "GNN", "STGAT", "Demand Prediction"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction <sec:intro>
// summarize the importance of the problem you are trying to solve and the reason that motivated you to select this project.
Docked bike-sharing, i.e. a system of stations where bicycles are docked and taken from are a common way of implementing a bicycle sharing system. As bicycle sharing becomes one of the most popular ways to commute, it plays a significant role in the public transportation system. 
Considered for the city of Toronto, according to the Bike Share Toronto 2023 Business Review, the total number of rides in 2023 is estimated to be about 5.5 million, projected to 2025 to become more than 6.2 million @Hanna_2023. The total number of stations deployed is planned to be more than 1, 000 with more than 10, 000 bikes available. However, the demand varies greatly between location, weekday, season and other factors, which leads to imbalances and congestions in the system. This results in customer dissatisfaction and unreliability of the system, jeopardizing the central role of bicycle in reaching emission-neutral transportation and providing convenience. 

To properly implement dynamic solutions, such as adaptive dynamic pricing and terminal extensions, the demand needs to be reliably predicted. Since bike-sharing usage fluctuates nontrivially based on a multitude of different influences, such as weekday, daytime, events, weather and more features, machine learning and especially neural network approaches have been successfully employed to predict general usage @Ashqar2017 @Dastjerdi2022 @Liang2024.

//Explain what was the problem or challenge that you were given? state the purpose of the project and how did you solve it? 
In this article, our goal is to learn a demand function from data that quantitatively encodes user demand. For this, we extend the STGAT @Zhang2022AGT prediction model to simultaneously predict usage and an infered demand function and define an adapted loss function to capture the notion of the demand.
// Enumerate the objectives of the project and describe in brief the structure of the article. 

The structure of this article is as follows:
- We give an overview on other approaches on bike-sharing usage prediction and the STGAT architecture.
- We then mathematize the notion of a demand function and formalize the bike-sharing joint prediction task.
- In the following section, we translate the notion the demand function into a proper regularizer and describe our variation of the STGAT architecture and the data processing.
- We then compare the predictive performance of our model to a linear baseline. We investigate its quality over the prediction horizon and show qualitative prediction results. We qualitatively compare different modeling choices in the resulting demand model and evaluate quantitative aspects of the demand model.

// Problem motivation
== Motivation <sec:motivation>
For an illustration of the difference between demand and usage, consider a bike-sharing system with docks. Bikes can be taken out at a dock and have to be docked in back at any dock, with the price of the ride being proportional to the time between the dockings. Consider a user that wants to go from A to B at some day, possibly to reach their workplace or possibly to spontaneously take a ride to a park as leisure activity. In the case that there are bikes available, the user takes one and the demand is equal to the number of bikes taken out at that station. However, if the station is empty, the user might either walk to a nearby station, take another transportation medium or cancel the planned ride all together. In this case, the demand can not be inferred from the rate, but probably exhibits other regularities: The station might not be full at similar dates and the local demand is greatly defined by people commuting regularly to work, or other complex regularities. However, manually extracting the components of the demand is infeasible: From the author's own experience with the Toronto bike-sharing system, what people do when a station is full depends on weather, close stations, personal and time considerations, daytime and other factors. Thus we propose to use a deep learning approach to learn a suitable demand function from usage data using proper regularization.

For those familiar with the notion of counterfactual reasoning, the demand can be seen as counterfactual quantity: It is the rate of bikes taken in or out if the station would not have been empty at that point. 

= Literature Review <sec:lit_review>
// conduct a critical survey on similar solutions and explain how your solution extends or differs from these solutions. 

= Problem Formulation and Modeling <sec:modeling>
// describe the include mathematical formulation of the problem and possible modeling approaches.
#let (ID, OD) = ("ID", "OD")
#let (IR, OR) = ("IR", "OR")
#let (estIR, estOR) = (math.hat(IR), math.hat(OR))
#let (estID, estOD) = (math.hat(ID), math.hat(OD))
#let (atmax, atmin) = ("atmax", "atmin")
#let IF = " if "
#let dt = "dt"

== Axioms of Demand
From the described characteristics of the demand in @sec:motivation, we postulate the following properties of an ideal demand function, which we translate into a loss minimization problem for machine learning. 

In the following, $IR(s, t), OR(s,t)$ denote the rates of bikes docked in or taken out at station $s ∈ ℕ$ and time $t ∈ ℝ$ and $ID, OD$ the respective demand functions. The predicates $atmax(s,t), atmin(s,t)$ are defined to be $1$ if the station is at its maximum capacity or empty respectively, and $0$ otherwise.

- If the station is empty at a time, the out-demand should be greater or equal than the bikes taken out: $OD(s,t) ≥ OR(s,t) IF atmin(s,t)$
  
- If the station is completely full, the in-demand should be lower than the rate of bikes docked. $OD(s,t) ≥ OR(s,t) IF atmax(s,t)$

- Otherwise, the demand should be equal to the respective in or out rates: 
  - $ID(s,t) = IR(s,t) IF "not" atmax(s,t)$ 
  - $OD(s,t) = OR(s,t) IF "not" atmin(s,t)$

- Both in-demand and out-demand functions should be smooth, i.e. its intuitively it should not change abruptly. This can be modeled in several different ways. We propose to demand that the third derivative is small.:  // motivation (?)
  - $d^3/dt^3 (ID(s,t) + OD(s,t)) ≪ 1$
  - The motivation for this is that the third derivative is the smallest derivative providing a natural shape for demand. Notably, the functions with minimal third derivatives are quadratic functions, while functions minimizing the first or second derivative are constants or affine linear functions respectively, which would be an unnatural way to extend a demand function. The minimization of the third derivative can be seen as analogous to motion planning in robotics, where one minimizes `jerk` (the third derivative of the position) in order to obtain a smooth trajectory with small changes in acceleration.

== Relaxation to Regularized Loss Function
From the ideal properties, we construct an analogous relaxed loss function, which allows to model the problem as minimization problem. We take the mean of the squares the respective differences and discretize the times:
$
  "loss" := \ 1/(N_s ⋅ N_t) sum_(s,t) 
    (estIR - IR)^2 
  + (estOR - OR)^2 \
  + (IR - estID)^2 ⋅ (atmax ⋅ [IR ≥ ID] + 1 - atmax) \
  + (OR - estOD)^2 ⋅ (atmin ⋅ [OR ≥ OD] + 1 - atmin) \
  + α ⋅ [S_3(estID) + S_3(estOD)],
$
where $S_3$ is a numerical finite difference approximation of the third derivative and $α$ is a constant that regulates the strength of the regularization.

= Proposed Solution <sec:solution>
//: describe the bio-inspired algorithms selected to solve the project problem.

In order to solve suitably solve the minimization problem 
// architecture

== Modeling Details
// Demand capping
// Figure: Demand capping at various stations
// 
 
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
