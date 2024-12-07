#import "@preview/charged-ieee:0.1.3": ieee

// ------ General Styling ---------
#show link: underline

#set table(align: left, inset: (x: 4pt, y: 3pt), stroke: (x, y) => if y <= 1 { (top: 0.5pt)}, fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0  { rgb("#efefef") })

#show: ieee.with(
  title: [Learning Demand Functions for Bike-Sharing Using Spatio-Temporal Graph Neural Networks],
  abstract: [
    Bike-sharing usage prediction has been implemented and analyzed using a variety of models, ranging from linear and logistic regression models with extensive spatial features [cit], decision features, ARIMA [cit] to deep learning models with convolutional [cit] and graph features [cit]. 
    However, modeling and prediction of the underlying demand function that drives the usage has not been rigorously attempted to the best of our knowledge, possibly due to the ill-posed nature of the problem. Our goal is to learn a demand function from data, that is suitable for short term demand prediction, such as adaptive pricing applications.
    We propose defining properties of a demand function and extend the biologically inspired Spatio-Temporal Graph Neural Network (STGAT) architecture from traffic prediction to jointly predict both the actual usage rate and a suitable demand function matching the data, which we take from the City of Toronto's open data portal [cit]. We analyze predictive performance and measures of demand prediction comparing three variations of the architecture and show the qualitative behavior of the learned demand function. 
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
Considered for the city of Toronto, according to the Bike Share Toronto 2023 Business Review, the total number of rides in 2023 is estimated to be about $5.5$ million, projected to 2025 to become more than $6.2$ million @Hanna_2023. The total number of stations deployed is planned to be more than $1,000$ with more than $10, 000$ bikes available. However, the demand varies greatly between location, weekday, season and other factors, which leads to imbalances and congestions in the system. This results in customer dissatisfaction and unreliability of the system, jeopardizing the central role of bicycle in reaching emission-neutral transportation and providing convenience. 

To properly implement dynamic solutions, such as adaptive dynamic pricing and terminal extensions, the demand needs to be reliably predicted. Since bike-sharing usage fluctuates nontrivially based on a multitude of different influences, such as weekday, daytime, events, weather and more features, machine learning and especially neural network approaches have been successfully employed to predict general usage @Ashqar2017 @Dastjerdi2022 @Liang2024. However, only predicting the number of bikes taken in or out is not sufficient for implementing a dynamic pricing system: For example, when a small station is predicted to be completely full at a time, how much should we incentivize taking out bikes? The demand in this case could range from zero (e.g. in the night) to almost arbitrarily high (e.g. during rush-hours in the day), but this quantity can not be directly computed from a purely predictive model.

//Explain what was the problem or challenge that you were given? state the purpose of the project and how did you solve it? 
In this article, our goal is to learn a demand function from data that quantitatively encodes user demand. For this, we extend the STGAT @Zhang2022AGT prediction model to simultaneously predict usage and an infered demand function and define an adapted loss function to capture the notion of the demand. Notably, it is not our goal to outperform other purely predictive approaches, but improve the learned demand function via simultaneous learning of the pure prediction task and verify the model capability.
// Enumerate the objectives of the project and describe in brief the structure of the article. 

The structure of this article is as follows:
- We give an overview on other approaches on bike-sharing usage prediction and the STGAT architecture.
- We then mathematize the notion of a demand function and formalize the bike-sharing joint prediction task.
- In the following section, we translate the notion of a demand function into a proper regularizer and describe our variation of the STGAT architecture and the data processing.
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

- Both in-demand and out-demand functions should be smooth, i.e. its intuitively it should not change abruptly. This can be modeled in several different ways. We propose to demand that the third derivative is small, i.e. $
d^3/dt^3 (ID(s,t) + OD(s,t)) ≪ 1.$

  The motivation for this is that the third derivative is the smallest derivative providing a natural shape for demand. Notably, the functions with minimal third derivatives are quadratic functions, while functions minimizing the first or second derivative are piecewise constant or affine linear functions respectively, which would be an unnatural way to extend a demand function. The minimization of the third derivative can be seen as analogous to motion planning in robotics, where one minimizes `jerk` (the third derivative of the position) in order to obtain a smooth trajectory with small changes in acceleration.

== Relaxation to Loss Function
#let (RateError, DemViol, SmoothViol, loss) = ("RateError", "DemViol", "SmoothViol", "Loss")
From the ideal properties, we construct an analogous relaxed loss function, which allows to model the problem as minimization problem. 
We define the loss in terms of three components:
- the rate error $(estIR - IR)^2 + (estOR - OR)^2$
- the demand constraint violation 
$
  (IR - estID)^2 ⋅ (atmax ⋅ [IR ≥ ID] + 1 - atmax) \
  + (OR - estOD)^2 ⋅ (atmin ⋅ [OR ≥ OD] + 1 - atmin) \ $
- the smoothness violation $
S_3(estID)^2 + S_3(estOD)^2
,$ where $S_3$ is a numerical finite difference approximation of the third derivative.

The final loss is now formed as mean over stations and times of the previous terms:
$
  \ 1/(N_s ⋅ N_t) sum_(s,t) RateError + DemViol + α ⋅ SmoothViol,
$ where $α ∈ ℝ$ is a regularization constant and we used abbreviations to denote the error components.

= Proposed Solution <sec:solution>
//: describe the bio-inspired algorithms selected to solve the project problem.

// STGAT architecture and advantages
// describe features of graph neural networks
In order to suitably solve the minimization problem, we adapt the spatio-temporal graph attention neural network architecture (STGAT) from @Kong_STGAT_2020. We base our adaptation on the open implementation from @wang_course_project. 
// describe bio inspired aspects
Generally, a graph neural network (GNN) is a biologically inspired approach to machine learning, which operates on graph-structured data. The fundamental layer employed in a graph neural network is a graph convolutional layer (GCN), which computes output node features by computing and then aggregating features from each incoming node. The graph attention layer (GAT) employed in the STGAT is an extension of this layer, which weights the computed features by a learned attention score in order to compute a more refined representation. Notably, the fundamental principle of an attention mechanism, itself biologically inspired, has proven to be successful in many machine learning domains. 

The STGAT architecture in @Kong_STGAT_2020 is constructed to predict car traffic velocities at measurement points, given historical velocities over all measurement points as graph.
We adapt the final linear layer to give bike in-, out-rate and in-, out-demand predictions. We choose to investigate the STGAT architecture because of its performance in predicting roughly the next $45$ minutes, given the last hour of information, which is a  time horizon that would be useful for demand prediction for a dynamic pricing problem, and the similarity of the problems. 

Additionally, because bike-sharing depends significantly on time feature information [@EREN2020101882], 

The base network consists of the following layers:
- A graph attention layer with $8$ heads, followed by dropout.
- An LSTM layer with hidden size $32$, which takes the reshaped historic data over all nodes. As modification of the original architecture, we concatenate additional daytime and day of week features to this layer's input. We encode both with sinusoidal encodings due to their periodic nature.
- A second LSTM layer with hidden size $128$.
- A final linear layer operating on the last LSTM prediction, outputting information for all $9$ timepoints.

// precise description; also transformer variation
// architecture: straight diagram [td]

== Modeling Details
We use the historic bike-sharing data provided from the city of Toronto's open data portal. For all computations, we use the data for the full month May 2024. For the station geographical locations, we use the current live information provided. As of November $2024$, there are $861$ stations in the data. Notably, because historic station capacities are not given and station capacities frequently change due to extension and relocation, we have to estimate when a station is full or empty from the bikes taken in and out. For an exploratory spatial data analysis of Toronto's bike-sharing system, where we also cover other details of the data processing, see our #link("https://medium.com/ai4sm/exploring-spatial-patterns-in-torontos-bike-sharing-system-7b5c486ae250", "Medium article").

=== Station Occupancy Estimation
The behavior at most stations follows a clear daily pattern, where bikes show wave-like patterns, however, the cumulative number of bikes either increases or declines over the month. This is probably due to bikes taken out for repairing or relocation, which are not logged in the ridership data. 
In order to estimate when a station is empty or full, we thus take daily minima and maxima and consider the station full if the cumulative number of bikes at that timepoint is within $2$ bikes close to the maximum or minimum, respectively, as marked in @figure:cap. Notably, we aim to err on the side of overestimation, because if a station is almost full at a time, usually users remember this behavior and refrain from using this station during that time, although the demand exists.

#figure(image("imgs/cum_bikes_daily_with_capacity_marked.png"), caption: [Cumulative number of bikes in a station over several days with estimated capacity bounds marked.]) <figure:cap>

=== Temporal Horizon and Rate Calculation
In modeling bike sharing demand, the prediction time horizon and the averaging horizon fundamental parameters. In @Dastjerdi2022, the number of pickups in $15$ minutes was found to be reliably predictable, which is equivalent to predicting a uniform average over $15$ minutes. 
In our approach, we follow @Kong_STGAT_2020 and predict $9$ timepoints in $5$ minute intervals over the next $45$ minutes, given $12$ timepoints in the past, for all stations at the same time. Due to our averaging approach, which averages over partial timepoints in the future, we choose the standard evaluation horizon to be $20$ minutes, which has in @Ashqar2017 also been found to be the most effective horizon for prediction using standard ML models.
Because we are interested in finding a suitable demand function over time, which is related to rate data, a way of averaging the bike pickups has to be chosen. Notably, if one chooses to predict the exact number of bikes taken out or in each minute, one finds almost random behavior, because whether one arrives a minute later or not depends on many other factors, which are insignificant to the demand. Thus, we choose to average with a gaussian filter with $σ = 10 "min"$ to calculate rate information. For this standard deviation of $10$ minutes, thus roughly $63%$ of the information accumulated lies in the interval $±10 "min"$ around each datapoint and $≈95%$ in the $±20 "min"$ interval. Empirically, this interval yields nontrivial prediction results. (Notably, smoothing with $σ = 60 "min"$ renders the prediction task trivial, yielding similar accuracies for both linear and more complex models.)

 // critical choice: smoothing window. We use gaussian, σ 10mins, i.e. ≈63% of the information are from ± 10 minutes, 95% are within 20 minutes (predicting the number of bikes taken out in the next 20 minutes has been identified as the most effective interval)
= Performance Evaluation <sec:evaluation>
For the evaluation of our demand prediction task, we choose to compare three reference models:
- The base STGAT model, with number of nodes and final output size adapted to the problem, without dropout.
- An upscaled and regularized variant, with LSTM sizes $(128, 256)$, dropout of $0.95$ and weight decay $0.4$.
- A variation, where we replace the LSTMs by a decoder-only transformer with $4$ layers, $8$ attention heads, and an embedding size of $32$.

We split the whole month data into separate days and use approximately $70%$ of these for training, $15%$ for validation for hyperparameter optimization and the rest for testing.
Notably, the predictive performance of the models depends strongly on the exact split chosen, which is likely due to the limited data investigated, as the testing days might fall on special holidays or other unusual days, where the behavior is significantly different than in the training set. However, this split ensures the model is tested on fully unseen days, as opposed to only unseen segments of these.

//: Establish a set of evaluation metrics and run some experiments with different values of algorithm parameters to quantitatively and qualitatively assess the performance of the developed solution. Students must identify the pros and cons of each technique and assess the quality of work as well as its fit with project objectives.
We evaluate

// Quantitative Comparison
In the following, if not noted otherwise, the root mean squared error (RMSE) and mean absolute error (MAE) will always be in $["Bikes"/"Hour"]$, while the mean squared error (MSE) is in $["Bikes"/"Hour"]^2$ and the loss is dimensionless. 

#figure(
table(columns: 5,
  table.header([Model], [RMSE], [MSE], [MAE], [Loss]),
  [Linear], [$2.28$], [$6.28$], [$1.64$], [$2.60$],
  [STGAT], [$1.55$], [$3.07$], [$0.857$], [$0.787$],
  [STGAT-upscaled], [$1.55$], [$3.09$], [$0.868$], [$0.812$],
  [STGAT-transformer], [$1.57$], [$3.11$], [$0.868$], [$0.8$]
),
caption: [Quantitative comparison of the predictive quality of models.]
)


#figure(
table(columns: 4,
  table.header([Model], [RDemMSViol], [RSmoothViol], [MADemRateDiff]),
  [Linear], [$2.83$], [$0.0603$], [$4.65$],
  [STGAT], [$1.27$], [$0.00529$], [$1.08$],
  [STGAT-upscaled], [$1.29$], [$0.00706$], [$1.23$],
  [STGAT-transformer], [$1.29$], [$0.00543$], [$1.08$]
  ),
  
  caption: [Comparison of demand metrics.]
)


// Qualitative Comparison:
// Demand function plotted for trained model, compared to prediction
// Demand function plotted as comparison for different regularizers, smoothness of demand function

// Root Demand violation in evaluation in N_bikes / hour 
// Smoothness violation in evaluation set in Bikes / hour^4

// Model and linear model performance over different all time horizons
// Table: predictive model performance
// Quantitative metrics
// difference of demand prediction versus true rate

// Investigation of demand
// Predictions of problematic areas
= Conclusions and Recommendations <sec:conclusion>
//: summarize the conclusion and future improvement. Explain how did you solve the problem; what problems were met? what did the results show? And how to refine the proposed solution? You may organize ideas using lists or numbered points, if appropriate, but avoid making your article into a check-list or a series of encrypted notes.

// Future Work and Improvement:
// Develop a dynamic pricing model based on the demand prediction and a simulation using the station data
// Train and compare on full data (multiple years)
// Adapt deeper graph-neural network approaches with edge features to improve generalization to extended bike networks
// add weather, seasonal and holiday features

= Code <sec:code>
// : provide a GitHub permanent link to the code that implements the proposed solution.

#if false [
= Appendix


]
