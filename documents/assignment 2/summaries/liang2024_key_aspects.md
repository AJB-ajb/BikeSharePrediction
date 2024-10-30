# Cross-Mode Knowledge Adaptation for Bike Sharing Demand Prediction Using Domain-Adversarial Graph Neural Networks

## Motivation
- Graph neural network generally achieve state of the art performance in bike demand prediction tasks as of 2024 (?)
- The precise usage of bike-sharing systems strongly depends on the availability of other local transportation modes, such as subways or buses. Thus, incorporating this information into the model is desirable.
- The authors define multiple graphs encoding the specific spatial dependencies and feed this into a multi-relational graph neural network

## Data Description
NYC Citi bike trip records, NYC Subway passenger usage data, NYC Ride-Hailing trip data.

The paper compares their MRGNN approach with several models.

## Discussion
Pros. The claim made that integrating subway and ride-hailing information significantly influences ride-sharing usage and thus improves prediction is intuitively consistent.

Cons.
However the integration of this multimodal data, with its own  severely complicates the model and 


