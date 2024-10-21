# Using Graph Structural Information to Enhance (Yang 2020)

## Total Approaches
- XGBoost, MLP, LSTM

Central Finding: graph-based features improve all modeling approaches compared and are more important than usual meteorological data

Deep learning approaches were found to better learn time-lagged graph variables, giving more accurate forecasting.

## Data Used
New York Citi Bike (2016-17 ; 785 Stations)

Chicago Divvy Bike (2016 - 2017, 569 stations)


## Features
From travel-flow graph
- out-degree (number of connected nodes)
- in- degree
- in-strength: total weight of ingoing trips (Here, this corresponds to the total number, this can be interpreted as demand.)
- PageRank

- Departure time

- predict short-term bike demand

## Temporal Modeling

## Spatial Modeling

## Additional Findings:
Two central modeling approaches: prediction at individual station level or aggregated groups or areas.
Advantages of the second approach: works also for dockless stations; allows changing stations; less noise in the data