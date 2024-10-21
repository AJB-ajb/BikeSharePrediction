# Regression with Spatially Varying Coefficients

- Data Source: They use data from the BIXI Service Montreal.
  
- Incorporation of spatial features: Wang et al. want to investigate the influence of land use, social-demographic and transportation affect the bike-sharing demand at different stations.
- modeled variable: average hourly departure demand at each station
- Depth: Shallow Approach, does not learn intelligent features.

## Used Features (Influential Factors):
- they use Thiessen polygons and circle buffers to calculate the catchment area of a station and use this 

- trip durations and times, Montreal 2019, 5.6M trip records
- All of the following in catchment area of a station
- Population in catchment area per station
- Number of commercial POI
- ratios of park in catchment area
- binary variables for University, Metro station
- log(number of bus routs + 1)
- total road length 
- Land use factors: Commercial | Service | Government
- Walk Score: measure of walkability
- Cycle path proportion
- Capacity
- POI: Points of Interest

## Temporal Modeling
- This approach does not feed the model with immediate previous data.

## Spatial Modeling
- 
   
- Pros: The SVR approach Wang et. al uses here is well suited to explain the heterogeneous effects of a specific factor on different stations.

- Cons: This model is only well suited to model the potential demand for new stations, not future observations.