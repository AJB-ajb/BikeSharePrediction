# Generalized Demand Prediction Using Spatio Temporal Graph Attention Networks
- [Our medium article on exploring spatial and temporal patterns of bike sharing in Toronto](https://medium.com/ai4sm/exploring-spatial-patterns-in-torontos-bike-sharing-system-7b5c486ae250)
## Notebooks 
+ [the main reference notebook, illustrating training and loss of the main adapted STGAT model](https://colab.research.google.com/drive/1Pg2e6z50IkK-yZIYnzHQsPGhUfJja5Qx?usp=sharing)
+ [the notebook for the exploratory spatial data analysis](https://colab.research.google.com/drive/1Vmkf_HsPUwCMqX1inZPrSJnfUMiurx1Y?usp=sharing)

## Installation
+ Clone the repository
+ Open the [notebook](./documents/midterm_article/visualizations_article.ipynb)
  + Set `download_data = True` and `process_data = True` and run the cell, this should download and process the relevant data

## Main Data Sources
- Life data and station information: https://open.toronto.ca/dataset/bike-share-toronto/
  - Main station information at https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information
- Historic ridership data: https://open.toronto.ca/dataset/bike-share-toronto-ridership-data/
  - Download for 2024: (https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset/7e876c24-177c-4605-9cef-e50dd74c617f/resource/9a9a0163-8114-447c-bf66-790b1a92da51/download/bikeshare-ridership-2024.zip)

## Other Projects
- Toronto 2021 Data Analysis: https://github.com/dailyLi/toronto_bike_share
