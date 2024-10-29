import pandas as pd
import bisect
from dataclasses import dataclass, field
import pickle
import numpy as np
import networkx as nx
import requests
import json
import math
from scipy.ndimage import gaussian_filter1d

def preprocess_ridership_table(df):
    print("Preprocessing ridership table")

#    df = df[df["Trip  Duration"] != 0]
    print(f"Number of rows with NaN End Station Id: {df['End Station Id'].isna().sum()}")
    df = df[df["End Station Id"].notna()]
    df['End Station Id'] = df['End Station Id'].astype(int)
    df['Start Time'] = pd.to_datetime(df['Start Time'], format='%m/%d/%Y %H:%M')
    df['End Time'] = pd.to_datetime(df['End Time'], format='%m/%d/%Y %H:%M')

    return df

# Load the CSV data into a DataFrame
# note: some of the station ids are parsed as floats, so integer conversion is needed for using them as keys
#file_path = '../data/bikeshare-ridership-2024/Bike share ridership 2024-06.csv'
ridership_table = pd.read_csv("../data/bikeshare-ridership-2024/Bike share ridership 2024-02.csv", encoding='cp1252')

# example date: 01/01/2024 00:00
def parse_time(time_str):
    return pd.to_datetime(time_str, format='%m/%d/%Y %H:%M')


class BikeShareData:
    def __init__(self, name):
        self.name = name

    def process_data_and_save(self, ridership_table):
        self.load_current_stations()
        ridership_table = preprocess_ridership_table(ridership_table)
        self._process_data_by_minute(ridership_table)
        self.calculate_in_out_rates()
        self.calc_capacities()
        #self.calc_graph()
        self.save_pickle()
        return ridership_table

    def _process_data_by_minute(self, ridership_table):
        """
            Compute numpy arrays indexed by the minute from the beginning time of the data that give the number of bikes taken in and out in each very minute at each station.
            The data size is roughly 45k (minutes in a month) * 1000 (stations) * 2 (in and out) ≈ 400MB in int32 precision.

            Assumes this BikeShareData object has been initialized with the stations and their capacities.
            Assumes `preprocessed` ridership table
            Returns (in_bikes, out_bikes) where in_bikes and out_bikes are numpy arrays of shape (num_stations, num_minutes) with the number of bikes taken in and out at each station in each minute.
        """
        df = ridership_table

        start_time = df['Start Time'].min().replace(second=0, microsecond=0, minute=0)
        end_time = df['End Time'].max()
        num_minutes = math.ceil((end_time - start_time).total_seconds() / 60) + 1
        # Here, we assume that the station ids are contiguous integers starting from 0
        # In practise, some stations are removed, but these then appear as pure 0 rows in the data
        # Here, take the number of stations from the table, which might differ from month to month; if merging data from multiple months, take the maximum 
        # num_stations = max(df['Start Station Id'].nunique(), df['End Station Id'].nunique())  # old
        # Here, use the stations from the current json data which gives some more stations than included but allows merging all months
        num_stations = len(self.stations)
        station_id0 = 7000

        start_dt_in_minutes = ((df['Start Time'] - start_time).dt.total_seconds() / 60).astype(np.int32)
        end_dt_in_minutes = ((df['End Time'] - start_time).dt.total_seconds() / 60).astype(np.int32)

        df2 = df.copy()
        df2['Start Time'] = start_dt_in_minutes
        df2['End Time'] = end_dt_in_minutes


        in_bikes, out_bikes = np.zeros((num_stations, num_minutes), dtype=np.int16), np.zeros((num_stations, num_minutes), dtype=np.int16)

        # group every ride by start station and start time and then count the number of rides out from each station in each minute
        # same for the end stations

        grouped_out = df2.groupby(['Start Station Id', 'Start Time']).count()
        grouped_in = df2.groupby(['End Station Id', 'End Time']).count()

        in_bikes[grouped_in.index.get_level_values(0) - station_id0, grouped_in.index.get_level_values(1)] = grouped_in['Trip Id'].values
        out_bikes[grouped_out.index.get_level_values(0) - station_id0, grouped_out.index.get_level_values(1)] = grouped_out['Trip Id'].values

        self.in_bikes, self.out_bikes = in_bikes, out_bikes

        return in_bikes, out_bikes
    
    def calculate_in_out_rates(self, σ_minutes = 15):
        """
            Calculate the average number of bikes taken out per minute for each station at each minute by using
            a moving gaussian average.
            Returns (in_rates, out_rates) where in_rates and out_rates are numpy arrays of shape (num_stations, num_minutes)
        """
        # todo: find a suitable σ, such that the rates are smooth enough to reflect demand patterns, but local enough to capture important patterns
        # probably between 5 and 60 minutes

        # convert types to float32 to calculate moving average
        in_bikes = self.in_bikes.astype(np.float32)
        out_bikes = self.out_bikes.astype(np.float32)
        
        self.in_rates = gaussian_filter1d(in_bikes, sigma=σ_minutes, axis=1, mode='reflect') # / σ_minutes
        self.out_rates = gaussian_filter1d(out_bikes, sigma=σ_minutes, axis=1, mode='reflect') # / σ_minutes
        self.σ_minutes = σ_minutes

        return self.in_rates, self.out_rates


    def load_current_stations(self):
        r = requests.get('https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information')

        self.stations = pd.DataFrame(json.loads(r.content)['data']['stations'])[['station_id', 'name', 'lat', 'lon', 'capacity', 'altitude', 'is_charging_station']].astype({
          'station_id': 'int',
            'name': 'string',
        })
        self.stations.sort_values('station_id', inplace=True)
        min_station_id = self.stations['station_id'].min()
        max_station_id = self.stations['station_id'].max()

        all_station_ids = set(range(min_station_id, max_station_id + 1))
        existing_station_ids = set(self.stations['station_id'])
        missing_station_ids = all_station_ids - existing_station_ids

        missing_stations = pd.DataFrame({
            'station_id': list(missing_station_ids),
            'name': ['Unknown'] * len(missing_station_ids),
            'lat': [np.nan] * len(missing_station_ids),
            'lon': [np.nan] * len(missing_station_ids),
            'capacity': [0] * len(missing_station_ids)
        })

        self.stations = pd.concat([self.stations, missing_stations]).sort_values('station_id').reset_index(drop=True)
        print(f'Loaded {len(self.stations)} stations')
    

    def save_pickle(self, file_path = None):
            if file_path is None:
                file_path = f'../data/{self.name}.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
                
    def load_from_pickle(file_path = None, name = None):
        if name is None:
            return pickle.load(open(file_path, 'rb'))
        else:
            return pickle.load(open(f'../data/{name}.pkl', 'rb'))
    
    def calc_capacities(self):
        # we assume here that the capacity of every station is fully utilized frequently at some point in the data and the bikes are empty at some point
        # this should be true for most stations that are frequently used
        # We assume that the capacity stays constant over the month
        # We estimate the capacity as the median of ((max_day - min_day)_day for day in month)
        
        cum_in = np.cumsum(self.in_bikes, axis=1)
        cum_out = np.cumsum(self.out_bikes, axis=1)
        Δbikes = cum_in - cum_out # the additional number of bikes relative to the number of bikes at the start of the data
        self.capacities = np.max(Δbikes, axis=1) - np.min(Δbikes, axis=1)
        self.N_bikes = Δbikes - np.min(Δbikes, axis=1)[:, None] # the number of bikes at each station

            
    def calc_graph(self):
        # Create a directed graph
        G = nx.DiGraph()

        # Initialize the graph with nodes for each station
        for station_id, station in self.stations.items():
            G.add_node(station_id, name=station.name, capacity=station.capacity, lat=station.lat, lon=station.lon)

        # Add edges with weights representing the number of trips between stations
        for i_row, row in ridership_table.iterrows():
            start_station_id = row['Start Station Id']
            end_station_id = row['End Station Id']
            if G.has_edge(start_station_id, end_station_id):
                G[start_station_id][end_station_id]['weight'] += 1
            else:
                G.add_edge(start_station_id, end_station_id, weight=1)

        self.graph = G

