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
    df = df[df["Trip  Duration"] != 0]
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

#initial_time = parse_time(ridership_table['Start Time'].iloc[0])

# model a sequence, where the value changes infrequently
class SparseSequence:
    def __init__(self, events):
        self.events = events # list of (time, new value)

    def __getitem__(self, index):
        return self.events[index]
    
    def get_value(self, t):
        # find the most recent event that is <= t
        index = bisect.bisect_right(self.events, t, key=lambda ev: ev[0]) - 1
        return self.events[index] if index >= 0 else None
    def set(self, t, value):
        index = bisect.bisect_right(self.events, t, key=lambda ev: ev[0])
        if index > 0 and self.events[index-1] == value: # if there is an event at this time, update it
            self.events[index-1] = (t, value)
        else:
            self.events.insert(index, (t, value))
    def __len__(self):
        return len(self.events)
    def __repr__(self):
        return f'SparseSequence({self.events})'
    
    


# data model
@dataclass
class BikeStation:
    station_id: int
    name: str
    lat: float = float('NaN')
    lon: float = float('NaN')
    capacity: int = -1
    num_bikes_t0: int = 0
    num_bikes: SparseSequence = SparseSequence([])

    def __post_init__(self):
        self.num_bikes = SparseSequence([])

class BikeShareData:
    def __init__(self, name):
        self.stations = dict() # station_id -> BikeStation
        self.name = name

    def _process_data_by_minute(self, ridership_table):
        """
            Compute numpy arrays indexed by the minute from the beginning time of the data that give the number of bikes taken in and out in each very minute at each station.
            The data size is roughly 45k (minutes in a month) * 1000 (stations) * 2 (in and out) ≈ 400MB in int32 precision.

            Assumes this BikeShareData object has been initialized with the stations and their capacities.
            Assumes `preprocessed` ridership table
            Returns (in_bikes, out_bikes) where in_bikes and out_bikes are numpy arrays of shape (num_stations, num_minutes) with the number of bikes taken in and out at each station in each minute.
        """
        df = ridership_table

        start_time = df['Start Time'].min()
        end_time = df['End Time'].max()
        num_minutes = math.ceil((end_time - start_time).total_seconds() / 60) + 1
        # Here, we assume that the station ids are contiguous integers starting from 0
        # In practise, some stations are removed, but these then appear as pure 0 rows in the data
        # Here, take the number of stations from the table, which might differ from month to month; if merging data from multiple months, take the maximum 
        num_stations = max(df['Start Station Id'].nunique(), df['End Station Id'].nunique())

        station_id0 = min(df['Start Station Id'].min(), df['End Station Id'].min())
        station_id_end = max(df['Start Station Id'].max(), df['End Station Id'].max())
        num_stations = station_id_end - station_id0 + 1

        start_dt_in_minutes = ((df['Start Time'] - start_time).dt.total_seconds() / 60).astype(np.int32)
        end_dt_in_minutes = ((df['End Time'] - start_time).dt.total_seconds() / 60).astype(np.int32)

        df2 = df.copy()
        df2['Start Time'] = start_dt_in_minutes
        df2['End Time'] = end_dt_in_minutes


        in_bikes, out_bikes = np.zeros((num_stations, num_minutes), dtype=np.float32), np.zeros((num_stations, num_minutes), dtype=np.float32) # for calculations later, we use float anyway

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
        in_rates = gaussian_filter1d(self.in_bikes, sigma=σ_minutes, axis=1, mode='reflect') # / σ_minutes
        out_rates = gaussian_filter1d(self.out_bikes, sigma=σ_minutes, axis=1, mode='reflect') # / σ_minutes

        return in_rates, out_rates




    def load_current_stations(self):
        r = requests.get('https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information')

        bikeshare_stations = pd.DataFrame(json.loads(r.content)['data']['stations'])[['station_id', 'name', 'lat', 'lon']].astype({
          'station_id': 'int',
            'name': 'string',
        })
        for i_row, row in bikeshare_stations.iterrows():
            station_id = row['station_id']
            if station_id not in self.stations:
                self.stations[station_id] = BikeStation(station_id, row['name'], row['lat'], row['lon'])
            else: # update the lat, lon, name
                self.stations[station_id].lat = row['lat']
                self.stations[station_id].lon = row['lon']
                self.stations[station_id].name = row['name']
            

    def load_data(self, ridership_table):
        initial_time = parse_time(ridership_table['Start Time'].iloc[0])
        for i_row, row in ridership_table.iterrows():
            start_station_id = int(row['Start Station Id'])
            end_station_id = int(row['End Station Id'])
            
            if start_station_id not in self.stations:
                self.stations[start_station_id] = BikeStation(start_station_id, row['Start Station Name'])

            if end_station_id not in self.stations:
                self.stations[end_station_id] = BikeStation(end_station_id, row['End Station Name'])
            
            start_station = self.stations[start_station_id]
            start_time = parse_time(row['Start Time'])
            start_station.num_bikes.set(start_time, start_station.num_bikes.get_value(start_time)[1] - 1)

            end_time = parse_time(row['End Time'])
            end_station = self.stations[end_station_id]
            end_station.num_bikes.set(end_time, end_station.num_bikes.get_value(end_time)[1] + 1)

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
        for station in self.stations.values():
            # we assume here that the capacity of every station is fully utilized at some point in the data and the bikes are empty at some point
            bike_max = np.max([num_bikes for _, num_bikes in station.num_bikes.events])
            bike_min = np.min([num_bikes for _, num_bikes in station.num_bikes.events])
            station.capacity = bike_max - bike_min
            station.num_bikes_t0 = -bike_min
            station.num_bikes.events = SparseSequence([(time, bikes + station.num_bikes_t0) for (time, bikes) in station.num_bikes.events])
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

