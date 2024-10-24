import pandas as pd
import bisect
from dataclasses import dataclass, field
import pickle
import numpy as np
import networkx as nx

# Load the CSV data into a DataFrame
# note: some of the station ids are parsed as floats, so integer conversion is needed for using them as keys
file_path = '../data/bikeshare-ridership-2024/Bike share ridership 2024-06.csv'
ridership_table = pd.read_csv("../data/bikeshare-ridership-2024/Bike share ridership 2024-01.csv")

# example date: 01/01/2024 00:00
def parse_time(time_str):
    return pd.to_datetime(time_str, format='%m/%d/%Y %H:%M')

initial_time = parse_time(ridership_table['Start Time'].iloc[0])

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
    num_bikes: SparseSequence = field(default_factory=lambda: SparseSequence([(initial_time, 0)]))

    def __post_init__(self):
        self.num_bikes = SparseSequence([(initial_time, self.num_bikes_t0)])

class BikeShareData:
    def __init__(self, name):
        self.stations = dict() # station_id -> BikeStation
        self.name = name

    def load_data(self, ridership_table):
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

