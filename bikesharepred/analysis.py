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
from pathlib import Path
import pathlib

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
# create the directories if they don't exist
pathlib.Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

def RAW_DIR(year):
    return DATA_DIR / f'bikeshare-ridership-{year}'

def preprocess_ridership_table(df):
    print("Preprocessing ridership table")

#    df = df[df["Trip  Duration"] != 0]
    print(f"Number of rows with NaN End Station Id: {df['End Station Id'].isna().sum()}")
    df = df[df["End Station Id"].notna()]
    df['End Station Id'] = df['End Station Id'].astype(int)
    df['Start Time'] = pd.to_datetime(df['Start Time'], format='%m/%d/%Y %H:%M')
    df['End Time'] = pd.to_datetime(df['End Time'], format='%m/%d/%Y %H:%M')

    return df


# example date: 01/01/2024 00:00
def parse_time(time_str):
    return pd.to_datetime(time_str, format='%m/%d/%Y %H:%M')


class BikeShareData:
    def __init__(self, name):
        self.name = name
    @staticmethod
    def load(month = '02', year = '2024', force_reprocess = False, σ = 2.5):
        file_end = f'{year}-{month}'
        processed_path = PROCESSED_DIR / f'{file_end}.pkl'

        if force_reprocess or not processed_path.exists():
            data = BikeShareData(file_end)
            # find the csv file; i.e. the file in data / bikeshare-ridership-2024 that ends with {name}.csv 
            raw_data_files = list(RAW_DIR(year).glob(f'*{file_end}.csv'))
            if len(raw_data_files) == 0:
                raise FileNotFoundError(f'No file found for {file_end}')
            
            month_file = raw_data_files[0]

            ridership_table = pd.read_csv(month_file, encoding='cp1252')
            data.process_data(ridership_table, σ = σ)
            pickle.dump(data, open(processed_path, 'wb'))
            
        return pickle.load(open(processed_path, 'rb'))


    def process_data(self, ridership_table, σ = 2.5):
        self.load_current_stations()
        ridership_table = preprocess_ridership_table(ridership_table)
        self.process_data_by_minute(ridership_table)
        self.calculate_in_out_rates(σ_minutes = σ)
        self.calc_mask()
        return ridership_table

    def process_data_by_minute(self, ridership_table):
        """
            Compute numpy arrays indexed by the minute from the beginning time of the data that give the number of bikes taken in and out in each very minute at each station.
            The data size is roughly 45k (minutes in a month) * 1000 (stations) * 2 (in and out) ≈ 400MB in int32 precision.

            Assumes this BikeShareData object has been initialized with the stations and their capacities.
            Assumes `preprocessed` ridership table
            Returns (in_bikes, out_bikes) where in_bikes and out_bikes are numpy arrays of shape (num_stations, num_minutes) with the number of bikes taken in and out at each station in each minute.
        """
        df = ridership_table

        start_time = df['Start Time'].min().replace(second=0, microsecond=0, minute=0)
        end_time = df['End Time'].max() # last time of ride, often many days after the end of the month
        end_of_month = pd.Timestamp(year=start_time.year, month=start_time.month, day=1) + pd.DateOffset(months=1) # after processing, cut off the data at the end of the month
        num_minutes_in_month = (end_of_month - start_time).days * 24 * 60

        num_minutes = math.ceil((end_time - start_time).total_seconds() / 60) + 1
        
        
        # Here, we assume that the station ids are contiguous integers starting from 0
        # In practice, some stations are removed, but these then appear as pure 0 rows in the data
        # Here, take the number of stations from the table, which might differ from month to month; if merging data from multiple months, take the maximum 
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

        self.in_bikes = in_bikes[:, :num_minutes_in_month]
        self.out_bikes = out_bikes[:, :num_minutes_in_month] # cut off the data at the end of the month

        return in_bikes, out_bikes
    
    def calculate_in_out_rates(self, σ_minutes = 2.5):
        """
            Calculate the average number of bikes taken out per minute for each station at each minute by using
            a moving gaussian average.
            Returns (in_rates, out_rates) where in_rates and out_rates are numpy arrays of shape (num_stations, num_minutes)
        """

        # convert types to float32 to calculate moving average
        in_bikes = self.in_bikes.astype(np.float32)
        out_bikes = self.out_bikes.astype(np.float32)
        
        self.in_rates = gaussian_filter1d(in_bikes, sigma=σ_minutes, axis=1, mode='reflect')
        self.out_rates = gaussian_filter1d(out_bikes, sigma=σ_minutes, axis=1, mode='reflect') 
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
    
    def calc_mask(self, Δtol = 2):
        """
            Calculate the at_min_mask and at_max_mask arrays which are boolean arrays of shape (num_stations, num_minutes) that are True if the number of bikes at the station is at max `Δtol` close to the minimum or maximum capacity, respectively.
        """
        Δbikes = np.cumsum(self.in_bikes - self.out_bikes, axis=1)
        
        # go through each day of the month
        num_days = Δbikes.shape[1] // (24 * 60)
        at_min_mask = np.zeros_like(Δbikes, dtype=bool)
        at_max_mask = np.zeros_like(Δbikes, dtype=bool)

        for day in range(num_days):
            day_start = day * 24 * 60
            day_end = (day + 1) * 24 * 60
            day_max = np.max(Δbikes[:, day_start:day_end], axis=1)
            day_min = np.min(Δbikes[:, day_start:day_end], axis=1)
            at_min_mask[:, day_start:day_end] = Δbikes[:, day_start:day_end] - day_min[:, None] <= Δtol
            at_max_mask[:, day_start:day_end] = day_max[:, None] - Δbikes[:, day_start:day_end] <= Δtol

        self.at_min_mask = at_min_mask
        self.at_max_mask = at_max_mask