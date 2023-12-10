#!/usr/bin/env python
# coding: utf-8

# In[28]:


import os
import pandas as pd
username = 'pankhuri'
file_name = 'dataset-3.csv'

file_path = os.path.join("/Users", username, "Downloads", file_name)

df = pd.read_csv(file_path)

file_path = os.path.join("~", "Downloads", "dataset-3.csv")
file_path = os.path.expanduser(file_path)  
df = pd.read_csv(file_path)


# In[3]:


#Question 1: Distance Matrix Calculation

import networkx as nx
def calculate_distance_matrix(csv_file_path):

    # Creating a directed graph to represent the relationships between toll locations
    G = nx.DiGraph()

    # Populating the graph with edge weights from the DataFrame
    for _, row in df.iterrows():
        G.add_edge(row[0], row[1], weight=row[2])
        G.add_edge(row[1], row[0], weight=row[2])  # Bidirectional edge

    # Creating a symmetric graph by adding reverse edges
    G = G.to_undirected()

    # Calculating the shortest path lengths between toll locations
    distance_matrix = nx.floyd_warshall_numpy(G)

    # Creating a DataFrame from the distance matrix
    distance_df = pd.DataFrame(distance_matrix, index=G.nodes, columns=G.nodes)

    return distance_df


csv_file_path = 'dataset-3.csv'  
resulting_distance_matrix = calculate_distance_matrix(csv_file_path)
print(resulting_distance_matrix)


# In[8]:


#Question 3: Finding IDs within Percentage Threshold

import pandas as pd

def find_ids_within_ten_percentage_threshold(df, reference_value):
    # Filtering the DataFrame based on the reference_value
    reference_df = df[df['id_start'] == reference_value]

    # Calculating the average distance for the reference value
    average_distance = reference_df['distance'].mean()

    # Calculating the lower and upper bounds within 10% of the average distance
    lower_bound = 0.9 * average_distance
    upper_bound = 1.1 * average_distance

    # Filtering the DataFrame to include rows within the threshold
    within_threshold_df = df[(df['id_start'] != reference_value) & (df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]

    # Getting the unique values from the 'id_start' column and sorting them
    result_ids = sorted(within_threshold_df['id_start'].unique())

    return result_ids


reference_value = 1001400
result_ids = find_ids_within_ten_percentage_threshold(df, reference_value)
print(result_ids)


# In[30]:


#Question 2: Unroll Distance Matrix

import pandas as pd
import networkx as nx

def calculate_distance_matrix(csv_file_path):
 

    # Creating a directed graph to represent the relationships between toll locations
    G = nx.DiGraph()

    # Populating the graph with edge weights from the DataFrame
    for _, row in df.iterrows():
        G.add_edge(row.index[0], row.index[1], weight=row.values[2])
        G.add_edge(row.index[1], row.index[0], weight=row.values[2])  # Bidirectional edge

    # Creating a symmetric graph by adding reverse edges
    G = G.to_undirected()

    # Calculating the shortest path lengths between toll locations
    distance_matrix = nx.floyd_warshall_numpy(G)

    # Creating a DataFrame from the distance matrix
    distance_df = pd.DataFrame(distance_matrix, index=G.nodes, columns=G.nodes)

    return distance_df

def unroll_distance_matrix(distance_matrix):
    # Creating a DataFrame to store unrolled distances
    unrolled_distances = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    # Iterate through the rows of the distance_matrix
    for i in distance_matrix.index:
        for j in distance_matrix.columns:
            if i != j:
                unrolled_distances = unrolled_distances.append({
                    'id_start': i,
                    'id_end': j,
                    'distance': distance_matrix.at[i, j]
                }, ignore_index=True)

    return unrolled_distances

input_file = ('/Users/pankhuri/Downloads/dataset-3.csv')
resulting_distance_matrix = calculate_distance_matrix(csv_file_path)
print("Distance Matrix:")
print(resulting_distance_matrix)

# Unroll the distance matrix
unrolled_distances = unroll_distance_matrix(resulting_distance_matrix)
print("\nUnrolled Distances:")
print(unrolled_distances)




# In[33]:


#Question 4: Calculate Toll Rate

import pandas as pd

def calculate_toll_rate(input_df):
    # Copy the input DataFrame to avoid modifying the original
    df = input_df.copy()

    # Calculate toll rates for each vehicle type
    df['moto'] = df['distance'] * 0.8
    df['car'] = df['distance'] * 1.2
    df['rv'] = df['distance'] * 1.5
    df['bus'] = df['distance'] * 2.2
    df['truck'] = df['distance'] * 3.6

    return df


resulting_toll_rates = calculate_toll_rate(unrolled_distances)

# Displaying the resulting DataFrame with toll rates
print(resulting_toll_rates[['id_start', 'id_end', 'distance', 'moto', 'car', 'rv', 'bus', 'truck']])



# In[38]:


#Question 5: Calculate Time-Based Toll Rates

import pandas as pd
import numpy as np
from datetime import time

def calculate_time_based_toll_rates(input_df):
    # Copying the input DataFrame to avoid modifying the original
    df = input_df.copy()

    weekday_time_ranges = [(time(0, 0, 0), time(10, 0, 0)),
                           (time(10, 0, 0), time(18, 0, 0)),
                           (time(18, 0, 0), time(23, 59, 59))]

    weekend_time_range = (time(0, 0, 0), time(23, 59, 59))

    # Function to apply discount factor based on time range
    def apply_discount_factor(row):
        if row['start_time'].weekday() < 5:  # Weekdays (Monday - Friday)
            for start, end in weekday_time_ranges:
                if start <= row['start_time'].time() <= end:
                    return row[['moto', 'car', 'rv', 'bus', 'truck']] * 0.8 if start == weekday_time_ranges[0][0] else row[['moto', 'car', 'rv', 'bus', 'truck']] * 1.2
            return row[['moto', 'car', 'rv', 'bus', 'truck']] * 0.8
        else:  # Weekends (Saturday and Sunday)
            return row[['moto', 'car', 'rv', 'bus', 'truck']] * 0.7

    # Applying discount factor based on time range
    df[['moto', 'car', 'rv', 'bus', 'truck']] = df.apply(apply_discount_factor, axis=1)

    # Create columns for start_day, start_time, end_day, and end_time
    df['start_day'] = df['start_time'].dt.strftime('%A')
    df['start_time'] = df['start_time'].dt.time
    df['end_day'] = df['end_time'].dt.strftime('%A')
    df['end_time'] = df['end_time'].dt.time

    return df





# In[ ]:




