#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd

username = 'pankhuri'
file_name = 'dataset-1.csv'

# Constructing the full file path
file_path = os.path.join("/Users", username, "Downloads", file_name)

df = pd.read_csv(file_path)

file_path = os.path.join("~", "Downloads", "dataset-1.csv")
file_path = os.path.expanduser(file_path)  # Expand the '~' to the actual home directory
df = pd.read_csv(file_path)


# In[2]:


#Question 1: Car Matrix Generation
import pandas as pd

def generate_car_matrix(dataset_path):


    # Pivot the DataFrame to create the desired matrix
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    for idx in car_matrix.index:
        car_matrix.at[idx, idx] = 0

    return car_matrix

# Replace 'dataset-1.csv' with the actual path to your CSV file
result_matrix = generate_car_matrix('dataset-1.csv')

# Display the resulting DataFrame
print(result_matrix)


# In[3]:


#Question 2: Car Type Count Calculation
import pandas as pd

def get_type_count(dataset_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataset_path)

    # Add a new categorical column 'car_type' based on the values of the 'car' column
    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.Series(np.select(conditions, choices), dtype='category')

    # Calculate the count of occurrences for each 'car_type' category
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    type_count = dict(sorted(type_count.items()))

    return type_count

# Example usage:
dataset_path = 'dataset-1.csv'
result_dict = get_type_count(dataset_path)
print(result_dict)


# In[4]:


import pandas as pd
import numpy as np
def get_type_count(dataset_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataset_path)

    # Adding a new categorical column 'car_type' based on the values of the 'car' column
    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.Series(np.select(conditions, choices), dtype='category')

    # Calculate the count of occurrences for each 'car_type' category
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    type_count = dict(sorted(type_count.items()))

    return type_count


dataset_path = 'dataset-1.csv'
result_dict = get_type_count(dataset_path)
print(result_dict)


# In[5]:


#Question 3: Bus Count Index Retrieval
import pandas as pd

def get_bus_indexes(dataset_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataset_path)

    # Calculate the mean value of the 'bus' column
    mean_bus = df['bus'].mean()

    # Identify indices where the 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus].index.tolist()

    # Sorting the indices in ascending order
    bus_indexes.sort()

    return bus_indexes


dataset_path = 'dataset-1.csv'
result_list = get_bus_indexes(dataset_path)
print(result_list)


# In[6]:


#Question 4: Route Filtering
import pandas as pd

def filter_routes(df):
    #calculating the average of truck column
    avg_truck_by_route = df.groupby('route')['truck'].mean()

    # Filtering routes where the average of 'truck' column is greater than 7
    selected_routes = avg_truck_by_route[avg_truck_by_route > 7].index.tolist()

    return selected_routes


# result
selected_routes = filter_routes(df)
print(selected_routes)


# In[7]:


#Question 5: Matrix Value Modification
import pandas as pd

def generate_car_matrix(dataset_path):
    # Loading the dataset into a DataFrame
    df = pd.read_csv(dataset_path)

    # Pivot the DataFrame to create the desired matrix
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    for idx in car_matrix.index:
        car_matrix.at[idx, idx] = 0

    return car_matrix

def multiply_matrix(input_matrix):
    modified_matrix = input_matrix.copy()

    # Applying the specified logic to modify the values
    for col in modified_matrix.columns:
        modified_matrix[col] = modified_matrix[col].apply(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Rounding the values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

df = pd.read_csv('dataset-1.csv')


car_matrix = generate_car_matrix('dataset-1.csv')

# Multiplying and round the values
modified_car_matrix = multiply_matrix(car_matrix)

print(modified_car_matrix)


# In[8]:


import os
import pandas as pd

username = 'pankhuri'
file_name = 'dataset-2.csv'

file_path = os.path.join("/Users", username, "Downloads", file_name)

df = pd.read_csv(file_path)

file_path = os.path.join("~", "Downloads", "dataset-2.csv")
file_path = os.path.expanduser(file_path)  # Expand the '~' to the actual home directory
df = pd.read_csv(file_path)


# In[10]:


import pandas as pd

def verify_timestamp_completeness(df):
    # Combining date and time columns into a single datetime column
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')

    # Creating a boolean series indicating if timestamps are correct
    completeness_check = (
        (df['start_datetime'].dt.floor('D') == df['start_datetime'].dt.floor('D')) &  # Check if timestamps have the same date
        (df['start_datetime'].dt.hour == 0) &  # Check if start time is midnight
        (df['end_datetime'].dt.hour == 23) & (df['end_datetime'].dt.minute == 59) & (df['end_datetime'].dt.second == 59)  # Check if end time is 11:59:59 PM
    )

    # doing completeness check
    completeness_check = completeness_check.groupby([df['id'], df['id_2']]).all()

    return completeness_check

df = pd.read_csv('dataset-2.csv')

completeness_result = verify_timestamp_completeness(df)
print(completeness_result)


# In[ ]:




