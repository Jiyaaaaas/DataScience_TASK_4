import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "vscode"
import folium
from folium.plugins import HeatMap

plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

print("Loading Dataset...")

df = pd.read_csv("US_Accidents_March23.csv")

print("Dataset Loaded")

print(df.head())

df['Start_Time'] = pd.to_datetime(
    df['Start_Time'],
    format='mixed',
    errors='coerce'
)

df['End_Time'] = pd.to_datetime(
    df['End_Time'],
    format='mixed',
    errors='coerce'
)

df['Year'] = df['Start_Time'].dt.year
df['Month'] = df['Start_Time'].dt.month
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.day_name()

def time_of_day(hour):

    if 5 <= hour < 12:
        return 'Morning'
    
    elif 12 <= hour < 17:
        return 'Afternoon'
    
    elif 17 <= hour < 21:
        return 'Evening'
    
    else: 
        return 'Night'

df['Time_of_Day'] = df['Hour'].apply(time_of_day)

# Accidents by Hour

plt.figure(figsize=(10,5))

sns.countplot(x='Hour', data=df)

plt.title("Accidents by Hour")

plt.show()

# Accidents By Time of Day

plt.figure(figsize=(8,5))

sns.countplot(x='Time_of_Day', data=df)

plt.title("Accidents by Time of Day")

plt.show()

# Weather Analysis

weather_counts = df['Weather_Condition'].value_counts().head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=weather_counts.values,
    y=weather_counts.index
)

plt.title("Weather Conditions")

plt.show()

# Road Condition Analysis

road_features = [

    'Bump',
    'Junction',
    'Traffic_Signal',
    'Roundabout',
    'Station',
    'Crossing',
    'Stop'

]

feature_counts = df[road_features].sum()

plt.figure(figsize=(10,6))

sns.barplot(
    x=feature_counts.values,
    y=feature_counts.index
)

plt.title("Road Features")

plt.show()

# Severity Analysis

plt.figure(figsize=(6,4))

sns.countplot(x='Severity', data=df)

plt.title("Severity Distribution")

plt.show()

# Plotly Hotspot Map

print("Creating Plotly Map....")

sample_df = df.sample(5000)

fig = px.scatter_map(

    sample_df,

    lat="Start_Lat",

    lon="Start_Lng",

    color="Severity",

    zoom=3,

    title="Accident Hotspots",

    height=800
)

fig.write_image(

    "hotspots_map.png",

    width=1200,

    height=800,

    scale=2
)

print("Hotspot map saved")

# Folium Heatmap

print("Creating Heatmap....")

map_centre = [

    df['Start_Lat'].mean(),

    df['Start_Lng'].mean()

]

m = folium.Map(

    location=map_centre,

    zoom_start=4

)

heat_data = df[

    ['Start_Lat', 'Start_Lng']

].dropna().sample(20000).values.tolist()

HeatMap(heat_data).add_to(m)

m

# Correlation HeatMap

numeric_cols = [

    'Temperature(F)',

    'Humidity(%)',

    'Pressure(in)',

    'Visibility(mi)',

    'Wind_Speed(mph)',

    'Severity'
]

plt.figure(figsize=(10,6))

sns.heatmap(

    df[numeric_cols].corr(),

    annot=True

)

plt.title("Correlation HeatMap")

plt.show()

print("Analysis Completed!")