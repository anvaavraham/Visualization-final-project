import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

# import plotly.figure_factory as ff

zip_path = 'data_new.zip'
csv_filename = 'data_new.csv'

# Open the zip folder
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extract(csv_filename, path='.')

# Read the extracted CSV file into a DataFrame
df = pd.read_csv(csv_filename)

st.title("Visualization- Final Project")
st.markdown("Can we predict having an accident?")
st.markdown("Reut Ben-Hamo & Anva Avraham")

accidents_per_state = df['State'].value_counts().reset_index()
accidents_per_state.columns = ['State', 'Accident Count']

# Box plot
# Define the desired colors for each severity level
colors = ['lightblue', 'mediumblue', 'lightpink', 'indianred']

st.subheader("Box Plot - How temperature affecting the severity of the accident?")
fig = go.Figure()
severity_levels = df['Severity'].unique()

for severity in sorted(severity_levels):
    fig.add_trace(go.Box(
        y=df[df['Severity'] == severity]['Temperature(F)'],
        name=f'Severity {severity}',
        marker_color=colors[int(severity)-1]  # Assign the specific color for each severity level
    ))

fig.update_layout(
    xaxis=dict(title='Category'),
    yaxis=dict(title='Temperature'),
    title='Severity vs Temperature'
)

st.plotly_chart(fig)

# Correlation matrix

st.subheader("Correlation Matrix between different road conditions")
selected_columns = ['Severity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
                    'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal']
new_df = df[selected_columns].copy()
# Compute correlation matrix
correlation_matrix = new_df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
# Create a heatmap plot of the correlation matrix
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.xlabel('Variables')
plt.ylabel('Variables')
plt.title('Correlation Matrix')
st.pyplot(heatmap.figure)

# Bar plot
top_10_conditions = df['Weather_Condition'].value_counts().nlargest(10).index
colors = ['lightblue', 'mediumblue', 'lightpink', 'indianred']
# Filter the DataFrame to include only the top 10 weather conditions
filtered_df = df[df['Weather_Condition'].isin(top_10_conditions)]
# Group the filtered data by weather condition and severity, and count the occurrences
grouped_data = filtered_df.groupby(['Weather_Condition', 'Severity']).size().unstack()

st.subheader("Bar Plot  - How different weather condition affecting accident severity?")
fig = px.bar(grouped_data, barmode='stack', color_discrete_sequence=colors)
fig.update_layout(
    xaxis=dict(title='Weather Condition'),
    yaxis=dict(title='Count'),
    title='Top 10 Weather Condition Severity Distribution'
)
st.plotly_chart(fig)

# Map
st.subheader("USA Map  - Does specific stated have more accidents?")
fig = px.choropleth(accidents_per_state,
                    locations='State',
                    locationmode="USA-states",
                    color='Accident Count',
                    color_continuous_scale='Reds',
                    scope="usa",
                    labels={'Accident Count': 'Accident Count'}
                    )
fig.update_layout(title_text='Accident Count by State')
st.plotly_chart(fig)

st.subheader("Data")
st.write(df)
