import streamlit as st
import pandas as pd
import plotly.express as px

# Sample data for the leaderboard
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Hannah', 'Ivy', 'Jack'],
    'Number of Reps': [10, 15, 8, 12, 20, 18, 9, 14, 16, 11],
    'Exercise': ['Push-ups', 'Squats', 'Shrugs', 'Push-ups', 'Squats', 'Shrugs', 'Push-ups', 'Squats', 'Shrugs', 'Push-ups']
}

# Create a Pandas DataFrame
df = pd.DataFrame(data)

# Title for the leaderboard
st.title('2-Metric Leaderboard')

# Create an interactive horizontal bar chart using Plotly Express
fig = px.bar(
    df, y='Name', x='Number of Reps',
    color='Exercise',  # Color bars based on the exercise type
    labels={'Number of Reps': 'Reps'},  # Customize axis labels
    title='Exercise vs. Number of Reps Leaderboard'
)

# Customize the plot (you can add more customization as needed)
fig.update_layout(xaxis_title='Reps', yaxis_title='Name')  # Update axis labels

# Display the chart using st.plotly_chart
st.plotly_chart(fig)
