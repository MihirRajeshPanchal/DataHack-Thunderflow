# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import matplotlib.pyplot as plt

# # generate random weight data for different periods
# np.random.seed(42) # set random seed for reproducibility
# dates_30 = pd.date_range(start="2023-10-01", end="2023-10-31") # create date range for last 30 days
# dates_90 = pd.date_range(start="2023-08-01", end="2023-10-31") # create date range for last 3 months
# dates_180 = pd.date_range(start="2023-05-01", end="2023-10-31") # create date range for last 6 months
# dates_365 = pd.date_range(start="2023-01-01", end="2023-12-31") # create date range for last year
# weight_30 = np.random.normal(loc=70, scale=5, size=len(dates_30)) # generate random weight data for last 30 days
# weight_90 = np.random.normal(loc=70, scale=5, size=len(dates_90)) # generate random weight data for last 3 months
# weight_180 = np.random.normal(loc=70, scale=5, size=len(dates_180)) # generate random weight data for last 6 months
# weight_365 = np.random.normal(loc=70, scale=5, size=len(dates_365)) # generate random weight data for last year

# # create dataframes for different periods
# df_30 = pd.DataFrame({"Date": dates_30, "Weight": weight_30}) # create dataframe for last 30 days
# df_90 = pd.DataFrame({"Date": dates_90, "Weight": weight_90}) # create dataframe for last 3 months
# df_180 = pd.DataFrame({"Date": dates_180, "Weight": weight_180}) # create dataframe for last 6 months
# df_365 = pd.DataFrame({"Date": dates_365, "Weight": weight_365}) # create dataframe for last year

# def get_reps(weight):
#     # set the random seed for reproducibility
#     np.random.seed(int(weight * 100))
#     # generate a random number between 0 and 10
#     correct = np.random.randint(0, 11)
#     # calculate the number of wrong reps
#     wrong = 10 - correct
#     # return a tuple of correct and wrong reps
#     return (correct, wrong)


# # set up the basic configuration of the dashboard
# st.set_page_config(
#     page_title="Weight Dashboard",
#     page_icon="🏋️‍♂️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # add title and subheader to the app
# st.title("Weight Dashboard")
# st.subheader("Track your weight progress with Streamlit")

# # add a selectbox widget to choose the period of interest
# period = st.selectbox("Select the period of interest", ["Last 30 days", "Last 3 months", "Last 6 months", "Last year"])

# # create a dictionary to map the period to the corresponding dataframe
# period_dict = {
#     "Last 30 days": df_30,
#     "Last 3 months": df_90,
#     "Last 6 months": df_180,
#     "Last year": df_365
# }

# # get the dataframe for the selected period
# df = period_dict[period]

# # create a line plot for the weight data using plotly express
# weight_line = px.line(df, x="Date", y="Weight", color_discrete_sequence=["blue"]) # create line chart for weight with plotly express
# weight_line.update_layout(title=f"Weight Progress - {period}", title_x=0.5) # update layout of weight chart

# # display the line plot in the app using plotly chart
# st.plotly_chart(weight_line)

# # add a date input widget to choose a date of interest
# date = st.date_input("Select a date of interest", min_value=df["Date"].min(), max_value=df["Date"].max())

# # filter the dataframe by the selected date and get the weight value
# # weight = df[df["Date"] == date]["Weight"].values[0]
# weight = df[df["Date"] == date]["Weight"].values

# # check if the array is not empty
# if weight.size != 0:
#     weight = weight[0]
# else:
#     weight = None # or some default value
# # generate random data for the correct and wrong reps
# correct, wrong = get_reps(weight)

# # create a pie chart for the correct and wrong reps using matplotlib.pyplot
# fig, ax = plt.subplots() # create a figure and an axis object
# ax.pie([correct, wrong], labels=["Correct", "Wrong"], autopct="%1.1f%%", colors=["green", "red"]) # create a pie chart with labels, percentages and colors
# ax.set_title(f"Reps Performance - {date}") # set the title of the pie chart

# # display the pie chart in the app using pyplot chart
# st.pyplot(fig)



import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

np.random.seed(42)
dates_30 = pd.date_range(start="2023-10-01", end="2023-10-31")
dates_90 = pd.date_range(start="2023-08-01", end="2023-10-31")
dates_180 = pd.date_range(start="2023-05-01", end="2023-10-31")
dates_365 = pd.date_range(start="2023-01-01", end="2023-12-31")
weight_30 = np.random.normal(loc=70, scale=5, size=len(dates_30))
weight_90 = np.random.normal(loc=70, scale=5, size=len(dates_90))
weight_180 = np.random.normal(loc=70, scale=5, size=len(dates_180))
weight_365 = np.random.normal(loc=70, scale=5, size=len(dates_365))

df_30 = pd.DataFrame({"Date": dates_30, "Weight": weight_30})
df_90 = pd.DataFrame({"Date": dates_90, "Weight": weight_90})
df_180 = pd.DataFrame({"Date": dates_180, "Weight": weight_180})
df_365 = pd.DataFrame({"Date": dates_365, "Weight": weight_365})

def get_reps(weight):
    np.random.seed(int(42))
    correct = np.random.randint(0, 11)
    wrong = 10 - correct
    return (correct, wrong)

st.set_page_config(
    page_title="Weight Dashboard",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Weight Dashboard")
st.subheader("Track your weight progress with Streamlit")

col1, col2 = st.columns(2)
col1.subheader("Select the period of interest")
period = col1.selectbox("", ["Last 30 days", "Last 3 months", "Last 6 months", "Last year"])

# period = st.selectbox("Select the period of interest", ["Last 30 days", "Last 3 months", "Last 6 months", "Last year"])

period_dict = {
    "Last 30 days": df_30,
    "Last 3 months": df_90,
    "Last 6 months": df_180,
    "Last year": df_365
}

df = period_dict[period]

weight_line = px.line(df, x="Date", y="Weight", color_discrete_sequence=["blue"])
weight_line.update_layout(title=f"Weight Progress - {period}", title_x=0.5)


col2.subheader("Select a date of interest")
date = col2.date_input("", min_value=df["Date"].min(), max_value=df["Date"].max())
# date = st.date_input("Select a date of interest", min_value=df["Date"].min(), max_value=df["Date"].max())

weight = df[df["Date"] == date]["Weight"].values

if weight.size != 0:
    weight = weight[0]
else:
    weight = None 

correct, wrong = get_reps(weight)

fig, ax = plt.subplots() 
ax.pie([correct, wrong], labels=["Correct", "Wrong"], autopct="%1.1f%%", colors=["green", "red"])
ax.set_title(f"Reps Performance - {date}")

# create two columns with equal width

# display the line plot in the first column
col1.plotly_chart(weight_line)

# display the pie plot in the second column
col2.pyplot(fig)

