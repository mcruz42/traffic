# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Reading the pickle file that we created before 
model_pickle = open('xgb_traffic.pickle', 'rb') 
xgb_model = pickle.load(model_pickle) 
model_pickle.close()

# Load the default dataset
df = pd.read_csv('Traffic_Volume.csv')
traffic_df = df.copy()
traffic_df['holiday'] = traffic_df['holiday'].replace({None: "None"})
# Make sure all the date_time entries are actually in date_time format
traffic_df['date_time'] = pd.to_datetime(df['date_time'])

# Extract date/time to month and day of week and the hour and treat them as categorical
# Extract month
traffic_df['month'] = traffic_df['date_time'].dt.month_name()

# Extract day of the week
traffic_df['weekday'] = traffic_df['date_time'].dt.day_name()

# Extract hour
traffic_df['hour'] = traffic_df['date_time'].dt.hour

# drop original date_time column
traffic_df.drop(['date_time'], axis=1, inplace=True)

# move traffic_volume to the end column
traffic_df = traffic_df[['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'month', 'weekday', 'hour', 'traffic_volume']] 
traffic_df.dropna(inplace = True)
traffic_df = traffic_df.drop(columns = ['traffic_volume']) # confirmed: has all the right columns

# Custom HTML for gradient title text
st.markdown(
    """
    <h1 style='text-align: center; font-size: 2.5em;'>
        <span style="background: -webkit-linear-gradient(left, orange, yellow, green); -webkit-background-clip: text; color: transparent;">
            Traffic Volume Predictor
        </span>
    </h1>
    <p style='text-align: center; font-size: 1.2em;'>Utilize our advanced Machine Learning application to predict traffic volume.</p>
    """,
    unsafe_allow_html=True
)
st.image("traffic_image.gif", use_column_width=True)
#TODO: message = st.info("Please choose input method")

# TODO: make the step funcs
# TODO: check the mins and maxes like rain
# Sidebar for user inputs with an expander
with st.sidebar:
    st.image("traffic_sidebar.jpg", use_column_width = True, caption="Traffic Volume Predictor")
    st.header("Input Features")
    st.write("You can either upload your data file or manually enter input features.")
    
    with st.expander("Option 1: Upload CSV File"):
        st.write("Upload a CSV file containing traffic details.")
        user = st.file_uploader("Choose a CSV file")
        st.header("Sample Data Format for Upload")
        st.write(traffic_df.head())
        st.warning("Ensure your uploaded file has the same column names and data types as shown above.")

    with st.expander("Option 2: Fill Out Form"):
        # Sidebar for user inputs with an expander
        with st.form("user_inputs_form"):
            st.write("Enter the traffic details manually by using the form below.")
            holiday = st.selectbox("Choose whether today is a designated holiday or not", options=["None", "Labor Day", "Thanksgiving Day", "Christmas Day", "New Years Day", 
                                                                                                   "Martin Luther King Jr Day", "Columbus Day", "Veterans Day", "Washingtons Birthday", 
                                                                                                   "Memorial Day", "Independence Day", "State Fair"])
            temp = st.number_input("Average temperature in Kelvin", 
                                    min_value=traffic_df['temp'].min(),
                                    max_value=traffic_df['temp'].max(),
                                    value=traffic_df['temp'].mean())
            rain = st.number_input("Amount in mm of rain that occurred in the hour", 
                                    min_value=traffic_df['rain_1h'].min(),
                                    max_value=traffic_df['rain_1h'].max(),
                                    value=traffic_df['rain_1h'].mean())
            snow = st.number_input("Amount in mm of snow that occurred in the hour", 
                                    min_value=traffic_df['snow_1h'].min(),
                                    max_value=traffic_df['snow_1h'].max(),
                                    value=traffic_df['snow_1h'].mean())
            clouds = st.number_input("Percentage of cloud cover", 
                                    min_value=float(traffic_df['clouds_all'].min()),
                                    max_value=float(traffic_df['clouds_all'].max()),
                                    value=traffic_df['clouds_all'].mean())
            weather = st.selectbox("Choose the current weather", options=["Clouds", "Clear", "Mist", "Rain", "Snow", "Drizzle", "Haze", "Thunderstorm", "Fog", "Smoke", "Squall"])
            month = st.selectbox("Choose month", options=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
            day = st.selectbox("Choose day of the week", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            hour = st.selectbox("Choose hour", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"])
            submit_button = st.form_submit_button("Submit Form Data")

# Get the prediction with its intervals
alpha = st.slider("Select alpha value for prediction intervals", min_value=0.01, max_value=0.5, value=0.1, step=0.01) # 0.1 for 90% confidence level

if user is not None:
    # Display input summary
    st.write("### Your File")
    user = pd.read_csv(user)
    user['holiday'] = user['holiday'].replace({None: "None"})

    # Input features (excluding traffic volume column)
    features = traffic_df[['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'month', 'weekday', 'hour']] 

    # One-hot encoding to handle categorical variables
    cat_var = ['holiday', 'weather_main', 'month', 'weekday', 'hour']
    features_encoded = pd.get_dummies(features, columns = cat_var)

    # Extract encoded user data
    user_encoded_df = features_encoded.tail(len(df))
    st.write("got to here")

    prediction, intervals = xgb_model.predict(user_encoded_df, alpha = alpha)
    st.write("got to here 2")
    user["Predicted value"] = prediction
    st.write("got to here 3")
    user["Lower value limit"] = np.maximum(0, intervals[:, 0])          # no negative values
    user["Upper value limit"] = intervals[:, 1]
    st.write("Prediction results with", ((1 - alpha)*100), "%", "confidence interval:")
    st.write(user)



# # Combine the list of user data as a row to default

# # Input features
# features = user[['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'month', 'weekday', 'hour']] 

# # One-hot encoding to handle categorical variables
# cat_var = ['holiday', 'weather_main', 'month', 'weekday', 'hour']
# features_encoded = pd.get_dummies(features, columns = cat_var)

# # Using predict() with new data provided by the user
# new_prediction = xgb_model.predict(features_encoded)

# # Store the predicted species
# user["Predicted"] = new_prediction

# # Display table with prediction
# user
