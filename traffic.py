# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mapie.regression import MapieRegressor

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Reading the pickle file that we created before 
model_pickle = open('xgb_traffic.pickle', 'rb') 
xgb_model = pickle.load(model_pickle) 
model_pickle.close()

# Load the default dataset
# df = pd.read_csv('Traffic_Volume.csv')
# traffic_df = df.copy()

traffic_df = pd.read_csv('Traffic_Volume.csv')

traffic_df['holiday'] = traffic_df['holiday'].replace({None: "None"})

# Make sure all the date_time entries are actually in date_time format
traffic_df['date_time'] = pd.to_datetime(traffic_df['date_time'])

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

# Custom HTML for gradient title text   # source: chatgpt
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
message = st.info("Please choose input method")

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

def make_tabs(feature_pic, residual_pic, pred_act_pic, cov_pic):
# Additional tabs for model performance
    st.subheader("Model Insights")
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                                "Histogram of Residuals", 
                                "Predicted Vs. Actual", 
                                "Coverage Plot"])
    with tab1:
        st.write("### Feature Importance")
        st.image(feature_pic)
        st.caption("Relative importance of features in prediction.")
    with tab2:
        st.write("### Histogram of Residuals")
        st.image(residual_pic)
        st.caption("Distribution of residuals to evaluate prediction quality.")
    with tab3:
        st.write("### Plot of Predicted Vs. Actual")
        st.image(pred_act_pic)
        st.caption("Visual comparison of predicted and actual values.")
    with tab4:
        st.write("### Coverage Plot")
        st.image(cov_pic)
        st.caption("Range of predictions with confidence intervals.")

def encode_csv(df, default):
    # Encode the inputs for model prediction
    encode_df = default.copy()

    # Combine the csv of user data under our default
    encode_df = pd.concat([encode_df, df], ignore_index=True)
    # encode_df

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df, columns=["holiday", "weather_main", "month", "weekday", "hour"], drop_first=True)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(len(df))

    # Convert all booleans to ints 0/1 so xgb can handle them
    user_encoded_df = user_encoded_df.astype({col: 'int' for col in user_encoded_df.select_dtypes('bool').columns}) # source: chatgpt

    return user_encoded_df

def encode_form(df, default):
    # Encode the inputs for model prediction
    encode_df = default.copy()

    # Combine the list of user data as a row to our default
    encode_df.loc[len(encode_df)] = df

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df, columns=["holiday", "weather_main", "month", "weekday", "hour"], drop_first=True)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(len(df))

    # Convert all booleans to ints 0/1 so xgb can handle them       # source: chatgpt
    user_encoded_df = user_encoded_df.astype({col: 'int' for col in user_encoded_df.select_dtypes('bool').columns}) # source: chatgpt

    return user_encoded_df

# Option 1: User uploads a CSV
if user is not None:
    st.header("Predicting Traffic")

    user = pd.read_csv(user)
    user['hour'] = user['hour'].astype(str)
    user['holiday'] = user['holiday'].astype(str)
    user['holiday'] = user['holiday'].replace({"nan": "None"})

    user_encoded_df = encode_csv(user, traffic_df)

    # prediction, intervals = xgb_model.predict(user_encoded_df, alpha = alpha)   # NOTE: This "predict" is where the problem arises

    # user["Predicted value"] = prediction
    # user["Lower value limit"] = np.maximum(0, intervals[:, 0])          # no negative values
    # user["Upper value limit"] = intervals[:, 1]
    # st.write("Prediction results with", ((1 - alpha)*100), "%", "confidence interval:")

    st.subheader("Predicted Volume")
    st.write("Predicted with", ((1 - alpha)*100), "%", "confidence")
    st.write(user)  
    make_tabs('feature_imp.svg', 'residual_plot.svg', 'pred_vs_actual.svg', 'coverage.svg')

# Option 2: User submits a form. Without clicking "submit," prediction from form default data will display.
if user is None:
    st.header("Predicting Traffic")
    # insert prediction here
    user_form = [holiday, temp, rain, snow, clouds, weather, month, day, hour]
    user_encoded_df = encode_form(user_form, traffic_df)

    # Drop duplicate columns based on their names # source: chatgpt
    user_encoded_df = user_encoded_df.loc[:, ~user_encoded_df.columns.duplicated()]

    # prediction, intervals = xgb_model.predict(user_encoded_df, alpha = alpha)   # NOTE: This "predict" is where the problem arises.
    # pred_value = prediction[0]
    # lower_limit = intervals[0][0][0]
    # upper_limit = intervals[0][1][0]

    st.subheader("Predicted Volume: 1372")     # placeholder for: st.subheader("Predicted Volume: ", pred_value)
    st.write("Predicted with", ((1 - alpha)*100), "%", "confidence")
    make_tabs('feature_imp.svg', 'residual_plot.svg', 'pred_vs_actual.svg', 'coverage.svg')
