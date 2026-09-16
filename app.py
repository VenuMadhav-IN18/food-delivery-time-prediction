import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load model.pkl")
    st.code(str(e))
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🍔 Food Delivery Time Prediction")
st.write(
    "Enter the order details below to predict the estimated delivery time."
)

st.divider()


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_category_options(column_name):
    """
    Try to extract the actual categories used during model training
    from OneHotEncoder inside the trained pipeline.
    """

    try:
        # Check pipeline steps
        if hasattr(model, "named_steps"):

            for step_name, step in model.named_steps.items():

                # ColumnTransformer
                if hasattr(step, "transformers_"):

                    for _, transformer, columns in step.transformers_:

                        if transformer == "drop" or transformer == "passthrough":
                            continue

                        # Convert columns to list
                        if isinstance(columns, str):
                            columns = [columns]

                        if hasattr(transformer, "categories_"):

                            for col, categories in zip(
                                columns,
                                transformer.categories_
                            ):

                                if col == column_name:
                                    return list(categories)

    except Exception:
        pass

    return None


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📦 Order Information")

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Order Hour
# ------------------------------------------------------------

with col1:

    order_hour = st.number_input(
        "Order Hour",
        min_value=0,
        max_value=23,
        value=13,
        step=1
    )


# ------------------------------------------------------------
# Day of Week
# ------------------------------------------------------------

with col2:

    day_of_week = st.selectbox(
        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )


# ------------------------------------------------------------
# Is Weekend
# ------------------------------------------------------------

with col3:

    is_weekend = st.selectbox(
        "Is Weekend",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


# ============================================================
# FESTIVAL / WEATHER
# ============================================================

st.header("🌦️ Date & Weather")

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Is Festival
# ------------------------------------------------------------

with col1:

    is_festival = st.selectbox(
        "Is Festival",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


# ------------------------------------------------------------
# Weather
# ------------------------------------------------------------

with col2:

    weather = st.selectbox(
        "Weather",
        [
            "Clear",
            "Rain",
            "Cloudy",
            "Storm",
            "Fog"
        ]
    )


# ------------------------------------------------------------
# Traffic Level
# ------------------------------------------------------------

with col3:

    traffic_level = st.selectbox(
            "Traffic Level",
            [
                'Low',
                'Moderate',
                'High',
                'Severe'
            ]
        )




# ============================================================
# LOCATION
# ============================================================

st.header("📍 Location Information")

col1, col2, col3 = st.columns(3)

# ------------------------------------------------------------
# Pickup Zone
# ------------------------------------------------------------

with col1:
    pickup_zone = st.selectbox(
        "Pickup Zone",
        [
            "Residential",
            "CBD",
            "Commercial",
            "Industrial",
            "Suburban"
        ]
    )

with col2:
    dropoff_zone = st.selectbox(
        "Dropoff Zone",
        [
            "Residential",
            "Commercial",
            "CBD",
            "Suburban",
            "Industrial"
        ]
    )

# ------------------------------------------------------------
# Delivery Distance Category
# ------------------------------------------------------------

with col3:

    delivery_distance_category = st.selectbox(
        "Delivery Distance Category",
        [
            "Short",
            "Medium",
            "Long"
        ]
    )


# ============================================================
# RESTAURANT INFORMATION
# ============================================================

st.header("🍽️ Restaurant Information")

col1, col2, col3 = st.columns(3)

with col1:
    cuisine_type = st.selectbox(
        "Cuisine Type",
        [
            "North Indian",
            "Biryani",
            "Pizza",
            "Chinese",
            "Burger",
            "South Indian",
            "Cafe",
            "Bakery",
            "Desserts"
        ]
    )

with col2:
    restaurant_load = st.selectbox(
        "Restaurant Load",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

with col3:
    order_items = st.number_input(
        "Number of Order Items",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )
# ============================================================
# RIDER INFORMATION
# ============================================================

st.header("🛵 Rider Information")

col1, col2, col3 = st.columns(3)

with col1:
    vehicle_type = st.selectbox(
        "Vehicle Type",
        [
            "Bike",
            "Scooter",
            "Electric Scooter",
            "Bicycle"
        ]
    )

with col2:
    rider_experience = st.number_input(
        "Rider Experience (Years)",
        min_value=0.0,
        max_value=30.0,
        value=2.0,
        step=0.5
    )

with col3:
    rider_rating = st.number_input(
        "Rider Rating",
        min_value=0.0,
        max_value=5.0,
        value=4.5,
        step=0.1
    )

# ============================================================
# RATINGS & PREPARATION
# ============================================================

st.header("⭐ Ratings & Preparation")

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Restaurant Rating
# ------------------------------------------------------------

with col1:

    restaurant_rating = st.number_input(
        "Restaurant Rating",
        min_value=0.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )


# ------------------------------------------------------------
# Preparation Time
# ------------------------------------------------------------

with col2:

    preparation_time = st.number_input(
        "Preparation Time (Minutes)",
        min_value=0.0,
        max_value=180.0,
        value=20.0,
        step=1.0
    )


# ------------------------------------------------------------
# Number of Signals
# ------------------------------------------------------------

with col3:

    number_of_signals = st.number_input(
        "Number of Signals",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )


# ============================================================
# DISTANCE
# ============================================================

st.header("🛣️ Delivery Distance")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Road Distance
# ------------------------------------------------------------

with col1:

    road_distance = st.number_input(
        "Road Distance (km)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.1
    )


# ------------------------------------------------------------
# Delivery Priority
# ------------------------------------------------------------

with col2:

    delivery_priority = st.selectbox(
        "Delivery Priority",
        [
            "Normal",
            "Priority",
            "VIP"
        ]
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🚀 Predict Delivery Time",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "Order_Hour": [order_hour],

        "Day_of_Week": [day_of_week],

        "Is_Weekend": [is_weekend],

        "Is_Festival": [is_festival],

        "Weather": [weather],

        "Pickup_Zone": [pickup_zone],

        "Dropoff_Zone": [dropoff_zone],

        "Vehicle_Type": [vehicle_type],

        "Rider_Experience_Years": [rider_experience],

        "Rider_Rating": [rider_rating],

        "Restaurant_Rating": [restaurant_rating],

        "Cuisine_Type": [cuisine_type],

        "Order_Items": [order_items],

        "Restaurant_Load": [restaurant_load],

        "Preparation_Time_Min": [preparation_time],

        "Road_Distance_km": [road_distance],

        "Delivery_Distance_Category": [
            delivery_distance_category
        ],

        "Traffic_Level": [traffic_level],

        "Number_of_Signals": [number_of_signals],

        "Delivery_Priority": [delivery_priority]
    })


    # --------------------------------------------------------
    # SHOW INPUT DATA
    # --------------------------------------------------------

    with st.expander("🔍 View Input Data"):

        st.dataframe(
            input_data,
            use_container_width=True
        )


    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    try:

        prediction = model.predict(input_data)

        predicted_time = float(prediction[0])


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.success("✅ Prediction completed successfully!")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Estimated Delivery Time",
                f"{int(round(predicted_time))} minutes"
            )


        with col2:

            st.metric(
                "Approximately",
                f"{predicted_time / 60:.2f} hours"
            )


    except Exception as e:

        st.error("❌ Prediction failed")

        st.write("Error details:")

        st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🍔 Food Delivery Time Prediction | Machine Learning Project"
)