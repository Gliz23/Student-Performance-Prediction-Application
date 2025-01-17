import streamlit as st
import pandas as pd
import pickle

# Load the saved scaler and model
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('exam_score_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the encoded column names
with open('encoded_columns.pkl', 'rb') as columns_file:
    encoded_columns = pickle.load(columns_file)

# Load the all_columns file
with open('all_columns.pkl', 'rb') as all_columns_file:
    all_columns = pickle.load(all_columns_file)

# Load the test data
test_data = pd.read_csv('StudentPerformanceFactors.csv')
df_test = test_data.copy()

# List of columns to apply one-hot encoding
encode = ['Parental_Involvement', 'Access_to_Resources',
           'Extracurricular_Activities', 'Motivation_Level',
          'Internet_Access', 'Family_Income', 'Teacher_Quality',
          'School_Type', 'Peer_Influence',
          'Learning_Disabilities', 'Gender',
          'Distance_from_Home', 'Parental_Education_Level']

# Apply one-hot encoding to test data
for col in encode:
    dummy = pd.get_dummies(df_test[col], prefix=col)
    df_test = pd.concat([df_test, dummy], axis=1)
    df_test.drop(columns=[col], inplace=True)

# Reorder the test data columns to match the training set
columns_without_target = [col for col in all_columns if col != 'Exam_Score']
df_test = df_test[columns_without_target]


# Scale the test features
scaled_test_features = scaler.transform(df_test)


st.markdown("""
    <style>
        .stApp {
            background-color: #000018; /* Dark blue background */
        }
        .header {
            text-align: center;
            padding-bottom: 4px;
        }
        .header h1 {
            font-size: 38px;  
            color: #4CAF50; /* Light green */
            margin-bottom: 5px;
        }
        .header p {
            font-size: 18px;  
            color: #FFFFFF; /* White text */
        }
    </style>

    <div class="header">
        <h1>📚Student Performance Prediction App</h1>
        <p>📊Predict the exam scores based on various student performance factors</p>
    </div>
""", unsafe_allow_html=True)


# Produces all the columns without the target 'Exam_Score'
feature_names = [col for col in all_columns if col != 'Exam_Score']



# Interactive sliders for feature inputs
st.markdown("""
    <div style="margin: 2rem auto; text-align: center;">
        <h3 style="color: #4CAF50;">Input Parameters</h3>
        <p>Adjust the sliders and dropdowns to predict the student's exam score.</p>
    </div>
""", unsafe_allow_html=True)



def user_input_features(df_test):
    feature_values = {}

    # List of categorical features
    categorical_features = [
        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 
        'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Gender', 
        'Parental_Education_Level', 'Distance_from_Home'
    ]

    # List of numerical features
    numerical_features = [
        'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
        'Tutoring_Sessions', 'Physical_Activity'
    ]

    encoded_columns_df = test_data[categorical_features]

    # Handle categorical features
    for feature in categorical_features:
        # Get unique categories for each categorical feature
        categories = encoded_columns_df[feature].unique()
        feature_values[feature] = st.sidebar.selectbox(
            f"{feature}",
            categories
        )
    
    # Handle numerical features
    for feature in numerical_features:
        min_value = df_test[feature].min()  # Adjust range dynamically
        max_value = df_test[feature].max()
        feature_values[feature] = st.sidebar.slider(
            f"{feature}",
            min_value=float(min_value),
            max_value=float(max_value),
            value=float((min_value + max_value) / 2),  # Default value is the midpoint
            step=1.0
        )

    # Convert the dictionary of selected feature values into a DataFrame
    features = pd.DataFrame(feature_values, index=[0])
    return features

# Get input features from the user
input_df = user_input_features(df_test)



# Apply the same encoding and scaling as the training data
for col in encode:
    if col in input_df.columns:
        dummy = pd.get_dummies(input_df[col], prefix=col)
        input_df = pd.concat([input_df, dummy], axis=1)
        input_df.drop(columns=[col], inplace=True)

for col in columns_without_target:
    if col not in input_df.columns:
        input_df[col] = 0 

# Reorder the columns to match the training data
input_df = input_df[columns_without_target]

# Scale the input features for prediction
scaled_input = scaler.transform(input_df)


# Custom Button Style
st.markdown("""
 <style>
        /* Style for Streamlit button */
        .stButton>button {
            background-color: #6CA0D9;  /* Darker Light Blue */
            color: white;
            font-size: 18px;
            border-radius: 10px;
            width: 200px;
            margin: 0 auto;
            display: block;
            padding: 10px;
            border: 2px solid transparent; /* Default border (transparent) */
        }

        /* Button hover state */
        .stButton>button:hover {
            background-color: #4B7BAE;  /* Even darker blue */
            color: black;  /* Text color turns black */
            border: 2px solid black;  /* Border turns black on hover */
        }

        /* Button active (clicked) state */
        .stButton>button:active {
            background-color: #4B7BAE !important;  /* Even darker blue */
            color: black !important;  /* Text color stays black */
            border: 2px solid black !important;  /* Border stays black when clicked */
        }

        /* Ensure the active state persists */
        .stButton>button:focus {
            background-color: #4B7BAE !important;  /* Make the background darker after click */
            color: black !important;  /* Text color remains black */
            border: 2px solid black !important;  /* Border stays black after click */
        }
    </style>
""", unsafe_allow_html=True)

# Display the prediction result in a centered, styled way
if st.button("Predict Exam Score", key="predict_button"):
    prediction = model.predict(scaled_input)
    result = f"{prediction[0]:.2f}"  # Correct string formatting

    st.markdown(f"""
        <div style="text-align: center; margin-top: 20px;">
            <h1 style="font-weight: 100; font-size: 20px; color: #F1F1F1;">Predicted Exam Score</h1>
            <div style="font-size: 30px; font-weight: bold; color: #4CAF50;">
                {result}
            </div>
        </div>
    """, unsafe_allow_html=True)


# Footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #2E7D32; 
            color: white;
            text-align: right;
            padding-top: 7px;
            padding-bottom: 7px;
            font-size: 16px;
            height: 36px;
        }
    </style>
    <div class="footer">
        <p>Created with ❤️ by Glory Elizabeth Akoto</p>
    </div>
""", unsafe_allow_html=True)
