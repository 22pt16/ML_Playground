import streamlit as st
import pickle
import pandas as pd
from sklearn.impute import SimpleImputer    #handle unknown data


#1.NAIVE BAYES
# Load the training datasets
def load_emails():
    GOOD_EMAILS_PATH = 'Datasets/good_emails.txt'
    SPAM_EMAILS_PATH = 'Datasets/spam_emails.txt'

    with open(GOOD_EMAILS_PATH, 'r') as f:
        good_emails = f.readlines()
    with open(SPAM_EMAILS_PATH, 'r') as f:
        spam_emails = f.readlines()
    return good_emails, spam_emails

# Train the Naive Bayes model
def train_naive_bayes():
    MODEL_PATH = 'Saved_models/naive_bayes_model.pkl'
    from CLASSIFICATION.NAIVE_BAYES.nb_train import NaiveBayes  # Import your training class
    good_emails, spam_emails = load_emails()

    nb = NaiveBayes()  # Instantiate your model
    nb.train_model(good_emails, spam_emails)  # Train with the loaded emails
    likelihood_table = nb.calculate_likelihood()  # Calculate the likelihood table

    # Save the model
    with open(MODEL_PATH, 'wb') as model_file:
        pickle.dump({'model': nb, 'likelihood_table': likelihood_table}, model_file)

    return "Naive Bayes model trained successfully!"

# Validate the email
def validate_email(email):
    MODEL_PATH = 'Saved_models/naive_bayes_model.pkl'
    from CLASSIFICATION.NAIVE_BAYES.nb_test import NaiveBayes  # Import your testing class
    with open(MODEL_PATH, 'rb') as model_file:
        data = pickle.load(model_file)
        model = data['model']
        likelihood_table = data['likelihood_table']

    result = model.classify_email(email)
    return result

# Streamlit interface
st.title("Machine Learning Playground")

# Dropdown for model type selection
model_type = st.selectbox("Select Model Type", ["Regression", "Classification", "Clustering"])

if model_type == "Classification":
    # Dropdown for classification methods
    classification_method = st.selectbox("Select Classification Method", [
        "Naive Bayes", 
        "Neural Network", 
        "SVM", 
        "Random Forest", 
        "Decision Tree Classifier", 
        "Gradient Boosting", 
        "KNN", 
        "Mixture of Gaussians"
    ])

    # Button to train the selected classification model
    if classification_method == "Naive Bayes":
        if st.button("Train Naive Bayes Model"):
            message = train_naive_bayes()
            st.success(message)

    # Input area for testing
    email_input = st.text_area("Enter email text for classification:")
    result_placeholder = st.empty()
    if st.button("Test Naive Bayes Model"):
        if email_input:  # Check if input is provided
            result = validate_email(email_input)  # Classify the email
            result_placeholder.text_area("Classification Result:", result, height=150)  # Display result
        else:
            st.error("Please enter an email to classify.")  # Error if input is empty

elif model_type == "Regression":
    # Dropdown for regression methods
    regression_method = st.selectbox("Select Regression Method", [
        "Linear Regression", 
        "Multiple Regression", 
        "Logistic Regression", 
        "Decision Tree Regression", 
        "Gradient Boosting (Regression)"
    ])
    # Placeholder for regression logic (to be implemented later)
    if regression_method == "Decision Tree Regression":
        DATA_PATH = 'Datasets/SydneyHousePrices.csv'
        from REGRESSION.DECISION_TREE.dtr_train import train_decision_tree_model, meta_data, handle_unknown
        from REGRESSION.DECISION_TREE.dtr_test import test_decision_tree_model
        
        #SELECT DATASET 
        feature_columns ,target_column = meta_data(1)   #as of now 1.SydneyHouseprices
        
        
        # Input hyperparameters for Decision Tree Regressor
        max_depth = st.slider("Max Depth of the Tree", min_value=1, max_value=50, value=10)
        min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2)
        min_samples_leaf = st.slider("Min Samples per Leaf", min_value=1, max_value=10, value=1)


        # Train the Decision Tree Regressor model
        if st.button("Train Decision Tree Model"):
            test_x, test_y, message = train_decision_tree_model(
                DATA_PATH, target_column, max_depth, min_samples_split, min_samples_leaf)
            st.success(message)

        # # Evaluate the trained model
        # if st.button("Evaluate Model"):
            mae, mse, r2, predictions = test_decision_tree_model(test_x, test_y)

            # Display performance metrics
            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
            st.write(f"R² Score: {r2:.4f}")
            

           

        # Section to accept custom input for testing
        st.subheader("Test the model with custom input")

        #NOTE: add datas from meta_Data file later
        #Get unknowns from metadata
        # user_input={}
        # for i in unknowns:
        #     user_input[i] = None

        # for i in feature_columns:   
        #     user_input[i] = st.number_input(f"i", min_value=metadata[i][1], max_value=metadata[i][2], value=metadata[i][1]),
        
        user_input = {
            "Id": None,
            'suburb': None,
            'postalCode': None,
            'propType': None,
            "bed": st.number_input("Number of Bedrooms", min_value=1, max_value=7, value=1),
            "bath": st.number_input("Number of Bathrooms",min_value=1, max_value=5, value=1),
            "car": st.number_input("Number of Car Spaces",min_value=1, max_value=5, value=1)
        }
       

        if st.button("Predict with Custom Input"):
           
            
            MODEL_PATH = 'Saved_models/decision_tree_regressor.pkl'
            with open(MODEL_PATH, 'rb') as model_file:
                decision_tree_model = pickle.load(model_file)
                 
            with open('Saved_models/modes.pkl', 'rb') as mode_file:
                modes = pickle.load(mode_file)   

             #Handling unknown by default values
            crct_input = handle_unknown(1, modes, user_input)
            input_df = pd.DataFrame([crct_input])
            # Predict the target variable using the crct user input
            prediction = decision_tree_model.predict(input_df)
            st.write(f"Predicted: ${prediction[0]:.3f}")
            print(input_df)
            print(f"Predicted: ${prediction[0]:.3f}")

elif model_type == "Clustering":
    # Dropdown for clustering methods
    clustering_method = st.selectbox("Select Clustering Method", [
        "K-Means Clustering", 
        "Hierarchical Clustering", 
        "K-Medoids Clustering", 
        "Spectral Clustering"
    ])
    # Placeholder for clustering logic (to be implemented later)

# Add logic for Regression and Clustering as needed