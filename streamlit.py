import streamlit as st
import pickle

# Paths to your dataset and model
GOOD_EMAILS_PATH = 'Datasets/good_emails.txt'
SPAM_EMAILS_PATH = 'Datasets/spam_emails.txt'
MODEL_PATH = 'Saved_models/naive_bayes_model.pkl'

# Load the training datasets
def load_emails():
    with open(GOOD_EMAILS_PATH, 'r') as f:
        good_emails = f.readlines()
    with open(SPAM_EMAILS_PATH, 'r') as f:
        spam_emails = f.readlines()
    return good_emails, spam_emails

# Train the Naive Bayes model
def train_naive_bayes():
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