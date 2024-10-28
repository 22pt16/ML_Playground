import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
    #handle unknown data

def train_svm():
    from CLASSIFICATION.SVM.svm_train import train_model  # Import your training function
    msg = train_model()  # Train and save model
    return msg
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
    if classification_method == "SVM":
        if st.button("Train Iris SVM Model"):
            msg = train_svm()
            st.success(msg)

        st.subheader("Iris Data Prediction")

        sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
        sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
        petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=4.0)
        petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=1.0)

        if st.button("Classify"):
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            from CLASSIFICATION.SVM.svm_test import predict_model
            prediction = predict_model(input_data)
            st.write("Prediction:", prediction)

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
   

    if regression_method == "Multiple Regression":
        DATA_PATH = 'Datasets/Students_Performance.csv'  # Adjust the path to your dataset
        from REGRESSION.MULTIPLEREGRESSION.mr_train import train_multiple_regression_model
        from REGRESSION.MULTIPLEREGRESSION.mr_test import test_multiple_regression_model

        target_column = 'Performance Index'  # Adjust to your actual target column

        # Train the Multiple Regression model
        if st.button("Train Multiple Regression Model"):
            test_x, test_y, mse, mae, r2, message = train_multiple_regression_model(DATA_PATH, target_column)
            st.success(message)

            # Display evaluation metrics
            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
            st.write(f"R² Score: {r2:.4f}")

        # Check if the model is trained
        if 'test_x' in locals () and 'test_y' in locals():
            # Display input sliders and dropdowns for prediction input
            st.subheader("Test the model with custom input")
            user_input = {
                'feature1': st.slider("Feature 1", min_value=0, max_value=100, value=50, step=1),
                'feature2': st.slider("Feature 2", min_value=0, max_value=100, value=50, step=1),
                # Add more features as needed
            }

            if st.button("Predict with Custom Input"):
                # Load model once
                MODEL_PATH = 'Saved_models/multiple_regression_model.pkl'
                with open(MODEL_PATH, 'rb') as model_file:
                    multiple_regression_model = pickle.load(model_file)

                # Handle unknown inputs by default values
                input_df = pd.DataFrame([user_input])

                # Predict the target variable using the corrected user input
                prediction = multiple_regression_model.predict(input_df)
                st.write(f"Predicted: {prediction[0]:.3f}")

    elif regression_method == "Decision Tree Regression":
        DATA_PATH = 'Datasets/SydneyHousePrices.csv'
        from REGRESSION.DECISION_TREE.dtr_train import train_decision_tree_model, meta_data, handle_unknown
        from REGRESSION.DECISION_TREE.dtr_test import test_decision_tree_model

        # SELECT DATASET 
        feature_columns, target_column = meta_data(1)  # as of now 1.SydneyHouseprices

        # Input hyperparameters for Decision Tree Regressor
        max_depth = st.slider("Max Depth of the Tree", min_value=1, max_value=50, value=10)
        min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2)
        min_samples_leaf = st.slider("Min Samples per Leaf", min_value=1, max_value=10, value=1)

        # Train the Decision Tree Regressor model
        if st.button("Train Decision Tree Model"):
            test_x, test_y, mse, mae, r2, unique_suburbs, unique_postal_codes, unique_prop_types, \
            (min_bed, max_bed), (min_bath, max_bath), (min_car, max_car), message = train_decision_tree_model(
                DATA_PATH, target_column, max_depth, min_samples_split, min_samples_leaf)
            st.success(message)

            # Display evaluation metrics
            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
            st.write(f"R² Score: {r2:.4f}")

            # Save unique values and range limits to session state
            st.session_state.unique_suburbs = unique_suburbs
            st.session_state.unique_postal_codes = unique_postal_codes
            st.session_state.unique_prop_types = unique_prop_types
            st.session_state.min_bed, st.session_state.max_bed = min_bed, max_bed
            st.session_state.min_bath, st.session_state.max_bath = min_bath, max_bath
            st.session_state.min_car, st.session_state.max_car = min_car, max_car

        # Check if unique values and range limits are available in session state
        if 'unique_suburbs' in st.session_state:
            # Display input sliders and dropdowns for prediction input
            st.subheader("Test the model with custom input")
            user_input = {
                'suburb': st.selectbox("Select Suburb", st.session_state.unique_suburbs),
                'postalCode': st.selectbox("Select Postal Code", st.session_state.unique_postal_codes),
                'propType': st.selectbox("Select Property Type", st.session_state.unique_prop_types),
                "bed": st.slider("Number of Bedrooms", min_value=st.session_state.min_bed, max_value=st.session_state.max_bed, value=st.session_state.min_bed, step = 1),
                "bath": st.slider("Number of Bathrooms", min_value=st.session_state.min_bath, max_value=st.session_state.max_bath, value=st.session_state.min_bath, step=1),
                "car": st.slider("Number of Car Spaces", min_value=st.session_state.min_car, max_value=st.session_state.max_car, value=st.session_state.min_car,step = 1)
            }

            if st.button("Predict with Custom Input"):
                # Load model once
                MODEL_PATH = 'Saved_models/decision_tree_regressor.pkl'
                with open(MODEL_PATH, 'rb') as model_file:
                    decision_tree_model = pickle.load(model_file)

                # Handle unknown inputs by default values
                # crct_input = handle_unknown(1, modes, user_input)
                input_df = pd.DataFrame([user_input])
                if target_column in input_df.columns:
                    input_df = input_df.drop(columns=[target_column])

                # Predict the target variable using the corrected user input
                prediction = decision_tree_model.predict(input_df)
                st.write(f"Predicted: ${prediction[0]:.3f}")

elif model_type == "Clustering":

    
    # Dropdown for clustering methods
    clustering_method = st.selectbox("Select Clustering Method", [
        "K-Means Clustering", 
        "Hierarchical Clustering", 
        "K-Medoids Clustering", 
        "Spectral Clustering"
    ])

    if clustering_method == "K-Medoids Clustering":

        from CLUSTERING.K_MEDOIDS.k_med_train import train_kmedoids
        from CLUSTERING.K_MEDOIDS.k_med_test import predict_cluster
        # Hyperparameters
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
        max_iter = st.number_input("Max Iterations", min_value=100, max_value=500, value=300)
        metric = st.selectbox("Distance Metric", ["euclidean"])

        if st.button("Train K-Medoids Model"):
            message, silhouette, data_pca, labels = train_kmedoids(n_clusters=n_clusters, max_iter=max_iter, metric=metric)
            st.success(message)

            # Store data_pca and labels in session state for later use
            st.session_state.data_pca = data_pca
            st.session_state.labels = labels
            
            # Plot training data
            fig, ax = plt.subplots()
            scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis')
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)
            st.pyplot(fig)

        # Prediction input
        if 'data_pca' in st.session_state:
            st.subheader("Predict Cluster for New Data")

            data_pca = st.session_state.data_pca
            labels = st.session_state.labels

            pca_min = np.min(data_pca, axis=0)
            pca_max = np.max(data_pca, axis=0)

            gdp_input = st.number_input("GDP per Capita", min_value=float(pca_min[0]), max_value=float(pca_max[0]), step=0.01)
            social_support_input = st.number_input("Social Support", min_value=float(pca_min[1]), max_value=float(pca_max[1]), step=0.01)
            life_expectancy = st.number_input("Life Expectancy", min_value=float(0.10), max_value=float(1.10), step=0.01)
            Freedom_of_choices = st.number_input("Freedom to make Life choices", min_value=float(0.01), max_value=float(0.60), step=0.01)

            if st.button("Predict Cluster"):
                new_data = [gdp_input, social_support_input, life_expectancy, Freedom_of_choices]
                cluster = predict_cluster(new_data)
                st.write(f"Predicted Cluster: {cluster}")

        
                # Plot prediction point on the existing data scatter plot
                fig, ax = plt.subplots()
                scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis')
                ax.scatter([new_data[0]], [new_data[1]], color='red', label="Prediction", s=100, edgecolor="black")
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)
                ax.legend()
                st.pyplot(fig)
                    
    # Placeholder for clustering logic (to be implemented later)

