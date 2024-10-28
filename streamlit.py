import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Handle unknown data

# 1. NAIVE BAYES
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

# KNN Functions
def load_iris_data():
    DATA_PATH = 'Datasets/IRIS.csv'
    data = pd.read_csv(DATA_PATH)
    X = data.drop('species', axis=1)
    y = data['species']
    return X, y

def train_knn():
    MODEL_PATH = 'Saved_models/knn_model.pkl'
    from CLASSIFICATION.KNN_Classification.knn_train import KNNClassifier  # Import your KNN training class
    X, y = load_iris_data()

    knn = KNNClassifier(n_neighbors=3)  # Adjust n_neighbors as needed
    knn.train_model(X, y)  # Train KNN model
    knn.save_model(MODEL_PATH)  # Save the model

    return "KNN model trained successfully!"

def validate_knn(sepal_length, sepal_width, petal_length, petal_width):
    MODEL_PATH = 'Saved_models/knn_model.pkl'
    with open(MODEL_PATH, 'rb') as model_file:
        knn_model = pickle.load(model_file)

    # Prepare the input for prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = knn_model.predict(input_data)

    return prediction[0]

# Streamlit interface
st.title("Machine Learning Playground")

# Dropdown for model type selection
model_type = st.selectbox("Select Model Type", ["Regression", "Classification", "Clustering"])

if model_type == "Classification":
    # Dropdown for classification methods
    classification_method = st.selectbox("Select Classification Method", [
        "Naive Bayes", 
        "KNN",  # Added KNN to the classification methods
        "Neural Network", 
        "SVM", 
        "Random Forest", 
        "Decision Tree Classifier", 
        "Gradient Boosting", 
        "Mixture of Gaussians"
    ])

    # Button to train the selected classification model
    if classification_method == "Naive Bayes":
        if st.button("Train Naive Bayes Model"):
            message = train_naive_bayes()
            st.success(message)

    elif classification_method == "KNN":
        if st.button("Train KNN Model"):
            message = train_knn()
            st.success(message)

        # Input area for testing KNN
        st.subheader("Test KNN Model")
        sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
        sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
        petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=4.0)
        petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=1.5)

        if st.button("Test KNN Model"):
            prediction = validate_knn(sepal_length, sepal_width, petal_length, petal_width)
            st.write(f"Predicted Species: {prediction}")

    # Placeholder for other classification methods (to be implemented later)

elif model_type == "Regression":
    # Dropdown for regression methods
    regression_method = st.selectbox("Select Regression Method", [
        "Linear Regression", 
        "Multiple Regression", 
        "Logistic Regression", 
        "Decision Tree Regression", 
        "Gradient Boosting (Regression)"
    ])

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

            # Save unique values and range limits to session state
            st.session_state.test_x = test_x
            st.session_state.test_y = test_y

        # Check if unique values and range limits are available in session state if 'test_x' in st.session_state:
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
                input_df = pd .DataFrame([user_input])
                if target_column in input_df.columns:
                    input_df = input_df.drop(columns=[target_column])

                # Predict the target variable using the corrected user input
                prediction = decision_tree_model.predict(input_df)
                st.write(f"Predicted: ${prediction[0]:.3f}")

                # Plot prediction point on the existing data scatter plot
                fig, ax = plt.subplots()
                scatter = ax.scatter(st.session_state.test_x[:, 0], st.session_state.test_x[:, 1], c=st.session_state.test_y, cmap='viridis')
                ax.scatter([user_input['bed']], [user_input['bath']], color='red', label="Prediction", s=100, edgecolor="black")
                legend1 = ax.legend(*scatter.legend_elements(), title="Prices")
                ax.add_artist(legend1)
                ax.legend()
                st.pyplot(fig)

elif model_type == "Clustering":

    # Dropdown for clustering methods
    clustering_method = st.selectbox("Select Clustering Method", [
        "K-Means Clustering", 
        "DBSCAN Clustering", 
        "K-Medoids Clustering", 
        "Spectral Clustering"
    ])

    if clustering_method == "K-Medoids Clustering":

        from CLUSTERING.K_MEDOIDS.k_med_train import train_kmedoids
        from CLUSTERING.K_MEDOIDS.k_med_test import predict_cluster              

        # User inputs for data generation
        shape = st.selectbox("Select Data Shape", ["blobs", "moons", "circles"])
        n_samples = st.number_input("Number of Samples", min_value=10, max_value=1000, value=100)
        noise = st.number_input("Noise Level", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

        # Hyperparameters for DBSCAN
        eps = st.number_input("Epsilon (eps)", min_value=0.01, max_value=5.0, value=0.5, step=0.01)
        min_samples = st.slider("Minimum Samples", min_value=1, max_value=50, value=5)

        if st.button("Train K-Medoids Model"):
            message, silhouette, data_pca, labels = train_kmedoids(shape=shape, n_samples=n_samples, eps=eps, min_samples=min_samples, noise=noise)
            st.success(message)

            # Display training plot
            st.image('Saved_models/kmedoids_plot.png')

            # Store data_pca and labels in session state
            st.session_state.data_pca = data_pca
            st.session_state.labels = labels

        # Prediction input
        if 'data_pca' in st.session_state:
            st.subheader("Predict Cluster for New Data")

            # User input for new data point
            feature1 = st.number_input("Feature 1", value=0.0)
            feature2 = st.number_input("Feature 2", value=0.0)

            if st.button("Predict Cluster"):
                new_data = [feature1, feature2]
                cluster = predict_cluster(new_data)
                st.write(f"Predicted Cluster: {cluster}")

                # Plot prediction point on the existing data scatter plot
                fig, ax = plt.subplots()
                scatter = ax.scatter(st.session_state.data_pca[:, 0], st.session_state.data_pca[:, 1], c=st.session_state.labels, cmap='viridis')
                ax.scatter(new_data[0], new_data[1], color='red', label="Prediction", s=100, edgecolor="black")
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)
                ax.legend()
                st.pyplot(fig)

    elif clustering_method == "DBSCAN Clustering" :
        from CLUSTERING.DBSCAN.dbscan_train import train_dbscan
        from CLUSTERING.DBSCAN.dbscan_test import predict_cluster              

        # User inputs for data generation
        shape = st.selectbox("Select Data Shape", ["blobs", "moons", "circles"])
        n_samples = st.number_input("Number of Samples", min_value=10, max_value=1000, value=100)
        noise = st.number_input("Noise Level", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

        # Hyperparameters for DBSCAN
        eps = st.number_input("Epsilon (eps)", min_value=0.01, max_value=5.0, value=0.5, step=0.01)
        min_samples = st.slider("Minimum Samples", min_value=1, max_value=50, value=5)

        if st.button("Train DBSCAN Model"):
            message, silhouette, data_pca, labels = train_dbscan(shape=shape, n_samples=n_samples, eps=eps, min_samples=min_samples, noise=noise)
            st.success(message)

            # Display training plot
            st.image('Saved_models/dbscan_plot.png')

            # Store data_pca and labels in session state
            st.session_state.data_pca = data_pca
            st.session_state.labels = labels

        # Prediction input
        if 'data_pca' in st.session_state:
            st.subheader("Predict Cluster for New Data")

            # User input for new data point
            feature1 = st.number_input("Feature 1", value=0.0)
            feature2 = st.number_input("Feature 2", value=0.0)

            if st.button("Predict Cluster"):
                new_data = [feature1, feature2]
                cluster = predict_cluster(new_data)
                st.write(f"Predicted Cluster: {cluster}")

                # Plot prediction point on the existing data scatter plot
                fig, ax = plt.subplots()
                scatter = ax.scatter(st.session_state.data_pca[:, 0], st.session_state.data_pca[:, 1], c=st.session_state.labels, cmap='viridis')
                ax.scatter(new_data[0], new_data[1], color='red', label="Prediction", s=100, edgecolor="black")
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)
                ax.legend()
                st.pyplot(fig)