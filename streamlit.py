import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Handle unknown data

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
        "Decision Tree Classifier", 
        "Gradient Boosting", 
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
    elif classification_method == "SVM":
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
    elif classification_method == "Decision Tree Classifier":
        from CLASSIFICATION.DT_CLASS.dtc_train import load_data, fit, predict, accuracy, calculate_metrics

        
        # Sidebar: Model parameters
        st.sidebar.header("Model Parameters")
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
        max_depth = st.sidebar.slider("Max Depth", 5, 25, 5)

        # Load dataset
        X, y = load_data()

        # Train model on button click
        if st.sidebar.button("Train Model"):
            # Train decision tree model
            model = fit(X, y, max_depth=max_depth, criterion=criterion)
            predictions = predict(model, X)
            acc = accuracy(y, predictions)
            precision, recall, f1 = calculate_metrics(y, predictions)

            st.sidebar.write(f"Training Accuracy: {acc:.2f}")
            st.sidebar.write(f"Precision: {precision:.2f}")
            st.sidebar.write(f"Recall: {recall:.2f}")
            st.sidebar.write(f"F1 Score: {f1:.2f}")

            # Save model to file for further use
            with open("decision_tree_model.pkl", "wb") as f:
                pickle.dump(model, f)

        # Test new data point
        st.header("Classify New Data Point")
        pclass = st.slider("Pclass (1=1st, 2=2nd, 3=3rd)", 1, 3, 3)
        sex = st.selectbox("Sex", ["male", "female"])
        sex = 1 if sex == "male" else 0
        age = st.slider("Age", 1, 80, 30)
        fare = st.slider("Fare", 0.0, 500.0, 8.05)

        # Classify the new data point
        if st.button("Classify New Data Point"):
            # Load model from pickle file
            try:
                with open("decision_tree_model.pkl", "rb") as f:
                    model = pickle.load(f)
                new_point = np.array([pclass, sex, age, fare])
                prediction = predict(model, np.array([new_point]))[0]
                st.write(f"The new data point is predicted to: {'Survive' if prediction == 1 else 'Not Survive'}")
            except FileNotFoundError:
                st.write("Model not found. Please train the model first by clicking 'Train Model'.")

        
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
       
        from REGRESSION.MULTIPLEREGRESSION.mr_train import train_multiple_regression_model

        DATA_PATH = 'Datasets/Students_Performance.csv'  
        target_column = 'Performance Index'  # Adjust to your actual target column

        # Initialize a session state variable to track model training
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
            st.session_state.test_x = None
            st.session_state.test_y = None
            st.session_state.model = None

        # Train the Multiple Regression model
        if st.button("Train Multiple Regression Model"):
            print("Training the model")
            test_x, test_y, model, mse, mae, r2, message = train_multiple_regression_model(DATA_PATH, target_column)
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
                'Hours Studied': st.slider("Hours Studied", min_value=0, max_value=100, value=50, step=1),
                'Previous Scores': st.slider("Previous Scores", min_value=0, max_value=100, value=50, step=1),
                'Extracurricular Activities': st.selectbox("Extracurricular Activities", ["Yes", "No"]),
                'Sleep Hours': st.slider("Sleep Hours", min_value=0, max_value=24, value=8, step=1),
                'Sample Question Papers Practiced': st.slider("Sample Question Papers Practiced", min_value=0, max_value=10, value=1, step=1)
            }

            if st.button("Predict with Custom Input"):
                # Load model from session state
                multiple_regression_model = st.session_state.model

                # Convert user input into DataFrame
                input_df = pd.DataFrame([user_input])

                # Encode categorical features (make sure to match the preprocessing done during training)
                input_df['Extracurricular Activities'] = input_df['Extracurricular Activities'].map({"Yes": 1, "No": 0})

                # Predict the target variable using the corrected user input
                prediction = multiple_regression_model.predict(input_df)
                st.write(f"Predicted Performance Index: {prediction[0]:.3f}")

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
        "PCA",
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
            message, silhouette, data_pca, labels = train_kmedoids( n_clusters, max_iter, metric)
            st.success(message)

            
           

            # Store data_pca and labels in session state
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
    elif clustering_method == "Spectral Clustering":
        from CLUSTERING.SPECTRAL.spectral_train import generate_data, spectral_clustering_from_scratch
        from CLUSTERING.SPECTRAL.spectral_test import predict_cluster
        # Select data type and parameters
        data_type = st.selectbox("Choose Data Type", ['circles', 'moons'])
        n_clusters = st.slider("Number of Clusters", min_value=8, max_value=50, value=8)
        n_samples = st.slider("Number of Samples", min_value=800, max_value=3000, value=1000)
        gamma = st.slider("Gamma for RBF Kernel", min_value=0.20, max_value=150.0, value=15.0)

        # Train and visualize clustering on button click
        if st.button("Train and Visualize Clustering"):
            # Generate data and perform clustering
            X, _ = generate_data(data_type, n_samples)
            labels, silhouette = spectral_clustering_from_scratch(X, n_clusters, gamma)

            # Display silhouette score
            st.write(f"Silhouette Score: {silhouette:.2f}")

            # Visualize clustering result
            fig, ax = plt.subplots()
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
            plt.colorbar(scatter, ax=ax, label="Cluster")
            plt.title("Spectral Clustering Visualization")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            st.pyplot(fig)

            # Store data for testing predictions
            st.session_state.X = X
            st.session_state.labels = labels

        # Testing part: Get cluster for a new data point
        st.write("## Test New Data Point")
        feature_1 = st.number_input("Feature 1", value=0.0)
        feature_2 = st.number_input("Feature 2", value=0.0)

        if st.button("Get Cluster for New Point"):
            if 'X' in st.session_state and 'labels' in st.session_state:
                # Predict the cluster for the new point
                new_point = np.array([feature_1, feature_2])
                cluster = predict_cluster(new_point, st.session_state.X, st.session_state.labels)
                st.write(f"The new point ({feature_1}, {feature_2}) is assigned to cluster: {cluster}")

                # Visualize the dataset with the new point marked
                fig, ax = plt.subplots()
                scatter = ax.scatter(st.session_state.X[:, 0], st.session_state.X[:, 1], c=st.session_state.labels, cmap='viridis', s=10)
                ax.scatter(new_point[0], new_point[1], c='red', marker='X', s=100, label="New Point")
                plt.colorbar(scatter, ax=ax, label="Cluster")
                plt.title("Spectral Clustering Visualization with New Point")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                plt.legend()
                st.pyplot(fig)
            else:
                st.write("Please train the clustering model first by clicking 'Train and Visualize Clustering'.")
    elif clustering_method == "PCA":
        from CLUSTERING.PCA.pca_train import generate_synthetic_data, train_pca_kmeans, load_models, pca_from_scratch

                # Sidebar for parameters
        st.sidebar.header("Parameters")
        n_samples = st.sidebar.slider("Number of Samples", 100, 2000, 1000)
        n_features = st.sidebar.slider("Number of Features", 2, 20, 10)
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
        n_components = st.sidebar.slider("PCA Components", 2, n_features, 2)

        # Generate and train model on button click
        if st.button("Train Model"):
            X, y = generate_synthetic_data(n_samples=n_samples, n_features=n_features, n_clusters=n_clusters)
            labels, X_reduced, silhouette = train_pca_kmeans(X, n_clusters=n_clusters, n_components=n_components)
            
            st.write(f"Silhouette Score: {silhouette:.2f}")
            
            # Store training data for visualization with test data later
            st.session_state['X_reduced'] = X_reduced
            st.session_state['labels'] = labels
            
            # Visualize the PCA-reduced data
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', label="Training Data")
            ax.set_title("PCA-Reduced Data and Clusters (Train Set)")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            st.pyplot(fig)

        # Test new data point
        st.header("Test Model with New Data Point")
        feature_values = [st.number_input(f"Feature {i+1}", value=0.0) for i in range(n_features)]

        if st.button("Classify New Data Point"):
            eigenvector_subset, mean_vector, kmeans = load_models()
            
            # PCA from scratch on new data point using training mean vector
            new_point = np.array([feature_values])
            new_point_meaned = new_point - mean_vector  # Use training mean vector
            new_point_reduced = np.dot(new_point_meaned, eigenvector_subset)
            predicted_cluster = kmeans.predict(new_point_reduced)[0]
            
            st.write(f"The new data point belongs to cluster: {predicted_cluster}")
            st.write(f"PCA-Reduced Coordinates: {new_point_reduced[0]}")

            # Visualize the new data point along with training data
            fig, ax = plt.subplots()
            # Plot the training data
            scatter = ax.scatter(st.session_state['X_reduced'][:, 0], st.session_state['X_reduced'][:, 1], 
                                c=st.session_state['labels'], cmap='viridis', label="Training Data")
            
            # Plot the new test data point
            ax.scatter(new_point_reduced[0, 0], new_point_reduced[0, 1], c="red", s=100, label="Test Data Point", marker="x")
            ax.set_title("PCA-Reduced Data with New Data Point (Test)")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.legend()
            st.pyplot(fig)
