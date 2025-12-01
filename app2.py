import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, silhouette_score
import plotly.express as px
import time
import os
from PIL import Image

def visualize_clusters(data_scaled, clusters, model_name, n_clusters_or_eps, num_features, feature_names):
    """Visualizes clusters using PCA or direct scatter plot."""
    plt.style.use('seaborn-v0_8-darkgrid') # Optional: set a plot style

    unique_clusters = np.unique(clusters)
    if len(unique_clusters) == 0 : # Handle case where fit fails and returns nothing
        st.warning("⚠️ No clusters were generated to visualize.")
        return
    elif len(unique_clusters) == 1 and unique_clusters[0] == -1: # Handle DBSCAN finding only noise
         st.warning("⚠️ Only noise points found by DBSCAN, no clusters to visualize.")
         return

    if num_features > 2:
        st.subheader(f"Visualizing {model_name} Clusters using PCA")
        try:
            pca = PCA(n_components=2, random_state=42)
            principal_components = pca.fit_transform(data_scaled)
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = clusters

            fig, ax = plt.subplots(figsize=(8, 6))
            # Use a categorical palette suitable for clusters
            palette = sns.color_palette("viridis", as_cmap=False, n_colors=len(unique_clusters))
            sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette=palette, data=pca_df, ax=ax, legend='full', s=50, alpha=0.7)

            ax.set_title(f'{model_name} Clusters ({n_clusters_or_eps} clusters/eps) - PCA Projection')
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            # Place legend outside the plot
            ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
            st.pyplot(fig)
            plt.close(fig) # Close the figure to free memory
        except Exception as e:
            st.error(f"Error during PCA visualization: {e}")

    elif num_features == 2:
        st.subheader(f"Visualizing {model_name} Clusters (2D)")
        try:
            scatter_df = pd.DataFrame(data_scaled, columns=feature_names)
            scatter_df['Cluster'] = clusters

            fig, ax = plt.subplots(figsize=(8, 6))
            palette = sns.color_palette("viridis", as_cmap=False, n_colors=len(unique_clusters))
            sns.scatterplot(x=feature_names[0], y=feature_names[1], hue='Cluster', palette=palette, data=scatter_df, ax=ax, legend='full', s=50, alpha=0.7)

            ax.set_title(f'{model_name} Clusters ({n_clusters_or_eps} clusters/eps)')
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
            ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error during 2D scatter plot visualization: {e}")

    elif num_features == 1:
         st.subheader(f"Visualizing {model_name} Clusters (1D)")
         try:
            scatter_df = pd.DataFrame({'Feature': data_scaled[:, 0], 'Cluster': clusters})

            fig, ax = plt.subplots(figsize=(8, 3)) # Adjust size for 1D
            palette = sns.color_palette("viridis", as_cmap=False, n_colors=len(unique_clusters))
            # Use stripplot for 1D visualization
            sns.stripplot(x='Feature', y=[''] * len(scatter_df), hue='Cluster', palette=palette, data=scatter_df, ax=ax, jitter=0.2, legend='full', s=5, alpha=0.7)

            ax.set_title(f'{model_name} Clusters ({n_clusters_or_eps} clusters/eps)')
            ax.set_xlabel(feature_names[0])
            ax.set_yticks([]) # Remove y-axis ticks
            ax.set_ylabel('') # Remove y-axis label
            ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            st.pyplot(fig)
            plt.close(fig)
         except Exception as e:
            st.error(f"Error during 1D strip plot visualization: {e}")
    else: # num_features == 0 - should not happen due to earlier checks
        st.warning("No features available for visualization.")

def main():
    logo_path = "logo.png"  # Replace with the actual path to your logo

    if os.path.exists(logo_path):
        st.set_page_config(page_title="DeepStat", page_icon=logo_path, layout="wide")

    # Image paths
    image_paths = ["preet.jpg", "aakanksha.jpeg", "sakshi.jpg"] 
    owner_names = ["Preet Jain (D023)", "Aakanksha Pote (D014)", "Sakshi Prabhu (D026)"]

    # Gradient blue effect for the title area with logo #483D8B, #8A2BE2
    st.markdown(
        f"""
        <style>
        .title-area {{
            background: linear-gradient(to right, #1e3c72, #2a5298); 
            padding: 20px;
            text-align: center;
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .title-area h1 {{
            display: inline;
        }}
        .logo-area {{
            position: absolute;
            top: 10px;
            left: 10px;
        }}
        .logo-area img {{
            max-height: 80px;  /* Adjust the logo size as needed */
        }}
        .footer {{
            background-color: #f0f2f6;
            text-align: center;
            padding: 20px 0;
            font-size: 14px;
            color: #555;
        }}
        .disclaimer {{
            background-color: #f0f2f6;
            text-align: center;
            padding: 10px 0;
            font-size: 12px;
            color: #888;
        }}
        .owner-images-container {{
            display: flex;
            justify-content: space-around;
            align-items: center;
            margin-bottom: 20px;
        }}
        .owner-image-wrapper {{
            text-align: center;
        }}
        .owner-image-wrapper img {{
            max-width: 200px; /* Adjust image size as needed */
            max-height: 200px;
            border-radius: 50%; /* Make images circular */
            object-fit: cover; /* Maintain aspect ratio */
            margin-bottom: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if os.path.exists(logo_path):
        logo_img = Image.open(logo_path)
        st.markdown(f'<div class="logo-area"><img src="data:image/png;base64,{open(logo_path, "rb").read().hex()}"></div>', unsafe_allow_html=True)

    st.markdown(f"<div class='title-area'><h1>DeepStat: From Raw Data to Clarity</h1></div>", unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("Data Input & Analysis Options")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])



    # Initial tabs before "Analyze" is clicked
    if not st.session_state.get("analyze", False):
        description_tab, about_team_tab = st.tabs(["Description", "About the Team"])

        with description_tab:
            st.header("Project Description")
            # Display the logo after the header
            if os.path.exists(logo_path):
                st.image(logo_path, width=100)  # Adjust width as needed
            else:
                st.write("Logo not found.")  # Optional message if logo is missing
            st.write("DeepStat is a comprehensive data analysis tool designed to simplify the process of extracting meaningful insights from raw data. It provides a user-friendly interface to upload, explore, clean, visualize, and model data, empowering users to make data-driven decisions.")
            st.write("Key Features Include:")
            st.write("-   Data Upload and Exploration: Easily upload CSV files and get a quick overview of the data.")
            st.write("-   Data Cleaning: Tools to handle missing values and outliers.")
            st.write("-   Data Visualization: Generate various plots to understand data patterns.")
            st.write("-   Model Training and Prediction: Train machine learning models and make predictions.")

            # You can add more details here

        with about_team_tab:
            st.header("About the Team")

            # Display the logo after the header
            if os.path.exists(logo_path):
                st.image(logo_path, width=100)  # Adjust width as needed
            else:
                st.write("Logo not found.")  # Optional message if logo is missing

            st.write("Welcome to DeepStat! We are a team of data enthusiasts passionate about transforming raw data into actionable insights.")
            # Display owner images and names side by side
            if len(image_paths) == len(owner_names):
                cols = st.columns(len(image_paths))  # Create columns for each image
                for i in range(len(image_paths)):
                    if os.path.exists(image_paths[i]):
                        cols[i].image(image_paths[i], caption=owner_names[i], width=200)  # Display image in each column
                    else:
                        st.warning(f"Image not found: {image_paths[i]}")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("File uploaded successfully!")

            columns_to_remove = st.sidebar.multiselect("Select columns to remove:", df.columns)
            if st.sidebar.button("Remove Selected Columns"):
                if columns_to_remove:
                    df = df.drop(columns=columns_to_remove)
                    st.session_state.df = df
                    st.success(f"Columns '{', '.join(columns_to_remove)}' removed.")
                else:
                    st.info("No columns selected for removal.")
            if st.sidebar.button("Analyze Data"):
                st.session_state.analyze = True

        except Exception as e:
            st.error(f"Error reading file: {e}")

    if 'df' in st.session_state:
        df = st.session_state.df

        if st.session_state.get("analyze", False):
            tab_titles = [
                "Description", "Data Exploration", "Correlation", "Missing Values",
                "Outliers", "Visualization", "Supervised Learning Model",
                "Unsupervised Learning Model", "Dashboard", "About the Team" # Added Dashboard, moved About
            ]
            # Ensure you list all 10 variables here
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(tab_titles)
            with tab1:
                st.header("Project Description")
                # Display the logo after the header
                if os.path.exists(logo_path):
                    st.image(logo_path, width=100)  # Adjust width as needed
                else:
                    st.write("Logo not found.")  # Optional message if logo is missing
                st.write("1. Basic Information About Data")
                st.write("a)Head: Shows the first few rows of the dataset.Helps you quickly understand what kind of data is stored and how it's organized.")
                st.write("b)Tail: Shows the last few rows of the dataset. Useful for checking if the end of the data is consistent or contains errors.")
                st.write("c)Info: Gives a summary of the dataset auch as number of rows and columns, column names and data types (e.g., numbers, text, dates)")
                st.write("d)Unique Values: Tells how many distinct values are in a column and helps identify if the column is categorical or has repeated entries.")
                st.write("")
                st.write("2. Correlation: Correlation tells you how strongly two variables are related.")
                st.write("Heatmap: A colored grid that shows which variables go up or down together.")
                st.write("Scatter Matrix: A grid of scatter plots showing how every variable relates to every other variable.")
                st.write("")
                st.write("3. Missing Values: Sometimes data is incomplete (some cells are empty). You can fix this by:")
                st.write("Replacing with mean or median: Use the average value to fill in the missing spot.")
                st.write("Removing the row: Delete the entry if it has too many missing parts or if it's not important.")
                st.write("")
                st.write("4. Outliers: Outliers are values that are much higher or lower than normal(If most people are aged 20 to 40, someone aged 90 is an outlier.).")
                st.write("Fix by: Replacing it with mean or median to bring it closer to the normal range.")
                st.write("")
                st.write("5. Graphs & Visualizations: Choose the graph based on the type of data")
                st.write("Bar Graph: For comparing categories (e.g., number of cars by color).")
                st.write("Line Chart: To show trends over time (e.g., sales over months).")
                st.write("Histogram: For seeing the distribution of a single variable (e.g., how many people fall into each age group)")
                st.write("Scatter Plot: To see the relationship between two numbers (e.g., height vs weight).")
                st.write("")
                st.write("6. Model Training & Prediction")
                st.write("a)Variables")
                st.write("Independent Variable: The input or cause (e.g., study hours).")
                st.write("Dependent Variable: The outcome you want to predict (e.g., exam score)")
                st.write("b)Preprocessing")
                st.write("One Hot Encoding: Turns categories (like “Red”, “Blue”) into numbers so that models can understand them")
                st.write("Min-Max Scaling: Makes all numeric values fall between 0 and 1, so one column doesn’t dominate the model.")
                st.write("")
                st.write("c) Machine Learning Models")
                st.write("Logistic Regression: Best for yes/no type problems (e.g., will someone buy or not).")
                st.write("Random Forest: Works well with both categories and numbers, even if the data is messy")
                st.write("SVC (Support Vector Classifier): Good when the data is clearly separated into groups.")
                st.write("KNN (K-Nearest Neighbors): Simple model that looks at similar past examples to make a prediction. Works best for small datasets and when patterns are clear.")
                st.write("d) Evaluation Metrics")
                st.write("Accuracy: How many predictions were correct.")
                st.write("Precision: Of the predicted positives, how many were actually positive.")
                st.write("Recall: Of all actual positives, how many did we correctly find.")
                st.write("F1 Score: A balance between precision and recall. Good when data is unbalanced.")
                    # You can add more details here



            with tab2:
                with st.expander("Basic Data Analysis", expanded=True):
                    st.write("### Head of the dataset")
                    st.dataframe(df.head())
                    st.write("### Tail of the dataset")
                    st.dataframe(df.tail())
                    st.write("### Full dataset")
                    st.dataframe(df)
                    st.write("### Unique Values per Column")
                    unique_values = df.nunique()
                    st.write(unique_values)
                    st.write("### Summary Statistics")
                    st.write(df.describe())

            with tab3:
                with st.expander("Correlation Analysis"):
                    correlation_type = st.radio("Choose correlation visualization:", ("None", "Heatmap", "Scatter Matrix"))

                    if correlation_type == "Heatmap":
                        df_corr = df.copy()
                        categorical_cols = df_corr.select_dtypes(include=['object', 'category']).columns
                        label_encoders = {}
                        for col in categorical_cols:
                            label_encoders[col] = LabelEncoder()
                            df_corr[col] = label_encoders[col].fit_transform(df_corr[col])

                        if not df_corr.empty:
                            corr = df_corr.corr()
                            fig, ax = plt.subplots()
                            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                            st.pyplot(fig)
                        else:
                            st.warning("No columns found for correlation heatmap after encoding.")
                    elif correlation_type == "Scatter Matrix":
                        df_scatter = df.copy()
                        categorical_cols_scatter = df_scatter.select_dtypes(include=['object', 'category']).columns
                        label_encoders_scatter = {}
                        for col in categorical_cols_scatter:
                            label_encoders_scatter[col] = LabelEncoder()
                            df_scatter[col] = label_encoders_scatter[col].fit_transform(df_scatter[col])

                        numeric_df_scatter = df_scatter.select_dtypes(include=np.number)
                        if len(numeric_df_scatter.columns) > 1:
                            try:
                                fig = sns.pairplot(numeric_df_scatter)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error creating scatter matrix: {e}")
                                st.info("Try selecting fewer columns or ensure they are suitable for a scatter plot.")
                        else:
                            st.warning("Need more than one column for scatter matrix after encoding.")
                    elif correlation_type == "None":
                        st.info("No correlation visualization selected.")

            with tab4:
                with st.expander("Handling Missing Values"):
                    missing_values = df.isnull().sum()
                    total_missing = missing_values.sum()
                    if total_missing > 0:
                        st.write(missing_values)
                        null_action = st.radio("How do you want to handle missing values?", ("None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"))
                        if null_action == "Drop Rows":
                            df = df.dropna()
                            st.success("Dropped rows with missing values!")
                        elif null_action == "Fill with Mean":
                            df = df.fillna(df.mean(numeric_only=True))
                            st.success("Filled missing values with column mean!")
                        elif null_action == "Fill with Median":
                            df = df.fillna(df.median(numeric_only=True))
                            st.success("Filled missing values with column median!")
                        elif null_action == "Fill with Mode":
                            df = df.fillna(df.mode().iloc[0])
                            st.success("Filled missing values with column mode!")
                        st.write("### Updated Missing Values Count:")
                        st.write(df.isnull().sum())
                    else:
                        st.write("✅ No missing values in the dataset!")

            with tab5:
                with st.expander("Outlier Detection & Handling"):
                    numeric_cols = df.select_dtypes(include=np.number).columns
                    if len(numeric_cols) > 0:
                        outlier_counts = {}
                        for col in numeric_cols:
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            outlier_counts[col] = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                        outlier_df = pd.DataFrame(list(outlier_counts.items()), columns=["Column", "Outlier Count"])
                        st.write("### Number of Outliers per Column (IQR Method):")
                        st.dataframe(outlier_df, hide_index=True)

                        # Box Plot Visualization
                        selected_boxplot_column_outliers = st.selectbox("Select a numeric column to visualize its box plot:", numeric_cols)
                        if selected_boxplot_column_outliers:
                            fig_boxplot_outliers, ax_boxplot_outliers = plt.subplots()
                            sns.boxplot(y=df[selected_boxplot_column_outliers], ax=ax_boxplot_outliers)
                            ax_boxplot_outliers.set_title(f"Box Plot of {selected_boxplot_column_outliers} (Before Handling)")
                            st.pyplot(fig_boxplot_outliers)

                        selected_columns = st.multiselect("Select column(s) to handle outliers", numeric_cols)
                        if selected_columns:
                            outlier_action = st.radio("How do you want to handle outliers?", ("None", "Remove Outliers", "Replace with Mean", "Replace with Median"))
                            for col in selected_columns:
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                if outlier_action == "Remove Outliers":
                                    df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
                                    st.success(f"Removed outliers from {col}!")
                                elif outlier_action == "Replace with Mean":
                                    mean_value = df[col].mean()
                                    df[col] = np.where((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)), mean_value, df[col])
                                    st.success(f"Replaced outliers in {col} with mean value!")
                                elif outlier_action == "Replace with Median":
                                    median_value = df[col].median()
                                    df[col] = np.where((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)), median_value, df[col])
                                    st.success(f"Replaced outliers in {col} with median value!")
                            updated_outlier_counts = {}
                            for col in numeric_cols:
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                updated_outlier_counts[col] = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                            updated_outlier_df = pd.DataFrame(list(updated_outlier_counts.items()), columns=["Column", "Updated Outlier Count"])
                            st.write("### Updated Outlier Counts:")
                            st.dataframe(updated_outlier_df, hide_index=True)

                            # Updated Box Plot Visualization
                            st.write("### Box Plot After Handling (if handling was done):")
                            if selected_boxplot_column_outliers:
                                fig_boxplot_updated, ax_boxplot_updated = plt.subplots()
                                sns.boxplot(y=df[selected_boxplot_column_outliers], ax=ax_boxplot_updated)
                                ax_boxplot_updated.set_title(f"Box Plot of {selected_boxplot_column_outliers} (After Handling)")
                                st.pyplot(fig_boxplot_updated)

                    else:
                        st.info("No numeric columns available for outlier detection and handling.")

            with tab6:
                with st.expander("Data Visualization"):
                    vis_cols = st.multiselect("Select columns to visualize", df.columns)
                    graph_type = st.selectbox("Choose a graph type", ["Histogram", "Boxplot", "Scatterplot", "Bar Chart", "Line Chart"])
                    if vis_cols:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        if graph_type == "Histogram":
                            for col in vis_cols:
                                sns.histplot(df[col], kde=True, ax=ax, label=col)
                            ax.legend()
                        elif graph_type == "Boxplot":
                            sns.boxplot(data=df[vis_cols], ax=ax)
                        elif graph_type == "Scatterplot" and len(vis_cols) >= 2:
                            sns.scatterplot(x=df[vis_cols[0]], y=df[vis_cols[1]], ax=ax)
                        elif graph_type == "Line Chart" and len(vis_cols) >= 2:
                            for col in vis_cols[1:]:
                                sns.lineplot(x=df[vis_cols[0]], y=df[col], ax=ax, label=col)
                            ax.legend()
                        st.pyplot(fig)

            with tab7:
                st.header("Model Training and Prediction")
                with st.expander("Feature Selection for Model Training"):
                    all_columns = df.columns.tolist()
                    independent_variables = st.multiselect("Select Independent Variables (Features)", all_columns)
                    dependent_variable = st.selectbox("Select Dependent Variable (Target)", all_columns)

                    if independent_variables and dependent_variable:
                        if dependent_variable in independent_variables:
                            st.error("Dependent variable cannot be in the independent variables.")
                        else:
                            X_original = df[independent_variables].copy()
                            y = df[dependent_variable]

                            st.write("### Original Independent Variables (X):")
                            st.dataframe(X_original)
                            st.write("### Dependent Variable (y):")
                            st.dataframe(y.head())

                            st.subheader("Data Preprocessing - One-Hot Encoding (Independent Variables)")
                            categorical_cols_X = X_original.select_dtypes(include=['object', 'category']).columns
                            if len(categorical_cols_X) > 0:
                                encoder_X = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                                X_encoded = encoder_X.fit_transform(X_original[categorical_cols_X])
                                encoded_df_X = pd.DataFrame(X_encoded, columns=encoder_X.get_feature_names_out(categorical_cols_X))
                                X_processed = X_original.drop(columns=categorical_cols_X, errors='ignore')
                                X_processed = pd.concat([X_processed.reset_index(drop=True), encoded_df_X.reset_index(drop=True)], axis=1)
                                st.session_state.encoder_X = encoder_X
                                st.success("One-Hot Encoding applied to independent variable(s).")
                            else:
                                X_processed = X_original.copy()
                                st.info("No categorical columns found in the independent variable(s) for One-Hot Encoding.")

                            st.write("### Independent Variables (X) After One-Hot Encoding:")
                            st.dataframe(X_processed)
                            st.write("### Dependent Variable (y):")
                            st.dataframe(y)

                            st.subheader("Min-Max Scaling of Independent Variables")
                            scaler = MinMaxScaler()
                            X_scaled = scaler.fit_transform(X_processed)
                            X_scaled_df = pd.DataFrame(X_scaled, columns=X_processed.columns)
                            st.session_state.scaler = scaler
                            st.success("Min-Max Scaling applied to independent variable(s).")
                            st.write("### Independent Variables (X) After Min-Max Scaling:")
                            st.dataframe(X_scaled_df)

                            st.subheader("Model Selection")
                            model_choice = st.selectbox("Select Model:", ("Logistic Regression", "Random Forest", "SVC", "KNN"))

                            if model_choice == "Logistic Regression":
                                model = LogisticRegression()
                            elif model_choice == "Random Forest":
                                model = RandomForestClassifier()
                            elif model_choice == "SVC":
                                model = SVC()
                            elif model_choice == "KNN":
                                model = KNeighborsClassifier()

                            test_size = st.slider("Select Test Set Size (%):", 0, 100, 20) / 100

                            if st.button("Train Model") or 'model' not in st.session_state or st.session_state.get("previous_test_size") != test_size:
                                if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
                                    label_encoder = LabelEncoder()
                                    y_encoded = label_encoder.fit_transform(y)
                                    st.session_state.label_encoder = label_encoder
                                    st.success("Dependent variable label encoded.")
                                else:
                                    y_encoded = y

                                X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_encoded, test_size=test_size, random_state=42)

                                model.fit(X_train, y_train)
                                st.session_state.model = model
                                st.session_state.trained_model_choice = model_choice
                                st.session_state.previous_test_size = test_size
                                st.success(f"{model_choice} model trained!")

                                y_pred = model.predict(X_test)
                                accuracy = accuracy_score(y_test, y_pred)
                                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                                st.subheader(f"{model_choice} Model Evaluation")
                                st.write(f"Accuracy: {accuracy:.4f}")
                                st.write(f"Precision: {precision:.4f}")
                                st.write(f"Recall: {recall:.4f}")
                                st.write(f"F1-Score: {f1:.4f}")
                                st.write("### Classification Report:")
                                st.text(classification_report(y_test, y_pred, zero_division=0))

                            if 'model' in st.session_state:
                                with st.form("prediction_form"):
                                    st.subheader("Make a Prediction with Original Data")
                                    st.write("Enter the values for the independent variables (in their original format and order):")
                                    prediction_data_original = []
                                    for col in X_original.columns:
                                        if pd.api.types.is_numeric_dtype(X_original[col]):
                                            prediction_data_original.append(st.number_input(f"Value for {col}", value=0.0))
                                        else:
                                            unique_vals = X_original[col].unique().tolist()
                                            prediction_data_original.append(st.selectbox(f"Value for {col}", unique_vals))
                                    submitted = st.form_submit_button("Predict")

                                if submitted:
                                    try:
                                        prediction_df_original = pd.DataFrame([prediction_data_original], columns=X_original.columns)

                                        if 'encoder_X' in st.session_state and len(categorical_cols_X) > 0:
                                            categorical_preds = prediction_df_original[categorical_cols_X]
                                            encoded_preds = st.session_state.encoder_X.transform(categorical_preds)
                                            encoded_preds_df = pd.DataFrame(encoded_preds, columns=st.session_state.encoder_X.get_feature_names_out(categorical_cols_X))
                                            numerical_preds = prediction_df_original.drop(columns=categorical_cols_X, errors='ignore')
                                            prediction_processed = pd.concat([numerical_preds.reset_index(drop=True), encoded_preds_df.reset_index(drop=True)], axis=1)
                                        else:
                                            prediction_processed = prediction_df_original.copy()

                                        if 'scaler' in st.session_state:
                                            prediction_scaled = st.session_state.scaler.transform(prediction_processed)
                                        else:
                                            prediction_scaled = prediction_processed.values

                                        if 'model' in st.session_state:
                                            predicted_raw = st.session_state.model.predict(prediction_scaled)[0]

                                        if 'label_encoder' in st.session_state:
                                            predicted_class = st.session_state.label_encoder.inverse_transform([predicted_raw])[0]
                                            st.write(f"### Predicted Class (Original Labels): {predicted_class}")
                                        else:
                                            st.write(f"### Predicted Value: {predicted_raw}")

                                    except Exception as e:
                                        st.error(f"Error during prediction: {e}")
                                        st.error("Please ensure you have entered the data in the correct format and order.")
                            else:
                                st.warning("Please train the model first.")


            with tab8:
                st.header("Unsupervised Learning Model")

                # Check if data exists *before* algorithm selection
                if 'df' not in st.session_state or st.session_state.df is None:
                    st.warning("⚠️ Please upload a dataset first on the 'Upload Data' tab.")
                    st.stop() # Stop execution in this tab if no data

                clustering_option = st.selectbox(
                    "Select Clustering Algorithm:",
                    # K-Medoids Removed from this list
                    ["K-Means Clustering", "DBSCAN", "Hierarchical Clustering", "Gaussian Mixture Models (GMM)"],
                    key="unsupervised_algo_select"
                )

                # --- K-Means Clustering ---
                if clustering_option == "K-Means Clustering":
                    st.subheader("K-Means Clustering")
                    with st.expander("K-Means Clustering Steps & Parameters"):
                        st.write("Assigns each data point to one of K clusters based on minimizing the within-cluster sum of squares (inertia). Uses cluster centroids (means).")
                        st.write("Steps involved:")
                        st.write("1. Drop non-numeric columns.")
                        st.write("2. Standardize the numeric data.")
                        st.write("3. Apply K-Means clustering with a user-defined number of clusters (K).")
                        st.write("4. Add cluster labels back to the original DataFrame.")
                        st.write("5. Reduce dimensions using PCA for visualization (if number of features > 2).")
                        st.write("6. Visualize the clusters.")
                        st.write("7. Display the final DataFrame with cluster labels.")
                        st.markdown("---")
                        n_clusters_kmeans = st.slider("Select the number of clusters (K):", min_value=2, max_value=15, value=3, key="kmeans_clusters")

                    # Data Prep & Execution for K-Means
                    # Note: Data preparation is repeated inside each algorithm block as per the requested structure.
                    # A more efficient approach would be to prepare data once before the if/elif blocks.
                    df_cluster_kmeans = st.session_state.df.copy()
                    non_numeric_cols_kmeans = df_cluster_kmeans.select_dtypes(exclude=np.number).columns
                    df_numeric_kmeans = df_cluster_kmeans.drop(columns=non_numeric_cols_kmeans)

                    if df_numeric_kmeans.empty:
                        st.warning("No numeric columns available for K-Means clustering.")
                    elif df_numeric_kmeans.isnull().all().all():
                        st.warning("Numeric columns contain only missing values. Cannot perform K-Means.")
                    else:
                        # Handle NaNs specifically for this algorithm's run
                        if df_numeric_kmeans.isnull().sum().any():
                            st.warning("Numeric columns contain missing values. Dropping rows with NaNs for K-Means.")
                            # Keep track of original indices to merge back correctly
                            original_indices_kmeans = df_numeric_kmeans.index
                            df_numeric_kmeans = df_numeric_kmeans.dropna()
                            # Filter the original copy to match dropped rows
                            df_cluster_kmeans = df_cluster_kmeans.loc[df_numeric_kmeans.index]

                        if df_numeric_kmeans.empty: # Check again after dropping NaNs
                            st.warning("No numeric data remaining after handling NaNs for K-Means.")
                        else:
                            st.subheader("Numeric Data for Clustering (K-Means)")
                            st.dataframe(df_numeric_kmeans.head())

                            st.subheader("Standardize Data (K-Means)")
                            from sklearn.preprocessing import StandardScaler # Keep import local if strictly following structure
                            scaler_kmeans = StandardScaler()
                            scaled_data_kmeans = scaler_kmeans.fit_transform(df_numeric_kmeans)
                            scaled_df_kmeans = pd.DataFrame(scaled_data_kmeans, columns=df_numeric_kmeans.columns)
                            st.dataframe(scaled_df_kmeans.head())

                            if st.button("Apply K-Means Clustering", key="apply_kmeans"):
                                from sklearn.cluster import KMeans # Keep import local if strictly following structure
                                kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
                                try:
                                    clusters_kmeans = kmeans.fit_predict(scaled_df_kmeans)

                                    # Add cluster labels back - align with the potentially filtered df_cluster_kmeans
                                    # Reset index of the potentially filtered original data before concat
                                    df_clustered_kmeans = df_cluster_kmeans.reset_index(drop=True)
                                    df_clustered_kmeans['Cluster'] = clusters_kmeans

                                    st.success(f"✅ K-Means clustering applied successfully with K={n_clusters_kmeans}!")
                                    st.subheader("DataFrame with Cluster Labels")
                                    st.dataframe(df_clustered_kmeans.head())

                                    # Evaluation
                                    if len(set(clusters_kmeans)) > 1:
                                        from sklearn.metrics import silhouette_score # Local import
                                        score_kmeans = silhouette_score(scaled_df_kmeans, clusters_kmeans)
                                        st.metric(label="Silhouette Score", value=f"{score_kmeans:.3f}")
                                        st.caption("Score range: -1 to 1. Higher is better.")
                                    else:
                                        st.warning("⚠️ Silhouette Score requires at least 2 clusters.")

                                    # Visualization (ensure visualize_clusters function is defined elsewhere)
                                    visualize_clusters(scaled_data_kmeans, clusters_kmeans, "K-Means", n_clusters_kmeans, scaled_df_kmeans.shape[1], scaled_df_kmeans.columns.tolist())

                                    # Store result if needed
                                    st.session_state.df_clustered_kmeans_result = df_clustered_kmeans

                                except Exception as e:
                                    st.error(f"❌ Error during K-Means clustering: {e}")

                # --- K-Medoids Clustering Block Removed ---

                # --- DBSCAN Clustering ---
                # NOTE: This is now elif, directly following the K-Means if block
                elif clustering_option == "DBSCAN":
                    st.subheader("DBSCAN Clustering")
                    with st.expander("DBSCAN Clustering Steps & Parameters"):
                        st.write("Density-Based Spatial Clustering of Applications with Noise. Good for arbitrarily shaped clusters and identifying noise points (labeled -1).")
                        st.write("Steps involved:")
                        st.write("1. Drop non-numeric columns.")
                        st.write("2. Standardize the numeric data (important for distance measure `eps`).")
                        st.write("3. Select `eps` (neighborhood distance) and `min_samples` (core point threshold).")
                        st.write("4. Apply DBSCAN clustering.")
                        st.write("5. Add cluster labels back (noise points are -1).")
                        st.write("6. Visualize clusters.")
                        st.markdown("---")
                        eps_dbscan = st.number_input("Select Epsilon (eps):", min_value=0.01, max_value=5.0, value=0.5, step=0.05, key="dbscan_eps", help="Max distance between samples for one to be considered as in the neighborhood of the other.")
                        min_samples_dbscan = st.slider("Select Minimum Samples (min_samples):", min_value=2, max_value=50, value=5, key="dbscan_min_samples", help="Number of samples in a neighborhood for a point to be considered as a core point.")

                    # Data Prep & Execution for DBSCAN (Repeated structure)
                    df_cluster_dbscan = st.session_state.df.copy()
                    non_numeric_cols_dbscan = df_cluster_dbscan.select_dtypes(exclude=np.number).columns
                    df_numeric_dbscan = df_cluster_dbscan.drop(columns=non_numeric_cols_dbscan)

                    if df_numeric_dbscan.empty:
                        st.warning("No numeric columns available for DBSCAN clustering.")
                    elif df_numeric_dbscan.isnull().all().all():
                        st.warning("Numeric columns contain only missing values. Cannot perform DBSCAN.")
                    else:
                        if df_numeric_dbscan.isnull().sum().any():
                            st.warning("Numeric columns contain missing values. Dropping rows with NaNs for DBSCAN.")
                            original_indices_dbscan = df_numeric_dbscan.index
                            df_numeric_dbscan = df_numeric_dbscan.dropna()
                            df_cluster_dbscan = df_cluster_dbscan.loc[df_numeric_dbscan.index]

                        if df_numeric_dbscan.empty:
                            st.warning("No numeric data remaining after handling NaNs for DBSCAN.")
                        else:
                            st.subheader("Numeric Data for Clustering (DBSCAN)")
                            st.dataframe(df_numeric_dbscan.head())
                            st.subheader("Standardize Data (DBSCAN)")
                            from sklearn.preprocessing import StandardScaler
                            scaler_dbscan = StandardScaler()
                            scaled_data_dbscan = scaler_dbscan.fit_transform(df_numeric_dbscan)
                            scaled_df_dbscan = pd.DataFrame(scaled_data_dbscan, columns=df_numeric_dbscan.columns)
                            st.dataframe(scaled_df_dbscan.head())

                            if st.button("Apply DBSCAN Clustering", key="apply_dbscan"):
                                from sklearn.cluster import DBSCAN
                                try:
                                    dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
                                    clusters_dbscan = dbscan.fit_predict(scaled_df_dbscan)

                                    df_clustered_dbscan = df_cluster_dbscan.reset_index(drop=True)
                                    df_clustered_dbscan['Cluster'] = clusters_dbscan

                                    n_clusters_found_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
                                    n_noise_dbscan = list(clusters_dbscan).count(-1)

                                    st.success(f"✅ DBSCAN clustering applied successfully!")
                                    st.metric(label="Number of Clusters Found", value=n_clusters_found_dbscan)
                                    st.metric(label="Number of Noise Points", value=n_noise_dbscan)
                                    st.subheader("DataFrame with Cluster Labels")
                                    st.dataframe(df_clustered_dbscan.head())

                                    # Evaluation (excluding noise)
                                    non_noise_mask_dbscan = clusters_dbscan != -1
                                    if n_clusters_found_dbscan > 1 and np.sum(non_noise_mask_dbscan) > 1:
                                        from sklearn.metrics import silhouette_score
                                        score_dbscan = silhouette_score(scaled_df_dbscan[non_noise_mask_dbscan], clusters_dbscan[non_noise_mask_dbscan])
                                        st.metric(label="Silhouette Score (excluding noise)", value=f"{score_dbscan:.3f}")
                                        st.caption("Score calculated only on non-noise points.")
                                    elif n_clusters_found_dbscan <= 1:
                                        st.warning("⚠️ Silhouette Score requires at least 2 clusters (excluding noise).")

                                    visualize_clusters(scaled_data_dbscan, clusters_dbscan, "DBSCAN", f"eps={eps_dbscan}", scaled_df_dbscan.shape[1], scaled_df_dbscan.columns.tolist())
                                    st.session_state.df_clustered_dbscan_result = df_clustered_dbscan

                                except Exception as e:
                                    st.error(f"❌ Error during DBSCAN clustering: {e}")

                # --- Hierarchical Clustering ---
                elif clustering_option == "Hierarchical Clustering":
                    st.subheader("Hierarchical Clustering (Agglomerative)")
                    with st.expander("Hierarchical Clustering Steps & Parameters"):
                        st.write("Builds a hierarchy of clusters bottom-up. Merges closest clusters iteratively based on linkage criteria.")
                        st.write("Steps involved:")
                        st.write("1. Drop non-numeric columns.")
                        st.write("2. Standardize the numeric data.")
                        st.write("3. Select number of clusters (K) and linkage method.")
                        st.write("4. Apply Agglomerative Clustering.")
                        st.write("5. Add cluster labels back.")
                        st.write("6. Visualize clusters.")
                        st.markdown("---")
                        n_clusters_agg = st.slider("Select the number of clusters:", min_value=2, max_value=15, value=3, key="agg_clusters")
                        linkage_agg = st.selectbox("Select Linkage Criterion:", ['ward', 'complete', 'average', 'single'], key="agg_linkage",
                                            help="'ward' minimizes variance within clusters, 'complete' uses max distance, 'average' uses average distance, 'single' uses min distance.")

                    # Data Prep & Execution for Hierarchical (Repeated structure)
                    df_cluster_agg = st.session_state.df.copy()
                    non_numeric_cols_agg = df_cluster_agg.select_dtypes(exclude=np.number).columns
                    df_numeric_agg = df_cluster_agg.drop(columns=non_numeric_cols_agg)

                    if df_numeric_agg.empty:
                        st.warning("No numeric columns available for Hierarchical clustering.")
                    elif df_numeric_agg.isnull().all().all():
                        st.warning("Numeric columns contain only missing values. Cannot perform Hierarchical Clustering.")
                    else:
                        if df_numeric_agg.isnull().sum().any():
                            st.warning("Numeric columns contain missing values. Dropping rows with NaNs for Hierarchical Clustering.")
                            original_indices_agg = df_numeric_agg.index
                            df_numeric_agg = df_numeric_agg.dropna()
                            df_cluster_agg = df_cluster_agg.loc[df_numeric_agg.index]

                        if df_numeric_agg.empty:
                            st.warning("No numeric data remaining after handling NaNs for Hierarchical Clustering.")
                        else:
                            st.subheader("Numeric Data for Clustering (Hierarchical)")
                            st.dataframe(df_numeric_agg.head())
                            st.subheader("Standardize Data (Hierarchical)")
                            from sklearn.preprocessing import StandardScaler
                            scaler_agg = StandardScaler()
                            scaled_data_agg = scaler_agg.fit_transform(df_numeric_agg)
                            scaled_df_agg = pd.DataFrame(scaled_data_agg, columns=df_numeric_agg.columns)
                            st.dataframe(scaled_df_agg.head())

                            if st.button("Apply Hierarchical Clustering", key="apply_agg"):
                                if linkage_agg == 'ward' and not np.isfinite(scaled_data_agg).all(): # Ward requires finite values, though scaling should handle most issues
                                    st.error("Ward linkage cannot be used with non-finite values (check data).")
                                else:
                                    from sklearn.cluster import AgglomerativeClustering
                                    try:
                                        # Affinity 'euclidean' is default and suitable for scaled data / required for 'ward'
                                        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters_agg, metric='euclidean', linkage=linkage_agg)
                                        clusters_agg = agg_clustering.fit_predict(scaled_df_agg)

                                        df_clustered_agg = df_cluster_agg.reset_index(drop=True)
                                        df_clustered_agg['Cluster'] = clusters_agg

                                        st.success(f"✅ Hierarchical clustering applied successfully with K={n_clusters_agg} and '{linkage_agg}' linkage!")
                                        st.subheader("DataFrame with Cluster Labels")
                                        st.dataframe(df_clustered_agg.head())

                                        if len(set(clusters_agg)) > 1:
                                            from sklearn.metrics import silhouette_score
                                            score_agg = silhouette_score(scaled_df_agg, clusters_agg)
                                            st.metric(label="Silhouette Score", value=f"{score_agg:.3f}")
                                            st.caption("Score range: -1 to 1. Higher is better.")
                                        else:
                                            st.warning("⚠️ Silhouette Score requires at least 2 clusters.")

                                        visualize_clusters(scaled_data_agg, clusters_agg, "Hierarchical", n_clusters_agg, scaled_df_agg.shape[1], scaled_df_agg.columns.tolist())
                                        st.session_state.df_clustered_agg_result = df_clustered_agg

                                    except Exception as e:
                                        st.error(f"❌ Error during Hierarchical clustering: {e}")

                # --- Gaussian Mixture Models (GMM) ---
                elif clustering_option == "Gaussian Mixture Models (GMM)":
                    st.subheader("Gaussian Mixture Models (GMM)")
                    with st.expander("GMM Steps & Parameters"):
                        st.write("Assumes data is generated from a mix of Gaussian distributions. Can model complex cluster shapes.")
                        st.write("Steps involved:")
                        st.write("1. Drop non-numeric columns.")
                        st.write("2. Standardize the numeric data.")
                        st.write("3. Select number of components (clusters) and covariance type.")
                        st.write("4. Apply GMM using Expectation-Maximization (EM).")
                        st.write("5. Add cluster labels back (hard assignment based on highest probability).")
                        st.write("6. Visualize clusters.")
                        st.markdown("---")
                        n_components_gmm = st.slider("Select the number of components (clusters):", min_value=1, max_value=15, value=3, key="gmm_components")
                        covariance_type_gmm = st.selectbox("Select Covariance Type:", ['full', 'tied', 'diag', 'spherical'], key="gmm_cov_type",
                                                    help="'full': Each component has own covariance. 'tied': All share same covariance. 'diag': Each has own diagonal covariance. 'spherical': Each has single variance.")

                    # Data Prep & Execution for GMM (Repeated structure)
                    df_cluster_gmm = st.session_state.df.copy()
                    non_numeric_cols_gmm = df_cluster_gmm.select_dtypes(exclude=np.number).columns
                    df_numeric_gmm = df_cluster_gmm.drop(columns=non_numeric_cols_gmm)

                    if df_numeric_gmm.empty:
                        st.warning("No numeric columns available for GMM clustering.")
                    elif df_numeric_gmm.isnull().all().all():
                        st.warning("Numeric columns contain only missing values. Cannot perform GMM.")
                    else:
                        if df_numeric_gmm.isnull().sum().any():
                            st.warning("Numeric columns contain missing values. Dropping rows with NaNs for GMM.")
                            original_indices_gmm = df_numeric_gmm.index
                            df_numeric_gmm = df_numeric_gmm.dropna()
                            df_cluster_gmm = df_cluster_gmm.loc[df_numeric_gmm.index]

                        if df_numeric_gmm.empty:
                            st.warning("No numeric data remaining after handling NaNs for GMM.")
                        else:
                            st.subheader("Numeric Data for Clustering (GMM)")
                            st.dataframe(df_numeric_gmm.head())
                            st.subheader("Standardize Data (GMM)")
                            from sklearn.preprocessing import StandardScaler
                            scaler_gmm = StandardScaler()
                            scaled_data_gmm = scaler_gmm.fit_transform(df_numeric_gmm)
                            scaled_df_gmm = pd.DataFrame(scaled_data_gmm, columns=df_numeric_gmm.columns)
                            st.dataframe(scaled_df_gmm.head())

                            if st.button("Apply GMM Clustering", key="apply_gmm"):
                                if n_components_gmm < 1:
                                    st.error("❌ Number of components must be at least 1.")
                                else:
                                    from sklearn.mixture import GaussianMixture
                                    try:
                                        gmm = GaussianMixture(n_components=n_components_gmm, covariance_type=covariance_type_gmm, random_state=42)
                                        clusters_gmm = gmm.fit_predict(scaled_df_gmm)

                                        df_clustered_gmm = df_cluster_gmm.reset_index(drop=True)
                                        df_clustered_gmm['Cluster'] = clusters_gmm

                                        st.success(f"✅ GMM clustering applied successfully with {n_components_gmm} components and '{covariance_type_gmm}' covariance!")
                                        st.subheader("DataFrame with Cluster Labels")
                                        st.dataframe(df_clustered_gmm.head())

                                        if len(set(clusters_gmm)) > 1:
                                            from sklearn.metrics import silhouette_score
                                            score_gmm = silhouette_score(scaled_df_gmm, clusters_gmm)
                                            st.metric(label="Silhouette Score", value=f"{score_gmm:.3f}")
                                            st.caption("Score range: -1 to 1. Higher is better.")
                                        elif len(set(clusters_gmm)) == 1:
                                            st.warning("⚠️ Only one cluster assigned. Silhouette Score requires at least 2 clusters.")
                                        else:
                                            st.warning("⚠️ No clusters assigned (check data/parameters). Cannot calculate Silhouette Score.")


                                        visualize_clusters(scaled_data_gmm, clusters_gmm, "GMM", n_components_gmm, scaled_df_gmm.shape[1], scaled_df_gmm.columns.tolist())
                                        st.session_state.df_clustered_gmm_result = df_clustered_gmm

                                    except Exception as e:
                                        st.error(f"❌ Error during GMM clustering: {e}")
                    
            with tab9: # If this is inside a tab structure

                # --- Changeable Dashboard Title ---
                def update_dashboard_title():
                    st.session_state.dashboard_title = st.session_state.dashboard_title_input # Update session state from widget

                st.text_input(
                    "Dashboard Title:",
                    value=st.session_state.get("dashboard_title", "📊 Interactive Dashboard"), # Get title or default
                    key="dashboard_title_input", # Unique key for the widget
                    on_change=update_dashboard_title # Callback to update session state
                )
                # Display the title (using the value from session state)
                st.header(st.session_state.get("dashboard_title", "📊 Interactive Dashboard"))
                st.markdown("Configure charts in the **sidebar** on the left, view results here.")
                st.markdown("---") # Visual separator

                # --- Check if data exists ---
                if 'df' in st.session_state and not st.session_state.df.empty:
                    df_dash = st.session_state.df.copy()

                    # --- Prepare lists of column types ---
                    try:
                        numeric_cols = df_dash.select_dtypes(include=np.number).columns.tolist()
                        categorical_cols = df_dash.select_dtypes(include=['object', 'category']).columns.tolist()
                        all_cols = df_dash.columns.tolist()
                        # datetime_cols = df_dash.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist() # Add if needed
                    except Exception as e:
                        st.error(f"Error identifying column types: {e}")
                        # Initialize lists to prevent errors later if detection fails
                        numeric_cols = []
                        categorical_cols = []
                        all_cols = []

                    # ============================================================
                    # SIDEBAR CONTROLS (Only appears when tab9 is active)
                    # ============================================================
                    with st.sidebar:
                        st.header("⚙️ Chart Controls") # This header ONLY appears when tab_dashboard is active
                        st.markdown(f"Configure **{st.session_state.get('dashboard_title', 'Dashboard')}** charts.")

                        # --- Pie Chart Controls ---
                        with st.expander("🥧 Pie Chart Options", expanded=False):
                            pie_cat_col_selected = None
                            pie_val_col_selected = None
                            if categorical_cols:
                                pie_cat_col_selected = st.selectbox("Category (Names)", categorical_cols, key="pie_cat_ctrl", index=0 if categorical_cols else None)
                                pie_val_options = ['Count'] + numeric_cols
                                pie_val_col_selected = st.selectbox("Values", pie_val_options, key="pie_val_ctrl", index=0) # Default to 'Count'
                            else:
                                st.info("❗ No categorical columns available for Pie Chart.")

                        # --- Bar Chart Controls ---
                        with st.expander("📊 Bar Chart Options", expanded=False):
                            bar_cat_col_selected = None
                            bar_num_col_selected = None
                            bar_agg_selected = None
                            bar_orient_selected = None
                            if categorical_cols and numeric_cols:
                                bar_cat_col_selected = st.selectbox("Category Axis", categorical_cols, key="bar_cat_ctrl", index=0 if categorical_cols else None)
                                bar_num_col_selected = st.selectbox("Numeric Axis", numeric_cols, key="bar_num_ctrl", index=0 if numeric_cols else None)
                                bar_agg_selected = st.selectbox("Aggregation", ['Sum', 'Mean', 'Count'], key="bar_agg_ctrl", index=0)
                                bar_orient_selected = st.selectbox("Orientation", ['Vertical', 'Horizontal'], key="bar_orient_ctrl", index=0)
                            else:
                                st.info("❗ Need both categorical & numeric columns for Bar Chart.")

                        # --- Stacked Bar Controls ---
                        with st.expander("📈 Stacked Bar Options", expanded=False):
                            stack_x_col_selected = None
                            stack_color_col_selected = None
                            stack_y_col_selected = None
                            stack_agg_selected = None
                            if len(categorical_cols) >= 2 and numeric_cols:
                                stack_x_col_selected = st.selectbox("X-axis (Category)", categorical_cols, key="stack_x_ctrl", index=0 if categorical_cols else None)
                                # Filter out the selected x-axis column for the color option
                                stack_color_options = [c for c in categorical_cols if c != stack_x_col_selected]
                                if stack_color_options: # Ensure there's at least one other categorical column
                                    stack_color_col_selected = st.selectbox("Color/Stack by", stack_color_options, key="stack_color_ctrl", index=0)
                                else:
                                    st.warning("Need at least 2 distinct categorical columns.")
                                    stack_color_col_selected = None # Prevent error later

                                stack_y_col_selected = st.selectbox("Y-axis (Value)", numeric_cols, key="stack_y_ctrl", index=0 if numeric_cols else None)
                                stack_agg_selected = st.selectbox("Aggregation ", ['Sum', 'Mean'], key="stack_agg_ctrl", index=0)
                            elif not numeric_cols:
                                st.info("❗ Need at least 1 numeric column for Stacked Bar.")
                            else: # <= 1 categorical column
                                st.info("❗ Need at least 2 categorical columns for Stacked Bar.")

                        # --- Line Chart Controls ---
                        with st.expander("📉 Line Chart Options", expanded=False):
                            line_x_col_selected = None
                            line_y_cols_selected = None
                            # Consider adding datetime columns here if available and relevant
                            line_x_options = numeric_cols + categorical_cols # Combine options
                            if line_x_options and numeric_cols:
                                line_x_col_selected = st.selectbox("X-axis ", line_x_options, key="line_x_ctrl", index=0 if line_x_options else None)
                                line_y_cols_selected = st.multiselect("Y-axis (Numeric) ", numeric_cols,
                                                                    default=numeric_cols[0] if numeric_cols else None, key="line_y_ctrl")
                            else:
                                st.info("❗ Need columns for X-axis and >=1 numeric column for Y-axis for Line Chart.")

                        # --- Scatter Plot Controls ---
                        with st.expander("✨ Scatter Plot Options", expanded=False):
                            scatter_x_selected = None
                            scatter_y_selected = None
                            scatter_color_selected = None
                            scatter_size_selected = None
                            if len(numeric_cols) >= 2:
                                scatter_x_selected = st.selectbox("X-axis (Numeric)", numeric_cols, key="scatter_x_ctrl", index=0 if numeric_cols else None)
                                # Ensure Y is different from X by default if possible
                                y_options_scatter = [c for c in numeric_cols if c != scatter_x_selected]
                                if not y_options_scatter: y_options_scatter = numeric_cols # Fallback if only one num col
                                scatter_y_selected = st.selectbox("Y-axis (Numeric)", y_options_scatter, key="scatter_y_ctrl", index=0 if y_options_scatter else None)

                                scatter_color_options = ['None'] + numeric_cols + categorical_cols
                                scatter_color_selected = st.selectbox("Color by (Optional)", scatter_color_options, key="scatter_color_ctrl", index=0)
                                if scatter_color_selected == 'None': scatter_color_selected = None

                                scatter_size_options = ['None'] + numeric_cols
                                scatter_size_selected = st.selectbox("Size by (Numeric, Optional)", scatter_size_options, key="scatter_size_ctrl", index=0)
                                if scatter_size_selected == 'None': scatter_size_selected = None
                            else:
                                st.info("❗ Need at least 2 numeric columns for Scatter Plot.")

                        # --- Histogram Controls ---
                        with st.expander("📊 Histogram Options", expanded=False):
                            hist_col_selected = None
                            hist_color_selected = None
                            hist_bins_selected = None
                            if numeric_cols:
                                hist_col_selected = st.selectbox("Column to Bin (Numeric)", numeric_cols, key="hist_col_ctrl", index=0 if numeric_cols else None)
                                hist_color_options = ['None'] + categorical_cols
                                hist_color_selected = st.selectbox("Color by (Categorical, Optional)", hist_color_options, key="hist_color_ctrl", index=0)
                                if hist_color_selected == 'None': hist_color_selected = None
                                hist_bins_selected = st.slider("Number of Bins", min_value=5, max_value=100, value=20, key="hist_bins_ctrl")
                            else:
                                st.info("❗ Need at least 1 numeric column for Histogram.")

                        # --- Box Plot Controls ---
                        with st.expander("📦 Box Plot Options", expanded=False):
                            box_cat_selected = None
                            box_num_selected = None
                            box_color_selected = None
                            if numeric_cols and categorical_cols:
                                box_cat_selected = st.selectbox("Category (X-axis)", categorical_cols, key="box_cat_ctrl", index=0 if categorical_cols else None)
                                box_num_selected = st.selectbox("Value (Y-axis, Numeric)", numeric_cols, key="box_num_ctrl", index=0 if numeric_cols else None)
                                box_color_options = ['None'] + [c for c in categorical_cols if c != box_cat_selected] # Optional different category for color
                                box_color_selected = st.selectbox("Color by (Categorical, Optional)", box_color_options, key="box_color_ctrl", index=0)
                                if box_color_selected == 'None': box_color_selected = None
                            else:
                                st.info("❗ Need at least 1 numeric and 1 categorical column for Box Plot.")


                    # =========================
                    # MAIN DASHBOARD AREA
                    # =========================

                    # --- KPI Section ---
                    st.markdown("#### Key Performance Indicators")
                    kpi_cols = st.columns(4)
                    with kpi_cols[0]:
                        st.metric("Total Records", f"{len(df_dash):,}")
                    with kpi_cols[1]:
                        st.metric("Total Features", len(df_dash.columns))
                    with kpi_cols[2]:
                        st.markdown("###### Mean Value (First Num Col)")
                        if numeric_cols:
                            first_num_col = numeric_cols[0]
                            try:
                                # Ensure column is numeric before calculating mean
                                numeric_series = pd.to_numeric(df_dash[first_num_col], errors='coerce')
                                if not numeric_series.isnull().all(): # Check if there are any valid numbers
                                    mean_val = numeric_series.mean()
                                    st.metric(f"Mean: {first_num_col}", f"{mean_val:,.2f}" if not pd.isna(mean_val) else "N/A")
                                else:
                                    st.metric(f"Mean: {first_num_col}", "N/A (No valid data)")
                            except Exception as e:
                                st.warning(f"Could not calculate mean: {e}")
                        else:
                            st.info("No numeric cols.")
                    with kpi_cols[3]:
                        st.markdown("###### Unique Count (First Cat Col)")
                        if categorical_cols:
                            first_cat_col = categorical_cols[0]
                            try:
                                unique_count = df_dash[first_cat_col].nunique()
                                st.metric(f"Unique: {first_cat_col}", f"{unique_count:,}")
                            except Exception as e:
                                st.warning(f"Could not count unique: {e}")
                        else:
                            st.info("No categorical cols.")

                    st.markdown("---") # Visual separator

                    # --- Chart Layout (Adjusted for more charts - e.g., 2x3 grid) ---
                    st.subheader("📊 Visualizations")
                    row1_col1, row1_col2 = st.columns(2)
                    row2_col1, row2_col2 = st.columns(2)
                    row3_col1, row3_col2 = st.columns(2) # Add a third row

                    # --- Display Line Chart ---
                    with row1_col1:
                        st.markdown("**Line Chart**")
                        if line_x_col_selected and line_y_cols_selected:
                            try:
                                line_data = df_dash.copy()
                                # Attempt to sort if x-axis is numeric or datetime-like
                                try:
                                    line_data['_sort_col'] = pd.to_datetime(line_data[line_x_col_selected], errors='ignore')
                                    if not pd.api.types.is_datetime64_any_dtype(line_data['_sort_col']):
                                        line_data['_sort_col'] = pd.to_numeric(line_data[line_x_col_selected], errors='ignore')

                                    if pd.api.types.is_numeric_dtype(line_data['_sort_col']) or pd.api.types.is_datetime64_any_dtype(line_data['_sort_col']):
                                        # Drop rows where sort key is NaN before sorting to avoid issues
                                        line_data = line_data.dropna(subset=['_sort_col']).sort_values(by='_sort_col')
                                    line_data = line_data.drop(columns=['_sort_col'], errors='ignore')
                                except Exception:
                                    pass # Use original order if sorting prep fails

                                # Ensure Y columns are numeric
                                valid_y_cols = []
                                for col in line_y_cols_selected:
                                    if col in line_data.columns:
                                        try:
                                            line_data[col] = pd.to_numeric(line_data[col], errors='coerce')
                                            valid_y_cols.append(col)
                                        except Exception:
                                            st.warning(f"Could not convert column '{col}' to numeric for Line Chart. Skipping.")
                                    else:
                                        st.warning(f"Column '{col}' selected for Y-axis not found. Skipping.")

                                if valid_y_cols:
                                    fig_line = px.line(line_data, x=line_x_col_selected, y=valid_y_cols,
                                                    title=f"{', '.join(valid_y_cols)} over {line_x_col_selected}", markers=True)
                                    fig_line.update_layout(xaxis_title=line_x_col_selected, yaxis_title="Values", legend_title_text='Series')
                                    st.plotly_chart(fig_line, use_container_width=True)
                                else:
                                    st.warning("No valid numeric Y-axis columns selected or available for Line Chart.")

                            except Exception as e:
                                st.error(f"Error generating line chart: {e}")
                        else:
                            st.info("Select Line Chart options in the sidebar.")

                    # --- Display Pie Chart ---
                    with row1_col2:
                        st.markdown("**Pie Chart**")
                        if pie_cat_col_selected and pie_val_col_selected:
                            try:
                                fig_pie = None
                                # Basic Column Existence Check
                                if pie_cat_col_selected not in df_dash.columns:
                                    st.error(f"Category column '{pie_cat_col_selected}' not found.")
                                elif pie_val_col_selected != 'Count' and pie_val_col_selected not in df_dash.columns:
                                    st.error(f"Value column '{pie_val_col_selected}' not found.")
                                else:
                                    # Prepare data
                                    if pie_val_col_selected == 'Count':
                                        pie_data = df_dash[pie_cat_col_selected].fillna('NaN').value_counts().reset_index()
                                        pie_data.columns = [pie_cat_col_selected, 'count'] # Rename for clarity
                                        # Check if data is suitable (avoid plotting single huge category)
                                        if len(pie_data) > 50:
                                            st.warning("Too many categories (>50) for an effective Pie Chart. Showing top 50.")
                                            pie_data = pie_data.nlargest(50, 'count')
                                        fig_pie = px.pie(pie_data, names=pie_cat_col_selected, values='count',
                                                    title=f"Distribution by {pie_cat_col_selected}", hole=0.3)
                                    elif pie_val_col_selected in numeric_cols:
                                        # Ensure value column is numeric
                                        df_dash[pie_val_col_selected] = pd.to_numeric(df_dash[pie_val_col_selected], errors='coerce')
                                        # Group, sum, handle NaNs
                                        pie_data = df_dash.groupby(df_dash[pie_cat_col_selected].fillna('NaN'), observed=False)[pie_val_col_selected].sum(skipna=True).reset_index()
                                        # Filter out zero or negative values which don't make sense in a pie chart sum
                                        pie_data = pie_data[pie_data[pie_val_col_selected] > 0]
                                        if pie_data.empty:
                                            st.warning(f"No positive values found for '{pie_val_col_selected}' to display in Pie Chart.")
                                        else:
                                            if len(pie_data) > 50:
                                                st.warning("Too many categories (>50) for an effective Pie Chart. Showing top 50 by value.")
                                                pie_data = pie_data.nlargest(50, pie_val_col_selected)
                                            fig_pie = px.pie(pie_data, names=pie_cat_col_selected, values=pie_val_col_selected,
                                                            title=f"Total {pie_val_col_selected} by {pie_cat_col_selected}", hole=0.3)

                                # Display chart if created
                                if fig_pie:
                                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                                    st.plotly_chart(fig_pie, use_container_width=True)
                                elif pie_cat_col_selected and pie_val_col_selected and not fig_pie: # Options selected but no fig generated (due to warnings above)
                                    pass # Warnings were already shown
                                # else: # This case should be covered by the column existence checks now

                            except Exception as e:
                                st.error(f"Error generating pie chart: {e}")
                        else:
                            st.info("Select Pie Chart options in the sidebar.")

                    # --- Display Bar Chart ---
                    with row2_col1:
                        st.markdown("**Bar Chart**")
                        if bar_cat_col_selected and bar_num_col_selected and bar_agg_selected and bar_orient_selected:
                            try:
                                agg_func = bar_agg_selected.lower()
                                x_title, y_title = "", ""
                                num_axis_val_col_name = "" # Stores the actual column name holding the value to plot

                                # Check column existence
                                if bar_cat_col_selected not in df_dash.columns:
                                    st.error(f"Categorical column '{bar_cat_col_selected}' not found.")
                                elif agg_func != 'count' and bar_num_col_selected not in df_dash.columns:
                                    st.error(f"Numeric column '{bar_num_col_selected}' not found.")
                                else:
                                    # Prepare data
                                    temp_df = df_dash[[bar_cat_col_selected]].copy() # Start with category column
                                    if agg_func == 'count':
                                        bar_data = temp_df.groupby(temp_df[bar_cat_col_selected].fillna('NaN'), observed=False).size().reset_index(name=agg_func)
                                        num_axis_val_col_name = agg_func
                                    else: # sum or mean
                                        # Add and coerce the numeric column
                                        temp_df[bar_num_col_selected] = pd.to_numeric(df_dash[bar_num_col_selected], errors='coerce')
                                        # Drop rows where the numeric value is NaN before aggregation
                                        temp_df = temp_df.dropna(subset=[bar_num_col_selected])
                                        if temp_df.empty:
                                            st.warning(f"No valid numeric data in '{bar_num_col_selected}' after handling NaNs.")
                                            bar_data = pd.DataFrame() # Empty dataframe
                                        else:
                                            bar_data = temp_df.groupby(temp_df[bar_cat_col_selected].fillna('NaN'), observed=False)[bar_num_col_selected].agg(agg_func).reset_index()
                                            num_axis_val_col_name = bar_num_col_selected # Plotting the aggregated numeric column

                                    if not bar_data.empty:
                                        # Sort by value for better visualization (optional, descending)
                                        try:
                                            bar_data = bar_data.sort_values(by=num_axis_val_col_name, ascending=False)
                                        except Exception:
                                            pass # Ignore if sorting fails

                                        plot_args = {'data_frame': bar_data, 'title': f"{bar_agg_selected} of {bar_num_col_selected if agg_func != 'count' else 'Records'} by {bar_cat_col_selected}"}

                                        if bar_orient_selected == 'Vertical':
                                            plot_args['x'] = bar_cat_col_selected
                                            plot_args['y'] = num_axis_val_col_name
                                            x_title=bar_cat_col_selected
                                            y_title=f"{bar_agg_selected}({bar_num_col_selected})" if agg_func != 'count' else "Count"
                                        else: # Horizontal
                                            plot_args['x'] = num_axis_val_col_name
                                            plot_args['y'] = bar_cat_col_selected
                                            plot_args['orientation'] = 'h'
                                            x_title=f"{bar_agg_selected}({bar_num_col_selected})" if agg_func != 'count' else "Count"
                                            y_title=bar_cat_col_selected

                                        fig_bar = px.bar(**plot_args)
                                        fig_bar.update_layout(xaxis_title=x_title, yaxis_title=y_title)
                                        st.plotly_chart(fig_bar, use_container_width=True)
                                    elif not temp_df.empty: # If temp_df wasn't empty initially but bar_data is
                                        st.warning(f"Could not generate Bar Chart. Check data in '{bar_num_col_selected}'.")

                            except Exception as e:
                                st.error(f"Error generating bar chart: {e}")
                        else:
                            st.info("Select Bar Chart options in the sidebar.")

                    # --- Display Stacked Bar Chart ---
                    with row2_col2:
                        st.markdown("**Stacked Bar Chart**")
                        # Check if all necessary variables have been assigned values in the sidebar
                        if stack_x_col_selected and stack_y_col_selected and stack_color_col_selected and stack_agg_selected:
                            try:
                                agg_func = stack_agg_selected.lower()

                                # Check column existence
                                if stack_x_col_selected not in df_dash.columns:
                                    st.error(f"X-axis column '{stack_x_col_selected}' not found.")
                                elif stack_color_col_selected not in df_dash.columns:
                                    st.error(f"Color/Stack column '{stack_color_col_selected}' not found.")
                                elif stack_y_col_selected not in df_dash.columns:
                                    st.error(f"Y-axis (Value) column '{stack_y_col_selected}' not found.")
                                else:
                                    # Prepare data
                                    temp_df = df_dash[[stack_x_col_selected, stack_color_col_selected, stack_y_col_selected]].copy()
                                    # Ensure numeric column is numeric, coerce errors
                                    temp_df[stack_y_col_selected] = pd.to_numeric(temp_df[stack_y_col_selected], errors='coerce')
                                    # Drop rows where the numeric value is NaN before aggregation
                                    temp_df = temp_df.dropna(subset=[stack_y_col_selected])

                                    if temp_df.empty:
                                        st.warning(f"No valid numeric data in '{stack_y_col_selected}' after handling NaNs for Stacked Bar.")
                                        stack_data = pd.DataFrame()
                                    else:
                                        # Group by both categorical columns (handle NaNs in categories) and aggregate
                                        stack_data = temp_df.groupby(
                                            [temp_df[stack_x_col_selected].fillna('NaN'), temp_df[stack_color_col_selected].fillna('NaN')],
                                            observed=False # Important for categorical dtypes
                                        )[stack_y_col_selected].agg(agg_func).reset_index()

                                    if not stack_data.empty:
                                        fig_stack = px.bar(stack_data, x=stack_x_col_selected, y=stack_y_col_selected, color=stack_color_col_selected,
                                                        title=f"{stack_agg_selected} of {stack_y_col_selected} by {stack_x_col_selected} (Stacked by {stack_color_col_selected})",
                                                        barmode='stack')
                                        fig_stack.update_layout(xaxis_title=stack_x_col_selected, yaxis_title=f"{stack_agg_selected}({stack_y_col_selected})", legend_title_text=stack_color_col_selected)
                                        st.plotly_chart(fig_stack, use_container_width=True)
                                    elif not temp_df.empty: # Warning only if initial data existed
                                        st.warning(f"Could not generate Stacked Bar Chart. Check aggregation results.")


                            except Exception as e:
                                st.error(f"Error generating stacked bar chart: {e}")
                        else:
                            st.info("Select Stacked Bar options in the sidebar (requires >=2 Cat & 1 Num columns).")

                    # --- Display Scatter Plot ---
                    with row3_col1:
                        st.markdown("**Scatter Plot**")
                        if scatter_x_selected and scatter_y_selected:
                            try:
                                # Check column existence
                                cols_to_check = [scatter_x_selected, scatter_y_selected]
                                if scatter_color_selected: cols_to_check.append(scatter_color_selected)
                                if scatter_size_selected: cols_to_check.append(scatter_size_selected)

                                missing_cols = [col for col in cols_to_check if col not in df_dash.columns]
                                if missing_cols:
                                    st.error(f"Scatter plot columns not found: {', '.join(missing_cols)}")
                                else:
                                    # Prepare data - coerce numeric columns, keep others
                                    scatter_data = df_dash[list(set(cols_to_check))].copy() # Use set to avoid duplicates
                                    scatter_data[scatter_x_selected] = pd.to_numeric(scatter_data[scatter_x_selected], errors='coerce')
                                    scatter_data[scatter_y_selected] = pd.to_numeric(scatter_data[scatter_y_selected], errors='coerce')
                                    if scatter_size_selected:
                                        scatter_data[scatter_size_selected] = pd.to_numeric(scatter_data[scatter_size_selected], errors='coerce')
                                        # Size must be positive, replace non-positive with small value or NaN
                                        scatter_data[scatter_size_selected] = scatter_data[scatter_size_selected].apply(lambda x: x if x > 0 else np.nan)


                                    # Drop rows where essential X or Y are NaN
                                    scatter_data = scatter_data.dropna(subset=[scatter_x_selected, scatter_y_selected])

                                    if scatter_data.empty:
                                        st.warning("No valid data points remaining for Scatter Plot after handling NaNs in X/Y axes.")
                                    else:
                                        fig_scatter = px.scatter(scatter_data, x=scatter_x_selected, y=scatter_y_selected,
                                                            color=scatter_color_selected, size=scatter_size_selected,
                                                            title=f"Scatter Plot: {scatter_y_selected} vs {scatter_x_selected}",
                                                            hover_data=scatter_data.columns) # Show all data on hover
                                        fig_scatter.update_layout(xaxis_title=scatter_x_selected, yaxis_title=scatter_y_selected)
                                        st.plotly_chart(fig_scatter, use_container_width=True)

                            except Exception as e:
                                st.error(f"Error generating scatter plot: {e}")
                        else:
                            st.info("Select Scatter Plot options in the sidebar (requires >=2 Num columns).")

                    # --- Display Histogram ---
                    with row3_col2:
                        st.markdown("**Histogram**")
                        if hist_col_selected and hist_bins_selected:
                            try:
                                # Check column existence
                                cols_to_check_hist = [hist_col_selected]
                                if hist_color_selected: cols_to_check_hist.append(hist_color_selected)

                                missing_cols_hist = [col for col in cols_to_check_hist if col not in df_dash.columns]
                                if missing_cols_hist:
                                    st.error(f"Histogram columns not found: {', '.join(missing_cols_hist)}")
                                else:
                                    # Prepare data - coerce numeric column
                                    hist_data = df_dash[list(set(cols_to_check_hist))].copy()
                                    hist_data[hist_col_selected] = pd.to_numeric(hist_data[hist_col_selected], errors='coerce')
                                    hist_data = hist_data.dropna(subset=[hist_col_selected]) # Drop NaN numeric values

                                    if hist_data.empty:
                                        st.warning(f"No valid numeric data remaining for Histogram in column '{hist_col_selected}'.")
                                    else:
                                        fig_hist = px.histogram(hist_data, x=hist_col_selected, color=hist_color_selected,
                                                                nbins=hist_bins_selected,
                                                                title=f"Histogram of {hist_col_selected}" + (f" by {hist_color_selected}" if hist_color_selected else ""))
                                        fig_hist.update_layout(xaxis_title=hist_col_selected, yaxis_title="Frequency")
                                        st.plotly_chart(fig_hist, use_container_width=True)

                            except Exception as e:
                                st.error(f"Error generating histogram: {e}")
                        else:
                            st.info("Select Histogram options in the sidebar (requires >=1 Num column).")

                    # --- Display Box Plot (Example - you might replace Histogram or add another row) ---
                    # Uncomment and adjust placement if needed
                    # with row3_col2: # Or another column/row
                    #     st.markdown("**Box Plot**")
                    #     if box_cat_selected and box_num_selected:
                    #         try:
                    #             # Check column existence
                    #             cols_to_check_box = [box_cat_selected, box_num_selected]
                    #             if box_color_selected: cols_to_check_box.append(box_color_selected)
                    #
                    #             missing_cols_box = [col for col in cols_to_check_box if col not in df_dash.columns]
                    #             if missing_cols_box:
                    #                  st.error(f"Box plot columns not found: {', '.join(missing_cols_box)}")
                    #             else:
                    #                 # Prepare data - coerce numeric column
                    #                 box_data = df_dash[list(set(cols_to_check_box))].copy()
                    #                 box_data[box_num_selected] = pd.to_numeric(box_data[box_num_selected], errors='coerce')
                    #                 # Drop rows where category or numeric value is NaN
                    #                 box_data = box_data.dropna(subset=[box_cat_selected, box_num_selected])
                    #
                    #                 if box_data.empty:
                    #                      st.warning(f"No valid data remaining for Box Plot after handling NaNs.")
                    #                 else:
                    #                      fig_box = px.box(box_data, x=box_cat_selected, y=box_num_selected, color=box_color_selected,
                    #                                          title=f"Box Plot of {box_num_selected} by {box_cat_selected}" + (f" colored by {box_color_selected}" if box_color_selected else ""))
                    #                      fig_box.update_layout(xaxis_title=box_cat_selected, yaxis_title=box_num_selected)
                    #                      st.plotly_chart(fig_box, use_container_width=True)
                    #
                    #         except Exception as e:
                    #             st.error(f"Error generating box plot: {e}")
                    #     else:
                    #         st.info("Select Box Plot options in the sidebar (requires 1 Cat & 1 Num column).")


                # --- Fallback if no data ---
                else:
                    st.warning("⚠️ Please upload and analyze data first using the 'Upload & Analyze' tab to view the dashboard.")

            # ===============================================
            # End of the code for the Interactive Dashboard Tab
            # ===============================================                            
            with tab10: 
                st.header("About the Team")
                
                # Display the logo after the header
                if os.path.exists(logo_path):
                    st.image(logo_path, width=100) # Adjust width as needed
                else:
                    st.write("Logo not found.") # Optional message if logo is missing

                st.write("Welcome to DeepStat! We are a team of data enthusiasts passionate about transforming raw data into actionable insights.")

                    # Display owner images and names side by side
                if len(image_paths) == len(owner_names):
                        cols = st.columns(len(image_paths))  # Create columns for each image
                for i in range(len(image_paths)):
                        if os.path.exists(image_paths[i]):
                            cols[i].image(image_paths[i], caption=owner_names[i], width=200)  # Display image in each column
                        else:
                            st.warning(f"Image not found: {image_paths[i]}")

    # Footer section
    footer_text = "© 2025 DeepStat. All rights reserved. | Name: Preet(D023), Aakanksha(D014), Sakshi(D026) | Address: NMIMS,Navi Mumbai | Contact:preet101104@gmail.com" # Add your details here.
    st.markdown(f'<div class="footer">{footer_text}</div>', unsafe_allow_html=True)

    # Disclaimer Section
    disclaimer_text = "DeepStat can make mistakes. Check important info."
    st.markdown(f'<div class="disclaimer">{disclaimer_text}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()