import pandas as pd
import streamlit as st
import io
from processing import handle_missing_values, normalize_data, encode_categorical, handle_outliers
from processing import select_features, bin_columns, handle_imbalance, generate_polynomial_features, suggest_columns_to_drop, drop_columns
from model_selection import smart_model_selection, evaluate_model

# Set Streamlit layout to wide
st.set_page_config(layout="wide")

# File Loading Function
def load_dataset_ui(file, file_type: str):
    try:
        if file_type == 'csv':
            df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
            st.success(f"🎉 Successfully loaded CSV file. Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df
        elif file_type == 'json':
            df = pd.read_json(io.StringIO(file.read().decode('utf-8')))
            st.success(f"🎉 Successfully loaded JSON file. Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns.")
            return df
        else:
            st.error("⚠️ Unsupported file format. Please provide a CSV or JSON file.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

# Main App
def main():
    st.title("🚀 AutoML System: Smart Preprocessing and Model Selection")

    # Use an expander for file upload section
    with st.expander("📂 Upload Dataset", expanded=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            uploaded_file = st.file_uploader("Upload your dataset (CSV or JSON)", type=['csv', 'json'])
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1]
            df = load_dataset_ui(uploaded_file, file_type)
            if df is not None:
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.write("📊 **Here are the first few rows of your dataset:**")
                    st.write(df.head())
                with col2:
                    st.write("🔍 **Data Summary**")
                    st.write(df.describe(include='all'))

    # Tabs for organizing the process
    tab1, tab2, tab3 = st.tabs(["1. Data Preprocessing 🛠", "2. Feature Engineering 🧬", "3. Model Selection 🎯"])

    # Data Preprocessing Tab
    with tab1:
        if uploaded_file is not None and df is not None:
            st.header("📋 Data Preprocessing")

            col1, col2 = st.columns(2)

            with col1:
                with st.expander("🧹 Handle Missing Values"):
                    missing_value_columns = st.multiselect("Select columns to handle missing values", df.columns)
                    strategy = st.selectbox("Select strategy for missing values", ["mean", "median", "mode"])
                    if st.button("🛠 Handle Missing Values"):
                        df = handle_missing_values(df, missing_value_columns, strategy)
                        st.success("✅ Missing values handled.")
                        st.write(df.head())

            with col2:
                with st.expander("🚫 Drop Columns"):
                    suggested_columns = suggest_columns_to_drop(df)
                    st.markdown("### Suggested Columns to Drop:")
                    st.write(suggested_columns) if suggested_columns else st.write("No columns suggested for dropping.")

                    selected_columns_to_drop = st.multiselect("Select columns to drop", df.columns)
                    if st.button("🗑 Drop Selected Columns"):
                        df = drop_columns(df, selected_columns_to_drop)
                        st.success("✅ Columns dropped.")
                        st.write(df.head())

            with st.expander("📏 Normalize/Standardize", expanded=True):
                col1, col2 = st.columns([2, 1])
                with col1:
                    normalization_columns = st.multiselect("Select columns to normalize/standardize", df.select_dtypes(include=['float64', 'int64']).columns)
                with col2:
                    method = st.radio("Normalization Method", ["Standard", "MinMax"])
                if st.button("🔄 Normalize Data"):
                    df = normalize_data(df, normalization_columns, method.lower())
                    st.success("✅ Data normalized.")
                    st.write(df.head())

    # Feature Engineering Tab
    with tab2:
        if df is not None:
            st.header("🔧 Feature Engineering")

            col1, col2 = st.columns(2)

            with col1:
                with st.expander("🏷 Categorical Encoding", expanded=True):
                    categorical_columns = st.multiselect("Select categorical columns for encoding", df.select_dtypes(include=['object']).columns)
                    encoding_method = st.radio("Encoding Method", ["One-Hot", "Label"])
                    if st.button("🔤 Encode Categorical Columns"):
                        df = encode_categorical(df, categorical_columns, encoding_method.lower())
                        st.success("✅ Categorical columns encoded.")
                        st.write(df.head())

            with col2:
                with st.expander("📊 Outlier Detection & Handling", expanded=True):
                    outlier_columns = st.multiselect("Select columns to handle outliers", df.select_dtypes(include=['float64', 'int64']).columns)
                    outlier_strategy = st.radio("Outlier Handling Strategy", ["IQR"])
                    if st.button("🧹 Handle Outliers"):
                        df = handle_outliers(df, outlier_columns, outlier_strategy.lower())
                        st.success("✅ Outliers handled.")
                        st.write(df.head())

            st.subheader("🏆 Feature Selection & Binning")

            col1, col2 = st.columns(2)

            with col1:
                if df.shape[1] > 5:
                    with st.expander("🔍 Feature Selection"):
                        target_col = st.text_input("Enter target column for feature selection:")
                        if target_col in df.columns:
                            num_features = st.slider("Select number of top features", min_value=2, max_value=df.shape[1] - 1, value=5)
                            if st.button("🎯 Select Features"):
                                df = select_features(df, target_col, num_features)
                                st.success("✅ Features selected.")
                                st.write(df.head())

            with col2:
                with st.expander("🔢 Binning/Discretization"):
                    binning_columns = st.multiselect("Select columns to discretize", df.select_dtypes(include=['float64', 'int64']).columns)
                    n_bins = st.slider("Select number of bins", min_value=2, max_value=10, value=5)
                    if st.button("📊 Discretize Columns"):
                        df = bin_columns(df, binning_columns, n_bins)
                        st.success("✅ Columns discretized.")
                        st.write(df.head())

    # Model Selection Tab
    with tab3:
        if df is not None:
            st.header("🤖 Model Selection")

            st.subheader("Model Configuration")
            col1, col2 = st.columns(2)

            with col1:
                task_type = st.selectbox("Select Task Type", options=["Classification", "Regression"])
            with col2:
                target_col = st.text_input("Enter target column for model training:")

            if target_col and target_col in df.columns:
                hyperparameter_tuning = st.checkbox("Perform Hyperparameter Tuning")

                if st.button("🚀 Train Model"):
                    task = "classification" if task_type == "Classification" else "regression"
                    st.info("Model training in progress...")

                    try:
                        model, metrics, best_params = smart_model_selection(df, target_col, task_type=task, hyperparameter_tuning=hyperparameter_tuning)

                        st.subheader("📈 Model Evaluation")
                        st.write(f"**Model:** {type(model).__name__}")
                        st.write(f"**Metrics:** {metrics}")
                        if hyperparameter_tuning:
                            st.write(f"**Best Hyperparameters:** {best_params}")

                        st.success("🎉 Model training and evaluation completed!")
                    except ValueError as e:
                        st.error(f"❌ Model training failed: {e}")
                    except Exception as e:
                        st.error(f"⚠️ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
