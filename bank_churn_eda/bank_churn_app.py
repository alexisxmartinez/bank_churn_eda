# models/bank_churn_app.py  
  
# 🚀 Bank Customer Churn Analysis - Streamlit Dashboard  
  
import streamlit as st  
import pandas as pd  
import plotly.express as px  
import json  
from pathlib import Path  
import os  
import joblib  # For model loading (optional)  
  
# ----------------------------  
# 🔧 Configuration and Paths  
# ----------------------------  
  
# Define the base directory (parent of the current script's directory)  
BASE_DIR = Path(__file__).parent.parent  
  
# Construct absolute paths  
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'Bank_Churn.csv'  
INSIGHTS_PATH = BASE_DIR / 'data' / 'processed' / 'insights.json'  
MODEL_PATH = BASE_DIR / 'models' / 'churn_model.pkl'           # Optional  
EXPLAINER_PATH = BASE_DIR / 'models' / 'shap_explainer.pkl'    # Optional  
  
# ----------------------------  
# 📄 Page Configuration  
# ----------------------------  
  
st.set_page_config(  
    page_title="🏦 Bank Churn Analysis",  
    page_icon="🏦",  
    layout="wide"  
)  
  
# ----------------------------  
# 📥 Data Loading with Caching  
# ----------------------------  
  
@st.cache_data(show_spinner=False)  
def load_data(data_path, insights_path):  
    """Load the dataset and insights JSON with error handling."""  
    # Debugging: Display paths  
    st.write(f"🔍 BASE_DIR: {BASE_DIR}")  
    st.write(f"📂 DATA_PATH: {data_path}")  
    st.write(f"📂 INSIGHTS_PATH: {insights_path}")  
  
    # Check if data file exists  
    if not data_path.exists():  
        st.error(f"❌ The data file was not found at {data_path}. Please check the path.")  
        return None, None  
    else:  
        st.success("✅ Data file exists.")  
  
    # Load DataFrame  
    try:  
        df = pd.read_csv(data_path)  
        st.success("✅ Data loaded successfully!")  
    except Exception as e:  
        st.error(f"❌ An error occurred while loading the data: {e}")  
        return None, None  
  
    # Check if insights file exists  
    if not insights_path.exists():  
        st.error(f"❌ The insights file was not found at {insights_path}. Please ensure it exists.")  
        insights = {}  
    else:  
        # Load Insights  
        try:  
            with open(insights_path, 'r') as f:  
                insights = json.load(f)  
            st.success("✅ Insights loaded successfully!")  
        except json.JSONDecodeError:  
            st.error(f"❌ Failed to decode JSON from {insights_path}. Please check the file format.")  
            insights = {}  
        except Exception as e:  
            st.error(f"❌ An error occurred while loading insights: {e}")  
            insights = {}  
  
    # Additional Debugging: Check if required columns exist  
    required_columns = ['Age', 'Geography', 'Exited', 'Balance', 'CreditScore',   
                        'NumOfProducts', 'IsActiveMember', 'Gender', 'Tenure',   
                        'HasCrCard', 'CustomerId']  
    missing_columns = [col for col in required_columns if col not in df.columns]  
    if missing_columns:  
        st.error(f"❌ The following required columns are missing in the data: {missing_columns}")  
        return None, None  
    else:  
        st.success("✅ All required columns are present in the data.")  
  
    return df, insights  
  
# Load Data  
df, insights = load_data(DATA_PATH, INSIGHTS_PATH)  
  
# If data failed to load, stop the app  
if df is None or insights is None:  
    st.stop()  
  
# ----------------------------  
# 🧩 Define Page Functions  
# ----------------------------  
  
def overview_page(df, insights):  
    """Overview Page: Display key metrics and basic visualizations."""  
    st.title("🏦 Bank Customer Churn Analysis Dashboard")  
  
    # Key Metrics in Columns  
    col1, col2, col3, col4 = st.columns(4)  
    with col1:  
        st.metric("Total Customers", f"{insights.get('total_customers', 0):,}")  
    with col2:  
        st.metric("Churn Rate", f"{insights.get('churn_rate', 0):.1f}%")  
    with col3:  
        st.metric("Avg Balance", f"${insights.get('avg_balance', 0):,.2f}")  
    with col4:  
        st.metric("Avg Age", f"{insights.get('avg_age', 0):.1f}")  
  
    st.markdown("---")  
  
    # Geographic Distribution  
    st.subheader("🌍 Geographic Distribution of Customers")  
    geo_data = df.groupby('Geography')['Exited'].agg(['count', 'mean']).reset_index()  
    geo_data['churn_rate'] = geo_data['mean'] * 100  
  
    fig_geo = px.bar(  
        geo_data,  
        x='Geography',  
        y='count',  
        color='churn_rate',  
        color_continuous_scale=px.colors.sequential.Viridis,  
        labels={  
            'count': 'Number of Customers',  
            'churn_rate': 'Churn Rate (%)',  
            'Geography': 'Geographic Region'  
        },  
        title='🌍 Customer Distribution and Churn Rate by Geography',  
        text='count'  
    )  
    fig_geo.update_traces(texttemplate='%{text:,}', textposition='outside')  
    fig_geo.update_layout(xaxis_title='Geographic Region', yaxis_title='Number of Customers')  
    st.plotly_chart(fig_geo, use_container_width=True)  
  
    st.markdown("---")  
  
    # Age Distribution  
    st.subheader("🎂 Age Distribution by Churn Status")  
    fig_age = px.histogram(  
        df,  
        x='Age',  
        color='Exited',  
        marginal='box',  
        nbins=30,  
        title='🎂 Age Distribution by Churn Status',  
        labels={'Exited': 'Churned'},  
        color_discrete_sequence=px.colors.qualitative.Pastel  
    )  
    st.plotly_chart(fig_age, use_container_width=True)  
  
def customer_analysis_page(df):  
    """Customer Analysis Page: Segment customers based on filters."""  
    st.title("👥 Customer Segmentation Analysis")  
  
    # Filters  
    st.sidebar.header("🔍 Filter Options")  
    selected_geography = st.sidebar.multiselect(  
        "🌍 Select Geography",  
        options=df['Geography'].unique(),  
        default=df['Geography'].unique()  
    )  
    selected_gender = st.sidebar.multiselect(  
        "👫 Select Gender",  
        options=df['Gender'].unique(),  
        default=df['Gender'].unique()  
    )  
    balance_min, balance_max = float(df['Balance'].min()), float(df['Balance'].max())  
    balance_range = st.sidebar.slider(  
        "💰 Balance Range ($)",  
        min_value=balance_min,  
        max_value=balance_max,  
        value=(balance_min, balance_max)  
    )  
    n_products_min, n_products_max = int(df['NumOfProducts'].min()), int(df['NumOfProducts'].max())  
    n_products_range = st.sidebar.slider(  
        "🛍️ Number of Products",  
        min_value=n_products_min,  
        max_value=n_products_max,  
        value=(n_products_min, n_products_max)  
    )  
    has_card = st.sidebar.checkbox("💳 Has Credit Card", value=False)  
  
    # Apply Filters  
    filtered_df = df[  
        (df['Geography'].isin(selected_geography)) &  
        (df['Gender'].isin(selected_gender)) &  
        (df['Balance'].between(balance_range[0], balance_range[1])) &  
        (df['NumOfProducts'].between(n_products_range[0], n_products_range[1])) &  
        (df['HasCrCard'] == int(has_card))  
    ].copy()  
  
    st.markdown(f"### 📊 Showing {filtered_df.shape[0]} customers based on selected filters.")  
  
    # Customer Segments  
    st.subheader("📈 Customer Segments and Churn Rate")  
  
    # Create Age Groups and Balance Segments  
    filtered_df = filtered_df.assign(  
        AgeGroup=pd.cut(  
            filtered_df['Age'],  
            bins=[0, 30, 40, 50, 60, 100],  
            labels=['18-30', '31-40', '41-50', '51-60', '60+']  
        ),  
        BalanceSegment=pd.qcut(  
            filtered_df['Balance'],  
            q=4,  
            labels=['Low', 'Medium-Low', 'Medium-High', 'High']  
        )  
    )  
  
    # Group by AgeGroup and BalanceSegment  
    segment_analysis = filtered_df.groupby(['AgeGroup', 'BalanceSegment'])['Exited'].agg(['count', 'mean']).reset_index()  
    segment_analysis['churn_rate'] = segment_analysis['mean'] * 100  
  
    fig_segments = px.treemap(  
        segment_analysis,  
        path=['AgeGroup', 'BalanceSegment'],  
        values='count',  
        color='churn_rate',  
        color_continuous_scale='RdBu',  
        title='🧩 Customer Segmentation Analysis',  
        hover_data={'churn_rate': ':.2f'}  
    )  
    st.plotly_chart(fig_segments, use_container_width=True)  
  
    st.markdown("---")  
  
    # Financial Distribution  
    st.subheader("💸 Credit Score vs. Balance")  
    fig_finance = px.scatter(  
        filtered_df,  
        x='CreditScore',  
        y='Balance',  
        color='Exited',  
        size='Age',  
        hover_data=['CustomerId', 'Geography', 'NumOfProducts'],  
        title='💸 Credit Score vs. Balance',  
        labels={'Exited': 'Churned'},  
        color_discrete_sequence=px.colors.qualitative.Set1  
    )  
    st.plotly_chart(fig_finance, use_container_width=True)  
  
def churn_analysis_page(df):  
    """Churn Analysis Page: Identify key churn factors."""  
    st.title("📈 Churn Risk Analysis")  
  
    st.markdown("---")  
  
    # Product Analysis  
    st.subheader("🛍️ Churn Rate by Number of Products")  
    product_churn = df.groupby('NumOfProducts')['Exited'].agg(['count', 'mean']).reset_index()  
    product_churn['churn_rate'] = product_churn['mean'] * 100  
  
    fig_products = px.bar(  
        product_churn,  
        x='NumOfProducts',  
        y='churn_rate',  
        color='churn_rate',  
        color_continuous_scale=px.colors.sequential.Viridis,  
        labels={  
            'NumOfProducts': 'Number of Products',  
            'churn_rate': 'Churn Rate (%)'  
        },  
        title='🛍️ Churn Rate by Number of Products',  
        text=product_churn['churn_rate'].apply(lambda x: f"{x:.2f}%")  
    )  
    fig_products.update_traces(textposition='outside')  
    fig_products.update_layout(xaxis_title='Number of Products', yaxis_title='Churn Rate (%)')  
    st.plotly_chart(fig_products, use_container_width=True)  
  
    # Active Member Analysis and Tenure Analysis Side by Side  
    st.markdown("---")  
    col1, col2 = st.columns(2)  
  
    with col1:  
        st.subheader("🔄 Churn Rate by Member Status")  
        active_churn = df.groupby('IsActiveMember')['Exited'].mean() * 100  
        active_churn.index = active_churn.index.map({0: 'Inactive', 1: 'Active'})  
  
        fig_active = px.pie(  
            names=active_churn.index,  
            values=active_churn.values,  
            title='🔄 Churn Rate by Member Status',  
            hole=0.4,  
            color_discrete_sequence=px.colors.sequential.Blues  
        )  
        fig_active.update_traces(textposition='inside', textinfo='percent+label')  
        st.plotly_chart(fig_active, use_container_width=True)  
  
    with col2:  
        st.subheader("📈 Churn Rate by Tenure (Years)")  
        tenure_churn = df.groupby('Tenure')['Exited'].mean() * 100  
  
        fig_tenure = px.line(  
            x=tenure_churn.index,  
            y=tenure_churn.values,  
            markers=True,  
            title='📈 Churn Rate by Tenure (Years)',  
            labels={'x': 'Tenure (Years)', 'y': 'Churn Rate (%)'},  
            color_discrete_sequence=['#FF5733']  
        )  
        st.plotly_chart(fig_tenure, use_container_width=True)  
  
def predictions_page(df, model=None):  
    """Predictions Page: Predict churn risk based on user inputs."""  
    st.title("🤖 Churn Prediction")  
  
    st.markdown("---")  
  
    st.subheader("📝 Enter Customer Information")  
  
    # Input Form  
    col1, col2, col3 = st.columns(3)  
    with col1:  
        credit_score = st.slider("📊 Credit Score", 300, 850, 600)  
        age = st.slider("🎂 Age", 18, 100, 35)  
        tenure = st.slider("📆 Tenure (Years)", 0, 10, 5)  
    with col2:  
        balance = st.number_input("💰 Balance ($)", min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0)  
        geography = st.selectbox("🌍 Geography", df['Geography'].unique())  
        n_products = st.slider("🛍️ Number of Products", 1, 4, 1)  
    with col3:  
        gender = st.selectbox("👫 Gender", ["Male", "Female"])  
        is_active = st.checkbox("🔄 Is Active Member")  
        has_card = st.checkbox("💳 Has Credit Card")  
  
    if st.button("📈 Predict Churn Risk"):  
        # Placeholder for model prediction  
        # ----------------------------  
        # To implement predictions, follow these steps:  
        # 1. Train a machine learning model on your dataset.  
        # 2. Save the trained model using joblib or pickle in the 'models/' directory.  
        # 3. Load the model here and use it to make predictions based on user inputs.  
        # 4. (Optional) Use SHAP or similar libraries to provide model explanations.  
  
        if model is None:  
            st.error("❌ ML model is not loaded. Prediction cannot be performed.")  
        else:  
            # Preprocess input data as per model's requirements  
            input_data = pd.DataFrame({  
                'CreditScore': [credit_score],  
                'Age': [age],  
                'Tenure': [tenure],  
                'Balance': [balance],  
                'NumOfProducts': [n_products],  
                'HasCrCard': [int(has_card)],  
                'IsActiveMember': [int(is_active)],  
                'Gender': [1 if gender == 'Male' else 0],  # Example encoding  
                'Geography': [geography]  # Ensure consistent encoding  
            })  
  
            # Example: Encode 'Geography' if necessary  
            # This depends on how the model was trained  
            # For example, using one-hot encoding:  
            # geo_dummies = pd.get_dummies(input_data['Geography'], prefix='Geography')  
            # input_data = pd.concat([input_data.drop('Geography', axis=1), geo_dummies], axis=1)  
  
            # Prediction  
            try:  
                prediction = model.predict(input_data)[0]  
                churn_probability = model.predict_proba(input_data)[0][1] * 100  
                if prediction == 1:  
                    st.warning(f"⚠️ The customer is **likely to churn** with a probability of {churn_probability:.2f}%.")  
                else:  
                    st.success(f"✅ The customer is **likely to stay** with a probability of {100 - churn_probability:.2f}%.")  
            except Exception as e:  
                st.error(f"❌ An error occurred during prediction: {e}")  
  
# ----------------------------  
# 🗺️ Main Navigation  
# ----------------------------  
  
def main():  
    """Main function to navigate between pages."""  
    # Load the ML model (optional)  
    @st.cache_resource(show_spinner=False)  
    def load_model(model_path):  
        """Load the trained machine learning model."""  
        if not model_path.exists():  
            st.error(f"❌ The model file was not found at {model_path}. Prediction feature will be disabled.")  
            return None  
        try:  
            model = joblib.load(model_path)  
            st.success("✅ ML Model loaded successfully!")  
            return model  
        except Exception as e:  
            st.error(f"❌ An error occurred while loading the model: {e}")  
            return None  
  
    model = load_model(MODEL_PATH)  
  
    # Sidebar Navigation  
    st.sidebar.title("📋 Navigation")  
    page = st.sidebar.radio("Go to", ["Overview", "Customer Analysis", "Churn Analysis", "Predictions"])  
  
    # Display the selected page  
    if page == "Overview":  
        overview_page(df, insights)  
    elif page == "Customer Analysis":  
        customer_analysis_page(df)  
    elif page == "Churn Analysis":  
        churn_analysis_page(df)  
    elif page == "Predictions":  
        predictions_page(df, model)  
  
    # Footer  
    st.markdown("---")  
    st.markdown("""  
    💼 **Created by [Your Name]**    
    🔗 [GitHub](https://github.com/your-github) | [LinkedIn](https://linkedin.com/in/your-linkedin)  
    """)  
  
if __name__ == "__main__":  
    main()  