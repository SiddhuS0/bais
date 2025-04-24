import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
import numpy as np
from prophet import Prophet
import json
import hashlib
import uuid
from pathlib import Path

# Adjust path for importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Page config
st.set_page_config(page_title="📊 Business Analytics Dashboard", layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>📈 Business Insights Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# Data storage setup
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
USERS_FILE = DATA_DIR / "users.json"
USER_DATASETS_DIR = DATA_DIR / "datasets"

# Initialize storage files
if not USERS_FILE.exists():
    USERS_FILE.write_text("{}")

USER_DATASETS_DIR.mkdir(exist_ok=True)

# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# User management functions
def load_users():
    return json.loads(USERS_FILE.read_text())

def save_users(users):
    USERS_FILE.write_text(json.dumps(users, indent=2))

def register_user(username, password, name):
    users = load_users()
    if username in users:
        return False
    users[username] = {
        "name": name,
        "password": hash_password(password),
        "datasets": []
    }
    save_users(users)
    return True

def verify_user(username, password):
    users = load_users()
    if username in users and users[username]["password"] == hash_password(password):
        return users[username]
    return None

def save_user_dataset(username, df, filename):
    user_dir = USER_DATASETS_DIR / username
    user_dir.mkdir(exist_ok=True)
    
    dataset_id = str(uuid.uuid4())
    filepath = user_dir / f"{dataset_id}.csv"
    df.to_csv(filepath, index=False)
    
    users = load_users()
    users[username]["datasets"].append({
        "id": dataset_id,
        "filename": filename,
        "upload_date": pd.Timestamp.now().isoformat()
    })
    save_users(users)
    return dataset_id

def load_user_datasets(username):
    users = load_users()
    if username not in users:
        return []
    
    datasets = []
    for dataset in users[username]["datasets"]:
        filepath = USER_DATASETS_DIR / username / f"{dataset['id']}.csv"
        if filepath.exists():
            dataset["df"] = pd.read_csv(filepath)
            datasets.append(dataset)
    return datasets

def auth_system():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        # Create a centered container with constrained width
        col1, col2, col3 = st.columns([1, 1 , 1])  # Middle column is 3x wider than side columns
        
        with col2:  # This centers the content
            # st.markdown("<h2 style='text-align: center;'>Business Insights Dashboard</h2>", 
            #            unsafe_allow_html=True)
            
            # Tab system for login/signup
            login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
            
            with login_tab:
                with st.form("login_form"):
                    st.subheader("Login", divider='gray')
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    
                    # Centered submit button
                    cols = st.columns(3)
                    with cols[1]:  # Middle column
                        login_submitted = st.form_submit_button("Login", use_container_width=True)
                    
                    if login_submitted:
                        user_info = verify_user(username, password)
                        if user_info:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.user_info = user_info
                            st.rerun()
                        else:
                            st.error("Invalid username or password", icon="⚠️")
            
            with signup_tab:
                with st.form("signup_form"):
                    st.subheader("Create Account", divider='gray')
                    new_username = st.text_input("Choose Username")
                    new_name = st.text_input("Full Name")
                    new_password = st.text_input("Choose Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    
                    # Centered submit button
                    cols = st.columns(3)
                    with cols[1]:  # Middle column
                        signup_submitted = st.form_submit_button("Sign Up", use_container_width=True)
                    
                    if signup_submitted:
                        if new_password != confirm_password:
                            st.error("Passwords don't match", icon="🔒")
                        elif len(new_password) < 6:
                            st.error("Password must be at least 6 characters", icon="🔐")
                        else:
                            if register_user(new_username, new_password, new_name):
                                st.success("Account created! Please login.", icon="✅")
                            else:
                                st.error("Username already exists", icon="⛔")
        
        st.stop()  # Stop execution if not authenticated

# Check authentication
auth_system()

# Main App (only accessible if authenticated)
st.write(f'Welcome {st.session_state.user_info["name"]} to Your Business Dashboard')

# Sidebar - User section
st.sidebar.header("👤 User Profile")
st.sidebar.write(f"Logged in as: {st.session_state.username}")

# Sidebar - Dataset management
st.sidebar.header("📂 Your Datasets")
user_datasets = load_user_datasets(st.session_state.username)

if user_datasets:
    selected_dataset = st.sidebar.selectbox(
        "Select a saved dataset",
        options=[d["filename"] for d in user_datasets],
        index=0
    )
    selected_data = next(d for d in user_datasets if d["filename"] == selected_dataset)
    user_df = selected_data["df"]
    st.sidebar.success(f"Loaded dataset: {selected_dataset}")
else:
    st.sidebar.info("No saved datasets yet")
    user_df = None

# Sidebar - Upload new dataset
st.sidebar.header("📤 Upload New Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv", key="file_uploader")

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    if st.sidebar.button("💾 Save Dataset"):
        dataset_id = save_user_dataset(st.session_state.username, user_df, uploaded_file.name)
        st.sidebar.success("Dataset saved successfully!")
        st.rerun()

# Main content
if user_df is not None:
    st.success("✅ Dataset loaded successfully!")

    with st.expander("🔍 Preview Data"):
        st.dataframe(user_df.head())

    from src.preprocessing.clean_data import load_and_validate_data, save_clean_data, REQUIRED_COLUMNS
    from src.features.extract_metrics import extract_features

    # Extract Insights button
    if st.button("📥 Extract Insights", key="extract_insights_button"):
        try:
            with st.spinner("🔍 Validating and processing your data..."):
                user_df_clean = load_and_validate_data(user_df)
                save_clean_data(user_df_clean)
                user_features = extract_features(user_df_clean)
                
                st.session_state["user_features"] = user_features
                st.session_state["extracted"] = True
                st.success("✅ Insights extracted successfully!")
                
        except ValueError as e:
            st.error(f"""
            ## ❌ Data Validation Failed
            
            **Error Details:**  
            {str(e)}
            
            ### How to fix this:
            1. Check your CSV file contains all required columns
            2. Verify date formats (YYYY-MM-DD)
            3. Ensure no missing values in numeric columns
            """)
            
            # Show sample of what was uploaded
            with st.expander("🔍 See uploaded data sample"):
                st.dataframe(user_df.head(3))
                
            # Show required columns
            with st.expander("ℹ️ Required columns help"):
                cols = [f"- **{col}**" for col in REQUIRED_COLUMNS]
                st.markdown("\n".join(cols))
                st.download_button(
                    "Download sample template",
                    data=pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(index=False),
                    file_name="template.csv",
                    mime="text/csv"
                )
            
            # Clear any invalid state
            st.session_state["extracted"] = False
            if "user_features" in st.session_state:
                del st.session_state["user_features"]
                
        except Exception as e:
            st.error(f"""
            ## 🛠️ Unexpected Error
            
            Our system encountered an unexpected problem:
            ```python
            {str(e)}
            ```
            
            Please try again or contact support if this persists.
            """)
            st.session_state["extracted"] = False

    if "user_features" in st.session_state and st.session_state["extracted"]:
        user_features = st.session_state["user_features"]

        # Filters
        st.markdown("### 🎛️ Apply Filters")
        with st.expander("🔧 Filter Options"):
            # Date filter
            min_date = pd.to_datetime(user_features["date"]).min()
            max_date = pd.to_datetime(user_features["date"]).max()
            date_range = st.date_input("Select Date Range", [min_date, max_date])

            # Revenue filter
            min_rev, max_rev = user_features["revenue"].min(), user_features["revenue"].max()
            revenue_filter = st.slider("Filter by Revenue", min_value=float(min_rev),
                                    max_value=float(max_rev),
                                    value=(float(min_rev), float(max_rev)))

        # Apply filters if insights are extracted
        filtered_df = user_features.copy()
        filtered_df["date"] = pd.to_datetime(filtered_df["date"])

        filtered_df = filtered_df[
            (filtered_df["date"] >= pd.to_datetime(date_range[0])) & 
            (filtered_df["date"] <= pd.to_datetime(date_range[1])) &
            (filtered_df["revenue"] >= revenue_filter[0]) &
            (filtered_df["revenue"] <= revenue_filter[1])
        ]

        # Display filtered KPIs
        st.markdown("## 🔢 Key Performance Indicators (KPIs)")
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Revenue", f"₹{filtered_df['revenue'].sum():,.2f}")
        col2.metric("📈 Net Profit", f"₹{filtered_df['net_profit'].sum():,.2f}")
        col3.metric("📊 Avg ROI", f"{filtered_df['ROI (%)'].mean():.2f}%")

        # Show filtered dataframe
        st.markdown("### 📋 Filtered Insights Table")
        st.dataframe(filtered_df)

        # Show visualizations if button is clicked
        if st.button("📊 Show Visualizations", key="show_visualizations_button"):
            st.markdown("## 📉 Visualizations")

            # Revenue & Net Profit Over Time (Line chart)
            fig1 = px.line(
                filtered_df,
                x="date",
                y=["revenue", "net_profit"],
                title="Revenue & Net Profit Over Time",
                markers=True,
                color_discrete_sequence=["#1f77b4", "#ff7f0e"]  # Blue, Orange
            )
            st.plotly_chart(fig1, use_container_width=True)

            # ROI & Profit Margin Over Time (Bar chart)
            fig2 = px.bar(
                filtered_df,
                x="date",
                y=["ROI (%)", "Profit_Margin (%)"],
                title="ROI & Profit Margin Over Time",
                barmode="group",
                color_discrete_sequence=["#2ca02c", "#1f77b4"]  # Green, Red
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Investment vs Revenue (Scatter chart)
            filtered_df["profit_magnitude"] = np.abs(filtered_df["net_profit"])
            filtered_df["profit_status"] = filtered_df["net_profit"].apply(lambda x: "Profit" if x >= 0 else "Loss")

            fig3 = px.scatter(
                filtered_df,
                x="investment_cost",
                y="revenue",
                size="profit_magnitude",
                color="profit_status",
                title="Investment vs Revenue (Size = Profit Magnitude, Color = Status)",
                hover_data=["net_profit"],
                color_discrete_map={
                    "Profit": "#2ECC71",  # Green
                    "Loss": "#E74C3C"     # Red
                }
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Cost break-down
            fig4 = px.bar(
                filtered_df,
                x="date",
                y=["operating_expense", "marketing_cost", "cogs"],
                title="Cost Breakdown Over Time",
                barmode="stack",
                color_discrete_sequence=["#f7b7a3", "#ffb9b9", "#ff6f61"]
            )
            st.plotly_chart(fig4, use_container_width=True)

            # Customer vs marketing cost
            filtered_df["customer_acquisition_cost"] = filtered_df["marketing_cost"] / filtered_df["new_customers_acquired"]
            fig5 = px.scatter(
                filtered_df,
                x="customer_acquisition_cost",
                y="revenue",
                size="units_sold",
                color="region",
                title="Customer Acquisition Cost vs. Revenue",
                hover_data=["customer_id"],
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig5, use_container_width=True)

            # Profit vs employee count
            fig6 = px.scatter(
                filtered_df,
                x="employee_count",
                y="net_profit",
                title="Profit vs Employee Count",
                hover_data=["product_name"],
                color="region",
                color_discrete_sequence=px.colors.qualitative.G10
            )
            st.plotly_chart(fig6, use_container_width=True)

            # Sales by Region (Pie chart)
            region_sales = filtered_df.groupby("region")["revenue"].sum().reset_index()
            fig7 = px.pie(
                region_sales,
                names="region",
                values="revenue",
                title="Sales by Region",
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            st.plotly_chart(fig7, use_container_width=True)

            # Top Performing Products (Bar chart)
            product_sales = filtered_df.groupby("product_name")["units_sold"].sum().reset_index()
            product_sales = product_sales.sort_values("units_sold", ascending=False).head(10)
            fig8 = px.bar(
                product_sales,
                x="product_name",
                y="units_sold",
                title="Top 10 Best Selling Products",
                color="product_name",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig8, use_container_width=True)

            from src.insights.insights_engine import generate_business_insights
            # Business Insights Section
            st.markdown("## 💡 Strategic Business Insights")
            st.markdown("Use these data-driven insights to improve performance, cut costs, and identify growth opportunities.")

            with st.spinner("Analyzing data and generating insights..."):
                insights = generate_business_insights(filtered_df)

            # Show insights with improved styling
            for key, value in insights.items():
                with st.expander(f"🔍 {key.replace('_', ' ').title()}"):
                    st.markdown(f"""
                    <div style='
                        font-size: 16px; 
                        padding: 15px; 
                        background-color: #f0f0f5; 
                        border-left: 6px solid #4CAF50;
                        color: #333333;
                        border-radius: 5px;
                        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.05);
                    '>
                    ✅ {value}
                    </div>
                    """, unsafe_allow_html=True)

        from src.prediction.revenue_forecast import forecast_metric, plot_forecast
        st.markdown("## 🔮 Forecasting Insights")
        st.markdown("Analyze upcoming trends in your **Revenue** and **Net Profit** for better planning.")

        # Toggle between Revenue and Net Profit
        forecast_option = st.radio("Choose a metric to forecast:", ("Revenue", "Net Profit"), horizontal=True)

        if st.button("📊 Generate Forecast", key="generate_forecast_button"):
            with st.spinner("Generating forecast..."):
                try:
                    if forecast_option == "Revenue":
                        forecast_df, model = forecast_metric(filtered_df, 'revenue')
                        fig = plot_forecast(model, forecast_df, filtered_df, 'revenue')
                    else:
                        forecast_df, model = forecast_metric(filtered_df, 'net_profit')
                        fig = plot_forecast(model, forecast_df, filtered_df, 'net_profit')

                    st.success(f"✅ {forecast_option} forecast complete!")
                    st.pyplot(fig)
                    with st.expander(f"📋 Forecast Data ({forecast_option})"):
                        selected_cols = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                        selected_cols.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']  
                        st.dataframe(selected_cols)

                except Exception as e:
                    st.error(f"❌ Forecasting failed: {e}")

    else:
        st.warning("Extract Insights first by clicking the '📥 Extract Insights' button")

else:
    st.warning("📎 Upload a CSV file from the sidebar to begin")

#LOGOUT :
if st.session_state.get('authenticated', False):
    if st.sidebar.button("🚪 Logout", key="unique_logout_button"):
        st.session_state.authenticated = False
        st.session_state.clear()
        st.rerun()

if user_datasets:
    with st.sidebar.expander("🗑️ Delete Datasets"):
        to_delete = st.multiselect(
            "Select datasets to delete",
            options=[d["filename"] for d in user_datasets]
        )
        if st.button("Delete Selected", key="delete_datasets_button"):
            users = load_users()
            for dataset in user_datasets:
                if dataset["filename"] in to_delete:
                    # Remove from user record
                    users[st.session_state.username]["datasets"] = [
                        d for d in users[st.session_state.username]["datasets"]
                        if d["id"] != dataset["id"]
                    ]
                    # Delete the file
                    filepath = USER_DATASETS_DIR / st.session_state.username / f"{dataset['id']}.csv"
                    if filepath.exists():
                        filepath.unlink()
            save_users(users)
            st.sidebar.success("Datasets deleted successfully!")
            st.rerun()