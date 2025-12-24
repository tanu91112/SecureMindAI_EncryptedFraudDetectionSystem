"""
SecureMindAI Fraud Detection System - Streamlit Dashboard
Real-time fraud detection visualization with encrypted vector database
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import sys

# Import custom modules
from fraud_model import FraudDetectionModel, train_and_evaluate_model
from cyborg_test import FraudDetectionVectorDB, CyborgDBSimulator

# Page configuration
st.set_page_config(
    page_title="SecureMindAI Fraud Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .fraud-alert {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
        font-weight: bold;
    }
    .normal-alert {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_or_train_model():
    """Load or train the fraud detection model"""
    model_path = 'fraud_model.pkl'
    
    if os.path.exists(model_path):
        st.info("Loading pre-trained model...")
        model = FraudDetectionModel()
        model.load_model(model_path)
        return model
    else:
        st.info("Training new model... This may take a few minutes.")
        with st.spinner("Training fraud detection model..."):
            model = train_and_evaluate_model('data/transactions.csv')
        st.success("Model trained successfully!")
        return model


@st.cache_resource
def initialize_vector_db(_model):
    """Initialize and populate vector database"""
    db_path = 'cyborg_db.pkl'
    
    vector_db = FraudDetectionVectorDB(_model)
    
    if os.path.exists(db_path):
        st.info("Loading existing vector database...")
        vector_db.db.load_database(db_path)
    else:
        st.info("Indexing transactions into CyborgDB...")
        with st.spinner("Building encrypted vector index..."):
            df = pd.read_csv('data/transactions.csv')
            vector_db.index_transactions(df)
            vector_db.db.save_database(db_path)
        st.success("Vector database initialized!")
    
    return vector_db


@st.cache_data
def load_transaction_data():
    """Load transaction dataset"""
    return pd.read_csv('data/transactions.csv')


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🔒 SecureMindAI Fraud Detection System</div>', 
                unsafe_allow_html=True)
    st.markdown("### AI-Powered Fraud Detection with Encrypted Vector Storage & Controlled Access")
    st.markdown("---")
    
    # Initialize session state
    if 'fraud_alerts' not in st.session_state:
        st.session_state.fraud_alerts = []
    if 'transaction_history' not in st.session_state:
        st.session_state.transaction_history = []
    
    # Load model and database
    try:
        model = load_or_train_model()
        vector_db = initialize_vector_db(model)
        df = load_transaction_data()
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.info("Please ensure 'data/transactions.csv' exists. Run data generator first.")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Control Panel")
    
    mode = st.sidebar.radio(
        "Select Mode",
        ["📊 Dashboard Overview", "🔍 Real-time Detection", "📈 Analytics", "⚡ Performance Metrics"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ Security Status")
    st.sidebar.success("✅ Encryption: ENABLED")
    st.sidebar.info(f"🔢 Vector Dimension: {vector_db.db.dimension}")
    st.sidebar.info(f"📦 Total Vectors: {vector_db.db.metrics['total_vectors']}")
    
    # Mode: Dashboard Overview
    if mode == "📊 Dashboard Overview":
        dashboard_overview(model, vector_db, df)
    
    # Mode: Real-time Detection
    elif mode == "🔍 Real-time Detection":
        real_time_detection(model, vector_db, df)
    
    # Mode: Analytics
    elif mode == "📈 Analytics":
        analytics_view(model, df)
    
    # Mode: Performance Metrics
    elif mode == "⚡ Performance Metrics":
        performance_metrics(model, vector_db)


def dashboard_overview(model, vector_db, df):
    """Dashboard overview with key metrics"""
    
    st.header("📊 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Transactions",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        fraud_count = df['is_fraud'].sum()
        st.metric(
            label="Fraudulent Transactions",
            value=f"{fraud_count:,}",
            delta=f"{fraud_count/len(df)*100:.2f}%"
        )
    
    with col3:
        if model.metrics:
            accuracy = model.metrics.get('accuracy', 0) * 100
            st.metric(
                label="Model Accuracy",
                value=f"{accuracy:.2f}%",
                delta="High" if accuracy > 95 else "Medium"
            )
    
    with col4:
        db_stats = vector_db.get_database_stats()
        avg_latency = db_stats.get('avg_query_latency_ms', 0)
        st.metric(
            label="Avg Query Latency",
            value=f"{avg_latency:.2f} ms",
            delta="Fast" if avg_latency < 10 else "Normal"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Amount Distribution")
        
        fig = px.histogram(
            df, 
            x='amount', 
            color='is_fraud',
            nbins=50,
            labels={'is_fraud': 'Fraud Status', 'amount': 'Amount ($)'},
            color_discrete_map={0: '#2e7d32', 1: '#c62828'},
            title="Amount Distribution by Fraud Status"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fraud by Merchant Category")
        
        fraud_by_category = df.groupby('merchant_category')['is_fraud'].agg(['sum', 'count'])
        fraud_by_category['fraud_rate'] = fraud_by_category['sum'] / fraud_by_category['count'] * 100
        fraud_by_category = fraud_by_category.sort_values('fraud_rate', ascending=False)
        
        fig = px.bar(
            fraud_by_category.reset_index(),
            x='merchant_category',
            y='fraud_rate',
            title="Fraud Rate by Merchant Category",
            labels={'fraud_rate': 'Fraud Rate (%)', 'merchant_category': 'Category'},
            color='fraud_rate',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series
    st.subheader("Transaction Timeline")
    df_time = df.copy()
    df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
    df_time['date'] = df_time['timestamp'].dt.date
    
    daily_stats = df_time.groupby(['date', 'is_fraud']).size().reset_index(name='count')
    
    fig = px.line(
        daily_stats,
        x='date',
        y='count',
        color='is_fraud',
        title="Daily Transaction Volume",
        labels={'count': 'Number of Transactions', 'date': 'Date', 'is_fraud': 'Fraud Status'},
        color_discrete_map={0: '#2e7d32', 1: '#c62828'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model performance
    if model.metrics:
        st.subheader("Model Performance Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Precision", f"{model.metrics.get('precision', 0)*100:.2f}%")
            st.metric("Recall", f"{model.metrics.get('recall', 0)*100:.2f}%")
        
        with col2:
            st.metric("F1-Score", f"{model.metrics.get('f1_score', 0)*100:.2f}%")
            st.metric("Accuracy", f"{model.metrics.get('accuracy', 0)*100:.2f}%")
        
        with col3:
            # Feature importance
            if model.feature_importance:
                st.markdown("**Top 5 Important Features:**")
                sorted_features = sorted(
                    model.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                for feat, imp in sorted_features:
                    st.text(f"• {feat}: {imp:.3f}")


def real_time_detection(model, vector_db, df):
    """Real-time fraud detection interface"""
    
    st.header("🔍 Real-Time Fraud Detection")
    
    st.markdown("""
    Test the fraud detection system with real or simulated transactions.
    The system uses encrypted vector similarity search combined with ML models.
    """)
    
    # Input method
    input_method = st.radio("Input Method", ["Random from Dataset", "Manual Input"])
    
    if input_method == "Random from Dataset":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🎲 Analyze Random Transaction", type="primary"):
                with st.spinner("Analyzing transaction..."):
                    # Select random transaction
                    random_idx = np.random.randint(0, len(df))
                    transaction = df.iloc[random_idx:random_idx+1].copy()
                    
                    # Detect fraud
                    start_time = time.time()
                    result = vector_db.detect_fraud(transaction, k=10)
                    detection_time = time.time() - start_time
                    
                    # Display result
                    st.markdown("---")
                    display_detection_result(transaction.iloc[0], result, detection_time)
                    
                    # Add to history
                    st.session_state.transaction_history.append({
                        'timestamp': datetime.now(),
                        'transaction_id': result['transaction_id'],
                        'amount': result['amount'],
                        'is_fraud': result['is_fraud'],
                        'score': result['fraud_score']
                    })
        
        with col2:
            st.info("📊 Click to test with a random transaction from the dataset")
    
    else:
        st.subheader("Manual Transaction Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Amount ($)", min_value=0.01, max_value=10000.0, value=100.0)
            merchant_category = st.selectbox(
                "Merchant Category",
                ['grocery', 'restaurant', 'gas_station', 'online_retail', 
                 'electronics', 'pharmacy', 'entertainment', 'travel']
            )
            location = st.selectbox(
                "Location",
                ['New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX',
                 'Phoenix, AZ', 'Miami, FL', 'Seattle, WA', 'Boston, MA']
            )
        
        with col2:
            card_age = st.number_input("Card Age (days)", min_value=1, max_value=3650, value=365)
            transaction_frequency = st.number_input("Transaction Frequency (7 days)", min_value=0, max_value=50, value=3)
            hour = st.slider("Hour of Day", 0, 23, 12)
        
        if st.button("🔍 Analyze Transaction", type="primary"):
            with st.spinner("Analyzing transaction..."):
                # Create transaction dataframe
                transaction_data = {
                    'transaction_id': f'MANUAL_{int(time.time())}',
                    'timestamp': datetime.now().replace(hour=hour),
                    'amount': amount,
                    'merchant_id': f'MERCH_{merchant_category.upper()}_9999',
                    'merchant_category': merchant_category,
                    'location': location,
                    'card_age': card_age,
                    'transaction_frequency': transaction_frequency,
                    'is_fraud': 0  # Unknown
                }
                
                transaction = pd.DataFrame([transaction_data])
                
                # Detect fraud
                start_time = time.time()
                result = vector_db.detect_fraud(transaction, k=10)
                detection_time = time.time() - start_time
                
                # Display result
                st.markdown("---")
                display_detection_result(transaction.iloc[0], result, detection_time)
                
                # Add to history
                st.session_state.transaction_history.append({
                    'timestamp': datetime.now(),
                    'transaction_id': result['transaction_id'],
                    'amount': result['amount'],
                    'is_fraud': result['is_fraud'],
                    'score': result['fraud_score']
                })
    
    # Transaction history
    if st.session_state.transaction_history:
        st.markdown("---")
        st.subheader("📜 Recent Detection History")
        
        history_df = pd.DataFrame(st.session_state.transaction_history[-10:])
        history_df['status'] = history_df['is_fraud'].map({True: '🚨 FRAUD', False: '✅ NORMAL'})
        history_df['score'] = history_df['score'].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(
            history_df[['timestamp', 'transaction_id', 'amount', 'status', 'score']],
            use_container_width=True
        )


def display_detection_result(transaction, result, detection_time):
    """Display fraud detection result"""
    
    is_fraud = result['is_fraud']
    fraud_score = result['fraud_score']
    
    # Alert box
    if is_fraud:
        st.markdown(
            f'<div class="fraud-alert">🚨 FRAUD DETECTED - Risk Score: {fraud_score:.1%}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="normal-alert">✅ TRANSACTION NORMAL - Risk Score: {fraud_score:.1%}</div>',
            unsafe_allow_html=True
        )
    
    # Details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Transaction Details**")
        st.write(f"ID: `{result['transaction_id']}`")
        st.write(f"Amount: **${result['amount']:.2f}**")
        st.write(f"Category: {transaction['merchant_category']}")
        st.write(f"Location: {transaction['location']}")
    
    with col2:
        st.markdown("**Detection Metrics**")
        st.write(f"Fraud Score: **{fraud_score:.1%}**")
        st.write(f"Model Probability: {result['model_probability']:.1%}")
        st.write(f"Similarity Score: {result['similarity_fraud_ratio']:.1%}")
        st.write(f"Detection Time: {detection_time*1000:.2f} ms")
    
    with col3:
        st.markdown("**Similar Transactions**")
        st.write(f"Total Found: {result['similar_transactions']}")
        st.write(f"Fraud Count: {result['similar_fraud_count']}")
        st.write(f"Fraud Ratio: {result['similarity_fraud_ratio']:.1%}")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Level"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if is_fraud else "darkgreen"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)


def analytics_view(model, df):
    """Advanced analytics view"""
    
    st.header("📈 Advanced Analytics")
    
    # Fraud patterns
    st.subheader("Fraud Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hour of day analysis
        fraud_by_hour = df.groupby('hour')['is_fraud'].agg(['sum', 'count'])
        fraud_by_hour['rate'] = fraud_by_hour['sum'] / fraud_by_hour['count'] * 100
        
        fig = px.bar(
            fraud_by_hour.reset_index(),
            x='hour',
            y='rate',
            title="Fraud Rate by Hour of Day",
            labels={'rate': 'Fraud Rate (%)', 'hour': 'Hour'},
            color='rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Card age analysis
        df['card_age_group'] = pd.cut(
            df['card_age'],
            bins=[0, 30, 90, 180, 365, 730, 3650],
            labels=['<1 month', '1-3 months', '3-6 months', '6-12 months', '1-2 years', '>2 years']
        )
        
        fraud_by_age = df.groupby('card_age_group')['is_fraud'].agg(['sum', 'count'])
        fraud_by_age['rate'] = fraud_by_age['sum'] / fraud_by_age['count'] * 100
        
        fig = px.bar(
            fraud_by_age.reset_index(),
            x='card_age_group',
            y='rate',
            title="Fraud Rate by Card Age",
            labels={'rate': 'Fraud Rate (%)', 'card_age_group': 'Card Age'},
            color='rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    if model.feature_importance:
        st.subheader("Feature Importance")
        
        importance_df = pd.DataFrame(
            list(model.feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True).tail(15)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 15 Most Important Features for Fraud Detection",
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    
    numeric_cols = ['amount', 'card_age', 'transaction_frequency', 'hour', 'day_of_week', 'is_fraud']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def performance_metrics(model, vector_db):
    """Performance metrics view"""
    
    st.header("⚡ Performance Metrics")
    
    db_stats = vector_db.get_database_stats()
    
    # Database metrics
    st.subheader("🗄️ CyborgDB Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Vectors", f"{db_stats['total_vectors']:,}")
    
    with col2:
        st.metric("Vector Dimension", db_stats['dimension'])
    
    with col3:
        st.metric("Total Queries", f"{db_stats['total_queries']:,}")
    
    with col4:
        st.metric("Total Insertions", f"{db_stats['total_insertions']:,}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Avg Insert Latency",
            f"{db_stats['avg_insert_latency_ms']:.3f} ms",
            delta=f"{'Fast' if db_stats['avg_insert_latency_ms'] < 1 else 'Normal'}"
        )
    
    with col2:
        st.metric(
            "Avg Query Latency",
            f"{db_stats['avg_query_latency_ms']:.3f} ms",
            delta=f"{'Fast' if db_stats['avg_query_latency_ms'] < 10 else 'Normal'}"
        )
    
    # Throughput calculation
    st.markdown("---")
    st.subheader("📊 Throughput Analysis")
    
    if db_stats['avg_insert_latency_ms'] > 0:
        insert_throughput = 1000 / db_stats['avg_insert_latency_ms']
        st.metric("Insert Throughput", f"{insert_throughput:,.0f} vectors/sec")
    
    if db_stats['avg_query_latency_ms'] > 0:
        query_throughput = 1000 / db_stats['avg_query_latency_ms']
        st.metric("Query Throughput", f"{query_throughput:,.0f} queries/sec")
    
    # Encryption status
    st.markdown("---")
    st.subheader("🔒 Security Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("✅ Encryption: ENABLED")
        st.info("🔐 Algorithm: Fernet (AES-128 CBC)")
    
    with col2:
        st.success(f"✅ Cache Status: {db_stats['cache_status'].upper()}")
        st.info("🛡️Data encrypted at rest with scoped in-memory access")
    
    # Model metrics
    if model.metrics:
        st.markdown("---")
        st.subheader("🤖 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = model.metrics.get('accuracy', 0) * 100
            st.metric("Accuracy", f"{accuracy:.2f}%")
        
        with col2:
            precision = model.metrics.get('precision', 0) * 100
            st.metric("Precision", f"{precision:.2f}%")
        
        with col3:
            recall = model.metrics.get('recall', 0) * 100
            st.metric("Recall", f"{recall:.2f}%")
        
        with col4:
            f1 = model.metrics.get('f1_score', 0) * 100
            st.metric("F1-Score", f"{f1:.2f}%")


if __name__ == "__main__":
    main()
