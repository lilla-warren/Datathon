# app.py - WORKING HOSPITAL ANALYTICS PLATFORM
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="MedAnalytics Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1a237e;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1a237e;
        border-bottom: 2px solid #1976d2;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1976d2;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 MedAnalytics Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Clinical Intelligence Platform • Predictive Analytics • Patient Risk Stratification")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Data Configuration")
        uploaded_file = st.file_uploader("Upload Clinical Dataset (CSV)", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Target selection
                target_variable = st.selectbox(
                    "Select Target Variable",
                    options=df.columns.tolist(),
                    index=len(df.columns)-1
                )
                
                # Show target distribution
                if target_variable:
                    st.write("**Target Distribution:**")
                    target_counts = df[target_variable].value_counts()
                    for value, count in target_counts.items():
                        st.write(f"- {value}: {count} ({count/len(df)*100:.1f}%)")
                
                # Model selection
                st.header("🤖 ML Models")
                models_to_run = st.multiselect(
                    "Select models to train:",
                    ["Logistic Regression", "Random Forest"],
                    default=["Logistic Regression", "Random Forest"]
                )
                
                test_size = st.slider("Test Set Size (%)", 20, 40, 30)
                
                if st.button("🚀 Run Analysis", type="primary"):
                    run_analysis(df, target_variable, models_to_run, test_size/100)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

def run_analysis(df, target_variable, selected_models, test_size):
    """Run the analysis pipeline"""
    
    # Data preparation
    with st.spinner("Preparing data..."):
        df_clean = df.copy()
        
        # Handle missing values
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Prepare features and target
        X = df_clean.drop(columns=[target_variable])
        y = df_clean[target_variable]
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    
    st.success(f"✅ Data prepared: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Results", 
        "📈 Performance", 
        "💡 Insights"
    ])
    
    # TAB 1: DATA OVERVIEW
    with tab1:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", len(X))
        with col2:
            st.metric("Features", X.shape[1])
        with col3:
            st.metric("Training Set", X_train.shape[0])
        with col4:
            st.metric("Test Set", X_test.shape[0])
        
        # Data preview
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.head(8))
        with col2:
            st.subheader("Statistics")
            st.dataframe(df.describe())
        
        # Target distribution
        st.subheader("Target Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Count plot
        pd.Series(y).value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])
        ax1.set_title('Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        
        # Pie chart
        pd.Series(y).value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2, colors=['lightblue', 'lightpink'])
        ax2.set_title('Class Proportions')
        ax2.set_ylabel('')
        
        st.pyplot(fig)
    
    # TAB 2: MODEL RESULTS
    with tab2:
        st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
        
        # Train models
        results = []
        trained_models = {}
        predictions = {}
        
        for model_name in selected_models:
            with st.spinner(f"Training {model_name}..."):
                try:
                    if model_name == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000, random_state=42)
                    elif model_name == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        continue
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    
                    results.append({
                        'Model': model_name,
                        'Accuracy': round(accuracy, 4),
                        'Precision': round(precision, 4),
                        'Recall': round(recall, 4),
                        'F1-Score': round(f1, 4),
                        'ROC-AUC': round(roc_auc, 4)
                    })
                    
                    trained_models[model_name] = model
                    predictions[model_name] = (y_pred, y_pred_proba)
                    
                except Exception as e:
                    st.warning(f"Could not train {model_name}: {str(e)}")
        
        if results:
            results_df = pd.DataFrame(results)
            st.subheader("Performance Metrics")
            st.dataframe(results_df.style.highlight_max(subset=['ROC-AUC', 'Accuracy']))
            
            # Best model
            best_model = results_df.loc[results_df['ROC-AUC'].idxmax()]
            st.success(f"**Best Model:** {best_model['Model']} (AUC: {best_model['ROC-AUC']:.3f})")
        else:
            st.error("No models were successfully trained")
            return
    
    # TAB 3: PERFORMANCE
    with tab3:
        st.markdown('<div class="section-header">Performance Analysis</div>', unsafe_allow_html=True)
        
        if not results:
            st.warning("Train models first in the Model Results tab")
            return
        
        # ROC Curves
        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name, (y_pred, y_pred_proba) in predictions.items():
            RocCurveDisplay.from_predictions(y_test, y_pred_proba, name=model_name, ax=ax)
        
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
        ax.set_title('ROC Curves - Model Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix - Best Model")
        best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
        y_pred_best, _ = predictions[best_model_name]
        
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {best_model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        
        # Feature Importance
        if "Random Forest" in trained_models:
            st.subheader("Feature Importance")
            rf_model = trained_models["Random Forest"]
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
            ax.set_title('Top 10 Feature Importance (Random Forest)')
            st.pyplot(fig)
    
    # TAB 4: INSIGHTS
    with tab4:
        st.markdown('<div class="section-header">Clinical Insights</div>', unsafe_allow_html=True)
        
        if not results:
            st.warning("Train models first in the Model Results tab")
            return
        
        best_auc = results_df['ROC-AUC'].max()
        best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏥 Clinical Applications")
            st.markdown("""
            **Risk Stratification**
            - Identify high-risk patients
            - Prioritize interventions
            - Optimize resources
            
            **Decision Support**
            - Augment clinical judgment
            - Reduce diagnostic variability
            - Improve treatment planning
            """)
            
            st.subheader("📋 Implementation")
            st.markdown("""
            **Phase 1: Validation (2-4 weeks)**
            - Model validation
            - Protocol development
            
            **Phase 2: Pilot (4-8 weeks)**
            - Limited deployment
            - Staff training
            """)
        
        with col2:
            st.subheader("💡 Key Findings")
            st.markdown(f"""
            **Model Performance**
            - Best Algorithm: {best_model_name}
            - Predictive Accuracy: {best_auc:.1%}
            - Clinical Utility: {'Excellent' if best_auc > 0.8 else 'Good'}
            - Patient Cohort: {len(X):,}
            - Features: {X.shape[1]}
            """)
            
            st.subheader("⚖️ Ethics & Safety")
            st.markdown("""
            **Key Principles**
            - AI supports, never replaces clinicians
            - Regular performance monitoring
            - Bias detection protocols
            - HIPAA compliance
            - Patient privacy protection
            """)

if __name__ == "__main__":
    main()
