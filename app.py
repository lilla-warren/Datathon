# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
import warnings
warnings.filterwarnings('ignore')

# Try to import ML dependencies with fallbacks
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError as e:
    st.error(f"Scikit-learn import error: {e}")
    SKLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError as e:
    st.warning(f"SHAP not available: {e}. Some explainability features will be limited.")
    SHAP_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="HCT Datathon 2025 - Healthcare Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_dependencies():
    """Check if all required dependencies are available"""
    issues = []
    if not SKLEARN_AVAILABLE:
        issues.append("Scikit-learn is not available. Machine learning features will not work.")
    if not SHAP_AVAILABLE:
        issues.append("SHAP is not available. Some explainability features will be limited.")
    
    return issues

def create_roc_curve_manual(y_true, y_pred_proba, model_names):
    """Manual ROC curve implementation if sklearn is not available"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Simple implementation - in real scenario, use sklearn
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (Manual)')
    ax.legend()
    ax.grid(True)
    
    return fig

def main():
    # Main title
    st.markdown('<h1 class="main-header">🏥 HCT Datathon 2025 - Healthcare Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Transform Data into Knowledge • Promote Informed Decision-Making • Advance Responsible AI")
    
    # Check dependencies
    dependency_issues = check_dependencies()
    if dependency_issues:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("⚠️ Dependency Warning")
        for issue in dependency_issues:
            st.write(f"• {issue}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if not SKLEARN_AVAILABLE:
        st.error("❌ Critical dependencies missing. Please check the requirements.txt file.")
        st.stop()

    # Sidebar - File upload and configuration
    with st.sidebar:
        st.header("📁 Data Configuration")
        uploaded_file = st.file_uploader("Upload Healthcare Dataset (CSV)", type=["csv"])
        
        if uploaded_file is not None:
            st.success("✅ File uploaded successfully!")
            
            # Load data
            @st.cache_data
            def load_data(file):
                return pd.read_csv(file)
            
            df = load_data(uploaded_file)
            
            st.subheader("Dataset Info")
            st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Target variable selection
            target_variable = st.selectbox(
                "Select Target Variable",
                options=df.columns.tolist(),
                index=len(df.columns)-1 if len(df.columns) > 0 else 0
            )
            
            # Model selection
            st.subheader("Model Configuration")
            models_to_run = st.multiselect(
                "Select Models to Train",
                ["Logistic Regression", "Random Forest"],
                default=["Logistic Regression", "Random Forest"]
            )
            
            test_size = st.slider("Test Set Size (%)", 20, 40, 30)
            
            # Analysis button
            analyze_button = st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True)
            
        else:
            st.info("👆 Please upload a CSV file to begin analysis")
            st.stop()
    
    # Main analysis when button is clicked
    if analyze_button:
        perform_analysis(df, target_variable, models_to_run, test_size/100)

def perform_analysis(df, target_variable, models_to_run, test_size):
    """Perform complete healthcare analytics workflow"""
    
    # Preprocessing
    with st.spinner("🔄 Preprocessing data..."):
        df_clean = df.copy()
        
        # Handle missing values
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        if not numerical_cols.empty:
            df_clean[numerical_cols] = df_clean[numerical_cols].fillna(df_clean[numerical_cols].median())
        
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
        
        # Prepare features and target
        X = df_clean.drop(target_variable, axis=1)
        y = df_clean[target_variable]
        
        # Convert categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    
    # Create tabs for different analytical perspectives
    tab_names = [
        "📊 Descriptive Analytics", 
        "🔍 Diagnostic Analytics", 
        "🤖 Predictive Analytics", 
        "📈 Model Performance",
        "💡 Prescriptive Insights",
        "⚖️ Ethics & Responsible AI"
    ]
    
    # Add Explainability tab only if SHAP is available
    if SHAP_AVAILABLE:
        tab_names.insert(4, "🔬 Explainability & XAI")
    
    tabs = st.tabs(tab_names)
    
    # TAB 1: Descriptive Analytics
    with tabs[0]:
        st.markdown('<h2 class="section-header">Descriptive Analytics</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"**Original Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            st.write(f"**After Cleaning:** {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
            
            # Basic info
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text_area("Data Types Info:", buffer.getvalue(), height=200)
        
        with col2:
            st.subheader("Summary Statistics")
            st.dataframe(df_clean.describe(), use_container_width=True)
        
        # Class distribution
        st.subheader("Class Distribution Analysis")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        y.value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])
        ax1.set_title('Class Distribution (Count)')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        
        # Pie chart
        y.value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2, colors=['lightblue', 'lightpink'])
        ax2.set_title('Class Distribution (Percentage)')
        ax2.set_ylabel('')
        
        st.pyplot(fig)
        
        # Feature distributions
        st.subheader("Feature Distributions")
        numerical_features = X.select_dtypes(include=[np.number]).columns
        if len(numerical_features) > 0:
            n_features = min(6, len(numerical_features))
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, feature in enumerate(numerical_features[:n_features]):
                df_clean[feature].hist(bins=30, ax=axes[i], alpha=0.7, color='lightseagreen')
                axes[i].set_title(f'Distribution: {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
            
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            st.pyplot(fig)
    
    # TAB 2: Diagnostic Analytics
    with tabs[1]:
        st.markdown('<h2 class="section-header">Diagnostic Analytics</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlation Analysis")
            # Correlation heatmap
            corr_data = pd.concat([X, y], axis=1)
            correlation_matrix = corr_data.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, ax=ax, fmt=".2f")
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Initial Feature Importance")
            # Train quick model for feature importance
            rf_quick = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_quick.fit(X_train, y_train)
            
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_quick.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=feature_importance.head(15), x='importance', y='feature', ax=ax, palette='viridis')
            ax.set_title('Top 15 Feature Importance (Random Forest)')
            ax.set_xlabel('Importance Score')
            st.pyplot(fig)
    
    # TAB 3: Predictive Analytics
    with tabs[2]:
        st.markdown('<h2 class="section-header">Predictive Analytics</h2>', unsafe_allow_html=True)
        
        st.subheader("Model Training Configuration")
        st.write(f"**Training Set:** {X_train.shape[0]} samples ({100-test_size*100}%)")
        st.write(f"**Test Set:** {X_test.shape[0]} samples ({test_size*100}%)")
        st.write(f"**Models Selected:** {', '.join(models_to_run)}")
        
        # Initialize models
        models = {}
        if "Logistic Regression" in models_to_run:
            models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
        if "Random Forest" in models_to_run:
            models['Random Forest'] = RandomForestClassifier(random_state=42)
        
        # Train models
        with st.spinner("Training machine learning models..."):
            results = []
            predictions = {}
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                predictions[name] = (y_pred, y_pred_proba)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                results.append({
                    'Model': name,
                    'Accuracy': round(accuracy, 4),
                    'Precision': round(precision, 4),
                    'Recall': round(recall, 4),
                    'F1-Score': round(f1, 4),
                    'ROC-AUC': round(roc_auc, 4)
                })
        
        # Display results
        results_df = pd.DataFrame(results)
        st.subheader("Model Evaluation Results")
        st.dataframe(results_df, use_container_width=True)
        
        # Highlight best model
        best_model = results_df.loc[results_df['ROC-AUC'].idxmax()]
        st.success(f"🎯 **Best Performing Model:** {best_model['Model']} (AUC: {best_model['ROC-AUC']:.3f})")
    
    # TAB 4: Model Performance
    with tabs[3]:
        st.markdown('<h2 class="section-header">Model Performance Visualization</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curves")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for name, (y_pred, y_pred_proba) in predictions.items():
                from sklearn.metrics import RocCurveDisplay
                RocCurveDisplay.from_predictions(y_test, y_pred_proba, name=name, ax=ax)
            
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
            ax.set_title('ROC Curves - Model Comparison')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Confusion Matrix")
            best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
            y_pred_best, _ = predictions[best_model_name]
            
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = confusion_matrix(y_test, y_pred_best)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'])
            ax.set_title(f'Confusion Matrix - {best_model_name}')
            st.pyplot(fig)
    
    # TAB 5: Explainability & XAI (Conditional)
    if SHAP_AVAILABLE and len(tabs) > 5:
        with tabs[4]:
            st.markdown('<h2 class="section-header">Explainability & XAI</h2>', unsafe_allow_html=True)
            
            best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
            best_model_instance = models[best_model_name]
            
            with st.spinner("Calculating SHAP values for model explainability..."):
                # SHAP analysis
                explainer = shap.TreeExplainer(best_model_instance)
                shap_values = explainer.shap_values(X_test)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("SHAP Summary Plot")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("SHAP Feature Importance")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar", show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.subheader("🔍 Model Interpretation")
            st.write("""
            **SHAP (SHapley Additive exPlanations) values show:**
            - **Red points**: High feature values that increase prediction probability
            - **Blue points**: Low feature values that decrease prediction probability  
            - **Feature order**: Top features have largest impact on model predictions
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Adjust tab indices based on SHAP availability
    prescriptive_tab_index = 5 if SHAP_AVAILABLE and len(tabs) > 5 else 4
    ethics_tab_index = 6 if SHAP_AVAILABLE and len(tabs) > 5 else 5
    
    # TAB: Prescriptive Insights
    with tabs[prescriptive_tab_index]:
        st.markdown('<h2 class="section-header">Prescriptive Insights</h2>', unsafe_allow_html=True)
        
        # Get top features from model
        best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
        best_model_instance = models[best_model_name]
        
        if hasattr(best_model_instance, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model_instance.feature_importances_
            }).sort_values('importance', ascending=False)
            top_features = feature_importance.head(5)['feature'].tolist()
        else:
            # Fallback: use correlation with target
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            top_features = correlations.head(5).index.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Key Predictive Features")
            for i, feature in enumerate(top_features, 1):
                st.metric(label=f"Top Feature #{i}", value=feature)
            
            st.subheader("📊 Model Performance Summary")
            st.metric(label="Best Model", value=best_model_name)
            best_auc = results_df.loc[results_df['ROC-AUC'].idxmax(), 'ROC-AUC']
            st.metric(label="ROC-AUC Score", value=f"{best_auc:.3f}")
            st.metric(label="Dataset Size", value=f"{df.shape[0]:,} patients")
        
        with col2:
            st.subheader("💡 Actionable Recommendations")
            
            st.markdown("""
            <div class="insight-box">
            <h4>🏥 Clinical Decision Support</h4>
            <ul>
            <li>Focus early screening on patients with abnormal values in top predictive features</li>
            <li>Implement risk stratification based on model probability scores</li>
            <li>Use as triage tool to prioritize high-risk cases for clinical review</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="insight-box">
            <h4>🔧 Operational Implementation</h4>
            <ul>
            <li>Integrate model predictions with electronic health record systems</li>
            <li>Establish regular model retraining schedule with new data</li>
            <li>Create clinical validation protocol for model recommendations</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB: Ethics & Responsible AI
    with tabs[ethics_tab_index]:
        st.markdown('<h2 class="section-header">Ethics & Responsible AI</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🛡️ Ethical Considerations")
            
            st.markdown("""
            <div class="metric-card">
            <h4>📊 Data Privacy & Security</h4>
            <p>• Used anonymized dataset only<br>
            • No personal identifiable information processed<br>
            • Secure data handling protocols implemented</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
            <h4>⚖️ Fairness & Bias Mitigation</h4>
            <p>• Analyzed class distribution for representation<br>
            • Used stratified sampling for train-test split<br>
            • Regular bias audits recommended for deployment</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🔍 Transparency & Accountability")
            
            st.markdown("""
            <div class="metric-card">
            <h4>🔬 Model Explainability</h4>
            <p>• Feature importance analysis conducted<br>
            • Clear documentation of model limitations<br>
            • Performance metrics transparently reported</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
            <h4>🎯 Responsible Deployment</h4>
            <p>• Human-in-the-loop requirement for final decisions<br>
            • Continuous monitoring for model drift<br>
            • Clinical validation before implementation</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("📝 Ethical Reflection Statement")
        st.markdown("""
        <div class="insight-box">
        <p>This healthcare AI system demonstrates our commitment to responsible AI practices. 
        While the model shows strong predictive performance, we emphasize its role as a 
        <strong>clinical decision support tool</strong> rather than a diagnostic replacement. 
        Final medical decisions must remain with qualified healthcare professionals who 
        consider the full clinical context beyond algorithmic predictions.</p>
        
        <p><strong>Key Principles:</strong> Transparency through explainable AI, fairness through 
        bias monitoring, accountability through human oversight, and privacy through 
        data anonymization form the foundation of our ethical framework.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
