import streamlit as st
import base64
from PIL import Image, ExifTags
import io
import time
import pandas as pd
import hashlib
import random
import logging
import html
from transformers import pipeline
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Basic Setup (Logger, Page Config, CSS) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="🌱 Plant Disease Detection", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-card h3 {
        color: #333333;
    }
    .result-card small {
        color: #333333;
        background-color: #EEEEEE;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .consensus-card {
        background: linear-gradient(145deg, #e8f5e8, #c8e6c9);
        border: 3px solid #4CAF50;
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
    }
    .confidence-high { color: #4CAF50; font-weight: bold; }
    .confidence-medium { color: #FF9800; font-weight: bold; }
    .confidence-low { color: #F44336; font-weight: bold; }
    .prediction-item { margin-bottom: 0.5rem; padding-left: 1rem; color: #333333; }
    .healthy { color: #4CAF50; font-weight: bold; font-size: 1.1rem; }
    .diseased { color: #F44336; font-weight: bold; font-size: 1.1rem; }
    .inference-card {
        background: linear-gradient(135deg, #f8f9ff, #e8f4f8);
        border: 2px solid #2196F3;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        color: #333333;
    }
    .insight-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
    }
    .reliability-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .reliability-high { background: #e8f5e8; color: #2e7d32; }
    .reliability-medium { background: #fff3e0; color: #f57c00; }
    .reliability-low { background: #ffebee; color: #c62828; }
    .model-performance-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .performance-metric {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .best-model-card {
        background: linear-gradient(145deg, #fff3e0, #ffe0b2);
        border: 3px solid #FF9800;
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# --- Models & Disease Info ---
MODELS = {
    "Vision Transformer": {
        "id": r"C:\Users\smi67\OneDrive\Desktop\plant-disease-detect\local_models\marwaALzaabi-vit",
        "accuracy": "98.1%",
    },
    "Swin Transformer": {
        "id": r"C:\Users\smi67\OneDrive\Desktop\plant-disease-detect\local_models\gianlab-swin-transformer",
        "accuracy": "98.6%",
    },
    "MobileNetV2": {
        "id": r"C:\Users\smi67\OneDrive\Desktop\plant-disease-detect\local_models\linkanjarad-mobilenet",
        "accuracy": "95.4%",
    },
    "ResNet-50": {
        "id": r"C:\Users\smi67\OneDrive\Desktop\plant-disease-detect\local_models\A2H0H0R1-resnet-50",
        "accuracy": "92.9%",
    }
}

# --- Helper Functions ---
def validate_image(image_file):
    try:
        image = Image.open(image_file)
        if image_file.size > 10 * 1024 * 1024: 
            return False, "Image file too large (max 10MB)"
        if image.width < 50 or image.height < 50: 
            return False, "Image too small (min 50x50 pixels)"
        return True, image
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

@st.cache_resource(ttl=3600)
def get_local_model(model_id):
    """Loads a local transformer model from the specified folder path."""
    logger.info(f"Attempting to load model from: {model_id}")
    try:
        if not os.path.isdir(model_id):
            raise FileNotFoundError(f"Model directory not found at: {model_id}")
        classifier = pipeline("image-classification", model=model_id)
        logger.info(f"Successfully loaded model: {model_id}")
        return classifier
    except Exception as e:
        logger.error(f"Failed to load model from {model_id}: {str(e)}")
        return None

def analyze_image_locally(classifier, image):
    """Analyzes an image and returns the top 3 predictions."""
    if classifier is None:
        return {"error": "Model classifier was not loaded."}
    try:
        predictions = classifier(image, top_k=3)
        return {"success": True, "data": predictions}
    except Exception as e:
        logger.error(f"Error during local analysis: {str(e)}")
        return {"error": f"Local analysis failed: {str(e)}"}

def process_model_response(response, model_name):
    """Processes the list of top 3 predictions from a model."""
    if "error" in response:
        return {"model": model_name, "status": "error", "description": response["error"], "predictions": []}
    try:
        processed_predictions = []
        for pred in response.get("data", []):
            disease_name = str(pred['label']).replace("_", " ").title()
            confidence = round(pred['score'] * 100, 1)
            processed_predictions.append({"disease": disease_name, "confidence": confidence})
        return {"model": model_name, "status": "success", "predictions": processed_predictions}
    except Exception as e:
        return {"model": model_name, "status": "error", "description": f"Failed to process results: {e}", "predictions": []}

def normalize_disease_name(disease_name):
    """Cleans up disease names to group similar predictions together."""
    name = disease_name.lower()
    words_to_remove = ["tomato", "potato", "corn", "maize", "with", "leaf", "bell", "pepper"]
    for word in words_to_remove:
        name = name.replace(word, "")
    return " ".join(name.split())

def get_consensus_analysis(results):
    """Calculates a consensus based on the top prediction of each successful model."""
    valid_results = [r for r in results if r["status"] == "success" and r.get("predictions")]
    if not valid_results:
        return None
    
    disease_votes = {}
    original_names = {}
    
    for result in valid_results:
        top_prediction = result["predictions"][0]
        original_disease = top_prediction["disease"]
        confidence = top_prediction["confidence"]
        normalized_disease = normalize_disease_name(original_disease)
        
        if normalized_disease in disease_votes:
            disease_votes[normalized_disease]["count"] += 1
            disease_votes[normalized_disease]["total_confidence"] += confidence
        else:
            disease_votes[normalized_disease] = {"count": 1, "total_confidence": confidence}
            original_names[normalized_disease] = original_disease
    
    if not disease_votes:
        return None
    
    best_normalized_disease = max(disease_votes, key=lambda d: disease_votes[d]["count"])
    best_display_disease = original_names[best_normalized_disease]
    avg_confidence = disease_votes[best_normalized_disease]["total_confidence"] / disease_votes[best_normalized_disease]["count"]
    
    return {
        "disease": best_display_disease,
        "confidence": round(avg_confidence, 1),
        "votes": disease_votes[best_normalized_disease]["count"],
        "total_models": len(valid_results)
    }

def identify_best_model(results, consensus):
    """Identifies the best performing model and provides clear inference."""
    if not results or not consensus:
        return None
    
    valid_results = [r for r in results if r["status"] == "success" and r.get("predictions")]
    if not valid_results:
        return None
    
    # Score each model based on multiple factors
    model_scores = []
    
    for result in valid_results:
        model_name = result["model"]
        top_prediction = result["predictions"][0]
        confidence = top_prediction["confidence"]
        agrees_with_consensus = normalize_disease_name(top_prediction["disease"]) == normalize_disease_name(consensus["disease"])
        
        # Get model's general accuracy
        general_accuracy = float(MODELS.get(model_name, {}).get("accuracy", "0%").replace("%", ""))
        
        # Calculate composite score
        # - Confidence: 40% weight
        # - Agreement with consensus: 35% weight  
        # - General accuracy: 25% weight
        score = (confidence * 0.4) + (agrees_with_consensus * 35) + (general_accuracy * 0.25)
        
        model_scores.append({
            "model": model_name,
            "confidence": confidence,
            "prediction": top_prediction["disease"],
            "agrees_with_consensus": agrees_with_consensus,
            "general_accuracy": general_accuracy,
            "composite_score": round(score, 1),
            "all_predictions": result["predictions"]
        })
    
    # Find the best model
    best_model = max(model_scores, key=lambda x: x["composite_score"])
    
    # Generate inference
    health_status = "healthy" if "healthy" in best_model["prediction"].lower() else "diseased"
    consensus_match = "✅ Matches" if best_model["agrees_with_consensus"] else "❌ Differs from"
    
    if best_model["confidence"] >= 85:
        confidence_level = "Very High"
        reliability = "Highly Reliable"
    elif best_model["confidence"] >= 70:
        confidence_level = "High" 
        reliability = "Reliable"
    elif best_model["confidence"] >= 55:
        confidence_level = "Moderate"
        reliability = "Moderately Reliable"
    else:
        confidence_level = "Low"
        reliability = "Less Reliable"
    
    inference = f"""
    **Best Performing Model:** {best_model['model']}
    
    **Diagnosis:** The plant appears to be **{health_status}** with '{best_model['prediction']}'.
    
    **Confidence Level:** {confidence_level} ({best_model['confidence']}%)
    
    **Consensus Agreement:** {consensus_match} overall consensus
    
    **Reliability Assessment:** {reliability} - This model achieved the highest composite score of {best_model['composite_score']} 
    based on its prediction confidence, agreement with other models, and general training accuracy.
    
    **Recommendation:** {'This diagnosis appears trustworthy and can guide your next steps.' if best_model['confidence'] >= 70 else 'Consider getting a second opinion or additional testing due to moderate confidence levels.'}
    """
    
    return {
        "best_model": best_model,
        "all_scores": sorted(model_scores, key=lambda x: x["composite_score"], reverse=True),
        "inference": inference
    }

def get_confidence_class(confidence):
    """Helper to get CSS class from confidence score."""
    if confidence >= 80: 
        return "confidence-high"
    elif confidence >= 60: 
        return "confidence-medium"
    else: 
        return "confidence-low"

def display_individual_result_card(result):
    """A dedicated function to display the top 3 predictions in a card."""
    status_icon = "✅" if result.get("status") == "success" else "❌"
    safe_model = html.escape(str(result.get("model", "N/A")))
    general_accuracy = MODELS.get(result.get("model", ""), {}).get("accuracy", "N/A")
    
    st.markdown(f"""
    <div class="result-card">
        <h3>{status_icon} {safe_model}</h3>
        <small>Model's General Training Accuracy: {general_accuracy}</small>
        <hr style="margin: 0.5rem 0;">
    """, unsafe_allow_html=True)
    
    if result.get("status") == "success":
        st.markdown("<h5>Top 3 Predictions:</h5>", unsafe_allow_html=True)
        for i, pred in enumerate(result.get("predictions", [])):
            safe_disease = html.escape(pred['disease'])
            confidence = pred['confidence']
            st.markdown(f"""
            <div class="prediction-item">
                {i+1}. <strong>{safe_disease}:</strong> <span class="{get_confidence_class(confidence)}">{confidence}%</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error(f"Analysis failed: {html.escape(result.get('description', 'Unknown error'))}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def create_performance_charts(best_model_analysis):
    """Create meaningful charts using Streamlit's built-in charting."""
    if not best_model_analysis:
        return None
    
    all_scores = best_model_analysis["all_scores"]
    
    # Prepare data for charts
    chart_data = []
    for score in all_scores:
        model_short_name = score["model"].split("(")[0].strip()
        chart_data.append({
            "Model": model_short_name,
            "Confidence": score["confidence"],
            "Composite Score": score["composite_score"],
            "Training Accuracy": score["general_accuracy"],
            "Agrees with Consensus": "Yes" if score["agrees_with_consensus"] else "No"
        })
    
    df = pd.DataFrame(chart_data)
    return df

def display_best_model_analysis(best_model_analysis):
    """Display the best model analysis with charts."""
    if not best_model_analysis:
        st.error("Could not determine the best performing model.")
        return
    
    best_model = best_model_analysis["best_model"]
    
    # Display Best Model Card
    st.markdown(f"""
    <div class="best-model-card">
        <h2>🏆 Best Performing Model</h2>
        <h3>{html.escape(best_model['model'])}</h3>
        <p><strong>Composite Score:</strong> {best_model['composite_score']}</p>
        <p><strong>Prediction Confidence:</strong> <span class="{get_confidence_class(best_model['confidence'])}">{best_model['confidence']}%</span></p>
        <p><strong>Training Accuracy:</strong> {best_model['general_accuracy']}%</p>
        <p><strong>Consensus Agreement:</strong> {'✅ Yes' if best_model['agrees_with_consensus'] else '❌ No'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create and display charts
    chart_df = create_performance_charts(best_model_analysis)
    
    if chart_df is not None:
        st.markdown("#### 📊 Model Confidence Comparison")
        st.bar_chart(chart_df.set_index("Model")["Confidence"])
        
        # Display data table
        st.markdown("#### 📈 Detailed Performance Metrics")
        st.dataframe(chart_df, use_container_width=True)
    
    # Display Model Rankings
    st.markdown("### 🏆 Model Performance Rankings")
    
    cols = st.columns(2)
    for i, model_score in enumerate(best_model_analysis["all_scores"]):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "4️⃣"
        agreement_icon = "✅" if model_score["agrees_with_consensus"] else "❌"
        
        with cols[i % 2]:
            st.markdown(f"""
            <div class="performance-metric">
                <h4 style="color: #333333;">{rank_emoji} {html.escape(model_score['model'])}</h4>
                <p style="color: #333333;"><strong>Composite Score:</strong> {model_score['composite_score']}</p>
                <p style="color: #333333;"><strong>Confidence:</strong> <span class="{get_confidence_class(model_score['confidence'])}">{model_score['confidence']}%</span></p>
                <p style="color: #333333;"><strong>Consensus Agreement:</strong> {agreement_icon}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display Final Inference
    st.markdown("### 🎯 Final Analysis & Inference")
    
    # Split the inference text by lines and display properly
    inference_lines = best_model_analysis['inference'].strip().split('\n')
    
    inference_content = ""
    for line in inference_lines:
        if line.strip():  # Skip empty lines
            if line.strip().startswith('**') and line.strip().endswith('**'):
                # Handle bold headers
                inference_content += f"<p><strong>{html.escape(line.strip()[2:-2])}</strong></p>"
            else:
                # Handle regular text
                inference_content += f"<p>{html.escape(line.strip())}</p>"
    
    st.markdown(f"""
    <div class="inference-card">
        <h2>🎯 Final Analysis & Inference</h2>
        <div style="line-height: 1.6; font-size: 1.1rem;">
            {inference_content}
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Main Application Logic ---
def main():
    st.markdown('<h1 class="main-header">🌱 Plant Disease Analysis Platform</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Models In Use")
        for name, info in MODELS.items():
            st.markdown(f"- **{name}** ({info['accuracy']})")
        st.markdown("---")
        st.header("📊 Model Status")
        st.info("Checks if local model folders are set up.")
        for model_name, model_info in MODELS.items():
            path = model_info["id"]
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
                st.success(f"✅ {model_name} found.")
            else:
                st.error(f"❌ {model_name} not found.")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📸 Upload Leaf Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            is_valid, result_or_image = validate_image(uploaded_file)
            if not is_valid:
                st.error(f"❌ {result_or_image}")
                st.stop()
            image = result_or_image
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)
    
    with col2:
        if 'image' in locals():
            st.header("🔍 Analysis Results")
            if st.button("🚀 Analyze Plant Disease", type="primary", use_container_width=True):
                processed_image = preprocess_image(image)
                results = []
                
                for model_name, model_info in MODELS.items():
                    with st.spinner(f"Analyzing with {model_name}..."):
                        classifier = get_local_model(model_info["id"])
                        if classifier is None:
                            analysis_result = {
                                "model": model_name, 
                                "status": "error", 
                                "description": f"Failed to load model from path.", 
                                "predictions": []
                            }
                        else:
                            response = analyze_image_locally(classifier, processed_image)
                            analysis_result = process_model_response(response, model_name)
                        results.append(analysis_result)
                
                st.success("✅ Analysis Complete!")
                
                # Display consensus
                consensus = get_consensus_analysis(results)
                if consensus:
                    st.markdown(f"""
                    <div class="consensus-card">
                        <h2>🎯 Consensus Diagnosis</h2>
                        <h3 class="{'healthy' if 'healthy' in consensus['disease'].lower() else 'diseased'}">{html.escape(consensus['disease'])}</h3>
                        <p>Based on the top prediction from <strong>{consensus['votes']} out of {consensus['total_models']}</strong> successful models.</p>
                        <p><strong>Average Confidence:</strong> <span class="{get_confidence_class(consensus['confidence'])}">{consensus['confidence']}%</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.header("🔬 Individual Model Results")
                for result in results:
                    display_individual_result_card(result)
                
                # --- BEST MODEL ANALYSIS & INFERENCE ---
                st.markdown("---")
                try:
                    best_model_analysis = identify_best_model(results, consensus)
                    if best_model_analysis:
                        display_best_model_analysis(best_model_analysis)
                    else:
                        st.warning("Could not perform best model analysis due to insufficient data.")
                except Exception as e:
                    st.error(f"Error in best model analysis: {str(e)}")
                    logger.error(f"Best model analysis error: {str(e)}")
        else:
            st.info("👆 Upload a plant image on the left and click 'Analyze' to begin.")

# Run the application
if __name__ == "__main__":
    main()