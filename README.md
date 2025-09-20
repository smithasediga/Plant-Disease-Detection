### 🌿 Plant Disease Detection
**Description:**  
This project is an AI-powered system for identifying plant diseases from leaf images using multiple deep learning models. Instead of relying on a single model, it compares predictions from different architectures, evaluates their performance, and selects the most reliable result. This ensures better accuracy, robustness, and trustworthiness in real-world usage.  

The system integrates models like **Swin Transformer, Vision Transformer, MobileNetV2, and ResNet-50**. Each model contributes to the decision-making process through a **composite scoring mechanism** that balances three factors:  
1. **Confidence** – how sure the model is about its current prediction.  
2. **Consensus** – agreement with other models.  
3. **Training Accuracy** – historical performance during training.  

This approach reduces the risk of misclassification and highlights the **best-performing model** for a given input. A web-based interface allows users to upload leaf images and get instant, consensus-driven predictions.  

**Key Features:**  
- Multi-model prediction system (Swin Transformer, Vision Transformer, MobileNetV2, ResNet-50).  
- Composite score calculation for more reliable results.  
- Identification of best-performing model per image.  
- User-friendly interface for image upload and disease detection.  
- Robust decision-making through consensus and validation.  

**Technologies:** Python, PyTorch, Transformers, Streamlit, PIL  

