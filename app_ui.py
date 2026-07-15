#for ui
!pip install -q gradio
import time
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. LOAD FINE-TUNED MODEL & TOKENIZER
# ==========================================
MODEL_PATH = "./best_model"
ui_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
ui_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
ui_model.eval()

# ==========================================
# 2. DEFINE THE ENHANCED PREDICTION FUNCTION
# ==========================================
def classify_ticket_enhanced(ticket_text):
    if not ticket_text.strip():
        return (
            {}, 
            "⚠️ Please enter a ticket description", 
            "0.0%", 
            "0.0 ms"
        )
    
    start_time = time.time()
    
    # Tokenize input (max 256 tokens as per model constraints)
    inputs = ui_tokenizer(
        ticket_text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=256
    )
    
    # Run model inference
    with torch.no_grad():
        outputs = ui_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    # Calculate inference latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Build dictionary of all class probabilities for the visualization plot
    label_probs = {}
    for idx, prob in enumerate(probabilities):
        label = ui_model.config.id2label[idx]
        label_probs[label] = float(prob.item())
        
    # Get the top predicted class and confidence
    top_class_idx = torch.argmax(probabilities).item()
    top_label = ui_model.config.id2label[top_class_idx]
    top_confidence = probabilities[top_class_idx].item() * 100
    
    # Format outputs for the dashboard metrics
    formatted_category = f"🎯 {top_label}"
    formatted_confidence = f"{top_confidence:.1f}%"
    formatted_latency = f"{latency_ms:.1f} ms"
    
    return label_probs, formatted_category, formatted_confidence, formatted_latency


# ==========================================
# 3. BUILD THE GRADIO DASHBOARD (UI)
# ==========================================
custom_css = """
.container { max-width: 1100px; margin: 0 auto; padding-top: 20px; }
.header { text-align: center; margin-bottom: 25px; padding: 20px; border-radius: 12px; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), css=custom_css) as demo:
    
    # Header Banner
    with gr.Row(elem_classes="header"):
        with gr.Column():
            gr.Markdown(
                """
                # 🛠️ AI Customer Support Ticket Classifier
                ### Powered by Fine-Tuned DistilBERT | Instant Multi-Class Classification
                """
            )

    # Main Dashboard Body
    with gr.Row():
        
        # LEFT COLUMN: User Input Area
        with gr.Column(scale=1):
            gr.Markdown("### **1. Submit Ticket Description**")
            gr.Markdown("ℹ️ *Model processes up to 256 tokens dynamically.*")
            
            input_text = gr.Textbox(
                lines=6,
                placeholder="Enter the full customer support ticket text here... (e.g., product setup issues, billing questions, cancellation requests)",
                label="Ticket Description",
                show_label=False
            )
            
            submit_btn = gr.Button("🚀 Classify Ticket", variant="primary")
            
            # Interactive Examples Helper
            gr.Examples(
                examples=[
                    ["My GoPro Hero screen is completely cracked and the device won't boot up. I tried following the troubleshooting manual but the issue persists."],
                    ["I received an accidental charge of $49.99 on my credit card this morning for a subscription I canceled last week. I need a chargeback or refund immediately."],
                    ["How do I sync my new LG Smart TV with my peripheral Bluetooth speakers? The user instructions don't mention compatibility setup steps."]
                ],
                inputs=input_text,
                label="Quick-Load Sample Tickets"
            )

        # RIGHT COLUMN: Model Insights & Results
        with gr.Column(scale=1):
            gr.Markdown("### **2. AI Prediction & Insights**")
            
            # Probabilities distribution chart
            output_chart = gr.Label(num_top_classes=5, label="Category Probability Distribution")
            
            # Native UI input boxes for metrics to prevent white-on-white text rendering issues
            metric_category = gr.Textbox(
                label="Final Predicted Category", 
                value="Awaiting input...", 
                interactive=False
            )
            
            with gr.Row():
                metric_confidence = gr.Textbox(
                    label="Model Confidence", 
                    value="-- %", 
                    interactive=False
                )
                metric_latency = gr.Textbox(
                    label="Inference Latency", 
                    value="-- ms", 
                    interactive=False
                )
                    
            # Explanatory Info Card
            with gr.Accordion("About the Model", open=False):
                gr.Markdown(
                    """
                    * **Base Architecture:** `distilbert-base-uncased`
                    * **Classes Evaluated:** 5 Support Categories (*Refund request, Technical issue, Cancellation request, Product inquiry, Billing inquiry*)
                    * **Under the Hood:** Tokenizes sequence strings, runs a forward pass on PyTorch weights, and calculates softmax confidence scores dynamically.
                    """
                )

    # Button Event Handlers
    submit_btn.click(
        fn=classify_ticket_enhanced,
        inputs=input_text,
        outputs=[output_chart, metric_category, metric_confidence, metric_latency]
    )
    
    # Real-time update as you press enter in the textbox
    input_text.submit(
        fn=classify_ticket_enhanced,
        inputs=input_text,
        outputs=[output_chart, metric_category, metric_confidence, metric_latency]
    )

# Launch the UI
demo.launch(inline=True, share=True)