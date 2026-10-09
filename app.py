"""Streamlit app for BANKING77 intent prediction."""

import os
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path(__file__).parent / "artifacts" / "banking77-distilbert"
DEFAULT_MODEL_ID = "mrm8488/distilroberta-finetuned-banking77"
MODEL_SOURCE = os.getenv("BANKING77_MODEL_ID") or (
    str(MODEL_DIR) if MODEL_DIR.is_dir() else DEFAULT_MODEL_ID
)
EXAMPLE = "I tried to withdraw ₹10,000, but the ATM didn't give me the cash."

st.set_page_config(page_title="Banking Intent Classifier", page_icon="B", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f4f6f2; color: #17211e; }
    .block-container { max-width: 1080px; padding-top: 2.2rem; }
    .eyebrow { color: #187b62; font-size: .75rem; font-weight: 800; letter-spacing: .12em; }
    .headline { font-size: 2.35rem; line-height: 1.1; font-weight: 750; color: #17211e; margin: .35rem 0; }
    .subhead { color: #58645f; max-width: 690px; font-size: 1.05rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 8px; }
    div.stButton > button[kind="primary"] { background: #187b62; border-color: #187b62; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">BANKING77 · CUSTOMER SUPPORT</div>', unsafe_allow_html=True)
st.markdown('<div class="headline">Banking intent classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subhead">Classify a customer message into one of 77 banking support intents.</div>',
    unsafe_allow_html=True,
)
st.divider()


@st.cache_resource
def load_model(model_source):
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


try:
    with st.spinner(f"Loading intent model from `{MODEL_SOURCE}`..."):
        tokenizer, model, device = load_model(MODEL_SOURCE)
except (OSError, ValueError) as error:
    st.error(f"Could not load the intent model from `{MODEL_SOURCE}`: {error}")
    st.stop()

left, right = st.columns([1.2, 0.8], gap="large")
with left:
    st.subheader("Customer message")
    message = st.text_area(
        "Message",
        value=EXAMPLE,
        height=180,
        label_visibility="collapsed",
        placeholder="Paste a banking customer message...",
    )
    predict = st.button("Predict intent", type="primary", use_container_width=True)

with right:
    st.subheader("Prediction")
    if predict:
        if not message.strip():
            st.info("Enter a customer message to get a prediction.")
        else:
            encoded = tokenizer(
                message, return_tensors="pt", truncation=True, max_length=128
            ).to(device)
            with torch.inference_mode():
                probabilities = torch.softmax(model(**encoded).logits[0], dim=-1)
            scores, indices = probabilities.topk(min(3, probabilities.numel()))
            best_id = indices[0].item()
            best_label = model.config.id2label.get(
                best_id, model.config.id2label.get(str(best_id), str(best_id))
            )
            st.metric("Most likely intent", best_label.replace("_", " ").title(),
                      f"{scores[0].item():.1%} confidence")
            st.caption("Other likely intents")
            for score, index in zip(scores[1:], indices[1:]):
                label = model.config.id2label.get(
                    index.item(), model.config.id2label.get(str(index.item()), str(index.item()))
                )
                st.write(f"**{label.replace('_', ' ').title()}** · {score.item():.1%}")
    else:
        st.info("Your prediction and the next closest intents will appear here.")

st.caption(f"Model: DistilBERT fine-tuned on BANKING77 · running on {device.type.upper()}")