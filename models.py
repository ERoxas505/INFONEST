import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer

@st.cache_resource
def load_bart_model():
    model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6", use_safetensors=False).to("cpu")
    return model

@st.cache_resource
def load_bart_tokenizer():
    tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    return tokenizer