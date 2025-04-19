import streamlit as st
from transformers.models.bart import BartForConditionalGeneration, BartTokenizer
import os

@st.cache_resource
def load_bart_model():
    token = os.getenv("HF_TOKEN")
    model = BartForConditionalGeneration.from_pretrained(
        "sshleifer/distilbart-cnn-12-6",
        use_safetensors=False,
        use_auth_token=token
    ).to("cpu")
    return model

@st.cache_resource
def load_bart_tokenizer():
    token = os.getenv("HF_TOKEN")
    tokenizer = BartTokenizer.from_pretrained(
        "sshleifer/distilbart-cnn-12-6",
        use_auth_token=token
    )
    return tokenizer




#ORIGINAL CODE:

#import streamlit as st
#from transformers import BartForConditionalGeneration, BartTokenizer
#from transformers.models.bart import BartForConditionalGeneration, BartTokenizer

#@st.cache_resource
#def load_bart_model():
    #model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6", use_safetensors=False).to("cpu")
    #return model
#@st.cache_resource
#def load_bart_tokenizer():
    #tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    #return tokenizer