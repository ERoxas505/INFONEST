import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article
import io
import nltk
import spacy
import re
#from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import pytextrank
import torch
from googlenewsdecoder import new_decoderv1
import time
from spacy.cli import download
from pathlib import Path

# Add logging setup at the top
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank", last=True)

# Page config
st.set_page_config(page_title='INFONest🇵🇭: Get the News!📰', page_icon='./Meta/newspaper1.ico')

#Import the Youtube summarizer
#from video_summarizer1 import run_youtube_summarizer 
#from video_summarizer1 import display_video_history_in_sidebar

@st.cache_resource
def get_actual_article_link(google_news_url):
    interval_time = 5  # Specify an interval to prevent rate-limiting issues
    try:
        decoded_url = new_decoderv1(google_news_url, interval=interval_time)
        if decoded_url.get("status"):
            return decoded_url["decoded_url"]
        else:
            st.warning("Could not decode URL.")
            return None
    except Exception as e:
        st.error(f"Error occurred while retrieving the article link: {e}")
        return None

# Load essential resources with caching
#@st.cache_resource
#def punkt_load():
    #return nltk.download('punkt')
#punkt = punkt_load()

#@st.cache_resource
#def stopwords_load():
    #nltk.download('stopwords')
    #stop_words = nltk.corpus.stopwords.words("english")
    #stop_words = stop_words + ['hi', 'im', 'hey']
    #return stop_words
#stop_words = stopwords_load()

@st.cache_resource
def punkt_load():
    logger.info("Downloading NLTK punkt data")
    result = nltk.download('punkt')
    logger.info("NLTK punkt data downloaded")
    return result
punkt = punkt_load()

@st.cache_resource
def stopwords_load():
    logger.info("Downloading NLTK stopwords data")
    nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words("english")
    stop_words = stop_words + ['hi', 'im', 'hey']
    logger.info("NLTK stopwords data downloaded")
    return stop_words

stop_words = stopwords_load()

from models import load_bart_model, load_bart_tokenizer

# Load the models with logging
logger.info("Starting to load BART model and tokenizer")
bart_model = load_bart_model()
bart_tokenizer = load_bart_tokenizer()
logger.info("BART model and tokenizer loaded successfully")
#@st.cache_resource
#def bart_tokenizer_load():
    #bart_tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    #bart_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    #bart_tokenizer =AutoTokenizer.from_pretrained("Angel0J/distilbart-multi_news-12-6")
    #return bart_tokenizer


#Load the bart model to GPU
#@st.cache_resource
#def bart_model_load():
    #bart_model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6", use_safetensors= False)
    #bart_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6", use_safetensors= False)
    #bart_model = BartForConditionalGeneration.from_pretrained("Angel0J/distilbart-multi_news-12-6", use_safetensors= True)
    #return bart_model

# Load the models
#loading_message = st.empty()  # Container for the "Now Loading..." message
#progress = st.progress(0)  # Initialize progress bar
#loading_message.markdown("**Now Loading...**", unsafe_allow_html=True)
#with st.empty():  # To prevent the spinner from blocking progress updates
    # Load the BART model
    #progress.progress(50)  # Set progress to 66% after loading BART model
    #bart_model = bart_model_load()

    # Load the BART tokenizer
    #progress.progress(100)  # Set progress to 100% after loading BART tokenizer
    #bart_tokenizer = bart_tokenizer_load()
    
# Once the models are loaded, remove the progress bar
#progress.empty()  # Remove the progress bar from the screen
#loading_message.empty()  # Remove the "Now Loading..." message
#time.sleep(1)

# Fetch news function
@st.cache_resource
def fetch_news_from_rss(url):
    op = urlopen(url)
    rd = op.read()
    op.close()
    return soup(rd, 'xml').find_all('item')

@st.cache_resource
def fetch_news_search_topic(topic):
    site = f'https://news.google.com/news/rss/search/section/q/{topic}?hl=en&gl=PH&ceid=PH%3Aen'
    return fetch_news_from_rss(site)

@st.cache_resource
def fetch_category_news(category):
    site = f'https://news.google.com/news/rss/headlines/section/topic/{category}?hl=en&gl=PH&ceid=PH%3Aen'
    return fetch_news_from_rss(site)

# Utility functions
@st.cache_data(ttl=None, max_entries=80)
def clean_text(text, stop_words):
    cleanT = re.sub(r"(@\[A-Za-z0-9]+)|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    cleanT = re.sub(r'[^a-zA-Z0-9\s.,\r\n-]+', '', cleanT)
    cleanT = re.sub(r'\s+', ' ', cleanT).strip()
    sentences = sent_tokenize(cleanT)
    return ' '.join([w for w in sentences if w.lower() not in stop_words and w])



def fetch_news_poster(poster_link):
    try:
        u = urlopen(poster_link)
        raw_data = u.read()
        image = Image.open(io.BytesIO(raw_data))
    except:
        image = Image.open('./Meta/no_image.jpg')
    st.image(image, use_column_width=True) #st.image(image, use_column_width=True)

@st.cache_data(ttl=None, max_entries=80)
def extract_entities(text):
    if text:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    return []

def prioritize_sentences_with_entities(text, entities, top_n=5):
    # Tokenize sentences
    sentences = sent_tokenize(text)
    
    # Create a dictionary to store scores
    sentence_scores = {}
    
    for sentence in sentences:
        score = 0
        for entity, _ in entities:
            # Boost score for each entity found in the sentence
            if entity in sentence:
                score += 1
        sentence_scores[sentence] = score
    
    # Sort sentences by score
    prioritized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    
    # Return the top N sentences with the highest scores
    return prioritized_sentences[:top_n]

@st.cache_data(ttl=None, max_entries=80)
def enhanced_textrank_summarize(text, num_sentences=5):
    if text:
        # Extract entities
        entities = extract_entities(text)  # No num_sentences argument here
        
        # Prioritize sentences with entities
        prioritized_sentences = prioritize_sentences_with_entities(text, entities, top_n=num_sentences)
        
        # Use TextRank for summarization
        doc = nlp(' '.join(prioritized_sentences))
        summary = ' '.join([str(sent) for sent in doc._.textrank.summary(limit_sentences=num_sentences)])
        return summary if summary else "Summary is Not Available..."
    return "Summary is not Available..."


def count_sentences(text):
    return len(sent_tokenize(text))

@st.cache_data(ttl=None, max_entries=80)
def bart_summarize(_bart_tokenizer, text, _bart_model, num_sentences=5):
    if text:
        # Pre-clean the input text
        text = re.sub(r"[{}:\"']", "", text)  # Remove unnecessary punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces

        inputs = _bart_tokenizer([text], return_tensors='pt', truncation=True, max_length=1024)
        inputs = {k: v.to(next(_bart_model.parameters()).device) for k, v in inputs.items()}
        
        # Define max_length and min_length based on the user-specified number of sentences
        max_length = 30 * num_sentences  # Estimate 50 tokens per sentence
        min_length = 20 * num_sentences  # Estimate 20 tokens per sentence
        
        with torch.no_grad():
            summary_ids = _bart_model.generate(
                inputs['input_ids'],
                max_length=max_length,
                min_length=min_length,
                num_beams=2,
                no_repeat_ngram_size=3
            )
        
        decoded_summary = _bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        final_summary = re.sub(r"([.!?])[^.!?]*$", r"\1", decoded_summary)  # Ensure it ends on a full sentence
        return final_summary
    return "Summary is Not Available..."



# Custom extraction function
def extract_main_content(html):
    soup_obj = soup(html, 'html.parser')
    # Modify the selector based on the actual HTML structure
    main_content = soup_obj.find('div', class_='article-content')  # Adjust this selector
    return main_content.get_text(strip=True) if main_content else ""

    
# Main display function
def display_news(list_of_news, news_quantity, stop_words, bart_tokenizer, bart_model, category):
    # Retrieve the history for the current category
    history = st.session_state["history"].get(category, [])

    # Keep track of titles that are already in the history to avoid duplication
    displayed_titles = set(article["title"] for article in history)

    for c, news in enumerate(list_of_news[:news_quantity], start=1):
        st.write(f'**({c}) {news.title.text}**')

        rss_link = news.link.text
        news_link = get_actual_article_link(rss_link)

        if not news_link:
            st.warning("Could not retrieve the article link.")
            continue

        news_data = Article(news_link)

        try:
            news_data.download()
            news_data.parse()
            raw_text = extract_main_content(news_data.html) or news_data.text
            clean_txt = clean_text(raw_text, stop_words)
        except Exception as e:
            error_message = str(e)
            if "403 Client Error" in error_message:
                st.info(f"Skipping article due to download restriction: {news.title.text}. [Read more at source.]({news_link})")
            #else:
                #st.error(f"Unexpected error fetching article: {e}")
                continue  # Skip to the next article

        fetch_news_poster(news_data.top_image)

        with st.expander(news.title.text):
            num_sentences = 5
            textrank_summary = enhanced_textrank_summarize(clean_txt, num_sentences)
            bart_summary = bart_summarize(bart_tokenizer, textrank_summary, bart_model, num_sentences)

            st.markdown(f'<h6 style="text-align: justify;">{bart_summary}</h6>', unsafe_allow_html=True)
            st.markdown(f"[Read more at source]({news_link})")

            # Check if the article has already been added to the history for this category
            if news.title.text not in displayed_titles:
                # Add the article title and summary to the history
                history.append({"title": news.title.text, "summary": bart_summary})
                displayed_titles.add(news.title.text)  # Update the set to prevent duplication

        st.success(f"Published Date: {news.pubDate.text}")
    # Ensure the history is capped at 10 most recent articles for this category
    st.session_state["history"][category] = history[-10:]  # Keep only the last 10 items


# Function to convert image to base64 format to use in the HTML img tag
def image_to_base64(image):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def run():
    if "history" not in st.session_state:
        st.session_state["history"] = {
            "Top News": [],
            "Hot Topics": [],
            "Search": []
        }
        
    stop_words = stopwords_load()

    st.title("INFONest🇵🇭: Get The News!📰")
    image = Image.open('./Meta/newspaper4.png')
    
    # Use st.empty() and markdown to center the image with a fixed width
    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{}" width="400"/>
        </div>
        """.format(image_to_base64(image)),
        unsafe_allow_html=True,
    )

    # Track category selection in session state
    if "selected_category" not in st.session_state:
        st.session_state["selected_category"] = None


    # Reset session state if category changes
    category = ['--Select--', 'Top News', 'Hot Topics', 'Search']
    cat_op = st.selectbox('Please Select:', category)

    if st.session_state["selected_category"] != cat_op:
        st.session_state["selected_category"] = cat_op


    if cat_op in category[0:3]:  # Show for 'Top News', 'Hot Topics', and 'Search'
        with st.expander("INSTRUCTIONS: How to use INFONest!"):
            st.write("""
                NOTE: Some articles may not be loaded at all as not all websites allow for the scraping of data.
            
                1. Select a category of your choice! (i.e. Top News!, Hot Topics, and Search)
                
                2. If you pick Top News, Hot Topics, or Search, the application will load 5 of the recent and newest articles 
                based on the category chosen. 
                
                3. The articles loaded will have their own summaries. (NOTE: Please wait as it may take time to load the articles and 
                summaries!)  

                4. Use the summaries as overview for what each of the news is about!            
            """) #removed Video News for now
    

    # If category is not selected yet, show a warning
    if cat_op == category[0]:
        st.warning('Please Select a Category!')

    elif cat_op == category[1]:
        with st.expander("PLEASE READ! : What is Top News?"):
            st.write("""
                NOTE: Please wait as the loading of the articles and summaries may take some time!

                - Top News are recent and relevant news about the Philippines gathered from different sources!

                - What it covers will be the recent developments or topics that are currently trending in the country. 

                
             """)
        st.subheader("Here Are the Top News For You!")
        no_of_news = 5  #st.slider('Number of News:', 5, 25, 10)
        news_list = fetch_news_from_rss('https://news.google.com/news/rss?hl=en&gl=PH&ceid=PH%3Aen')
        display_news(news_list, no_of_news, stop_words, bart_tokenizer, bart_model, "Top News")


    elif cat_op == category[2]:
        with st.expander("PLEASE READ! : What is Hot Topics?"):
            st.write("""
                  NOTE: Please wait as the loading of the articles and summaries may take some time!

                - Hot Topics offers a selection of topics from which the user can choose from. These news can have articles 
                from different countries not just the Philippines. 

                - The topics are : WORLD, NATION, BUSINESS, TECHNOLOGY, ENTERTAINMENT, SPORTS, SCIENCE, and HEALTH. This will
                provide news articles that are about what the currently selected topic is along with summaries of each of the article.
                 
             """)
        av_topics = ['--Please Select A Topic!--', 'WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SPORTS', 'SCIENCE', 'HEALTH']
        chosen_topic = st.selectbox("Choose a Topic:", av_topics)
        
        # Initialize news_list to avoid UnboundLocalError
        news_list = []

        if chosen_topic == av_topics[0]:
            st.warning("Please select a valid topic to proceed.")
        else:
            no_of_news = 5 #st.slider('Number of News:', 5, 25, 10)
            news_list = fetch_category_news(chosen_topic)
            
        if news_list:
            st.subheader("Here are the {} News for you!".format(chosen_topic))
            display_news(news_list, no_of_news, stop_words, bart_tokenizer, bart_model, "Hot Topics")
    
        
    elif cat_op == category[3]:
        with st.expander("PLEASE READ!: Instructions for Search"):
            st.write(""" 
                NOTE: Please wait as the loading of the articles and summaries may take some time!

                1. Enter a topic in the search bar below to find news articles.
                2. The app will fetch up to 5 news articles related to your topic.
            """)
        
        # Number of news articles to display
        no_of_news = 5  # Fixed at 5, as in your code

        # Text input for search
        user_topic = st.text_input(
            "Enter a topic to search:",
            placeholder="e.g., Sports, Technology",
            key="search_topic_input"
        )
        
        # Fetch and display news if a topic is provided
        if user_topic:
            # Clean and store the topic in session state
            user_topic = re.sub(r'[^\w\s]', '', user_topic).strip()  # Remove special characters except spaces
            st.session_state["user_topic"] = user_topic
            
            # Fetch news for the topic
            try:
                with st.spinner("Fetching news articles..."):
                    from urllib.parse import quote_plus
                    encoded_topic = quote_plus(user_topic)  # Encode spaces and special characters
                    news_list = fetch_news_search_topic(encoded_topic)
                    st.session_state["search_news_list"] = news_list
            except Exception as e:
                st.error(f"Error fetching news: {e}")
                st.session_state["search_news_list"] = []
            
            # Display news if available
            if st.session_state["search_news_list"]:
                st.subheader(f"Here are some {user_topic.capitalize()} News for you!")
                display_news(
                    st.session_state["search_news_list"],
                    no_of_news,
                    stop_words,
                    bart_tokenizer,
                    bart_model,
                    "Search"
                )
            else:
                st.warning("No news articles found for this topic.")
        else:
            st.warning("Please enter a topic to search.")

    #elif cat_op == category[4]:  # video_summarizer
        #run_youtube_summarizer()  # Call the function from youtube_summarizer.py


    # Inject custom CSS to change the font size of the sidebar header
    st.markdown(
    """
    <style>
    /* Apply to all sidebar headers */
    .sidebar .sidebar-header {
        font-size: 30px !important;  /* Adjust the font size as needed */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    # Sidebar to display history for all categories
    st.sidebar.header("History")
    spacer = st.sidebar.empty()  # Create an empty widget for flexibility
    spacer.text("\n")  # Add line breaks to push content down


    categories = ["Top News", "Hot Topics", "Search"] #removed Video News for now...

    # Create separate expanders for each category
    for cat in categories:
        with st.sidebar.expander(f"{cat} History", expanded=True):
            history_data = st.session_state["history"].get(cat, [])

            if history_data:
                for i, article in enumerate(history_data[-10:], 1):  # Display the last 10
                    st.markdown(f"**{i}. {article['title']}**")
                    st.markdown(f"Summary: {article['summary']}")
                    st.markdown(f"---")
            else:
                st.write("No history available yet.")

    #display_video_history_in_sidebar()


run()

#streamlit run news_summarizer1.py
