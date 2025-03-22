import os
import json
import glob
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import re
import random
import math

def extract_relevant_segment(text, category, keywords):
    """Extract the most relevant segment of text for a given category"""
    if not isinstance(text, str):
        return ""
    text_lower = text.lower()
    sentences = text.split('. ')
    relevant_sentences = []
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in keywords):
            relevant_sentences.append(sentence)
    if relevant_sentences:
        return '. '.join(relevant_sentences[:2]) + '.'  # 2 sentences max
    return '. '.join(sentences[:2]) + '.' if sentences else text

def generate_sentiment_summary(text, sentiment_score, category):
    """Generate a one-sentence summary of the sentiment and main point"""
    if not isinstance(text, str) or not text.strip():
        return "No summary available."
    
    sentiment_type = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
    
    # hard code for now
    summaries = {
        "Code Completion": {
            "positive": "User appreciates the quality and accuracy of code suggestions.",
            "negative": "User is frustrated with inaccurate or unhelpful code completions.",
            "neutral": "User has mixed feelings about the code completion capabilities."
        },
        "Code Quality": {
            "positive": "User finds the generated code clean and well-structured.",
            "negative": "User reports issues with code quality and correctness.",
            "neutral": "User has neutral opinions about the code quality."
        },
        "Performance Speed": {
            "positive": "User is impressed with the tool's speed and responsiveness.",
            "negative": "User experiences slowness or lag issues with the tool.",
            "neutral": "User finds the performance acceptable but not remarkable."
        },
        "Integration": {
            "positive": "User appreciates how well the tool integrates with their workflow.",
            "negative": "User struggles with integration or compatibility issues.",
            "neutral": "User has no strong feelings about the integration capabilities."
        },
        "Learning Curve": {
            "positive": "User finds the tool intuitive and easy to learn.",
            "negative": "User finds the tool difficult to understand or use.",
            "neutral": "User has a balanced view on the learning requirements."
        },
        "Context Understanding": {
            "positive": "User is impressed with how well the tool understands context.",
            "negative": "User finds the tool lacks proper context understanding.",
            "neutral": "User has mixed experiences with the context awareness."
        },
        "Customization": {
            "positive": "User appreciates the flexibility and customization options.",
            "negative": "User feels limited by lack of customization options.",
            "neutral": "User has neutral opinions about customization capabilities."
        },
        "Reliability": {
            "positive": "User finds the tool stable and dependable.",
            "negative": "User experiences crashes, bugs, or inconsistent behavior.",
            "neutral": "User has mixed experiences with reliability."
        },
        "Documentation": {
            "positive": "User appreciates the helpful documentation and examples.",
            "negative": "User finds documentation lacking or unclear.",
            "neutral": "User has neutral views on the documentation."
        },
        "Value": {
            "positive": "User feels they're getting good value for the cost.",
            "negative": "User questions whether the tool is worth its price.",
            "neutral": "User has balanced views on the value proposition."
        }
    }
    
    # use category specific sumary if u can
    if category in summaries and sentiment_type in summaries[category]:
        return summaries[category][sentiment_type]
    generic_summaries = {
        "positive": f"User has positive experiences with the tool's {category.lower()} capabilities.",
        "negative": f"User reports issues with the tool's {category.lower()} functionality.",
        "neutral": f"User expresses mixed opinions about the tool's {category.lower()}."
    }
    return generic_summaries[sentiment_type]


def render_competitor_section(df):
    """Render a tabbed section with intelligence on multiple competitors"""
    st.markdown("## Competitor Intelligence")
    # limit to 5 for now because of rate limits; picked according to the growth team
    competitors = ["GitHub Copilot", "Cursor", "Lovable", "Devin", "Sourcegraph Cody"]
    tabs = st.tabs(competitors)
    competitor_map = {
        "GitHub Copilot": {"prefix": "copilot", "api_name": "copilot"},
        "Cursor": {"prefix": "cursor", "api_name": "cursor"},
        "Lovable": {"prefix": "lovable", "api_name": "lovable"},
        "Devin": {"prefix": "devin", "api_name": "devin"},
        "Sourcegraph Cody": {"prefix": "cody", "api_name": "cody"}
    }
    for i, competitor in enumerate(competitors):
        with tabs[i]:
            competitor_data = load_data(competitor=competitor_map[competitor]["api_name"])
            if competitor_data is None or len(competitor_data) == 0:
                st.info(f"No data available for {competitor}. Run data collection scripts first.")
                continue
            avg_sentiment = competitor_data['sentiment_score'].mean()
            sentiment_score = int((avg_sentiment + 1) * 50)  
            st.markdown(f"### Sentiment Score: {sentiment_score}/100")
            sentiment_color = "green" if sentiment_score >= 70 else "orange" if sentiment_score >= 40 else "red"
            st.markdown(f"""
            <div style="background: linear-gradient(to right, #f0f0f0, #f0f0f0); height: 20px; border-radius: 10px; margin-bottom: 10px;">
                <div style="background: linear-gradient(to right, {sentiment_color}, {sentiment_color}); width: {sentiment_score}%; height: 20px; border-radius: 10px;"></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**{len(competitor_data)} recent mentions**")
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"### How Users Describe {competitor}")
                def extract_descriptive_words(texts):
                    descriptors = []
                    # hard code for now
                    positive_words = [
                        "amazing", "excellent", "great", "good", "impressive", "powerful", 
                        "smart", "helpful", "intuitive", "fast", "reliable", "better", "best",
                        "love", "awesome", "fantastic", "wonderful", "superb", "brilliant",
                        "quality", "efficient", "accurate", "precise", "clean", "consistent", 
                        "seamless", "innovative", "nice", "perfect", "easy", "convenient", 
                        "superior", "fun", "useful", "favorite", "recommended", "smooth"
                    ]
                    negative_words = [
                        "slow", "buggy", "limited", "expensive", "frustrating", "confusing", 
                        "disappointing", "unreliable", "broken", "clunky", "terrible", "worst",
                        "bad", "poor", "awful", "annoying", "useless", "overpriced", "glitchy",
                        "difficult", "mediocre", "insufficient", "lacking", "problematic", "costly",
                        "cumbersome", "complicated", "inconsistent", "inadequate", "unstable",
                        "error", "crash", "fail", "fails", "failed", "waste", "horrible", "trouble"
                    ]
                    neutral_words = [
                        "new", "different", "unique", "interesting", "complex", "simple", 
                        "basic", "standard", "common", "specialized", "normal", "typical",
                        "average", "alternative", "familiar", "surprising", "unusual", "fresh",
                        "competitive", "expected", "unexpected", "particular", "specific",
                        "notable", "comparable", "distinctive", "similar", "okay", "decent"
                    ]
                    all_descriptive_words = positive_words + negative_words + neutral_words
                    for text in texts:
                        if not isinstance(text, str):
                            continue
                        text = text.lower()
                        words = re.findall(r'\b\w+\b', text)
                        for word in words:
                            if len(word) >= 3 and word in all_descriptive_words:  
                                sentiment_type = "positive" if word in positive_words else "negative" if word in negative_words else "neutral"
                                descriptors.append((word, sentiment_type))
                    word_counts = {}
                    for word, sentiment in descriptors:
                        if word not in word_counts:
                            word_counts[word] = {"count": 0, "sentiment": sentiment}
                        word_counts[word]["count"] += 1

                    result = [(word, data["sentiment"], data["count"]) 
                             for word, data in word_counts.items()]
                    result.sort(key=lambda x: x[2], reverse=True)
                    
                    return result[:40] 
                
                texts = competitor_data['text'].tolist()
                descriptors = extract_descriptive_words(texts)
                
                if descriptors:
                    html_words = []
                    for word, sentiment, count in descriptors:
                        color = "#4CAF50" if sentiment == "positive" else "#F44336" if sentiment == "negative" else "#9E9E9E"
                        size = 1 + min(count / 2, 2) # add emphasis on count
                        html_words.append(f'<span style="color: {color}; font-size: {size}em; margin: 0.3em; display: inline-block;">{word}</span>')
                    
                    import random
                    random.shuffle(html_words)
                    
                    st.markdown(f"<div style='line-height: 2.5em; text-align: center;'>{''.join(html_words)}</div>", unsafe_allow_html=True)
                else:
                    st.info(f"Not enough descriptive words found for {competitor} in the data.")
            with col2:
                st.markdown(f"### Top Discussion Topics")
                # define topic areas
                topic_areas = {
                    "Code Generation": ["code", "generate", "generation", "programming", "coding", "write", "creates"],
                    "UI/UX": ["interface", "ui", "ux", "design", "look", "feel", "user experience", "dark mode"],
                    "Accuracy": ["accurate", "accuracy", "correct", "mistake", "error", "hallucination", "wrong", "right"],
                    "Performance": ["speed", "fast", "slow", "quick", "performance", "efficient", "responsive"],
                    "Integration": ["api", "endpoint", "integration", "function", "vscode", "editor", "ide", "plugin"],
                    "Cost": ["price", "cost", "expensive", "cheap", "free", "worth", "value", "money"],
                    "Features": ["feature", "capability", "ability", "function", "tool", "does", "can do"],
                    "Quality": ["quality", "good", "bad", "excellent", "poor", "best", "worst", "great", "terrible"],
                    "Context": ["context", "memory", "history", "remember", "conversation", "tokens"],
                    "Documentation": ["docs", "documentation", "tutorial", "guide", "examples", "help"]
                }
                topic_counts = {topic: 0 for topic in topic_areas}
                for _, row in competitor_data.iterrows():
                    text = str(row['text']).lower() if pd.notna(row['text']) else ""
                    for topic, keywords in topic_areas.items():
                        if any(keyword in text for keyword in keywords):
                            topic_counts[topic] += 1
                top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]  # show top 5

                for topic, count in top_topics:
                    if count > 0:
                        st.markdown(f"**{topic}** ({count} mentions)")
                    
            # define strengths and weaknesses
            categories = {
                "Code Completion": ["autocomplete", "completion", "suggestion", "predict", "code generation", 
                                   "generates", "writing", "finish", "complete", "fill", "writes", "coding"],
                
                "Code Quality": ["quality", "accurate", "accuracy", "error", "clean", "readable", "correct", 
                                "bugs", "fix", "precise", "typo", "refactor", "format", "code quality"],
                
                "Performance Speed": ["fast", "slow", "speed", "quick", "lag", "performance", "responsive", 
                                     "instant", "efficient", "latency", "delay", "wait", "snappy", "realtime"],
                
                "Integration": ["ide", "editor", "vscode", "intellij", "integration", "plugin", "extension", 
                               "workflow", "environment", "platform", "compatible", "connect", "works with"],
                
                "Learning Curve": ["learn", "learning", "intuitive", "easy", "difficult", "complex", "simple", 
                                  "confusing", "straightforward", "understand", "beginner", "novice", "expert"],
                
                "Context Understanding": ["context", "understand", "intelligent", "smart", "detection", 
                                         "comprehension", "aware", "knowledge", "recognize", "relevant", "intent"],
                
                "Customization": ["custom", "settings", "configure", "options", "preferences", "personalize", 
                                 "adjust", "modify", "tailor", "adapt", "tweak", "control", "flexibility"],
                
                "Reliability": ["reliable", "stable", "crash", "bug", "glitch", "consistent", "issue", 
                               "problem", "fails", "breaks", "broken", "depend", "trust", "robust", "solid"],
                
                "Documentation": ["docs", "documentation", "tutorial", "example", "guide", "help", "support", 
                                 "reference", "explanation", "manual", "instruction", "learn", "resources"],
                
                "Value": ["price", "cost", "value", "worth", "expensive", "cheap", "free", "subscription",
                         "pricing", "pay", "purchase", "premium", "plan", "roi", "money", "afford"]
            }
            category_sentiments = {category: {"positive": [], "negative": []} for category in categories}
            
            for _, row in competitor_data.iterrows():
                text = str(row['text']).lower() if pd.notna(row['text']) else ""
                sentiment = row['sentiment_score']
                for category, keywords in categories.items():
                    if any(keyword in text for keyword in keywords):
                        if sentiment > 0.05:  
                            category_sentiments[category]["positive"].append(row)
                        elif sentiment < -0.05: 
                            category_sentiments[category]["negative"].append(row)
            
            positive_categories = [(category, len(data["positive"])) for category, data in category_sentiments.items()]
            negative_categories = [(category, len(data["negative"])) for category, data in category_sentiments.items()]
            
            positive_categories.sort(key=lambda x: x[1], reverse=True)
            negative_categories.sort(key=lambda x: x[1], reverse=True)
            
            # make cols for strengths and weaknesses
            st.markdown("### Strengths & Weaknesses")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div style="background-color: rgba(0, 128, 0, 0.1); padding: 15px; border-radius: 5px; border-left: 5px solid green;">', unsafe_allow_html=True)
                st.markdown(f"#### 🌟 What People Like About {competitor}")
                # make sure we dont post the same post ID twice
                used_post_ids = set()
                top_positive_shown = 0
                for i, (category, count) in enumerate(positive_categories):
                    if count > 0 and top_positive_shown < 3:
                        # get the representative examples
                        unused_examples = [
                            example for example in category_sentiments[category]["positive"] 
                            if example['id'] not in used_post_ids
                        ]
                        
                        if not unused_examples:
                            continue  
                        top_positive_shown += 1
                        # sort by sentiment score
                        examples = sorted(unused_examples, key=lambda x: x['sentiment_score'], reverse=True)[:2]
                        
                        st.markdown(f"**{top_positive_shown}. {category} ({count} positive mentions)**")
                
                        if examples:
                            example = examples[0]
                            used_post_ids.add(example['id'])
                            
                            # show the relevant segment of the quote
                            quote = str(example['text'])
                            relevant_quote = extract_relevant_segment(quote, category, categories[category])
                            
                            # only show 20 words
                            short_quote = " ".join(relevant_quote.split()[:20]) 
                            shortened_quote = short_quote + "..." if len(relevant_quote.split()) > 20 else short_quote
                            
                            # create sentiment summary
                            sentiment_summary = generate_sentiment_summary(quote, example['sentiment_score'], category)
                            st.markdown(f"*\"{shortened_quote}\"*")
                            st.markdown(f"**Summary:** *{sentiment_summary}*")
                            
                            # "more examples" option
                            with st.expander("More examples"):
                                for ex in examples:
                                    used_post_ids.add(ex['id'])
                                    st.markdown(f"**Source:** {ex['source']} | **Sentiment:** {ex['sentiment_score']:.2f}")
                                    # Limit text length
                                    relevant_text = extract_relevant_segment(str(ex['text']), category, categories[category])
                                    short_comment = " ".join(relevant_text.split()[:50])
                                    summary = generate_sentiment_summary(str(ex['text']), ex['sentiment_score'], category)
                                    st.markdown(f"**Comment:** {short_comment}...")
                                    st.markdown(f"**Summary:** *{summary}*")
                                    if 'url' in ex and ex['url']:
                                        st.markdown(f"[View original]({ex['url']})")
                
                # no positive categories found
                if top_positive_shown == 0:
                    st.markdown("**No strong positive mentions found. More data collection may be needed.**")
                    
                st.markdown('</div>', unsafe_allow_html=True)
            # negative section
            with col2:
                st.markdown('<div style="background-color: rgba(255, 0, 0, 0.1); padding: 15px; border-radius: 5px; border-left: 5px solid red;">', unsafe_allow_html=True)
                st.markdown(f"#### ⚠️ What People Dislike About {competitor}")
                top_negative_shown = 0
                for i, (category, count) in enumerate(negative_categories):
                    if count > 0 and top_negative_shown < 3:
                        unused_examples = [
                            example for example in category_sentiments[category]["negative"] 
                            if example['id'] not in used_post_ids
                        ]
                        if not unused_examples:
                            continue  
                        top_negative_shown += 1
                        # sentiment score, most negative first
                        examples = sorted(unused_examples, key=lambda x: x['sentiment_score'])[:2]
                        st.markdown(f"**{top_negative_shown}. {category} ({count} negative mentions)**")
                        if examples:
                            example = examples[0]
                            used_post_ids.add(example['id'])
                            
                            # show relevant segment of the quote
                            quote = str(example['text'])
                            relevant_quote = extract_relevant_segment(quote, category, categories[category])
                            
                            # only show 20 words
                            short_quote = " ".join(relevant_quote.split()[:20]) 
                            shortened_quote = short_quote + "..." if len(relevant_quote.split()) > 20 else short_quote

                            # create sentiment summary
                            sentiment_summary = generate_sentiment_summary(quote, example['sentiment_score'], category)
                            
                            st.markdown(f"*\"{shortened_quote}\"*")
                            st.markdown(f"**Summary:** *{sentiment_summary}*")
                            
                            # "more examples" option
                            with st.expander("More examples"):
                                for ex in examples:
                                    used_post_ids.add(ex['id'])
                                    st.markdown(f"**Source:** {ex['source']} | **Sentiment:** {ex['sentiment_score']:.2f}")
                                    relevant_text = extract_relevant_segment(str(ex['text']), category, categories[category])
                                    short_comment = " ".join(relevant_text.split()[:50])
                                    summary = generate_sentiment_summary(str(ex['text']), ex['sentiment_score'], category)
                                    st.markdown(f"**Comment:** {short_comment}...")
                                    st.markdown(f"**Summary:** *{summary}*")
                                    if 'url' in ex and ex['url']:
                                        st.markdown(f"[View original]({ex['url']})")
                
                # no negative categories
                if top_negative_shown == 0:
                    st.markdown("**No strong negative mentions found. More data collection may be needed.**")
                    
                st.markdown('</div>', unsafe_allow_html=True)

def get_current_time():
    """Get current time formatted as string"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def update_refresh_timestamp():
    """Update the timestamp file with current time"""
    refresh_time_file = os.path.join(DATA_DIR, "last_refresh_time.txt")
    current_time = get_current_time()
    with open(refresh_time_file, "w") as f:
        f.write(current_time)
    return current_time

st.set_page_config(
    page_title="Windsurf & Codeium Mentions Dashboard",
    page_icon="🏄",
    layout="wide",
    initial_sidebar_state="expanded"
)
#paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

#css
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .metric-card {
        border-radius: 5px;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
    }
    .positive {
        background-color: rgba(0, 128, 0, 0.1);
        border-left: 5px solid green;
    }
    .neutral {
        background-color: rgba(128, 128, 128, 0.1);
        border-left: 5px solid gray;
    }
    .negative {
        background-color: rgba(255, 0, 0, 0.1);
        border-left: 5px solid red;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # cache for 1 hour
def load_data(competitor=None):
    """
    Load the latest aggregated data
    If competitor is specified, only load data for that competitor
    """
    # get the latest aggregated data file
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    data_files = glob.glob(os.path.join(DATA_DIR, f"aggregated_data_{today}.csv"))
    
    if not data_files:
        # if none from today, try finding the most recent one
        data_files = glob.glob(os.path.join(DATA_DIR, "aggregated_data_*.csv"))
        
    if not data_files:
        return None
    
    # sort by filename (which contains the date) to get the most recent
    latest_file = sorted(data_files)[-1]
    # load CSV file
    df = pd.read_csv(latest_file)
    
    # convert created_at to datetime
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # parse JSON columns
    df['platform_specific_metrics'] = df['platform_specific_metrics'].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    df['keywords_found'] = df['keywords_found'].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    
    # filter by competitor, keywords for each
    if competitor:
        if competitor == "codeium":
            df = df[df['keywords_found'].apply(
                lambda x: any(keyword in ['codeium', 'windsurf', 'windsurfing'] for keyword in x)
            )]
        elif competitor == "copilot":
            df = df[df['text'].str.lower().str.contains('github copilot|copilot|github co-pilot', 
                                                       na=False, regex=True)]
        elif competitor == "cursor":
            df = df[df['text'].str.lower().str.contains('cursor.so|cursor ai|cursor editor', 
                                                       na=False, regex=True)]
        elif competitor == "lovable":
            df = df[df['text'].str.lower().str.contains('lovable|lovable.ai|lovable editor', 
                                                       na=False, regex=True)]
        elif competitor == "devin":
            df = df[df['text'].str.lower().str.contains('devin ai|cognition labs|cognition ai', 
                                                       na=False, regex=True)]
        elif competitor == "cody":
            df = df[df['text'].str.lower().str.contains('sourcegraph cody|cody ai|sourcegraph', 
                                                       na=False, regex=True)]
    
    return df

def render_header():
    """Render the dashboard header"""
    st.markdown('<div class="main-header">Windsurf & Codeium Mentions Dashboard</div>', unsafe_allow_html=True)
    st.markdown('Track mentions and sentiment across social media and developer communities')

def render_filters(df):
    """Render filter controls"""
    st.sidebar.header("Filters")
    
    # date range filter
    min_date = df['created_at'].min().date()
    max_date = df['created_at'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(max_date - datetime.timedelta(days=7), max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # single date selection
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range
    
    # source filter
    sources = df['source'].unique().tolist()
    selected_sources = st.sidebar.multiselect(
        "Sources",
        options=sources,
        default=sources
    )
    
    # sentiment filter
    sentiments = df['sentiment_label'].unique().tolist()
    selected_sentiments = st.sidebar.multiselect(
        "Sentiment",
        options=sentiments,
        default=sentiments
    )
    
    # apply filters
    filtered_df = df[
        (df['created_at'].dt.date >= start_date) &
        (df['created_at'].dt.date <= end_date) &
        (df['source'].isin(selected_sources)) &
        (df['sentiment_label'].isin(selected_sentiments))
    ]
    
    return filtered_df

def render_metrics(df):
    """Render key metrics"""
    st.markdown('<div class="sub-header">Overview</div>', unsafe_allow_html=True)
    
    total_mentions = len(df)
    positive_mentions = len(df[df['sentiment_label'] == 'Positive'])
    neutral_mentions = len(df[df['sentiment_label'] == 'Neutral'])
    negative_mentions = len(df[df['sentiment_label'] == 'Negative'])
    
    if total_mentions > 0:
        positive_pct = (positive_mentions / total_mentions) * 100
        neutral_pct = (neutral_mentions / total_mentions) * 100
        negative_pct = (negative_mentions / total_mentions) * 100
    else:
        positive_pct = neutral_pct = negative_pct = 0
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_mentions}</div>
            <div class="metric-label">Total Mentions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card positive">
            <div class="metric-value">{positive_mentions} ({positive_pct:.1f}%)</div>
            <div class="metric-label">Positive Mentions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card neutral">
            <div class="metric-value">{neutral_mentions} ({neutral_pct:.1f}%)</div>
            <div class="metric-label">Neutral Mentions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card negative">
            <div class="metric-value">{negative_mentions} ({negative_pct:.1f}%)</div>
            <div class="metric-label">Negative Mentions</div>
        </div>
        """, unsafe_allow_html=True)

def render_codeium_sentiment_overview(df):
    """Render an overview of Codeium sentiment with score and word bubble"""
    st.markdown('<div class="sub-header">Codeium Sentiment Overview</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # calc sentiment score
        avg_sentiment = df['sentiment_score'].mean()
        sentiment_score = int((avg_sentiment + 1) * 50)  # Convert from -1 to 1 scale to 0-100 scale
        
        # progress bar
        st.markdown(f"### Sentiment Score: {sentiment_score}/100")
        
        # sentiment gauge
        sentiment_color = "green" if sentiment_score >= 70 else "orange" if sentiment_score >= 40 else "red"
        st.markdown(f"""
        <div style="background: linear-gradient(to right, #f0f0f0, #f0f0f0); height: 20px; border-radius: 10px; margin-bottom: 10px;">
            <div style="background: linear-gradient(to right, {sentiment_color}, {sentiment_color}); width: {sentiment_score}%; height: 20px; border-radius: 10px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # what the score means, hardcoded for now
        if sentiment_score >= 80:
            st.markdown("**Overall sentiment is very positive.** Users are highly satisfied with Codeium.")
        elif sentiment_score >= 70:
            st.markdown("**Overall sentiment is positive.** Most users have favorable opinions about Codeium.")
        elif sentiment_score >= 50:
            st.markdown("**Overall sentiment is moderately positive.** Users generally like Codeium but have some concerns.")
        elif sentiment_score >= 30:
            st.markdown("**Overall sentiment is mixed.** Users have both positive and negative experiences with Codeium.")
        else:
            st.markdown("**Overall sentiment needs improvement.** Users are expressing significant concerns.")
    
    with col2:
        st.markdown("### How Users Describe Codeium")
        
        # get descriptive words- same as in competitor section
        def extract_descriptive_words(texts):
            descriptors = []
            positive_words = [
                "amazing", "excellent", "great", "good", "impressive", "powerful", 
                "smart", "helpful", "intuitive", "fast", "reliable", "better", "best",
                "love", "awesome", "fantastic", "wonderful", "superb", "brilliant",
                "quality", "efficient", "accurate", "precise", "clean", "consistent", 
                "seamless", "innovative", "nice", "perfect", "easy", "convenient", 
                "superior", "fun", "useful", "favorite", "recommended", "smooth"
            ]
            
            negative_words = [
                "slow", "buggy", "limited", "expensive", "frustrating", "confusing", 
                "disappointing", "unreliable", "broken", "clunky", "terrible", "worst",
                "bad", "poor", "awful", "annoying", "useless", "overpriced", "glitchy",
                "difficult", "mediocre", "insufficient", "lacking", "problematic", "costly",
                "cumbersome", "complicated", "inconsistent", "inadequate", "unstable",
                "error", "crash", "fail", "fails", "failed", "waste", "horrible", "trouble"
            ]
            
            neutral_words = [
                "new", "different", "unique", "interesting", "complex", "simple", 
                "basic", "standard", "common", "specialized", "normal", "typical",
                "average", "alternative", "familiar", "surprising", "unusual", "fresh",
                "competitive", "expected", "unexpected", "particular", "specific",
                "notable", "comparable", "distinctive", "similar", "okay", "decent"
            ]
            
            all_descriptive_words = positive_words + negative_words + neutral_words
            for text in texts:
                if not isinstance(text, str):
                    continue
                    
                text = text.lower()
                words = re.findall(r'\b\w+\b', text)
                
                for word in words:
                    if len(word) >= 3 and word in all_descriptive_words:
                        sentiment_type = "positive" if word in positive_words else "negative" if word in negative_words else "neutral"
                        descriptors.append((word, sentiment_type))

            word_counts = {}
            for word, sentiment in descriptors:
                if word not in word_counts:
                    word_counts[word] = {"count": 0, "sentiment": sentiment}
                word_counts[word]["count"] += 1

            result = [(word, data["sentiment"], data["count"]) 
                     for word, data in word_counts.items()]

            result.sort(key=lambda x: x[2], reverse=True)
            
            return result[:50]  # return top 50 words for Codeium
        
        texts = df['text'].tolist()
        descriptors = extract_descriptive_words(texts)
        
        if descriptors:
            # generate word bubble
            html_words = []
            for word, sentiment, count in descriptors:
                # set color based on sentiment
                color = "#4CAF50" if sentiment == "positive" else "#F44336" if sentiment == "negative" else "#9E9E9E"
                # size based on count (1-3em)
                size = 1 + min(count / 2, 2)  # More emphasis on count
                html_words.append(f'<span style="color: {color}; font-size: {size}em; margin: 0.3em; display: inline-block;">{word}</span>')
            
            # shuffle words 
            import random
            random.shuffle(html_words)
            
            # display word bubble
            st.markdown(f"<div style='line-height: 2.5em; text-align: center;'>{''.join(html_words)}</div>", unsafe_allow_html=True)
            
            # legend for the colors
            st.markdown("""
            <div style='text-align: center; margin-top: 10px; font-size: 0.8em;'>
                <span style='color: #4CAF50;'>■</span> Positive &nbsp;
                <span style='color: #F44336;'>■</span> Negative &nbsp;
                <span style='color: #9E9E9E;'>■</span> Neutral
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Not enough descriptive words found in the data.")

def render_time_series(df):
    """Render time series charts"""
    st.markdown('<div class="sub-header">Trends</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # mentions over time
        mentions_by_date = df.groupby(df['created_at'].dt.date).size().reset_index(name='count')
        mentions_by_date.columns = ['date', 'count']
        
        fig = px.line(
            mentions_by_date, 
            x='date', 
            y='count',
            title='Mentions Over Time',
            labels={'date': 'Date', 'count': 'Number of Mentions'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # sentiment over time
        sentiment_by_date = df.groupby(df['created_at'].dt.date)['sentiment_score'].mean().reset_index()
        sentiment_by_date.columns = ['date', 'sentiment_score']
        
        # sentiment graph
        fig = px.line(
            sentiment_by_date, 
            x='date', 
            y='sentiment_score',
            title='Sentiment Trend',
            labels={'date': 'Date', 'sentiment_score': 'Average Sentiment Score'},
            color_discrete_sequence=['purple']
        )
        
        # add zero line
        fig.add_shape(
            type="line",
            x0=sentiment_by_date['date'].min(),
            y0=0,
            x1=sentiment_by_date['date'].max(),
            y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_source_distribution(df):
    """Render source distribution charts"""
    col1, col2 = st.columns(2)
    
    with col1:
        # source counts
        source_counts = df['source'].value_counts().reset_index()
        source_counts.columns = ['source', 'count']
        
        fig = px.bar(
            source_counts,
            x='source',
            y='count',
            title='Mentions by Source',
            labels={'source': 'Source', 'count': 'Number of Mentions'},
            color='source'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # sentiment by source
        sentiment_by_source = df.groupby('source')['sentiment_score'].mean().reset_index()
        
        fig = px.bar(
            sentiment_by_source,
            x='source',
            y='sentiment_score',
            title='Average Sentiment by Source',
            labels={'source': 'Source', 'sentiment_score': 'Average Sentiment Score'},
            color='sentiment_score',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
def render_product_insights(df):
    """Render standardized product feature sentiment analysis with historical navigation"""
    if len(df) == 0:
        st.warning("No data available for analysis.")
        return
    
    st.markdown('<div class="sub-header"> Feature Sentiment Analysis (Daily)</div>', unsafe_allow_html=True)
    # hard code for now
    standard_features = {
        "Code Completion": ["autocomplete", "completion", "suggestion", "predict", "completes", "autofill", 
                          "code generation", "generates", "writing", "finish", "complete", "fill", "writes"],
        
        "Code Quality": ["quality", "accurate", "accuracy", "error", "errors", "mistake", "clean", "readable",
                       "correct", "bugs", "buggy", "fix", "precise", "typo", "refactor", "format"],
        
        "Performance Speed": ["fast", "slow", "speed", "quick", "lag", "performance", "responsive", "instant",
                           "efficient", "latency", "overhead", "delay", "wait", "snappy", "real-time", "rapid"],
        
        "Integration": ["ide", "editor", "vscode", "intellij", "integration", "plugin", "extension", "workflow",
                      "environment", "platform", "compatible", "works with", "supports", "ecosystem", "tool"],
        
        "User Experience": ["learn", "learning", "intuitive", "easy", "difficult", "complex", "simple", "confusing",
                         "straightforward", "understand", "beginner", "novice", "expert", "onboarding", "get started",
                         "UX", "usability", "user-friendly", "approachable", "frustrating", "pleasant", "navigation"],
        
        "Context Understanding": ["context", "understand", "intelligent", "smart", "detection", "comprehension",
                               "aware", "knowledge", "recognize", "relevant", "understand code", "surrounding", "intent"],
        
        "Customization": ["custom", "settings", "configure", "options", "preferences", "personalize", "adjust",
                        "modify", "tailor", "adapt", "tweak", "control", "flexibility", "set up"],
        
        "Reliability": ["reliable", "stable", "crash", "bug", "glitch", "consistent", "issue", "problem",
                      "fails", "breaks", "broken", "depend", "trust", "robust", "solid", "dependable"],
        
        "Documentation": ["docs", "documentation", "tutorial", "example", "guide", "help", "support", "reference",
                        "explanation", "manual", "instruction", "how-to", "learn", "resources"],
        
        "Value": ["price", "cost", "value", "worth", "expensive", "cheap", "free", "subscription",
                "pricing", "pay", "purchase", "premium", "plan", "roi", "justify"]
    }
    # find multi-word phrases in the text
    def text_contains_feature(text, keywords):
        # 1. check for exact matches of keywords
        if any(keyword in text for keyword in keywords):
            return True
        
        # 2. check for semantic matches
        if "code" in text and any(word in text for word in ["finish", "complete", "write", "generate", "autocomplete"]):
            if "Code Completion" in keywords:
                return True
                
        if "doesn't work" in text or "not working" in text or "failed" in text:
            if "Reliability" in keywords:
                return True
            
        if "hard to" in text or "difficult to" in text or "confusing" in text:
            if "User Experience" in keywords:
                return True
            
        if "doesn't understand" in text or "misunderstands" in text:
            if "Context Understanding" in keywords:
                return True
        return False
    
    # get date range for historical data & calc past week
    today = datetime.datetime.now().date()
    days = []
    for i in range(7):
        days.append((today - datetime.timedelta(days=i)).strftime('%Y-%m-%d'))
    # get historial data for each day 
    historical_data = {}
    data_count = {}
    
    for day in days:
        daily_files = glob.glob(os.path.join(DATA_DIR, f"aggregated_data_{day}.csv"))
        if daily_files:
            try:
                daily_df = pd.read_csv(daily_files[0])
                # convert to datetime
                daily_df['created_at'] = pd.to_datetime(daily_df['created_at'])
                # process this day's data
                day_feature_data = []
                # count total mentions across all features for weighted sentiment 
                total_mentions = 0
                # first pass to get total mentions
                for feature, keywords in standard_features.items():
                    feature_mentions = 0
                    # find mentions of this feature across posts
                    for _, row in daily_df.iterrows():
                        text = str(row['text']).lower() if pd.notna(row['text']) else ""
                        # check if this text contains the feature
                        if text_contains_feature(text, keywords):
                            feature_mentions += 1
                    
                    total_mentions += feature_mentions
                # second pass to calculate statistics and weighted sentiment
                for feature, keywords in standard_features.items():
                    feature_mentions = 0
                    feature_sentiment = 0
                    positive_mentions = 0
                    negative_mentions = 0
                    neutral_mentions = 0
                    
                    # find mentions of this feature across posts
                    for _, row in daily_df.iterrows():
                        text = str(row['text']).lower() if pd.notna(row['text']) else ""
                        sentiment = row['sentiment_score']
                        
                        # check if this text contains the feature
                        if text_contains_feature(text, keywords):
                            feature_mentions += 1
                            feature_sentiment += sentiment
                            
                            if sentiment > 0.1:
                                positive_mentions += 1
                            elif sentiment < -0.1:
                                negative_mentions += 1
                            else:
                                neutral_mentions += 1
                    
                    # calculate average sentiment for feature
                    avg_sentiment = feature_sentiment / feature_mentions if feature_mentions > 0 else 0
                    
                    # calc weighted sentiment: (avg_sentiment * mentions) / sum_of_all_mentions
                    # this represents how confident we can be in the sentiment based on sample size
                    confidence_factor = min(feature_mentions / 30, 1)  # reaches max confidence at 30+ mentions

                    norm_sentiment = (avg_sentiment + 1) / 2  # convert from [-1,1] to [0,1]

                    # for low mention counts, this will pull toward 0.5 (neutral)
                    # for high mention counts, it will be closer to the actual sentiment
                    weighted_sent = norm_sentiment * confidence_factor + 0.5 * (1 - confidence_factor)
                    
                    # add feature data
                    if feature_mentions > 0:
                        day_feature_data.append({
                            'Feature': feature,
                            'Mentions': feature_mentions,
                            'Average Sentiment': avg_sentiment,
                            'Weighted Sentiment': weighted_sent,
                            'Positive': positive_mentions,
                            'Neutral': neutral_mentions,
                            'Negative': negative_mentions,
                            'Positive %': (positive_mentions / feature_mentions * 100) if feature_mentions > 0 else 0,
                            'Negative %': (negative_mentions / feature_mentions * 100) if feature_mentions > 0 else 0,
                        })
                
                # store today's data
                historical_data[day] = day_feature_data
                data_count[day] = len(daily_df)
            except Exception as e:
                st.warning(f"Could not load data for {day}: {e}")
    
    # get all days with data
    available_days = sorted([day for day in historical_data.keys() if historical_data[day]], reverse=True)
    
    # no data available
    if not available_days:
        st.info("No historical data available for analysis.")
        return
    
    # display timeline navigation
    st.markdown('<div style="font-size: 24px; font-weight: bold; margin-bottom: 10px;"></div>', unsafe_allow_html=True)
    
    # create dropdown for date selection
    selected_day_index = available_days.index(available_days[0])
    selected_day = st.selectbox(
        "Select Date:", 
        available_days,
        index=selected_day_index,
        format_func=lambda d: f"{datetime.datetime.strptime(d, '%Y-%m-%d').strftime('%b %d, %Y')} ({data_count.get(d, 0)} posts)"
    )
    
    # get data for selected day
    feature_data = historical_data.get(selected_day, [])
    
    # format date for display
    display_date = datetime.datetime.strptime(selected_day, '%Y-%m-%d').strftime('%b %d, %Y')
    total_posts = data_count.get(selected_day, 0)
    
    if not feature_data:
        st.info(f"No data available for {display_date}.")
        return
    
    # add one week trend calculation
    if len(available_days) > 1:
        # find the day from a week ago (or the oldest available)
        for day in available_days:
            day_date = datetime.datetime.strptime(day, '%Y-%m-%d').date()
            days_diff = (datetime.datetime.strptime(selected_day, '%Y-%m-%d').date() - day_date).days
            if days_diff >= 7:
                week_ago_day = day
                break
        else:
            # use the oldest available
            week_ago_day = available_days[-1]
        
        # get data from a week ago
        week_ago_data = historical_data.get(week_ago_day, [])
        week_ago_dict = {item['Feature']: item['Average Sentiment'] for item in week_ago_data}
        
        # calc week change
        for item in feature_data:
            feature = item['Feature']
            if feature in week_ago_dict:
                week_pct_change = ((item['Average Sentiment'] - week_ago_dict[feature]) / (abs(week_ago_dict[feature]) if week_ago_dict[feature] != 0 else 0.01)) * 100
                item['Week %'] = week_pct_change
            else:
                item['Week %'] = 0
    
    # calc day change if prev day available
    if len(available_days) > 1:
        prev_day_index = available_days.index(selected_day) + 1
        if prev_day_index < len(available_days):
            prev_day_data = historical_data.get(available_days[prev_day_index], [])
            prev_day_dict = {item['Feature']: item['Average Sentiment'] for item in prev_day_data}
            
            # calc day change
            for item in feature_data:
                feature = item['Feature']
                if feature in prev_day_dict:
                    day_pct_change = ((item['Average Sentiment'] - prev_day_dict[feature]) / (abs(prev_day_dict[feature]) if prev_day_dict[feature] != 0 else 0.01)) * 100
                    item['Day %'] = day_pct_change
                else:
                    item['Day %'] = 0
    
    # sort by mentions by default
    feature_data = sorted(feature_data, key=lambda x: x['Mentions'], reverse=True)
    
    # display as a bar chart
    if feature_data:
        # create horizontal bar chart showing sentiment by feature
        features = [item['Feature'] for item in feature_data]
        sentiments = [item['Average Sentiment'] for item in feature_data]
        mentions = [item['Mentions'] for item in feature_data]
        
        # determine dynamic x-axis range based on actual data
        min_sentiment = min(sentiments) if sentiments else 0
        max_sentiment = max(sentiments) if sentiments else 0.5
        
        # if all sentiments are positive, start from 0
        if min_sentiment >= 0:
            min_x = 0
            max_x = math.ceil(max_sentiment * 2) / 2  # round up to nearest 0.5
            if max_x <= 0.5:
                max_x = 0.5
        # if all sentiments are negative, end at 0
        elif max_sentiment <= 0:
            min_x = math.floor(min_sentiment * 2) / 2  # round down to nearest 0.5
            max_x = 0
            if min_x >= -0.5:
                min_x = -0.5
        # if mixed, round to nearest 0.5 in each direction
        else:
            min_x = math.floor(min_sentiment * 2) / 2
            max_x = math.ceil(max_sentiment * 2) / 2
        
        # create color map for sentiment (more cohesive colors - all green shades)
        colors = []
        for sentiment in sentiments: # diff shades of green
            if sentiment > 0.7:  # very positive
                colors.append('rgba(0, 100, 0, 0.9)')  
            elif sentiment > 0.5:  
                colors.append('rgba(34, 139, 34, 0.85)')  
            elif sentiment > 0.3:  
                colors.append('rgba(60, 179, 113, 0.8)')  
            elif sentiment > 0.1:  
                colors.append('rgba(144, 238, 144, 0.8)')  
            elif sentiment > -0.1:  
                colors.append('rgba(211, 211, 211, 0.8)')  
            elif sentiment > -0.3:  
                colors.append('rgba(240, 128, 128, 0.8)')  
            elif sentiment > -0.5:  
                colors.append('rgba(220, 20, 60, 0.8)')  
            else:  # very negative
                colors.append('rgba(139, 0, 0, 0.9)')  
        
        # create horizontal bar chart
        fig = go.Figure()
        
        # add bars for features with mentions
        fig.add_trace(go.Bar(
            y=[f"{features[i]} ({mentions[i]})" for i in range(len(features))],
            x=sentiments,
            orientation='h',
            marker_color=colors,
            text=[f"{s:.3f}" for s in sentiments],
            textposition='auto',
            name='Average Sentiment',
            textfont=dict(size=14)  
        ))
        sentiment_tooltip = """
        <b>How to interpret sentiment scores:</b><br>
        • +1.0: Extremely positive mentions<br>
        • +0.5: Moderately positive mentions<br>
        • 0.0: Neutral mentions<br>
        • -0.5: Moderately negative mentions<br>
        • -1.0: Extremely negative mentions<br><br>
        """
        
        # customize layout with dynamic axis and larger text
        fig.update_layout(
            title=dict(
                text=f'Product Feature Sentiment Analysis ({display_date} - {total_posts} posts processed)',
                font=dict(size=18)
            ),
            xaxis=dict(
                title=dict(
                    text="Sentiment Score (-1 to +1)",
                    font=dict(size=14)
                ),
                range=[min_x, max_x],
                tickvals=[min_x, 0, max_x] if min_x < 0 else [0, max_x/2, max_x],
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title=dict(
                    text='Product Feature',
                    font=dict(size=14)
                ),
                tickfont=dict(size=14)
            ),
            height=700,  
            margin=dict(l=20, r=20, t=60, b=120)  
        )
        
        if min_x < 0 and max_x > 0:
            fig.add_shape(
                type="line",
                x0=0, y0=-0.5,
                x1=0, y1=len(features)-0.5,
                line=dict(color="black", width=1, dash="dash")
            )
        
        fig.add_annotation(
            x=(min_x + max_x) / 2,
            y=-1.5,  
            xref="x",
            yref="paper",
            text="Hover for info on sentiment scores",
            showarrow=False,
            font=dict(size=14, color="blue"),
            hovertext=sentiment_tooltip,
            hoverlabel=dict(bgcolor="white"),
            opacity=0.8
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f'<div style="font-size: 18px; font-weight: bold; margin-top: 20px;">Feature Mentions & Sentiment Trends ({display_date})</div>', unsafe_allow_html=True)
        
        trend_timeframe = st.radio(
            "Trend Timeframe:",
            ["Day", "Week"],
            horizontal=True,
            key="trend_timeframe"
        )
        
        # initialize a dataframe for the table
        table_df = pd.DataFrame(feature_data)
        
        if trend_timeframe == "Day" and 'Day %' in table_df.columns:
            table_df.loc[:, 'Trend'] = table_df['Day %'].apply(
                lambda x: f"↑ +{x:.3f}%" if x > 0.01 else (f"↓ {x:.3f}%" if x < -0.01 else "→ 0.000%")
            )
        elif trend_timeframe == "Week" and 'Week %' in table_df.columns:
            table_df.loc[:, 'Trend'] = table_df['Week %'].apply(
                lambda x: f"↑ +{x:.3f}%" if x > 0.01 else (f"↓ {x:.3f}%" if x < -0.01 else "→ 0.000%")
            )
        else:
            table_df.loc[:, 'Trend'] = "N/A"
        table_df.loc[:, 'Average Sentiment'] = table_df['Average Sentiment'].round(3)
        table_df.loc[:, 'Weighted Sentiment'] = table_df['Weighted Sentiment'].round(3)
        table_df.loc[:, 'Positive %'] = table_df['Positive %'].round(3)
        table_df.loc[:, 'Negative %'] = table_df['Negative %'].round(3)
        formatted_df = table_df[['Feature', 'Mentions', 'Average Sentiment', 'Weighted Sentiment', 'Positive %', 'Negative %', 'Trend']]
        column_renames = {
            'Average Sentiment': 'Avg Sentiment',
            'Weighted Sentiment': 'Weighted Sent',
            'Positive %': 'Pos %',
            'Negative %': 'Neg %'
        }
        formatted_df = formatted_df.rename(columns=column_renames)
        
        styled_df = formatted_df.style.background_gradient(
            cmap='Greens', subset=['Avg Sentiment', 'Weighted Sent']
        ).background_gradient(
            cmap='Blues', subset=['Mentions']
        ).apply(
            lambda x: ['color: green' if '↑' in str(v) else 'color: red' if '↓' in str(v) else '' for v in x], 
            subset=['Trend']
        ).format({
            'Avg Sentiment': '{:.3f}',
            'Weighted Sent': '{:.3f}',
            'Pos %': '{:.1f}',
            'Neg %': '{:.1f}'
        }).set_properties(**{'font-size': '14px'})  
        st.caption("Click on column headers to sort the table")
        st.dataframe(styled_df, use_container_width=True, height=450)
        with st.expander("ⓘ  Learn more about what each stat means"):
            st.markdown(f"""
            <div style="font-size: 14px;">
            <ul>
                <li><strong>Avg Sentiment:</strong> The average sentiment score (-1 to +1) of all mentions for this feature. Higher values indicate more positive sentiment.</li>
                <li><strong>Weighted Sent:</strong> The actual sentiment value. Higher mention counts increase our confidence in this value but do not alter it directly. Features with many mentions likely represent more accurate sentiment scores.</li>
                <li><strong>Pos %:</strong> Percentage of mentions with positive sentiment (score > 0.1).</li>
                <li><strong>Neg %:</strong> Percentage of mentions with negative sentiment (score < -0.1).</li>
                <li><strong>Trend:</strong> Percentage change in sentiment compared to the previous day or week. Upward arrows (↑) indicate improving sentiment.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info(f"No feature data available for {display_date}.")

def render_recent_mentions(df):
    """Render table of recent mentions"""
    st.markdown('<div class="sub-header">Recent Mentions</div>', unsafe_allow_html=True)
    recent_df = df.sort_values('created_at', ascending=False).head(10)
    
    for idx, row in recent_df.iterrows():
        sentiment_class = row['sentiment_label'].lower()
        created_date = row['created_at'].strftime("%Y-%m-%d %H:%M")
        
        expander_label = f"{row['source']} | {created_date} | {row['sentiment_label']}"
        
        with st.expander(expander_label):
            st.markdown(f"""
            <div class="{sentiment_class}">
                <p><strong>Text:</strong> {row['text']}</p>
                <p><strong>Author:</strong> {row['author']}</p>
                <p><strong>Sentiment Score:</strong> {row['sentiment_score']:.2f}</p>
                <p><strong>Source:</strong> <a href="{row['url']}" target="_blank">View Original</a></p>
            </div>
            """, unsafe_allow_html=True)

def render_feature_sentiment_trends(df):
    """Render sentiment trends for each feature category over time with improved interactivity and explanations"""
    if len(df) == 0:
        st.warning("No data available for analysis.")
        return
    
    st.markdown('<div class="sub-header">Feature Sentiment Analysis (Over Time)</div>', unsafe_allow_html=True)
    
    # hard code for now
    feature_definitions = {
        "Code Completion": {
            "description": "How well the tool generates, completes, and suggests code",
            "high_sentiment": "Users find code suggestions accurate and helpful",
            "low_sentiment": "Users are frustrated with inaccurate or unhelpful completions",
            "keywords": ["autocomplete", "completion", "suggestion", "predict", "completes", "autofill", 
                        "code generation", "generates", "writing", "finish", "complete", "fill", "writes"]
        },
        "Code Quality": {
            "description": "Quality, accuracy, and correctness of generated code",
            "high_sentiment": "Generated code is clean, correct, and follows best practices",
            "low_sentiment": "Code contains errors, bugs, or poor practices",
            "keywords": ["quality", "accurate", "accuracy", "error", "errors", "mistake", "clean", "readable",
                      "correct", "bugs", "buggy", "fix", "precise", "typo", "refactor", "format"]
        },
        "Performance Speed": {
            "description": "Speed and responsiveness of the tool",
            "high_sentiment": "Tool responds quickly with minimal lag",
            "low_sentiment": "Users experience delays, slowness, or latency issues",
            "keywords": ["fast", "slow", "speed", "quick", "lag", "performance", "responsive", "instant",
                      "efficient", "latency", "overhead", "delay", "wait", "snappy", "real-time", "rapid"]
        },
        "Integration": {
            "description": "How well the tool integrates with IDEs and workflows",
            "high_sentiment": "Seamless integration with development environments",
            "low_sentiment": "Integration issues or compatibility problems",
            "keywords": ["ide", "editor", "vscode", "intellij", "integration", "plugin", "extension", "workflow",
                      "environment", "platform", "compatible", "works with", "supports", "ecosystem", "tool"]
        },
        "Learning Curve": {
            "description": "Ease of learning and getting started with the tool",
            "high_sentiment": "Tool is intuitive and easy to learn",
            "low_sentiment": "Tool is difficult to understand or use effectively",
            "keywords": ["learn", "learning", "intuitive", "easy", "difficult", "complex", "simple", "confusing",
                      "straightforward", "understand", "beginner", "novice", "expert", "onboarding", "get started"]
        },
        "Context Understanding": {
            "description": "How well the tool understands code context and intent",
            "high_sentiment": "Tool accurately interprets context and provides relevant suggestions",
            "low_sentiment": "Tool misunderstands context or makes irrelevant suggestions",
            "keywords": ["context", "understand", "intelligent", "smart", "detection", "comprehension",
                      "aware", "knowledge", "recognize", "relevant", "understand code", "surrounding", "intent"]
        },
        "Customization": {
            "description": "Ability to customize and configure the tool",
            "high_sentiment": "Users appreciate flexibility and customization options",
            "low_sentiment": "Tool is too rigid or lacks configuration options",
            "keywords": ["custom", "settings", "configure", "options", "preferences", "personalize", "adjust",
                      "modify", "tailor", "adapt", "tweak", "control", "flexibility", "set up"]
        },
        "Reliability": {
            "description": "Stability and dependability of the tool",
            "high_sentiment": "Tool works consistently without crashes or issues",
            "low_sentiment": "Tool crashes, has bugs, or behaves unpredictably",
            "keywords": ["reliable", "stable", "crash", "bug", "glitch", "consistent", "issue", "problem",
                      "fails", "breaks", "broken", "depend", "trust", "robust", "solid", "dependable"]
        },
        "Documentation": {
            "description": "Quality and helpfulness of documentation and support",
            "high_sentiment": "Documentation is clear, comprehensive, and helpful",
            "low_sentiment": "Documentation is lacking, unclear, or unhelpful",
            "keywords": ["docs", "documentation", "tutorial", "example", "guide", "help", "support", "reference",
                      "explanation", "manual", "instruction", "how-to", "learn", "resources"]
        },
        "Value": {
            "description": "Perceived value for cost or price considerations",
            "high_sentiment": "Users feel they get good value for the price",
            "low_sentiment": "Tool is perceived as overpriced or not worth the cost",
            "keywords": ["price", "cost", "value", "worth", "expensive", "cheap", "free", "subscription",
                      "pricing", "pay", "purchase", "premium", "plan", "roi", "justify"]
        }
    }
    
    def text_contains_feature(text, keywords):
        # 1. check for exact matches of keywords
        if any(keyword in text for keyword in keywords):
            return True
        
        # 2. check for semantic matches - more complex patterns
        if "code" in text and any(word in text for word in ["finish", "complete", "write", "generate", "autocomplete"]):
            if "Code Completion" in keywords:
                return True
                
        if "doesn't work" in text or "not working" in text or "failed" in text:
            if "Reliability" in keywords:
                return True
            
        if "hard to" in text or "difficult to" in text or "confusing" in text:
            if "Learning Curve" in keywords:
                return True
            
        if "doesn't understand" in text or "misunderstands" in text:
            if "Context Understanding" in keywords:
                return True
        
        return False
    
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    min_date = df['created_at'].min().date()
    max_date = df['created_at'].max().date()
    
    df['date'] = df['created_at'].dt.date
    
    trend_data = []
    
    # process each feature
    for feature_name, feature_info in feature_definitions.items():
        feature_by_date = {}
        current_date = min_date
        while current_date <= max_date:
            feature_by_date[current_date] = {'mentions': 0, 'sentiment_total': 0}
            current_date += datetime.timedelta(days=1)
        
        for _, row in df.iterrows():
            text = str(row['text']).lower() if pd.notna(row['text']) else ""
            sentiment = row['sentiment_score']
            post_date = row['date']
            
            # see if this post discusses the feature
            if text_contains_feature(text, feature_info["keywords"]):
                feature_by_date[post_date]['mentions'] += 1
                feature_by_date[post_date]['sentiment_total'] += sentiment
        
        # calculate daily average sentiment
        date_sentiments = []
        
        for date, values in feature_by_date.items():
            if values['mentions'] > 0:
                avg_sentiment = values['sentiment_total'] / values['mentions']
                date_sentiments.append({
                    'Feature': feature_name,
                    'Date': date,
                    'Sentiment': avg_sentiment,
                    'Mentions': values['mentions']
                })
        
        # only add to trend data if we have some mentions
        if date_sentiments:
            trend_data.extend(date_sentiments)
    
    # convert to dataframe
    if trend_data:
        trend_df = pd.DataFrame(trend_data)
        
        # get unique features for selection
        unique_features = sorted(trend_df['Feature'].unique())
        
        # add feature selection checkboxes with tooltips
        st.write("**Select features to display:**")
        cols = st.columns(3)
        selected_features = []

        for i, feature in enumerate(unique_features):
            col_idx = i % 3
            with cols[col_idx]:
                tooltip_text = (
                    f"{feature_definitions[feature]['description']}\n\n"
                    f"High sentiment: {feature_definitions[feature]['high_sentiment']}\n\n"
                    f"Low sentiment: {feature_definitions[feature]['low_sentiment']}"
                )
                
                if st.checkbox(feature, value=True, key=f"feature_{feature}", help=tooltip_text):
                    selected_features.append(feature)
        
        if selected_features:
            filtered_trend_df = trend_df[trend_df['Feature'].isin(selected_features)]
            
            fig = px.line(
                filtered_trend_df, 
                x='Date', 
                y='Sentiment', 
                color='Feature',
                title='Feature Sentiment Trends Over Time',
                markers=True,
                hover_data=['Mentions']  
            )
            
            fig.add_shape(
                type="line",
                x0=min_date,
                y0=0,
                x1=max_date,
                y1=0,
                line=dict(color="gray", width=1, dash="dash")
            )
            
            fig.update_layout(
                yaxis_title='Sentiment Score (-1 to +1)',
                xaxis_title='Date',
                legend_title='Product Feature',
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(range=[-1, 1]), 
                hovermode="closest" 
            )
            
            fig.update_traces(
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Date: %{x|%Y-%m-%d}<br>' +
                              'Sentiment: %{y:.2f}<br>' +
                              'Mentions: %{customdata[0]}<extra></extra>'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **How to read this chart:**
            - Each line represents sentiment for a specific product feature
            - Higher values (above 0) indicate positive sentiment
            - Lower values (below 0) indicate negative sentiment
            - Hover over the ⓘ icon next to a feature to learn more about it
            - Hover over data points to see specific values
            """)
        else:
            st.warning("Please select at least one feature to display.")
    else:
        st.info("Not enough data to display feature sentiment trends over time.")

def render_engineer_insights(df):
    """Render key insights for engineering teams with properly categorized feedback"""
    if len(df) == 0:
        st.warning("No data available for analysis.")
        return
    
    st.markdown('<div class="sub-header">Key Insights for Engineers</div>', unsafe_allow_html=True)
    
    # standard categories
    categories = [
        "Integration", "Documentation", "Code Quality", "Context Understanding", 
        "Value", "User Experience", "Reliability", "Performance Speed", 
        "Code Completion", "Customization"
    ]
    
    # classify text into the most appropriate category
    def classify_into_category(text, sentiment_score):
        text = str(text).lower() if pd.notna(text) else ""
        
        # hard code for now
        category_keywords = {
            "Integration": ["ide", "editor", "vscode", "intellij", "integration", "plugin", "extension", 
                           "workflow", "environment", "platform", "compatible", "install"],
            
            "Documentation": ["docs", "documentation", "tutorial", "example", "guide", "help", "support", 
                             "reference", "explanation", "manual", "instruction", "learn"],
            
            "Code Quality": ["quality", "accurate", "accuracy", "error", "clean", "readable", "correct", 
                            "bugs", "fix", "precise", "typo", "refactor", "format"],
            
            "Context Understanding": ["context", "understand", "intelligent", "smart", "detection", 
                                     "comprehension", "aware", "knowledge", "recognize", "surrounding"],
            
            "Value": ["price", "cost", "value", "worth", "expensive", "cheap", "free", "subscription",
                     "pricing", "pay", "purchase", "premium", "plan", "roi"],
            
            "User Experience": ["ux", "ui", "interface", "design", "easy", "intuitive", "simple", 
                               "user-friendly", "learn", "navigate", "layout", "theme", "dark mode"],
            
            "Reliability": ["reliable", "stable", "crash", "bug", "glitch", "consistent", "issue", 
                           "problem", "fails", "breaks", "broken", "depend", "trust", "robust"],
            
            "Performance Speed": ["fast", "slow", "speed", "quick", "lag", "performance", "responsive", 
                                 "instant", "efficient", "latency", "delay", "wait", "snappy"],
            
            "Code Completion": ["autocomplete", "completion", "suggestion", "predict", "code generation", 
                               "generates", "writing", "finish", "complete", "fill", "writes"],
            
            "Customization": ["custom", "settings", "configure", "options", "preferences", "personalize", 
                             "adjust", "modify", "tailor", "adapt", "tweak", "control", "flexibility"]
        }
        
        # score each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            # count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > 0:
                category_scores[category] = matches
        
        # If no category was detected, use a fallback
        if not category_scores:
            return "General Feedback"
            
        # Return the category with the highest score
        return max(category_scores.items(), key=lambda x: x[1])[0]
    
    df['category'] = df.apply(lambda row: classify_into_category(row['text'], row['sentiment_score']), axis=1)
    
    positive_df = df[df['sentiment_score'] > 0.1].copy()
    negative_df = df[df['sentiment_score'] < -0.1].copy()
    
    positive_counts = positive_df.groupby('category').size().reset_index(name='count')
    negative_counts = negative_df.groupby('category').size().reset_index(name='count')
    
    positive_categories = {}
    for _, row in positive_counts.iterrows():
        category = row['category']
        count = row['count']
        positive_categories[category] = {
            'count': count,
            'posts': positive_df[positive_df['category'] == category].sort_values('sentiment_score', ascending=False)
        }
    
    negative_categories = {}
    for _, row in negative_counts.iterrows():
        category = row['category']
        count = row['count']
        negative_categories[category] = {
            'count': count,
            'posts': negative_df[negative_df['category'] == category].sort_values('sentiment_score', ascending=True)
        }
    
    top_positive = sorted(positive_categories.items(), key=lambda x: x[1]['count'], reverse=True)
    top_negative = sorted(negative_categories.items(), key=lambda x: x[1]['count'], reverse=True)
    
    used_categories = set()
    final_positives = []
    final_negatives = []
    
    # first pass - add categories with the most mentions
    for category, data in top_positive:
        if len(final_positives) < 3 and category not in used_categories:
            final_positives.append((category, data))
            used_categories.add(category)
    
    for category, data in top_negative:
        if len(final_negatives) < 3 and category not in used_categories:
            final_negatives.append((category, data))
            used_categories.add(category)
    
    # if we don't have enough categories use leftover ones
    remaining_positives = [item for item in top_positive if item[0] not in used_categories]
    remaining_negatives = [item for item in top_negative if item[0] not in used_categories]
    
    while len(final_positives) < 3 and remaining_positives:
        category, data = remaining_positives.pop(0)
        final_positives.append((category, data))
    
    while len(final_negatives) < 3 and remaining_negatives:
        category, data = remaining_negatives.pop(0)
        final_negatives.append((category, data))
    
    # hard code for now
    category_summaries = {
        "Integration": {
            "positive": "Users praise how well the tool integrates with their development environments, with seamless setup and workflow.",
            "negative": "Users report difficulties with IDE integration, including installation problems and compatibility issues."
        },
        "Documentation": {
            "positive": "Documentation is well-received, with users finding examples and guides helpful for getting started.",
            "negative": "Documentation needs improvement, with users requesting more examples, clearer explanations, or better organized resources."
        },
        "Code Quality": {
            "positive": "Users are satisfied with the quality of code suggestions, noting fewer errors and better formatting.",
            "negative": "Code quality is a concern, with users reporting errors, poor formatting, or incorrect suggestions."
        },
        "Context Understanding": {
            "positive": "The tool's ability to understand code context is appreciated, with relevant suggestions that match user intent.",
            "negative": "Users find that the tool struggles to understand context, often providing irrelevant suggestions."
        },
        "Value": {
            "positive": "Users feel they're getting good value, with the tool's benefits justifying its cost.",
            "negative": "Pricing is a concern, with users questioning whether the tool provides enough value for its cost."
        },
        "User Experience": {
            "positive": "The interface is intuitive and user-friendly, making the tool easy to learn and enjoyable to use.",
            "negative": "The user interface could be improved with more intuitive design, better organization, or clearer options."
        },
        "Reliability": {
            "positive": "Users find the tool reliable and stable, with consistent performance across different usage scenarios.",
            "negative": "Reliability issues have been reported, including crashes, inconsistent behavior, or unexpected errors."
        },
        "Performance Speed": {
            "positive": "Users appreciate the tool's speed and responsiveness, with minimal lag even during complex tasks.",
            "negative": "Performance speed is an issue, with users reporting delays, lag, or slow response times."
        },
        "Code Completion": {
            "positive": "Code completion features are praised for their accuracy and helpfulness in speeding up development.",
            "negative": "Code completion could be improved, with users wanting more accurate or context-aware suggestions."
        },
        "Customization": {
            "positive": "The ability to customize settings and preferences is valued, giving users control over their experience.",
            "negative": "Users want more customization options to tailor the tool to their specific workflows and preferences."
        },
        "General Feedback": {
            "positive": "Overall feedback is positive, with users expressing satisfaction with the tool's functionality.",
            "negative": "General feedback suggests areas for improvement across various aspects of the tool."
        }
    }
    
    # display the insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div style="background-color: rgba(0, 128, 0, 0.1); padding: 15px; border-radius: 5px; border-left: 5px solid green;">', unsafe_allow_html=True)
        st.markdown("### 🌟 What's Working Well")
        
        for i, (category, data) in enumerate(final_positives):
            count = data['count']
            example = data['posts'].iloc[0] if not data['posts'].empty else None
            
            st.markdown(f"**{i+1}. {category} ({count} mentions)**")
            
            st.markdown(category_summaries[category]["positive"])
            
            if example is not None:
                with st.expander("See example"):
                    st.markdown(f"**Source:** {example['source']} | **Score:** {example['sentiment_score']:.2f}")
                    st.markdown(f"**Comment:** {str(example['text'])[:300]}...")
                    if 'url' in example and pd.notna(example['url']):
                        st.markdown(f"**Link:** [View original]({example['url']})")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="background-color: rgba(255, 0, 0, 0.1); padding: 15px; border-radius: 5px; border-left: 5px solid red;">', unsafe_allow_html=True)
        st.markdown("### 🔨 Areas for Improvement")
        
        for i, (category, data) in enumerate(final_negatives):
            count = data['count']
            example = data['posts'].iloc[0] if not data['posts'].empty else None
            
            st.markdown(f"**{i+1}. {category} ({count} mentions)**")
            
            st.markdown(category_summaries[category]["negative"])
            
            if example is not None:
                with st.expander("See example"):
                    st.markdown(f"**Source:** {example['source']} | **Score:** {example['sentiment_score']:.2f}")
                    st.markdown(f"**Comment:** {str(example['text'])[:300]}...")
                    if 'url' in example and pd.notna(example['url']):
                        st.markdown(f"**Link:** [View original]({example['url']})")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("ⓘ About this analysis"):
        st.markdown("""
        **Methodology:**
        - Comments are categorized into specific feature areas based on content analysis
        - Each category is evaluated independently for positive and negative sentiment
        - Categories are placed in either "What's Working Well" or "Areas for Improvement" based on overall sentiment
        - Examples show representative comments for each category
        - Categories are kept distinct between positive and negative sections for clearer insights
        """)

def main():
    render_header()
    df = load_data(competitor="codeium")
    if df is None or len(df) == 0:
        st.error("No data available for Codeium. Please run the data collection scripts first.")
        st.info("Check the README.md file for instructions on how to collect data.")
        return
    filtered_df = render_filters(df)
    refresh_time_file = os.path.join(DATA_DIR, "last_refresh_time.txt")
    if os.path.exists(refresh_time_file):
        with open(refresh_time_file, "r") as f:
            last_refresh = f.read().strip()
        st.sidebar.info(f"Data last updated: {last_refresh}")
    else:
        data_date = filtered_df['created_at'].max().strftime("%Y-%m-%d %H:%M:%S")
        st.sidebar.info(f"Data last updated: {data_date}")
    
    if st.sidebar.button("Refresh Data"):
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        status_placeholder.info("Starting data collection...")
        progress_bar = progress_placeholder.progress(0)
        
        try:
            import subprocess
            import time
            import re
            
            # script paths
            scripts = [
                ("Reddit Scraper", "../scripts/scrape_reddit.py"),
                ("Sentiment Analysis", "../scripts/sentiment_analysis.py"),
                ("Data Aggregation", "../scripts/aggregate_data.py")
            ]
            
            # total number of steps for progress calculation
            total_steps = len(scripts)
            current_step = 0
            
            for script_name, script_path in scripts:
                # update status to show current script
                status_placeholder.info(f"Running {script_name}...")
                
                # run script with output capture
                process = subprocess.Popen(
                    ["python", script_path], 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # process output in real-time to show progress
                for line in iter(process.stdout.readline, ''):
                    # display subreddit info if present
                    if "Searching in r/" in line:
                        # get subreddit name
                        subreddit_match = re.search(r'Searching in r/(\w+)', line)
                        if subreddit_match:
                            subreddit = subreddit_match.group(1)
                            status_placeholder.info(f"Analyzing subreddit: r/{subreddit}")
                    
                    # Update status for other progress indicators
                    if "Collected" in line or "Processing" in line:
                        status_placeholder.info(line.strip())

                process.stdout.close()
                process.wait()

                current_step += 1
                progress_bar.progress(current_step / total_steps)

            update_refresh_timestamp()
            status_placeholder.success("Data collection completed successfully!")
            progress_bar.progress(1.0)

            time.sleep(2)
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            status_placeholder.error(f"Error collecting data: {str(e)}")
    
    render_metrics(filtered_df)
    render_time_series(filtered_df)
    render_source_distribution(filtered_df)
    render_feature_sentiment_trends(filtered_df)
    render_product_insights(filtered_df)
    render_engineer_insights(filtered_df)
    render_codeium_sentiment_overview(filtered_df)
    render_competitor_section(filtered_df)
    render_recent_mentions(filtered_df)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Sentiment Aggregator v1.0")

if __name__ == "__main__":
    main()