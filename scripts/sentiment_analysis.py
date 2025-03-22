
import os
import glob
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import datetime
import re

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')

# directory for storing data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def clean_text(text):
    """Clean and preprocess text for sentiment analysis"""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs, mentions, hashtag symbols, special chars and white space
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_vader_sentiment(text):
    """Get sentiment scores using VADER"""
    if not text:
        return {"compound": 0, "pos": 0, "neu": 0, "neg": 0}
    
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)

def get_textblob_sentiment(text):
    """Get sentiment scores using TextBlob"""
    if not text:
        return {"polarity": 0, "subjectivity": 0}
    
    analysis = TextBlob(text)
    return {
        "polarity": analysis.sentiment.polarity,
        "subjectivity": analysis.sentiment.subjectivity
    }

def determine_sentiment_label(vader_score, textblob_score, text=""):
    """Determine overall sentiment label based on combined scores and text analysis"""
    compound = vader_score["compound"]
    polarity = textblob_score["polarity"]
    
    # calculate weighted average
    weighted_score = (compound * 0.7) + (polarity * 0.3)
    
    # adjust for comparative statements
    text = text.lower() if isinstance(text, str) else ""
    
    # check for specific comparison patterns that might be misinterpreted
    comparison_adjustments = [
        ("better than", 0.3),           # pos comparison
        ("worse than", -0.3),           # neg comparison
        ("in comparison to", 0.0),      # neutral comparison indicator
        ("compared to", 0.0),           # neutral comparison indicator
        ("unlike", 0.0),                # contrast indicator
    ]
    
    for phrase, adjustment in comparison_adjustments:
        if phrase in text:
            # see if our product is being favorably compared
            our_products = ["windsurf", "codeium"]
            competitors = ["copilot", "claude", "gpt", "tabnine", "kite"]
            
            # case when our product better than competitor
            for our in our_products:
                for comp in competitors:
                    if our in text and comp in text:
                        if phrase == "better than" and text.find(our) < text.find(comp):
                            weighted_score += 0.3  # Boost positive sentiment for our product
                        if phrase == "better than" and text.find(comp) < text.find(our):
                            weighted_score -= 0.2  # Negative comparison for our product
    
    # check for negations that might flip sentiment
    negations = ["not", "n't", "no", "never", "without"]
    if any(neg in text.split() for neg in negations):
        # when negations appear with positive words, sentiment is likely negative
        positive_words = ["good", "great", "excellent", "awesome", "like", "love"]
        if any(pos in text.split() for pos in positive_words):
            weighted_score -= 0.2
    
    # final sentiment classification with adjusted thresholds
    if weighted_score >= 0.05:
        return "Positive"
    elif weighted_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment_for_file(filepath):
    """Analyze sentiment for all content in a CSV file"""
    print(f"Analyzing sentiment for {filepath}...")
    
    # read file
    df = pd.read_csv(filepath)
    total_items = len(df)
    print(f"Found {total_items} items to analyze")
    
    # check if sentiment has already been analyzed
    if all(col in df.columns for col in ['sentiment_label', 'sentiment_score']):
        print(f"Sentiment already analyzed for {filepath}")
        return df
    
    # make a text column that combines title and text if applicable
    if 'title' in df.columns and 'text' in df.columns:
        print("Combining title and text fields...")
        df['combined_text'] = df.apply(
            lambda row: f"{row['title']} {row['text']}" if pd.notna(row['title']) else row['text'], 
            axis=1
        )
    else:
        # if only one text column exists
        text_col = 'text' if 'text' in df.columns else 'content'
        print(f"Using {text_col} field for sentiment analysis...")
        df['combined_text'] = df[text_col]
    
    # clean text
    print("Cleaning and preprocessing text...")
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # analyze sentiment with progress updates
    print("Running VADER sentiment analysis...")
    batch_size = max(1, total_items // 10)  
    
    # process in batches for progress reporting
    for i in range(0, total_items, batch_size):
        end_idx = min(i + batch_size, total_items)
        batch = df.iloc[i:end_idx]
        batch_texts = batch['cleaned_text'].tolist()
        vader_results = [get_vader_sentiment(text) for text in batch_texts]
        for j, vader_score in enumerate(vader_results):
            df.loc[i + j, 'vader_sentiment'] = str(vader_score)
            df.loc[i + j, 'vader_compound'] = vader_score['compound']
        
        # report progress
        progress_pct = min(100, int((end_idx / total_items) * 100))
        print(f"VADER analysis: {progress_pct}% complete ({end_idx}/{total_items} items)")
    
    print("Running TextBlob sentiment analysis...")
    for i in range(0, total_items, batch_size):
        end_idx = min(i + batch_size, total_items)
        batch = df.iloc[i:end_idx]
        batch_texts = batch['cleaned_text'].tolist()
        textblob_results = [get_textblob_sentiment(text) for text in batch_texts]
        for j, tb_score in enumerate(textblob_results):
            df.loc[i + j, 'textblob_sentiment'] = str(tb_score)
            df.loc[i + j, 'textblob_polarity'] = tb_score['polarity']
            df.loc[i + j, 'textblob_subjectivity'] = tb_score['subjectivity']
        
        progress_pct = min(100, int((end_idx / total_items) * 100))
        print(f"TextBlob analysis: {progress_pct}% complete ({end_idx}/{total_items} items)")
    
    # determine overall sentiment label
    print("Determining overall sentiment classification...")
    df['sentiment_label'] = df.apply(
        lambda row: determine_sentiment_label(eval(row['vader_sentiment']) if isinstance(row['vader_sentiment'], str) else row['vader_sentiment'], 
                                             eval(row['textblob_sentiment']) if isinstance(row['textblob_sentiment'], str) else row['textblob_sentiment'], 
                                             row['cleaned_text']), 
        axis=1
    )
    
    # use vader compound score as the main sentiment score
    df['sentiment_score'] = df['vader_compound']
    
    # get sentiment distribution
    sentiment_counts = df['sentiment_label'].value_counts()
    print(f"Sentiment distribution:")
    for label, count in sentiment_counts.items():
        percentage = (count / total_items) * 100
        print(f"  {label}: {count} items ({percentage:.1f}%)")
    
    # save the updated file
    output_filepath = filepath.replace('.csv', '_with_sentiment.csv')
    df.to_csv(output_filepath, index=False)
    print(f"Saved sentiment analysis to {output_filepath}")
    
    return df

def process_all_files():
    """Process all data files in the data directory"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    competitors = ["codeium", "claude", "xai", "gemini", "copilot", "cursor", "lovable", "devin", "cody"]
    competitors.append("reddit")  
    
    # process files for each competitor
    for competitor in competitors:
        filepath = os.path.join(DATA_DIR, f"{competitor}_mentions_{today}.csv")
        if os.path.exists(filepath):
            print(f"Processing sentiment for {competitor}...")
            analyze_sentiment_for_file(filepath)
        else:
            print(f"No data file found for {competitor}")

def main():
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    competitors = ["copilot", "cursor", "lovable", "devin", "cody", "codeium"]
    
    files_found = False
    
    for competitor in competitors:
        filepath = os.path.join(DATA_DIR, f"{competitor}_mentions_{today}.csv")
        if os.path.exists(filepath):
            files_found = True
            print(f"Processing sentiment for {competitor}...")
            analyze_sentiment_for_file(filepath)
        else:
            older_files = glob.glob(os.path.join(DATA_DIR, f"{competitor}_mentions_*.csv"))
            if older_files:
                latest_file = sorted(older_files)[-1]  # Get the most recent file
                files_found = True
                print(f"No file for today found for {competitor}. Processing {os.path.basename(latest_file)}...")
                analyze_sentiment_for_file(latest_file)
            else:
                print(f"⚠️ No data file found for {competitor}")
    
    # alert if no files were found at all
    if not files_found:
        print("⚠️ No data files found for any competitor. Please run the data collection scripts first.")
        
    # process regular reddit mentions for dashboard
    reddit_file = os.path.join(DATA_DIR, f"reddit_mentions_{today}.csv")
    if os.path.exists(reddit_file):
        print("Processing main dashboard data...")
        analyze_sentiment_for_file(reddit_file)

if __name__ == "__main__":
    main()