import os
import glob
import pandas as pd
import datetime
import json

# directory for storing data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)
STANDARD_COLUMNS = [
    'id', 'text', 'created_at', 'source', 'url', 
    'sentiment_label', 'sentiment_score',
    'author', 'platform_specific_metrics', 'keywords_found'
]

def standardize_twitter_data(df):
    """Standardize Twitter data to the common format"""
    if df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    
    # create a new dataframe with the standard columns
    standardized = pd.DataFrame()
    
    # map the twitter-specific columns to the standard ones
    standardized['id'] = df['id']
    standardized['text'] = df['text']
    standardized['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)
    standardized['source'] = df['source']
    standardized['url'] = df['url']
    
    # add sentiment columns if they exist
    if 'sentiment_label' in df.columns:
        standardized['sentiment_label'] = df['sentiment_label']
    else:
        standardized['sentiment_label'] = 'Unknown'
    
    if 'sentiment_score' in df.columns:
        standardized['sentiment_score'] = df['sentiment_score']
    else:
        standardized['sentiment_score'] = 0.0
    
    # author info
    standardized['author'] = df['author_username'].apply(lambda x: f"@{x}" if pd.notna(x) else "Unknown")
    # platform-specific metrics
    standardized['platform_specific_metrics'] = df.apply(
        lambda row: json.dumps({
            'retweet_count': int(row.get('retweet_count', 0)),
            'like_count': int(row.get('like_count', 0)) if 'like_count' in row else int(row.get('favorite_count', 0)),
            'reply_count': int(row.get('reply_count', 0)) if 'reply_count' in row else 0,
            'verified_author': bool(row.get('author_verified', False))
        }),
        axis=1
    )
    # extract keywords found
    def extract_keywords(text):
        keywords = []
        if 'windsurf' in text.lower() or 'windsurfing' in text.lower():
            keywords.append('windsurf')
        if 'codeium' in text.lower():
            keywords.append('codeium')
        return keywords
    
    standardized['keywords_found'] = df['text'].apply(
        lambda x: json.dumps(extract_keywords(x)) if pd.notna(x) else json.dumps([])
    )
    return standardized

#same as above but for reddit
def standardize_reddit_data(df):
    """Standardize Reddit data to the common format"""
    if df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    
    standardized = pd.DataFrame()
    
    standardized['id'] = df['id']
    
    def combine_title_text(row):
        if row.get('is_post', False) and pd.notna(row.get('title', '')):
            return f"{row['title']}\n\n{row.get('text', '')}"
        return row.get('text', '')
    
    standardized['text'] = df.apply(combine_title_text, axis=1)
    standardized['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)
    standardized['source'] = df.apply(
        lambda row: f"Reddit (r/{row['subreddit']})" if pd.notna(row.get('subreddit', None)) else 'Reddit',
        axis=1
    )
    standardized['url'] = df['url']
    
    if 'sentiment_label' in df.columns:
        standardized['sentiment_label'] = df['sentiment_label']
    else:
        standardized['sentiment_label'] = 'Unknown'
    
    if 'sentiment_score' in df.columns:
        standardized['sentiment_score'] = df['sentiment_score']
    else:
        standardized['sentiment_score'] = 0.0
    
    standardized['author'] = df['author'].apply(lambda x: f"u/{x}" if pd.notna(x) and x != '[deleted]' else "Anonymous")
    
    standardized['platform_specific_metrics'] = df.apply(
        lambda row: json.dumps({
            'score': int(row.get('score', 0)),
            'num_comments': int(row.get('num_comments', 0)),
            'upvote_ratio': float(row.get('upvote_ratio', 0)),
            'is_post': bool(row.get('is_post', False))
        }),
        axis=1
    )
    
    def extract_keywords(text):
        if not pd.notna(text):
            return []
        
        keywords = []
        if 'windsurf' in text.lower() or 'windsurfing' in text.lower():
            keywords.append('windsurf')
        if 'codeium' in text.lower():
            keywords.append('codeium')
        return keywords
    
    standardized['keywords_found'] = df['text'].apply(
        lambda x: json.dumps(extract_keywords(x)) if pd.notna(x) else json.dumps([])
    )
    
    return standardized

def aggregate_data():
    """Aggregate all data from various sources"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    competitors = ["copilot", "cursor", "lovable", "devin", "cody", "codeium"]
    
    all_data = []
    item_counts = {}
    
    # process all competitor files
    for competitor in competitors:
        # get files with sentiment analysis
        sentiment_files = glob.glob(os.path.join(DATA_DIR, f"{competitor}_mentions_{today}_with_sentiment.csv"))
        
        # if no file from today, try finding the most recent one
        if not sentiment_files:
            sentiment_files = glob.glob(os.path.join(DATA_DIR, f"{competitor}_mentions_*_with_sentiment.csv"))
        
        if sentiment_files:
            latest_file = sorted(sentiment_files)[-1]
            print(f"Processing {competitor} data from {os.path.basename(latest_file)}...")
            df = pd.read_csv(latest_file)
            item_count = len(df)
            item_counts[competitor] = item_count
            
            # standardize based on source
            if "reddit" in latest_file:
                standardized = standardize_reddit_data(df)
            elif "twitter" in latest_file:
                standardized = standardize_twitter_data(df)
            else:
                # default to Reddit standardization for now
                standardized = standardize_reddit_data(df)
            
            all_data.append(standardized)
    
    # also process regular Reddit mentions for the dashboard
    reddit_files = glob.glob(os.path.join(DATA_DIR, f"reddit_mentions_{today}_with_sentiment.csv"))
    if not reddit_files:
        reddit_files = glob.glob(os.path.join(DATA_DIR, f"reddit_mentions_*_with_sentiment.csv"))
    
    if reddit_files:
        latest_file = sorted(reddit_files)[-1]
        print(f"Processing main dashboard data from {os.path.basename(latest_file)}...")
        df = pd.read_csv(latest_file)
        item_counts["dashboard"] = len(df)
        standardized = standardize_reddit_data(df)
        all_data.append(standardized)
    
    # combine all data
    if not all_data:
        print("No data found to aggregate.")
        return None
    
    print("Combining all data sources...")
    aggregated_df = pd.concat(all_data, ignore_index=True)
    total_items = len(aggregated_df)
    
    # sort by newest first
    print("Sorting by date (newest first)...")
    aggregated_df = aggregated_df.sort_values('created_at', ascending=False)
    
    # summary statistics
    print("\nData Summary:")
    for source, count in item_counts.items():
        print(f"{source.capitalize()} items: {count}")
    print(f"Total items after aggregation: {total_items}")
    
    # count by sentiment
    sentiment_counts = aggregated_df['sentiment_label'].value_counts()
    print("\nSentiment distribution:")
    for label, count in sentiment_counts.items():
        percentage = (count / total_items) * 100
        print(f"  {label}: {count} items ({percentage:.1f}%)")
    
    # save the aggregated data
    output_path = os.path.join(DATA_DIR, f"aggregated_data_{today}.csv")
    aggregated_df.to_csv(output_path, index=False)
    print(f"Aggregated data saved to {output_path}")
    
    # create a JSON version for the dashboard
    json_path = os.path.join(DATA_DIR, f"aggregated_data_{today}.json")
    
    # convert datetime to string for JSON serialization
    print("Creating JSON version of the data...")
    agg_json = aggregated_df.copy()
    agg_json['created_at'] = agg_json['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    agg_json.to_json(json_path, orient='records', date_format='iso')
    print(f"JSON data saved to {json_path}")
    
    return aggregated_df

def main():
    aggregate_data()

if __name__ == "__main__":
    main()