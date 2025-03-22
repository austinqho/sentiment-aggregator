
import os
import json
import time
import datetime
import pandas as pd
from dotenv import load_dotenv
import tweepy
# hit rate limits on this 
# load environment variables (API keys, etc.)
load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET')

if not TWITTER_API_KEY or not TWITTER_API_SECRET:
    print("Error: Twitter API credentials not found in environment variables.")
    print("Create a .env file with TWITTER_API_KEY and TWITTER_API_SECRET.")
    exit(1)

def get_twitter_client():
    """Initialize and return Tweepy client"""
    if TWITTER_BEARER_TOKEN:
        # For v2 API
        client = tweepy.Client(
            bearer_token=TWITTER_BEARER_TOKEN,
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_API_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_SECRET
        )
        return client
    else:
        # For v1.1 API
        auth = tweepy.OAuth1UserHandler(
            TWITTER_API_KEY, TWITTER_API_SECRET,
            TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
        )
        api = tweepy.API(auth)
        return api

def search_tweets_v2(client, query, max_results=100):
    """Search tweets using Twitter API v2"""
    tweets = []
    
    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=['created_at', 'public_metrics', 'author_id', 'lang'],
            user_fields=['name', 'username', 'profile_image_url', 'verified'],
            expansions=['author_id']
        )
        
        if response.data:
            users = {user.id: user for user in response.includes['users']}
            
            for tweet in response.data:
                author = users[tweet.author_id]
                tweets.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'retweet_count': tweet.public_metrics['retweet_count'],
                    'like_count': tweet.public_metrics['like_count'],
                    'reply_count': tweet.public_metrics['reply_count'],
                    'quote_count': tweet.public_metrics['quote_count'],
                    'author_id': tweet.author_id,
                    'author_name': author.name,
                    'author_username': author.username,
                    'author_verified': author.verified,
                    'language': tweet.lang,
                    'source': 'Twitter',
                    'url': f"https://twitter.com/{author.username}/status/{tweet.id}"
                })
    
    except Exception as e:
        print(f"Error searching tweets: {e}")
    
    return tweets

def search_tweets_v1(api, query, max_results=100):
    """Search tweets using Twitter API v1.1 (fallback)"""
    tweets = []
    
    try:
        search_results = api.search_tweets(
            q=query,
            count=max_results,
            tweet_mode='extended',
            lang='en'
        )
        
        for tweet in search_results:
            tweets.append({
                'id': tweet.id_str,
                'text': tweet.full_text if hasattr(tweet, 'full_text') else tweet.text,
                'created_at': tweet.created_at,
                'retweet_count': tweet.retweet_count,
                'favorite_count': tweet.favorite_count,
                'author_id': tweet.user.id_str,
                'author_name': tweet.user.name,
                'author_username': tweet.user.screen_name,
                'author_verified': tweet.user.verified,
                'language': tweet.lang,
                'source': 'Twitter',
                'url': f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}"
            })
    
    except Exception as e:
        print(f"Error searching tweets: {e}")
    
    return tweets

def save_tweets_to_csv(tweets, filename):
    """Save the collected tweets to a CSV file"""
    if not tweets:
        print("No tweets to save.")
        return
    
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    filepath = os.path.join(DATA_DIR, f"{filename}_{today}.csv")
    
    df = pd.DataFrame(tweets)
    
    # if file exists, append; otherwise create new
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        # remove duplicates based on tweet id
        combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['id'])
        combined_df.to_csv(filepath, index=False)
        print(f"Updated {filepath} with {len(df)} new tweets. Total: {len(combined_df)}")
    else:
        df.to_csv(filepath, index=False)
        print(f"Created {filepath} with {len(df)} tweets.")

def main():
    client = get_twitter_client()
    
    query = '(windsurf) (codeium)'
    
    if isinstance(client, tweepy.Client):
        # use v2 API
        print("Using Twitter API v2")
        tweets = search_tweets_v2(client, query)
    else:
        # use v1.1 API
        print("Using Twitter API v1.1")
        tweets = search_tweets_v1(client, query)
    
    # save tweets
    save_tweets_to_csv(tweets, 'twitter_mentions')
    
    print(f"Collected {len(tweets)} tweets mentioning 'windsurf' and 'codeium'")

if __name__ == "__main__":
    main()