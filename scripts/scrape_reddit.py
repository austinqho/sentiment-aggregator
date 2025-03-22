import os
import json
import time
import datetime
import pandas as pd
from dotenv import load_dotenv
import praw

load_dotenv()

# directory for storing data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Reddit API 
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'script:sentiment-aggregator:v1.0 (by /u/YOUR_USERNAME)')

# debugging
if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
    print("Error: Reddit API credentials not found in environment variables.")
    print("Create a .env file with REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET.")
    exit(1)

def get_reddit_client():
    """Initialize and return Reddit API client"""
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    return reddit

def search_reddit_posts(reddit, keywords, subreddits, time_filter='week', limit=100):
    """Search for Reddit posts containing keywords"""
    posts = []
    
    query = ' OR '.join([f'"{keyword}"' for keyword in keywords])
    
    try:
        total_subreddits = len(subreddits)
        for idx, subreddit_name in enumerate(subreddits):
            print(f"Searching in r/{subreddit_name}... ({idx+1}/{total_subreddits})")
            
            try:
                subreddit = reddit.subreddit(subreddit_name)
                
                post_count = 0
                comment_count = 0
                
                try:
                    for submission in subreddit.search(query, limit=limit, time_filter=time_filter):
                        # check if any keyword is in title or selftext
                        title_text = submission.title.lower() if submission.title else ""
                        self_text = submission.selftext.lower() if submission.selftext else ""
                        combined_text = title_text + " " + self_text
                        
                        if any(keyword.lower() in combined_text for keyword in keywords):
                            post_data = {
                                'id': submission.id,
                                'title': submission.title,
                                'text': submission.selftext,
                                'created_at': datetime.datetime.fromtimestamp(submission.created_utc),
                                'score': submission.score,
                                'num_comments': submission.num_comments,
                                'upvote_ratio': submission.upvote_ratio,
                                'subreddit': submission.subreddit.display_name,
                                'author': str(submission.author) if submission.author else '[deleted]',
                                'is_post': True,
                                'source': f'Reddit (r/{submission.subreddit.display_name})',
                                'url': f"https://www.reddit.com{submission.permalink}"
                            }
                            posts.append(post_data)
                            post_count += 1
                        
                        # also get comments for this submission with improved error handling
                        try:
                            submission.comments.replace_more(limit=0)  # only get top-level comments
                            for comment in submission.comments.list():
                                comment_text = comment.body.lower() if comment.body else ""
                                if any(keyword.lower() in comment_text for keyword in keywords):
                                    comment_data = {
                                        'id': comment.id,
                                        'title': '',  # comments don't have titles
                                        'text': comment.body,
                                        'created_at': datetime.datetime.fromtimestamp(comment.created_utc),
                                        'score': comment.score,
                                        'num_comments': 0,
                                        'upvote_ratio': 0,
                                        'subreddit': comment.subreddit.display_name,
                                        'author': str(comment.author) if comment.author else '[deleted]',
                                        'is_post': False,
                                        'source': f'Reddit (r/{comment.subreddit.display_name})',
                                        'url': f"https://www.reddit.com{comment.permalink}"
                                    }
                                    posts.append(comment_data)
                                    comment_count += 1
                        except Exception as e:
                            print(f"  ⚠️ Error processing comments for submission {submission.id}: {e}")
                except Exception as e:
                    print(f"  ⚠️ Error searching submissions in r/{subreddit_name}: {e}")
                
                print(f"Collected {post_count} posts and {comment_count} comments from r/{subreddit_name}")
            except Exception as e:
                print(f"  ⚠️ Error accessing subreddit r/{subreddit_name}: {e}")
            
            # could add an add exponential backoff here
            time.sleep(2)
    
    except Exception as e:
        print(f"Error searching Reddit: {e}")
    
    return posts
# devin got so little mentions that we need to search more broadly
def search_for_devin(reddit):
    """Special function to search for Devin AI content more broadly"""
    print("\n--- Special search for Devin AI content ---")
    devin_posts = []
    
    # add more search queries that might find Devin mentions
    search_queries = [
        "Devin AI autonomous developer",
        "Cognition Labs Devin",
        "AI that can code entire projects",
        "autonomous AI developer",
        "AI developer assistant full projects",
        "AI developer agent"
    ]
    
    # additional subreddits focused on AI and technology news
    broader_subreddits = [
        'artificial', 'AINews', 'technews', 'tech', 'agi', 'machinelearning',
        'deeplearning', 'computervision', 'nlproc', 'LanguageTechnology',
        'generative', 'AIGeneratedContent'
    ]
    
    # try broader search queries in tech/AI-focused subreddits
    for query in search_queries:
        for subreddit_name in broader_subreddits:
            try:
                print(f"Searching for Devin in r/{subreddit_name} with query: {query}")
                subreddit = reddit.subreddit(subreddit_name)
                
                # get submissions matching query
                for submission in subreddit.search(query, limit=25, time_filter="year"):
                    # check if mentions Devin
                    title_text = submission.title.lower()
                    self_text = submission.selftext.lower()
                    combined_text = title_text + " " + self_text
                    
                    if "devin" in combined_text or "cognition labs" in combined_text:
                        post_data = {
                            'id': submission.id,
                            'title': submission.title,
                            'text': submission.selftext,
                            'created_at': datetime.datetime.fromtimestamp(submission.created_utc),
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'upvote_ratio': submission.upvote_ratio,
                            'subreddit': submission.subreddit.display_name,
                            'author': str(submission.author) if submission.author else '[deleted]',
                            'is_post': True,
                            'source': f'Reddit (r/{submission.subreddit.display_name})',
                            'url': f"https://www.reddit.com{submission.permalink}"
                        }
                        devin_posts.append(post_data)
                        
                        # Also check comments on this submission for Devin mentions
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments.list():
                            comment_text = comment.body.lower() if comment.body else ""
                            if "devin" in comment_text or "cognition labs" in comment_text:
                                comment_data = {
                                    'id': comment.id,
                                    'title': '',
                                    'text': comment.body,
                                    'created_at': datetime.datetime.fromtimestamp(comment.created_utc),
                                    'score': comment.score,
                                    'num_comments': 0,
                                    'upvote_ratio': 0,
                                    'subreddit': comment.subreddit.display_name,
                                    'author': str(comment.author) if comment.author else '[deleted]',
                                    'is_post': False,
                                    'source': f'Reddit (r/{comment.subreddit.display_name})',
                                    'url': f"https://www.reddit.com{comment.permalink}"
                                }
                                devin_posts.append(comment_data)
                
                # rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error searching r/{subreddit_name} for Devin: {e}")
    
    print(f"Found {len(devin_posts)} additional posts/comments mentioning Devin")
    return devin_posts

def save_reddit_data_to_csv(posts, filename):
    """Save the collected Reddit data to a CSV file"""
    if not posts:
        print(f"No Reddit posts to save for {filename}.")
        return
    
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    filepath = os.path.join(DATA_DIR, f"{filename}_{today}.csv")
    
    # convert to DataFrame and save
    df = pd.DataFrame(posts)
    
    # if file exists, append; otherwise create new
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        # convert date column back to datetime for proper merging
        existing_df['created_at'] = pd.to_datetime(existing_df['created_at'])
        # remove duplicates based on post/comment id
        combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['id'])
        combined_df.to_csv(filepath, index=False)
        print(f"Updated {filepath} with {len(df)} new Reddit items. Total: {len(combined_df)}")
    else:
        df.to_csv(filepath, index=False)
        print(f"Created {filepath} with {len(df)} Reddit items.")

def main():
    reddit = get_reddit_client()
    
    # hard code for now
    competitor_data = {
        "codeium": {
            "keywords": ['windsurf', 'windsurfing', 'codeium'],
            "subreddits": [
                'programming', 'webdev', 'devops', 'javascript', 'Python', 'coding', 
                'learnprogramming', 'AskProgramming', 'codeium', 'vscode', 'vim', 'emacs', 
                'IDEs', 'ChatGPTCoding', 'WindsurfAI'
            ]
        },
        "copilot": {
            "keywords": ['github copilot', 'copilot', 'copilot x', 'microsoft copilot', 'gh copilot'],
            "subreddits": [
                'github', 'vscode', 'programming', 'coding', 'learnprogramming',
                'MachineLearning', 'webdev', 'javascript', 'computerscience', 
                'artificial', 'VSCodeExtensions', 'code'
            ]
        },
        "cursor": {
            "keywords": ['cursor', 'cursor.so', 'cursor ai', 'cursor editor', 'cursor.sh'],
            "subreddits": [
                'programming', 'webdev', 'javascript', 'Python', 'coding',
                'learnprogramming', 'AskProgramming', 'developertools', 'codeeditors', 'cursor'
            ]
        },
        "lovable": {
            "keywords": ['lovable', 'lovable.ai', 'lovable editor', 'lovable code', 'lovable assistant'],
            "subreddits": [
                'programming', 'webdev', 'javascript', 'Python', 'coding',
                'learnprogramming', 'AskProgramming', 'Supabase', 'codeeditors', 'developertools'
            ]
        },
        "devin": {
            "keywords": [
                'devin ai', 'devin', 'cognition labs', 'cognition ai', 'dev assistant',
                'ai developer', 'ai coding assistant', 'autonomous developer', 'autonomous agent'
            ],
            "subreddits": [
                'programming', 'learnprogramming', 'MachineLearning', 'coding',
                'Python', 'javascript', 'webdev', 'computerscience', 'ArtificialIntelligence',
                'singularity', 'technology', 'futurology', 'softwareengineering', 'artificialintelligence',
                'OpenAI', 'MLQuestions', 'AIdev', 'ChatGPT', 'GPT', 'llm'
            ]
        },
        "cody": {
            "keywords": ['sourcegraph cody', 'cody ai', 'sourcegraph', 'cody assistant', 'cody code'],
            "subreddits": [
                'programming', 'webdev', 'javascript', 'Python', 'coding',
                'sourcegraph', 'AskProgramming', 'vscode', 'codeeditors', 'developertools'
            ]
        }
    }
    
    competitors_to_collect = ["copilot", "cursor", "lovable", "devin", "cody"]
    
    posts_by_competitor = {}
    
    for competitor in competitors_to_collect:
        if competitor in competitor_data:
            print(f"\n--- Collecting {competitor.capitalize()} mentions ---")
            print(f"Keywords: {competitor_data[competitor]['keywords']}")
            print(f"Subreddits: {competitor_data[competitor]['subreddits']}")
            
            posts = search_reddit_posts(
                reddit, 
                competitor_data[competitor]["keywords"], 
                competitor_data[competitor]["subreddits"], 
                time_filter="month",  
                limit=100  
            )
            posts_by_competitor[competitor] = posts
            
            print(f"Found {len(posts)} posts/comments for {competitor}")
            # if no posts
            if len(posts) == 0:
                print(f"⚠️ No results found for {competitor}! Check keywords and subreddits.")
            save_reddit_data_to_csv(posts, f"{competitor}_mentions")
            
            print(f"Collected {len(posts)} Reddit posts/comments mentioning {competitor}")
        else:
            print(f"⚠️ No configuration found for {competitor}")
    
    # handling for Devin - run the additional search
    if "devin" in competitors_to_collect:
        # run special Devin search to find more content
        devin_special_posts = search_for_devin(reddit)
        if devin_special_posts:
            # combine with any previously found posts
            if "devin" in posts_by_competitor:
                posts_by_competitor["devin"].extend(devin_special_posts)
            else:
                posts_by_competitor["devin"] = devin_special_posts
            print(f"Total Devin mentions after special search: {len(posts_by_competitor.get('devin', []))}")
            save_reddit_data_to_csv(posts_by_competitor.get("devin", []), "devin_mentions")
    
    # also save the original codeium data for backward compatibility
    print("\n--- Collecting data for main dashboard ---")
    codeium_posts = search_reddit_posts(
        reddit, 
        competitor_data["codeium"]["keywords"], 
        competitor_data["codeium"]["subreddits"]
    )
    save_reddit_data_to_csv(codeium_posts, "reddit_mentions")
    
    print("\nData collection complete!")

if __name__ == "__main__":
    main()