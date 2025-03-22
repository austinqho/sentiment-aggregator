#!/bin/bash

# Set the base directory to where the script is located
BASEDIR="$(dirname "$(realpath "$0")")"

# Navigate to the project directory
cd "$BASEDIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment
source venv/bin/activate

# Set timestamp for logging
timestamp=$(date "+%Y-%m-%d %H:%M:%S")
echo "[$timestamp] Starting data update..." >> logs/update.log

# Run the data pipeline with error handling
echo "[$timestamp] Running Twitter scraper..." >> logs/update.log
python scripts/scrape_twitter.py >> logs/update.log 2>&1
if [ $? -ne 0 ]; then
  echo "[$timestamp] ERROR: Twitter scraper failed" >> logs/update.log
fi

echo "[$timestamp] Running Reddit scraper..." >> logs/update.log
python scripts/scrape_reddit.py >> logs/update.log 2>&1
if [ $? -ne 0 ]; then
  echo "[$timestamp] ERROR: Reddit scraper failed" >> logs/update.log
fi

echo "[$timestamp] Running sentiment analysis..." >> logs/update.log
python scripts/sentiment_analysis.py >> logs/update.log 2>&1
if [ $? -ne 0 ]; then
  echo "[$timestamp] ERROR: Sentiment analysis failed" >> logs/update.log
fi

echo "[$timestamp] Aggregating data..." >> logs/update.log
python scripts/aggregate_data.py >> logs/update.log 2>&1
if [ $? -ne 0 ]; then
  echo "[$timestamp] ERROR: Data aggregation failed" >> logs/update.log
fi

echo "[$timestamp] Data update completed." >> logs/update.log