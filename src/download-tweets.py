import os
import json
import re
import sys

from twitter import Api

# #abortionrights #prolife #prochoice #AbortionIsHealthcare #AbortionIsHealthcare #antiabortion


PROLIFE_USERS = ['prolifecampaign','ProLifeAction','AntiAbortionAF','endthekillingTO','ProLifeAction']
PROCHOICE_USERS = ['prochoiceMT','Triangle4Choice','ProChoiceOH']
NEUTRAL_USERS = ['MovingMalteser', 'philtalkradio', '1800PetMeds', 'SMWikiOfficial', 'LitWorks', 'PoetrySociety']


twitter_oauth_access_token_url = 'https://api.twitter.com/oauth/access_token'
twitter_oauth_authorize_url = 'https://api.twitter.com/oauth/authorize'
twitter_api_base_url = 'https://api.twitter.com/1.1/'
twitter_oauth_request_token_url = 'https://api.twitter.com/oauth/request_token'
twitter_api_info_file = "../config/userinfo.txt"

tweets_search_file = '../datasets/tweets/search-tweets.txt'
tweets_stream_file = '../datasets/tweets/stream-tweets.txt'

# Retrieve Twitter API user info:
with open(twitter_api_info_file, 'r', encoding="utf-8") as f:
  twitter_info_lines = f.readlines()
  twitter_consumer_key = twitter_info_lines[0][:-1]
  twitter_consumer_secret = twitter_info_lines[1][:-1]
  twitter_access_token = twitter_info_lines[2][:-1]
  twitter_access_token_secret = twitter_info_lines[3][:-1]
  twitter_oauth_callback_url = twitter_info_lines[4][:-1]


api = Api(consumer_key=twitter_consumer_key,
          consumer_secret=twitter_consumer_secret,
          access_token_key=twitter_access_token,
          access_token_secret=twitter_access_token_secret,
		  tweet_mode='extended')
				  

# Retrieve the tweets of a user:
# "Important notice:  On June 19, 2019, we will begin limiting total GET requests to the v1.1
# /statuses/mentions_timeline and /statuses/user_timeline endpoints to 100,000 requests per day..."
# --> https://developer.twitter.com/en/docs/tweets/timelines/overview
def get_tweets(api=None, screen_name=None, count=200, exclude_replies=True, include_rts=False):
  timeline = api.GetUserTimeline(screen_name=screen_name, count=count, exclude_replies=exclude_replies, include_rts=include_rts)
  earliest_tweet = min(timeline, key=lambda x: x.id).id
  print("getting tweets before:", earliest_tweet)
  while True:
    tweets = api.GetUserTimeline(
      screen_name=screen_name, max_id=earliest_tweet, count=count, exclude_replies=exclude_replies, include_rts=include_rts
    )
    new_earliest = min(tweets, key=lambda x: x.id).id
    if not tweets or new_earliest == earliest_tweet:
      break
    else:
      earliest_tweet = new_earliest
      print("getting tweets before:", earliest_tweet)
      timeline += tweets
  return timeline
  

# Download the tweets from users:
def download_tweets(users_to_watch, stance, file_mode="w", min_chars=75):
  for screen_name in users_to_watch:
    tweets = get_tweets(api=api, screen_name=screen_name)
    with open(tweets_search_file, file_mode, encoding="utf-8") as f:
      for tweet in tweets:
        line = tweet._json
        tw_created_at = line['created_at']
        tw_id = str(line['id'])[-5:]
        tw_user_id = str(line['user']['id'])
        tw_user_screen_name = line['user']['screen_name']
		# Structure the tweets for the machine learning:
        tw_text = clean_tweet(line['full_text'])
        if len(tw_text) >= min_chars:
          f.write('\t'.join([tw_id, "Legalization of Abortion", tw_text, stance]))	
          f.write('\n')
		  
		  
def download_tweets2(users_to_watch, stance, file_mode="w", min_chars=75):
  for screen_name in users_to_watch:
    tweets = get_tweets(api=api, screen_name=screen_name)
    with open(tweets_search_file, file_mode, encoding="utf-8") as f:
      for tweet in tweets:
        line = tweet._json
        tw_created_at = line['created_at']
        tw_id = str(line['id'])[-5:]
        tw_user_id = str(line['user']['id'])
        tw_user_screen_name = line['user']['screen_name']
		# Structure the tweets for the machine learning:
        tw_text = clean_tweet(line['full_text'])
        if len(tw_text) >= min_chars:
          f.write('"')
          f.write('","'.join([tw_id, "Legalization of Abortion", tw_text.replace('"', "'"), stance]))	
          f.write('"\n')		  
		

# Stream tweets in real-time:
def stream_tweets(file_mode="w"):		
  with open(tweets_stream_file, mode=file_mode, encoding="utf-8") as f:
    f.write('\t'.join(["ID", "Target", "Tweet", "Stance"]))
    f.write("\n")
    for line in api.GetStreamFilter(track=['@twitter'], languages=['en'], stall_warnings=True):
      print(line)
      tw_created_at = line['created_at']
      tw_id = str(line['id'])[-5:]
      tw_user_id = str(line['user']['id'])
      tw_user_screen_name = line['user']['screen_name']
	  # Structure the tweets for the machine learning:
      tw_text = clean_tweet(line['text'])
      f.write('\t'.join([tw_id, "Legalization of Abortion", tw_text, "UNKNOWN"]))
      f.write('\n')
	  
	  
# Clean & normalize a tweet:
def clean_tweet(tweet):
  tweet = tweet.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ").replace("  ", " ")
  tweet = re.sub(r'https?:\/\/[\.a-zA-Z0-9\/\-_\?\=\+\#\:\,]+', '', tweet)
  tweet = re.sub(r'@[^\s]+', '@XYZ', tweet)
  return tweet

if __name__ == '__main__':
  if "download" in sys.argv:
    download_tweets2(PROLIFE_USERS, "AGAINST", "a")
    download_tweets2(PROCHOICE_USERS, "FAVOR", "a")
    download_tweets2(NEUTRAL_USERS, "NONE", "a")
  elif "stream" in sys.argv:
    stream_tweets()