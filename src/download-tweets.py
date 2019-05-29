import os
import json
import sys

from twitter import Api

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

				  
users_to_watch = ['prolifecampaign','ProLifeAction','AntiAbortionAF']


api = Api(consumer_key=twitter_consumer_key,
          consumer_secret=twitter_consumer_secret,
          access_token_key=twitter_access_token,
          access_token_secret=twitter_access_token_secret)
				  

def get_tweets(api=None, screen_name=None):
  timeline = api.GetUserTimeline(screen_name=screen_name, count=200)
  earliest_tweet = min(timeline, key=lambda x: x.id).id
  print("getting tweets before:", earliest_tweet)
  while True:
    tweets = api.GetUserTimeline(
      screen_name=screen_name, max_id=earliest_tweet, count=200
    )
    new_earliest = min(tweets, key=lambda x: x.id).id
    if not tweets or new_earliest == earliest_tweet:
      break
    else:
      earliest_tweet = new_earliest
      print("getting tweets before:", earliest_tweet)
      timeline += tweets
  return timeline


def main():
  # TEST: Get tweets from pro-life users:
  for screen_name in users_to_watch:
    timeline = get_tweets(api=api, screen_name=screen_name)
    with open(tweets_search_file, 'a', encoding="utf-8") as f:
      for tweet in timeline:
        line = tweet._json
        tw_created_at = line['created_at']
        tw_id = str(line['id'])[-5:]
        tw_user_id = str(line['user']['id'])
        tw_user_screen_name = line['user']['screen_name']
		# Structure the tweets for the machine learning:
        tw_text = line['text'].replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ").replace("  ", " ")
        f.write('\t'.join([tw_id, "Legalization of Abortion", tw_text, "UNKNOWN"]))	
        f.write('\n')
  
  # TEST: Stream tweets:
  with open(tweets_stream_file, mode="a", encoding="utf-8") as f:
    f.write('\t'.join(["ID", "Target", "Tweet", "Stance"]))
    f.write("\n")
    for line in api.GetStreamFilter(track=['@twitter'], languages=['en'], stall_warnings=True):
      print(line)
      tw_created_at = line['created_at']
      tw_id = str(line['id'])[-5:]
      tw_user_id = str(line['user']['id'])
      tw_user_screen_name = line['user']['screen_name']
	  # Structure the tweets for the machine learning:
      tw_text = line['text'].replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ").replace("  ", " ")
      f.write('\t'.join([tw_id, "Legalization of Abortion", tw_text, "UNKNOWN"]))
      f.write('\n')


if __name__ == '__main__':
  main()