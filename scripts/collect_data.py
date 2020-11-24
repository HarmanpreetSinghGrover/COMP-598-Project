import argparse
import json
import requests

"""
    Run this script from /scripts by :
        python3 collect_data.py ../data/data.json

    This script collects "iterations * num_posts" #posts from "/r/politics", "/r/conservative" and appends them to the file provided.

    Output : 
        Appends "iterations * num_posts" #posts to ../data/data.json file
        Each line of "data.json" contains one post  
"""

def get_posts(subreddit, after_field=None, num_posts=100):
    if after_field == None:
        data = requests.get(f'http://api.reddit.com{subreddit}/hot?limit={num_posts}', headers={'User-Agent' : 'mac:requests (by /u/harmanpreet)'})
        content = data.json()['data']
        return content['children']
    else:
        data = requests.get(f'http://api.reddit.com{subreddit}/hot?limit={num_posts}&after={after_field}', headers={'User-Agent' : 'mac:requests (by /u/harmanpreet)'})
        content = data.json()['data']
        return content['children']

def collect(subreddit, iterations = 4, num_posts = 100):
    result = []
    after_field = None
    for item in range(iterations):
        data = get_posts(subreddit, after_field, num_posts)
        for i in range(len(data)):
            if data[i]['data']['stickied'] == False:
                result.append(data[i]['data'])
        after_field = result[-1]['name']

    return result, len(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', help= "File in which you wish to save(with path)!")
    args = parser.parse_args()
    subreddits = ["/r/politics", "/r/conservative"]
    for subreddit in subreddits:
        iterations = 3
        num_posts = 100
        result, total_posts = collect(subreddit, iterations, num_posts)
        with open(args.output_file, 'a') as sample1_file:
            for i in range(total_posts):
                sample1_file.write(json.dumps(result[i]))
                sample1_file.write("\n")

if __name__ == '__main__':
    main()