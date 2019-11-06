import re
import requests
from bs4 import BeautifulSoup
import argparse


def get_title_from_link(links, tweet):
    ignore_urls = ['http://t.co/foXvuGfnaN', 'http://t.co/3aKDXDk4Vh']
    for link in links:
        try:
            if link not in ignore_urls:
                response = requests.get(link, timeout=None)

                print('ori', tweet)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    title = getattr(soup.title, 'string', '')
                    tweet = tweet.replace(link, title)
                else:
                    print('.....................Not 200............................................')
        except:
            print('.....................Not found............................................')
    return tweet


def load_dataset(in_file, out_file):
    with open(in_file, 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()
        tweets_with_no_links = []

        if '\n' in lines:
            lines.remove('\n')

        for line in lines:
            line = line.strip().split('\t')
            print(line)
            tweet_id, tweet, target = line

            # Replace links with their titles
            links_in_tweet = re.findall(r'http\S+', tweet)
            tweet = get_title_from_link(links_in_tweet, tweet)
            tweets_with_no_links.append([tweet_id, tweet.replace('\n', ' '), target])

    with open(out_file, 'w') as f:
        for tweet_row in tweets_with_no_links:
            f.write('\t'.join(tweet_row))
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', type=str)
    parser.add_argument('out_file', type=str)
    opts = parser.parse_args()

    load_dataset(opts.in_file, opts.out_file)
