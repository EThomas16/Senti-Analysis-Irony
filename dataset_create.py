import sys
import time

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from decorators import timer

CHROME_PATH = "C:/Users/Erik/chromedriver"
BROWSER = webdriver.Chrome(CHROME_PATH)
PARSER = "html.parser"
BASE_URL = "https://twitter.com/search?q="
# TODO: change to check installed browsers to use

@timer
def get_tweets(query: str, pages_to_scroll: int = 5):
    url = BASE_URL + query

    BROWSER.get(url)
    time.sleep(1)

    text_body = BROWSER.find_element_by_tag_name('body')

    for _ in range(pages_to_scroll):
        text_body.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)

    tweets = BROWSER.find_elements_by_class_name('tweet-text')

    for tweet in tweets:
        print(tweet.text)

@timer
def get_tweets_bs4(query: str):
    """
    Proof of concept as bs4 cannot load the webpage fully
    Selenium is required to have the scraped page return information
    """
    webpage = requests.get(f"{BASE_URL}{query}")
    b_soup = BeautifulSoup(webpage.text, PARSER)

    tweets = []

    for p in b_soup.findAll('p', class_='tweet-text'):
        tweets.append(p.text)

    print(tweets)

def check_browser():
    # TODO: add checks for browsers here
    pass

get_tweets("%23sarcasm&src=typd")