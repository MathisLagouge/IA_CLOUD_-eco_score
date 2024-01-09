import time
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import csv
import pandas as pd

# Path of the CSV file where we will write the reviews
path = 'Scraping/reviews.csv'

# the category for which we seek reviews
CATEGORY = "restaurants"

# the location
LOCATION = "New York"

# google's main URL
URL = "https://www.google.com/"

with sync_playwright() as pw:
    # creates an instance of the Chromium browser and launches it
    browser = pw.chromium.launch(headless=False)

    # creates a new browser page (tab) within the browser instance
    page = browser.new_page()

    # go to url with Playwright page element
    page.goto(URL)

    # deal with cookies
    page.click('.QS5gu.sy4vM')

    # write what you're looking for
    page.fill("textarea", f"{CATEGORY} near {LOCATION}")

    # press enter
    page.keyboard.press('Enter')

    # change to english
    page.locator("text='Change to English'").click()
    time.sleep(4)

    # click in the "Maps" HTML element
    page.click('.GKS7s')
    time.sleep(4)

    # scrolling
    for i in range(10):
        # tackle the body element
        html = page.inner_html('body')

        # create beautiful soup element
        soup = BeautifulSoup(html, 'html.parser')

        # select items
        categories = soup.select('.hfpxzc')
        last_category_in_page = categories[-1].get('aria-label')

        # scroll to the last item
        last_category_location = page.locator(
            f"text={last_category_in_page}")
        last_category_location.scroll_into_view_if_needed()

    # get links of all categories after scroll
    links = [item.get('href') for item in soup.select('.hfpxzc')]

    # get links of all categories after scroll
    places = [item.get('aria-label') for item in soup.select('.hfpxzc')]

    for i in range(len(links)):
        # Check if the restaurant is already in the CSV
        df = pd.read_csv(path)
        df = df[df['PLACE'] == places[i]]
        if not df.empty:
            print("Skipping : ", LOCATION, places[i])
            continue

        # go to subject link
        page.goto(links[i])
        time.sleep(4)

        # load  reviews
        page.locator("text='Reviews'").first.click()
        time.sleep(4)

        # create new soup
        html = page.inner_html('body')

        # Scroll down to load more reviews until no more are loaded
        while True:
            page.mouse.wheel(0, 15000)
            time.sleep(2)
            new_html = page.inner_html('body')
            if new_html == html:
                break
            html = new_html

        # create beautiful soup element
        soup = BeautifulSoup(html, 'html.parser')

        # get all reviews
        reviews = soup.select('.MyEned')
        reviews = [review.find('span').text for review in reviews]
        print("Number of reviews : ", len(reviews))

        # Write to CSV with | CITY | PLACE | REVIEW |
        with open(path, 'a') as f:
            writer = csv.writer(f)
            for review in reviews:
                print("Writing : ", LOCATION, places[i], review)
                writer.writerow([LOCATION, places[i], review])
