import time
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import csv
import pandas as pd
import os
import categorization

def scrape_reviews(page, location, places, links, path):
    for i in range(len(links)):
        df = pd.read_csv(path)
        if df.empty:
            with open(path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(["CITY", "PLACE", "REVIEW", "LINK"])

        df = df[df['PLACE'] == places[i]]
        if not df.empty or places[i] == 'New York':
            print("Skipping:", location, places[i])
            continue

        page.goto(links[i])
        time.sleep(4)

        page.locator("text='Reviews'").first.click()
        time.sleep(4)

        html = page.inner_html('body')

        while True:
            page.mouse.wheel(0, 15000)
            time.sleep(2)
            new_html = page.inner_html('body')
            if new_html == html:
                break
            html = new_html

        soup = BeautifulSoup(html, 'html.parser')
        reviews = [review.find('span').text for review in soup.select('.MyEned')]
        print("Number of reviews:", len(reviews))

        with open(path, 'a') as f:
            writer = csv.writer(f)
            for review in reviews:
                writer.writerow([location, places[i], review, links[i]])

def scrape(path, location):
    category = "restaurants"
    url = "https://www.google.com/"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        try:
            page.goto(url)
            page.click('.QS5gu.sy4vM')
            page.fill("textarea", f"{category} near {location}")
            page.keyboard.press('Enter')
            page.locator("text='Change to English'").click()
            time.sleep(4)
            page.click('.GKS7s')
            time.sleep(4)

            for _ in range(2):
                html = page.inner_html('body')
                soup = BeautifulSoup(html, 'html.parser')
                categories = soup.select('.hfpxzc')
                last_category_in_page = categories[-1].get('aria-label')
                page.locator(f"text={last_category_in_page}").scroll_into_view_if_needed()

            links = [item.get('href') for item in soup.select('.hfpxzc')]
            places = [item.get('aria-label') for item in soup.select('.hfpxzc')]

            if not os.path.exists(path):
                with open(path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(["CITY", "PLACE", "REVIEW", "LINK"])

            scrape_reviews(page, location, places, links, path)

        finally:
            page.close()
            
        
