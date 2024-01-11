import sys
sys.path.append('Scraping')
from Scraping import scrape_reviews
from Scraping import categorization


if __name__ == "__main__":
    path = 'Scraping/reviews.csv'
    location = 'Houston'
    
    # Scrape the reviews on google maps
    scrape_reviews.scrape(path, location)
    
    # Categorize reviews
    categorization.categorize(path)

