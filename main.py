import sys
sys.path.append('Scraping')
from Scraping import scrape_reviews
from Scraping import categorization
sys.path.append('Modele')
from Modele import rate_restaurant


if __name__ == "__main__":
    path = 'Scraping/reviews.csv'
    location = 'Houston'
    
    # Scrape the reviews on google maps
    scrape_reviews.scrape(path, location)
    
    # Categorize reviews
    categorization.categorize(path, threshold=0.7)

    # Create a csv which rates the restaurants of a city
    rate_restaurant.creation_csv(location)
