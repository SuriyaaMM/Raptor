from article_analyzer import newsapi_extractor, article_scraper
from natural_language_processer import nlp_pipeline

if __name__ == "__main__":

    newsapi = newsapi_extractor()

    #newsapi.request_articles()

    scraper = article_scraper()

    # scraper.scrape()

    nlp = nlp_pipeline()
    