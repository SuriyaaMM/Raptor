from article_analyzer import newsapi_extractor, article_scraper, yfnews_extractor
from natural_language_processer import nlp_pipeline
import foundation

if __name__ == "__main__":

    # newsapi = newsapi_extractor()
    # newsapi.request_articles()

    # yfnews = yfnews_extractor()
    # yfnews.request_articles()

    # scraper = article_scraper()
    # # scraper.scrape()
    # scraper.scrape(article_extractor_class = "yfnews_extractor")

    nlp = nlp_pipeline()
    nlp.process(foundation.env.fnews_processed_yf_filename)
    