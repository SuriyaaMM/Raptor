import foundation
import os

class newsapi_extractor(object):

    def __str__(self):
        return "newsapi_extractor"

    def __init__(self):
        self.url = f'https://newsapi.org/v2/everything?q=finance&apiKey={foundation.env.NEWSAPI_API_KEY}'

    def request_articles(self):
        """
            request's articles from NewsAPI and stores them to the file in
            env.latest_finance_news_filename
        """
        try:
            response = foundation.requests.get(self.url, timeout=20)
            response.raise_for_status()

            response_json = response.json()

            if not response_json["articles"]:
                foundation.log(f"NewsAPI response for url({self.url}) had no articles.", self.__str__(), "warn")
            # write to file latest_finance_news_filename
            with open(foundation.env.latest_finance_news_filename, "w") as _file:
                foundation.json.dump(response_json["articles"], _file)
            foundation.log(f"fetched {len(response_json['articles'])} articles from NewsAPI.", self.__str__())

        # ----- exception management -----
        except foundation.requests.exceptions.Timeout:
            foundation.log(f"NewsAPI request timed out for URL: {self.url}", self.__str__(), "error")
        except foundation.requests.exceptions.RequestException as req_e:
            foundation.log(f"Network or API request error for URL {self.url}: {req_e}", self.__str__(), "error")
        except foundation.json.JSONDecodeError as json_e:
            foundation.log(f"JSON decoding error from NewsAPI response: {json_e}", self.__str__(), "error")
        except Exception as e:
            foundation.log(f"An unexpected error occurred in request_articles: {e}", self.__str__(), "error")
        # ----- exception management -----

class article_scraper(object):

    def __str__(self):
        return "article_scraper"

    def __init__(self):
        self.articles_to_scrape = []

    def scrape(self):

        """
            scrapes the URLs extracted from NewsAPI, processes them using bs4,
            and writes them to env.processed_finance_news_filename as JSONL.
        """
        try:
            with open(foundation.env.latest_finance_news_filename, "r") as _file:
                self.articles_to_scrape = foundation.json.load(_file)

        # ----- exception management -----
        except FileNotFoundError:
            foundation.log(f"'{foundation.env.latest_finance_news_filename}' not found. Run newsapi_extractor first.", self.__str__(), "error")
            return
        except foundation.json.JSONDecodeError:
            foundation.log(f"Could not parse JSON from '{foundation.env.latest_finance_news_filename}'. File might be corrupted or empty.", self.__str__(), "error")
            return
        except Exception as e:
            foundation.log(f"An unexpected error occurred reading NewsAPI articles: {e}", self.__str__(), "error")
            return
        # ----- exception management -----

        if not self.articles_to_scrape:
            foundation.log("No URLs found in the latest articles to scrape.", self.__str__(), "warn")

        self._batch_process_articles()


    def _batch_process_articles(self, batch_size: int = 256, timeout: int = 15):

        """
        processes articles in batches, scraping content and writing to a JSONL file.

        Args:
            batch_size (int): number of articles to process in each batch.
            timeout (int): timeout in seconds for individual HTTP requests to article URLs.
        """

        total_articles = len(self.articles_to_scrape)
        
        with foundation.tqdm(total = total_articles, desc = "scraping articles") as pbar:

            with open(foundation.env.processed_finance_news_filename, "w") as _file:
                # batch processing
                for start in range(0, total_articles, batch_size):
                    # range calculations
                    end = min(start + batch_size, total_articles)
                    batch_articles = self.articles_to_scrape[start:end]
                    # debug message
                    foundation.log(f"Processing batch from index {start} to {end-1}...", self.__str__())

                    for i, article_meta in enumerate(batch_articles):
                    
                        url                 = article_meta["url"]
                        scraped_content     = "" 
                        article_status      = "success"
                        
                        if article_meta is None:  
                            foundation.log(f"Skipping null article metadata at original index {i}.", self.__str__(), "warn")
                            continue
                        
                        if not url:
                            foundation.log(f"Skipping article at original index {i}: No URL provided.", self.__str__(), "warn")
                            continue

                        processed_article_data = {
                            "title"             : article_meta.get("title"),
                            "url"               : url,
                            "publishedAt"       : article_meta["publishedAt"],
                            "author"            : article_meta["author"],
                            "scraped_content"   : "", 
                            "status"            : article_status
                        }

                        try:

                            response = foundation.requests.get(url, timeout=timeout)
                            response.raise_for_status()

                            soup = foundation.BeautifulSoup(response.text, features="lxml")
                            paragraphs = soup.find_all("p")
                            scraped_content = "\n".join(p.get_text() for p in paragraphs)

                            if not scraped_content.strip():
                                article_status = "no_content"
                                foundation.log(f"No meaningful content scraped from URL: {url}", self.__str__(), "warn")
                            
                            processed_article_data["scraped_content"]   = scraped_content
                            processed_article_data["status"]            = article_status

                        # ----- exception management -----
                        except foundation.requests.exceptions.Timeout:
                            article_status = "timeout_error"
                            foundation.log(f"Request timeout for URL: {url}", self.__str__(), "warn")
                            processed_article_data["status"] = article_status
                        except foundation.requests.exceptions.HTTPError as http_error:
                            status_code = http_error.response.status_code if http_error.response else "unknown"
                            article_status = f"http_error_{status_code}"
                            foundation.log(f"HTTP error {status_code} for URL: {url}", self.__str__(), "warn")
                            processed_article_data["status"] = article_status
                        except foundation.requests.exceptions.RequestException as req_e:
                            article_status = "network_error"
                            foundation.log(f"Network error for URL: {url} - {req_e}", self.__str__(), "warn")
                            processed_article_data["status"] = article_status
                        except Exception as e:
                            article_status = "unexpected_exception"
                            foundation.log(f"Unknown exception while scraping URL {url}: {e}", self.__str__(), "warn")
                            processed_article_data["status"] = article_status
                            processed_article_data["error_details"] = str(e) # Add error details for debugging
                        # ----- exception management -----
                        
                        # dump each processed article
                        _file.write(foundation.json.dumps(processed_article_data) + '\n')
                        pbar.update(1)