import foundation

class newsapi_extractor(object):

    def __str__(self):
        return "newsapi_extractor"

    def __init__(self):
        self.url = f'https://newsapi.org/v2/everything?q=finance&apiKey={foundation.env.NEWSAPI_API_KEY}'

    def request_articles(self):
        """
            request's articles from NewsAPI and stores them to news file
        """
        # try to get the response from the url
        try:
            # request for response
            response = foundation.requests.get(self.url, timeout=20)
            # raise status error
            response.raise_for_status()
            # convert to json
            response_json = response.json()
            """
                response_json FORMAT:
                    <class 'dict'>
                    dict_keys(['status', 'totalResults', 'articles'])

                we need only items in "articles" key, which is list of dicts
                
                response_json["articles"] type: <class 'list'> of dicts

                keys in response_json["articles"][0]: dict_keys(['source', 'author', 'title', 'description', 'url', 'urlToImage', 'publishedAt', 'content'])
                we need only title, author and url which are are dumping to the json

            """
            # warn if nothing in articles
            if not response_json["articles"]:
                foundation.log(f"NewsAPI response for url({self.url}) had no articles.", self.__str__(), "warn")
            # debug info
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
        # debug info
        foundation.log(f"writing news to {foundation.env.fnews_newsapi_filename}", self.__str__())
        # try writing to the file
        try:
            # write to file
            with open(foundation.env.fnews_newsapi_filename, "w") as _file:
                # FOREACH article in articles write them to file!
                for article in response_json["articles"]:
                    # skip if it not an article or if it is None
                    if article is None or not article:
                        foundation.log(f"skipping article because of its invalidity!", self.__str__(), "warn!")
                        continue
                    # create article_object for dumping
                    article_object = {
                        "title"     : article["title"] if article["title"] is not None else None,
                        "author"    : article["author"] if article["author"] is not None else None,
                        "url"       : article["url"] if article["url"] is not None else None
                    }
                    # dump with newline for jsonl format
                    _file.write(foundation.json.dumps(article_object) + "\n")
        # ----- exception managemenet -----
        except FileNotFoundError as fe:
            foundation.log(f"file not found {fe}", self.__str__(), "error")
        except IndexError as ie:
            foundation.log(f"index error {ie}", self.__str__(), "error")
        except Exception as e:
            foundation.log(f"unknown exception {e}", self.__str__(), "error")
        # ----- exception management -----

class yfnews_extractor(object):
    
    def __str__(self):
        return "yfnews_extractor"
    
    def __init__(self):
        # load ticker symbols from s and p csv
        self.symbols = foundation.pd.read_csv(foundation.env.s_and_p_500_tickers_filename)["Symbol"]
        # convert into python list
        self.symbols = list(self.symbols)

    def request_articles(self):
        """
            requests articles from yahoo finance api and writes them to news file
        """
        # download symbol related metadata
        self.yf_data = foundation.yf.Tickers(self.symbols)
        # debug info
        foundation.log("downloading news for each symbol!", self.__str__())
        # fetch latest news for symbols
        self.yf_news = self.yf_data.news()
        """
            yf_news is of <class 'dict'> with keys ["AAPL", "MMM" etc of ticker symbols]
            Each of these keys again map to item with rich info
        """
        # try writing to news file
        try:
            # debug info
            foundation.log(f"writing to news data to {foundation.env.fnews_yf_filename}", self.__str__())
            # open new file handle
            with open(foundation.env.fnews_yf_filename, "w") as _file:
                # FOREACH symbol in symbols
                for symbol in self.symbols:
                    # initialize a article for dumping to json
                    articles_metadata = []
                    # gets the list of news for this symbol
                    symbol_news_list = self.yf_news[symbol]
                    # ITERATE through symbol news and then add to article
                    for symbol_news in symbol_news_list:
                        # extract title
                        title = symbol_news["content"]["title"]
                        # extract url
                        url = symbol_news["content"]["canonicalUrl"]["url"]
                        # extract author (in this case provider)
                        author = symbol_news["content"]["provider"]["displayName"]
                        # append to articles metadata
                        articles_metadata.append({
                            "title"     : title, 
                            "author"    : author,
                            "url"       : url
                        })
                    # create object for dumping
                    article_object = {symbol : articles_metadata}
                    # dump article in jsonl format
                    _file.write(foundation.json.dumps(article_object) + "\n")

        # ----- exception management -----
        except FileNotFoundError as fe:
            foundation.log(f"file not found {fe}", self.__str__(), "error")
        except KeyError as ke:
            foundation.log(f"key error {ke}", self.__str__(), "error")
        except IndexError as ie:
            foundation.log(f"index error {ie}", self.__str__(), "error")
        except Exception as e:
            foundation.log(f"unexpected exception {e}", self.__str__(), "error")
        # ----- exception management -----

class article_scraper(object):

    def __str__(self):
        return "article_scraper"

    def __init__(self):
        pass

    def scrape(self, article_extractor_class : str = "newsapi_extractor"):
        """
            scrapes the URLs extracted from NewsAPI | YFinance, processes them using bs4,
            and writes them to env.processed_finance_news_filename as JSONL.

            Args:
                article_extractor_class : str, base extractor class to scrape news from!
                valid values: "newsapi_extractor", "yfnews_extractor"
        """

        if article_extractor_class == "newsapi_extractor":
            # debug info
            foundation.log(f"reading from {foundation.env.fnews_newsapi_filename}", self.__str__())
            foundation.log(f"writing to {foundation.env.fnews_processed_newsapi_filename}", self.__str__())
            self._batch_process_url(foundation.env.fnews_newsapi_filename, foundation.env.fnews_processed_newsapi_filename)
        
        elif article_extractor_class == "yfnews_extractor":
            # debug info
            foundation.log(f"reading from {foundation.env.fnews_yf_filename}", self.__str__())
            foundation.log(f"writing to {foundation.env.fnews_processed_yf_filename}", self.__str__())
            self._batch_process_url(foundation.env.fnews_yf_filename, foundation.env.fnews_processed_yf_filename)

        else:
            # invalid extractor class
            foundation.log(f"PARAMETER(article_extractor_class) is invalid", self.__str__(), "error")

    def _scrape_url(self, meta: dict, timeout:int = 15) -> dict | None:
        # warn skip meta is None
        if meta is None:  
            foundation.log(f"skpping", self.__str__(), "warn")
            return None
        
        # extract info for dumping to json
        url                 = meta["url"]
        title               = meta["title"]
        author              = meta["author"]
        scraped_content     = ""
        status              = "success"
        
        # warn and skip if no valid url
        if not url or url == "" or url is None:
            foundation.log(f"Skipping article, lack of valid url", self.__str__(), "warn")
            return None
                            
        # processed object
        processed_article_object = {
            "title"             : title,
            "url"               : url,
            "author"            : author,
            "scraped_content"   : "", 
            "status"            : status,
            "error_details"     : ""
        }

        # try scraping the url
        try:
            # get response
            response = foundation.requests.get(url, timeout=timeout)
            # raise for status to validate
            response.raise_for_status()
            # beautiful soup parser
            soup = foundation.BeautifulSoup(response.text, features="lxml")
            # try finding main content using div id's and classes
            main_content = soup.find(class_=[
                                                "article-body", "post-content", "entry-content", "main-content", 
                                                "td-post-content", "content-wrapper", "l-article-body", 
                                                "article__body", "article-content", "story-body", "body-content"
                                            ]) or \
                                soup.find(id=[
                                                "article-body", "content", "main", "primary", 
                                                "article-content", "story-page", "main-article-text"
                                            ]) or \
                                soup.find("article")
            # IF main_content exists
            if main_content:
                # find all paragraphs
                paragraphs = main_content.find_all("p")
                # find scraped content
                scraped_content = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
                # IF there is no scraped content
                if not scraped_content.strip():
                    # get all the text from this div
                    scraped_content = main_content.get_text(separator="\n", strip=True)
            # ELSE fallback
            else:
                # tags to extract
                tags_to_extract = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]
                fallback_texts = []
                # FOREACH tag in tags to extract
                for tag in soup.find_all(tags_to_extract):
                    # get all the text in that tag
                    text = tag.get_text(strip=True)
                    # IF there is text, then append to list
                    if text:
                        fallback_texts.append(text)
                # convert to single string
                scraped_content = "\n".join(fallback_texts)

            processed_article_object["scraped_content"]   = scraped_content
            processed_article_object["status"]            = status

        # ----- exception management -----
        except foundation.requests.exceptions.Timeout:
            article_status = "timeout_error"
            foundation.log(f"Request timeout for URL: {url}", self.__str__(), "warn")
            processed_article_object["status"] = article_status
        except foundation.requests.exceptions.HTTPError as http_error:
            status_code = http_error.response.status_code if http_error.response else "unknown"
            article_status = f"http_error_{status_code}"
            foundation.log(f"HTTP error {status_code} for URL: {url}", self.__str__(), "warn")
            processed_article_object["status"] = article_status
        except foundation.requests.exceptions.RequestException as req_e:
            article_status = "network_error"
            foundation.log(f"Network error for URL: {url} - {req_e}", self.__str__(), "warn")
            processed_article_object["status"] = article_status
        except KeyError as ke:
            foundation.log(f"key error {ke}", self.__str__(), "error")
        except IndexError as ie:
            foundation.log(f"index error {ie}", self.__str__(), "error")
        except Exception as e:
            article_status = "unexpected_exception"
            foundation.log(f"Unknown exception while scraping URL {url}: {e}", self.__str__(), "warn")
            processed_article_object["status"] = article_status
            processed_article_object["error_details"] = str(e)
        # ----- exception management -----

        return processed_article_object
    
    def _yield_batch_from_file(self, file_object, batch_size: int):
        # initialize a batch
        batch = []
        # FOREACH line in _file
        for line_num, line in enumerate(file_object):
            line = line.strip()
            # skip if it doesn't have any valid info
            if not line:
                foundation.log(f"skipping {line_num}, it didn't contain any info", self.__str__(), "warn")
                continue
                    
            # try appending to the batch
            try:
                # append loaded object to the batch
                batch.append(foundation.json.loads(line))
                # if batch is complete yield it and reset it
                if len(batch) >= batch_size:
                    yield batch
                    batch.clear()
            # ----- exception management -----
            except foundation.json.JSONDecodeError as je:
                foundation.log(f"json decode error, {je}", self.__str__(), "error")
            except Exception as e:
                foundation.log(f"unexpected exception, {e}", self.__str__(), "error")
            # ----- exception management -----

        # yield any remanining incompletet batch
        if batch:
            yield batch

    def _batch_process_url(self, news_file_path : str, write_file_path: str, batch_size : int = 64, timeout : int = 15):
        """
        processes urls in batches, scraping content and writing to write_file_path
        Args:
            news_file_path (str): path to read news jsonl
            write_file_path (str): path to write processed jsonl
            batch_size (int): number of articles to process in each batch.
            timeout (int): timeout in seconds for individual HTTP requests to article URLs.
        """
        # total length of read object
        total = 0
        # try calculating length of the file
        try:
            # create object for new_file
            with open(news_file_path, "r") as _file:
                # calculate total
                total = sum(1 for _ in _file)
        # ----- exception management -----
        except FileNotFoundError as fe:
            foundation.log(f"file({news_file_path}) was not found, {fe}", self.__str__(), "error")
        except Exception as e:
            foundation.log(f"unexpected exception, {e}", self.__str__(), "error")
        # ----- exception management -----

        # try creating the file streams
        try:
            # create read object
            read_object     = open(news_file_path, "r", encoding = "utf-8")
            write_object    = open(write_file_path, "w", encoding = "utf-8")
        # ----- exception management -----
        except FileNotFoundError as fe:
            foundation.log(f"file({write_file_path}) not found!", self.__str__(), "error")
        except Exception as e:
            foundation.log(f"unexpected exception, {e}", self.__str__(), "error")
        # ----- exception management -----
        
        # create threadpool object
        executor = foundation.ThreadPoolExecutor(max_workers = batch_size)

        # tqdm bar
        with foundation.tqdm(total = total, desc = "scarping urls") as pbar:
            # iterate through all the generator batches
            for this_batch_metadata in self._yield_batch_from_file(read_object, batch_size):
                # debug info
                foundation.log(f"processing batch of size {len(this_batch_metadata)}", self.__str__())
                # submit jobs
                future_jobs = {
                    executor.submit(self._scrape_url, meta, timeout) for meta in this_batch_metadata
                }
                # for completed jobs, dump their result to json
                for future in foundation.as_completed(future_jobs):
                    # get the processed article object
                    processed_article_object = future.result()
                    # dump and flush
                    write_object.write(foundation.json.dumps(processed_article_object) + "\n")
                    write_object.flush()
                    # update tqdm
                    pbar.update(1)

        executor.shutdown(wait = True)

        read_object.close()
        write_object.close()