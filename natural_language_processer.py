import foundation
import ml_models
class entity_linker(object):

    def __str__(self):
        return "entity_linker"
    
    """
        entity_linker, finds the entities's labels in the list financial_entity_labels
        and links them to relevant sentences
        fills in financial_entities dict with the structure
        financial_entities : dict
        ----- entity-text : str, [stores the entity text, ex: US, Apple, Amazon]
        ----- analysis : dict, [stores the data for processing]
                ----- sentence : str, [sentence in which this entity was found]
                ----- sentiment : dict, [sentiment analysis results performed by a transformer]
                        ----- results : str, [POSITIVE or NEGATIVE],
                        ----- score : float32, [confidence score]

        Example:

        'chevron': {'analysis': {'sentence': 'Sign up \n'
                                      'chevron down icon\n'
                                      'An icon in the shape of an angle '
                                      'pointing down.',
                          'sentiment': {'label': 'Neutral',
                                        'score': 0.9998425245285034}}}}
    """
    def __init__(self, spacy_doc):

        # these are the entity lables that we want to find links
        self.financial_entity_labels = ["ORG", "MONEY", "PERSON", "GPE", "PRODUCT"]
        self.financial_entities = {}
        # preparation for next stage of analysis
        self.relevant_sentence_for_sentiment_analysis = []
        # org entities for ticker matching
        self.org_entities = []

        # loop through entities in Doc.entities
        for entities in spacy_doc.ents:
            # ORG entities are needed for ticker mapping
            if entities.label_ == "ORG":
                self.org_entities.append(entities.text)
            # IF that entity label is in our required financial_entity_labels list
            # THEN process them, ELSE just ignore (we don't want unwanted entities to pile up)
            if entities.label_ in self.financial_entity_labels:
                # IF the entity ex (Apple) is not in our financial_entities dict
                # THEN add it to our dict analysis which is a dict of two keys "sentence" and "sentiment"
                if entities.text not in self.financial_entities:
                    self.financial_entities[entities.text] = {"analysis" : {
                        "sentences" : [],
                        "sentiment" : {}
                    }, "ticker" : None}
                # append those sentences where the entitity is seen
                # later when performing sentiment analysis, sentiment will be added
                self.financial_entities[entities.text]["analysis"]["sentences"].append(entities.sent.text.strip())
                # also those mentions are the only ones we are going to perform sentiment analysis on
                self.relevant_sentence_for_sentiment_analysis.append(entities.sent.text.strip())
        
        # remove duplices from org_entities
        self.org_entities = list(set(self.org_entities))

        # list ticker map that stores "title" : "ticker"
        self.ticker_map = {}
        # titles and tickers lists
        self.ticker_titles  = []
        self.tickers        = []
        # embedder
        self.embedder       = foundation.SentenceTransformer(foundation.env.model_name_embedder, cache_folder = foundation.env.cache_dir)
        # embeddings for similarity search
        self.embeddings     = []
        # # populate the ticker map
        # try:
        #     with open("companies_ticker_map.json", "r") as _file:
                
        #         raw_object = foundation.json.load(_file)

        #         for keys, company_info in raw_object.items():
        #             ticker_title    = company_info["title"]
        #             ticker          = company_info["ticker"]
        #             embedded_title  = self.embedder.encode(ticker_title)

        #             self.ticker_map[ticker_title] = ticker
        #             self.ticker_titles.append(ticker_title)
        #             self.tickers.append(ticker)
        #             self.embeddings.append(embedded_title)
        # # ----- exception management -----
        # except FileNotFoundError as fe:
        #     foundation.log(f"file not found : {fe}",self.__str__(), "error")
        # except KeyError as ke:
        #     foundation.log(f"key error {ke}", self.__str__(), "error")
        # # ----- exception management -----
        
        # # convert into np array for faiss indexing
        # self.embeddings = foundation.np.array(self.embeddings)
        # dimensions = self.embeddings.shape[1]
        # # inner product indexer
        # self.index = foundation.faiss.IndexFlatIP(dimensions)
        # self.index.add(self.embeddings) # type: ignore
        
        # for org_entity in self.org_entities:

        #     # org entity embedding
        #     org_entity_embedding = self.embedder.encode(org_entity)

        #     distance, index = self.index.search(org_entity_embedding.reshape(1, -1), k = 1) # type:ignore

        #     foundation.log(f"similarity : {distance[0][0]}", self.__str__())
        #     foundation.log(f"matched entity : {index[0][0]}", self.__str__())
        #     foundation.log(f"org : {org_entity}", self.__str__())

        # ----------------------------------------------------------------------------------
        # DOESNT WORK WELL WITH ALIASES AND FULL FORMS
        # # fuzzy match organization entities
        # self.matched_org_entities = {}
        # # find best match for each organization entity
        # for org_entity in self.org_entities:
            
        #     best_match = foundation.process.extract(org_entity,
        #                                                self.ticker_map.keys(),
        #                                                scorer = foundation.fuzz.token_sort_ratio)
        #     # if best_match:
        #     #     ticker_title, score, _ = best_match
        #     #     self.financial_entities[org_entity]["ticker"] = self.ticker_map[ticker_title]
        # ----------------------------------------------------------------------------------

        self.tickers_df = foundation.pd.read_csv(foundation.env.s_and_p_500_tickers_filename)
        self.symbol             = self.tickers_df["Symbol"]
        self.gics_sector        = self.tickers_df["GICS Sector"]
        self.gics_sub_industry  = self.tickers_df["GICS Sub-Industry"]

        for i, symbol in enumerate(self.symbol):

            self.embeddings = self.embedder.encode(self.symbol[i] + self.gics_sector[i] + self.gics_sub_industry[i])
        
        # convert into np array for faiss indexing
        self.embeddings = foundation.np.array(self.embeddings)
        dimensions = self.embeddings.shape[1]
        # inner product indexer
        self.index = foundation.faiss.IndexFlatIP(dimensions)
        self.index.add(self.embeddings) # type: ignore
        
        for org_entity in self.org_entities:

            # org entity embedding
            org_entity_embedding = self.embedder.encode(org_entity)

            distance, index = self.index.search(org_entity_embedding.reshape(1, -1), k = 1) # type:ignore

            foundation.log(f"similarity : {distance[0][0]}", self.__str__())
            foundation.log(f"matched entity : {index[0][0]}", self.__str__())
            foundation.log(f"org : {org_entity}", self.__str__())
        
class sentiment_analyzer(object):

    def __str__(self):
        return "sentiment_analyzer"
    
    def __init__(self, financial_entities: dict):

        self.model_name = "yiyanghkust/finbert-tone"

        # keywords list for spacy to find relevant sentences with these keywords
        self.financial_keywords =  {
        "profit", "loss", "revenue", "earnings", "dividend", "merger", "acquisition", "default",
        "bankruptcy", "downgrade", "upgrade", "regulation", "compliance", "fine", "lawsuit",
        "fraud", "volatility", "inflation", "interest", "credit", "loan", "debt", "equity",
        "bond", "stock", "shareholder", "investment", "portfolio", "risk", "return", "yield",
        "growth", "decline", "forecast", "outlook", "capital", "market", "economy", "trade",
        "shares", "assets", "liabilities", "valuation", "fund", "index"
        }
        
        # tokenizer and model from hugging face for sentiment analysis
        self.tokenizer_sentiment    = foundation.AutoTokenizer.from_pretrained(self.model_name, cache_dir=foundation.env.cache_dir)
        self.model_sentiment        = foundation.AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=foundation.env.cache_dir)
        # sentiment analysis pipeline
        self.sentiment_analyzer_pipeline = foundation.pipeline("sentiment-analysis", model=self.model_sentiment, tokenizer=self.tokenizer_sentiment)
        # hashmap for fast lookups and save time re-analyzing the sentiment
        sentence_sentiment_hashmap = {}
        # refer to structure of financial_entities
        # LOOP through financial_entities
        for entity, entity_data in financial_entities.items():
            # LOOP through each sentence in sentences
            for sentence in entity_data["analysis"]["sentences"]:
                # IF sentiment for that sentence has already been calculated, then add that to financial_entities
                if sentence in sentence_sentiment_hashmap.keys():
                    financial_entities[entity]["analysis"]["sentiment"] = sentence_sentiment_hashmap[sentence]
                # ELSE pass it through sentiment_analyzer then add that to hashmap & financial_entities
                else:
                    entity_sentiment_result = self.sentiment_analyzer_pipeline(sentence)
                    sentence_sentiment_hashmap[sentence] = entity_sentiment_result
                    financial_entities[entity]["analysis"]["sentiment"] = entity_sentiment_result
                

class nlp_pipeline(object):

    def __str__(self):
        return "nlp_pipeline"
    
    def __init__(self):
        
        # load our base nlp model from spacy
        self.nlp = foundation.spacy.load(foundation.env.model_name_spacy)
        # sentiment analyzer
        self.sentiment_analyzer = ml_models.sentiment_analyzer()
        # try:
        #     # try opening the processed file if one exists
        #     with open(foundation.env.fnews_processed_newsapi_filename, "r") as _file:
        #         # enumrate throug the file
        #         for i, line in enumerate(_file):
        #             # IF the line is empty then safely skip it
        #             # THEN continue
        #             if not line.strip():
        #                 continue
        #             # try performing nlp tasks on the processed articles in the file
        #             try:
        #                 processed_article_data = foundation.json.loads(line.strip())
        #             # ----- exception management -----
        #             except foundation.json.JSONDecodeError as json_e:
        #                 foundation.log(f"JSON decoding error on line {i+1}: {json_e} - Skipping line.", self.__str__(), "warn")
        #                 continue
        #             # ----- exception management -----
                    
        #             # IF processed article has status success, only then process it
        #             # ELSE just continue the loop
        #             if processed_article_data["status"] != "success":  
        #                 foundation.log(f"skipping article {processed_article_data.get('url', f'index {i}')}, status was '{processed_article_data.get('status')}'", self.__str__(), "warn")
        #                 continue
        #             # get scraped content from the processed article
        #             # this almost always has something due to robust error checking and exception
        #             # management in scraper
        #             article_content = processed_article_data["scraped_content"]
        #             # debug message
        #             foundation.log(f"Processing article: {processed_article_data["title"]}", self.__str__(), "info")
        #             # build nlp pipeline for our article_content
        #             doc = self.nlp(article_content)
                    
        #             # -------------------- STAGE 1: perform entity linking --------------------
        #             el  = entity_linker(doc)
        #             # debug messages
        #             if el.financial_entities:
        #                 foundation.log(f"Extracted entities for: {processed_article_data["title"]}", self.__str__(), "info")
        #             else:
        #                 foundation.log(f"No financial entities found in: {processed_article_data["title"]}", self.__str__(), "info")
        #             # -------------------- STAGE 2: perform sentiment analysis --------------------
        #             sa = sentiment_analyzer(el.financial_entities)

        #             foundation.pprint(el.financial_entities)

        #             if i == 1: break

        # # ----- exception management -----
        # except FileNotFoundError as fe:
        #     foundation.log(f"Processed finance news file '{foundation.env.fnews_processed_newsapi_filename}' not found! Ensure scraping is complete.\n cwd = {foundation.os.getcwd()}", self.__str__(), "error")
        # except Exception as e:
        #     foundation.log(f"An unexpected exception occurred in nlp_pipeline: {e}", self.__str__(), "error")
        # # ----- exception management -----

    def process(self, processed_file: str, entity_link: bool = False) -> list[str]:
        tickers = []
        # try processing the file
        try:
            # create write object
            with open(foundation.env.ml_sentiment_analysis_report, "w") as _write_file:
                # create read object
                with open(processed_file, "r") as _file:
                    # get and strip the line
                    for line in _file:
                        line = line.strip()
                        # if no content, then skip
                        if not line:
                            continue 
                        # load the object
                        article_object = foundation.json.loads(line)
                        # processed article content
                        processed_article_content = article_object["scraped_content"]
                        # process title if it isn't success!
                        if article_object["status"] != "success":
                            foundation.log(f"analyzing title because status was not success", self.__str__(), "warn")                        
                            processed_article_content = article_object["title"]
                        # perform sentiment analysis
                        result = self.sentiment_analyzer.analyze(processed_article_content)
                        # append symbol to result
                        result["symbol"] = article_object["symbol"] # type: ignore
                        # append ticker to ticker lisr
                        tickers.append(str(result["symbol"])) # type: ignore
                        # dump the object in jsonl format
                        _write_file.write(foundation.json.dumps(result) + "\n")
        # ----- exception management -----
        except foundation.json.JSONDecodeError as je:
            foundation.log(f"json decode error ({processed_file}), {je}", self.__str__(), "error")
        except FileNotFoundError as fe:
            foundation.log(f"file({processed_file}) not found, {fe}", self.__str__(), "error")
        except KeyError as ke:
            foundation.log(f"key error, {ke}", self.__str__(), "error")
        except Exception as e:
            foundation.log(f"unexpected exception, {e}", self.__str__(), "error")
        # ----- exception management
        # get unique tickers
        tickers = list(set(tickers))
        return tickers