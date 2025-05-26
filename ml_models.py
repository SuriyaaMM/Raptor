import foundation

class sentence_embedder(object):

    def __str__(self):
        return "sentence_embedder"
    
    def __init__(self, embedder_name : str = foundation.env.model_name_embedder):
        # load/download embedder
        self.embedder = foundation.SentenceTransformer(embedder_name, cache_folder = foundation.env.cache_dir)

    def embed(self, sentence: str):
        """
            embeds the sentence by utilizing the sentence transformer intialized when constructing
            
            Args:
                sentence (str): sentence to be embedded, must be less than 384 tokens!, else
                won't be embedded properly!
        """
        return self.embedder.encode(sentence)
    

class sentiment_analyzer(object):

    def __str__(self):
        return "sentiment_analyzer"
    
    def __init__(self, 
                 model_name : str = foundation.env.model_name_sentiment_analyzer,
                 chunk_overlap: int = 32):
        # load/download classification model and tokenizer
        self.model = foundation.AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir = foundation.env.cache_dir)
        self.tokenizer = foundation.AutoTokenizer.from_pretrained(model_name, cache_dir = foundation.env.cache_dir)
        # pipeline
        self.pipeline = foundation.pipeline(task = "text-classification", model = self.model, tokenizer = self.tokenizer)
        # max context window or fallback to 512
        self.max_context_window = getattr(self.tokenizer, "max_model_length", 512)
        # initialize text splitter
        self.text_splitter = foundation.RecursiveCharacterTextSplitter(
            chunk_size = (int)(0.9 * self.max_context_window * 3.5),
            chunk_overlap = chunk_overlap,
            length_function = len,
            is_separator_regex = False)
        
        # Check if the sentence exceeds the model's approximate token limit (using character count as heuristic)
        # Rough heuristic: average English word is ~5 chars, 1 token ~ 0.75 words, so 1 token ~ 3-4 chars.
        # Let's use 3.5 characters per token as a rough estimate.
        self.approx_char_limit = self.max_context_window* 3.5
        # default report object
        self.default_report_object = {"label" : "Neutral", "score" : 0.0, "details" : "default"}

    def analyze(self, sentence: str):
        """
            performs sentiment-analyzes (text-classification) and produces result and confidence

            Args:
                sentence (str): sentence to be analyzed, must be less than context window!, else
                won't be analyzed properly!
        """
        # IF it exceeds threshold then chunk it and processe it
        if(len(sentence) > self.approx_char_limit):
            # split into chunks
            chunks = self.text_splitter.split_text(sentence)
            # IF unable to split, then return default object
            if not chunks:
                foundation.log("unable to split into chunks, returning default", self.__str__(), "warn")
                return self.default_report_object
            # chunk results
            chunk_results = []
            # process each chunk
            for i, chunk in enumerate(chunks):
                # IF empty chunk, skip it
                if not chunk.strip():
                    foundation.log("skipping chunk because it didn't contain any info", self.__str__(), "warn")
                # analyze the chunk and append it
                chunk_results.append(self.pipeline(chunk)[0]) # type:ignore
            # return aggegrated sentiment analysis
            return self._aggregate_sentiment_analysis(chunk_results)

        # ELSE return just passing it through the pipeline
        else:
            return self.pipeline(sentence)[0] # type:ignore
    
    def _aggregate_sentiment_analysis(self, chunk_results: list, neutral_suppression_factor: float = 0.8):
        # aggregated result
        aggregated_result = 0.0
        # label map
        label_map = {"Positive" : 0, "Negative" : 0, "Neutral" : 0}
        # try aggregating
        try:
            # FOREACH chunk in chunk result
            for i, chunk_result in enumerate(chunk_results):
                # increase count in label map
                label_map[chunk_result["label"]] += 1
                # IF chunk result is NEUTRAL, then suppress it and add
                if chunk_result["label"] == "Neutral":
                    aggregated_result += (1 - 0.8) * chunk_result["score"]
                # ELSE just add it
                else:
                    aggregated_result += chunk_result["score"]
        # ----- exception management -----
        except KeyError as ke:
            foundation.log(f"key error, {ke}", self.__str__(), "error")
        except Exception as e:
            foundation.log(f"unexpected exception, {e}", self.__str__(), "error")
        # ----- exception management -----

        # majority label
        majority_label = "Neutral"
        majority_count = 0
        # total labels
        total_labels = len(chunk_results)
        # find majority label
        for label, count in label_map.items():

            if count > majority_count:
                majority_label = label
                majority_count = count

        return {"label" : majority_label, "score" : aggregated_result/total_labels, "details" : "aggregated"}
        

