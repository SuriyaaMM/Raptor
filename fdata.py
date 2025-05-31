import foundation

class raw_fdata(object):
    def __str__(self):
        return "raw_fdata"
    
    def __init__(self):
        # format string for yfinance api
        self._format_string = "%Y-%m-%d"
        # 1 day difference
        _delta_1day = foundation.datetime.timedelta(days = -1)
        # 8 week difference
        _delta_8week = foundation.datetime.timedelta(weeks = -8)
        # time interval to download
        self._interval = "1d"
        # yesterday object
        self._end= foundation.datetime.datetime.now() + _delta_1day
        # last week object
        self._start = foundation.datetime.datetime.now() + _delta_8week
        # format object to strings
        self._start = self._start.strftime(self._format_string)
        self._end = self._end.strftime(self._format_string)
        # training data time periods
        self._start_train = "2020-01-01"
        self._end_train = "2023-12-31"
        # initialize fdata object
        self.fdata = foundation.pd.DataFrame()
        self.fdata_train = foundation.pd.DataFrame()

    def process_train_test_data(self):
        # process train data & test data
        self.fdata = self._process_raw_fdata(foundation.env.raw_fdata_csv, foundation.env.processed_fdata_csv)
        self.fdata_train = self._process_raw_fdata(foundation.env.raw_fdata_csv_train, foundation.env.processed_fdata_csv_train)

    def _fetch_raw_fdata(self, tickers: list):
        """
            fetches finance data from yahoo finance for the tickers
            start = yesterday
            end = today

            writes them to csv
        """
        foundation.log(f"downloading data for({tickers} from {self._start} to {self._end}) with interval {self._interval}", self.__str__())
        self.fdata = foundation.pd.DataFrame(foundation.yf.download(tickers = tickers, 
                                                                    start = self._start, 
                                                                    end = self._end, 
                                                                    interval = self._interval))
        self.fdata_train = foundation.pd.DataFrame(foundation.yf.download(tickers = tickers, 
                                                                    start = self._start_train, 
                                                                    end = self._end_train, 
                                                                    interval = self._interval))
        # stack the data
        self.fdata = self.fdata.stack(future_stack = True)
        self.fdata_train = self.fdata_train.stack(future_stack = True)
        # write to csv
        self.fdata.to_csv(foundation.env.raw_fdata_csv)
        self.fdata_train.to_csv(foundation.env.raw_fdata_csv_train)

    # simple function to apply for labelling
    def _label_returns_percentile(self, future_return_pct, upper_thresh, lower_thresh):
        if future_return_pct >= upper_thresh:
            return 1
        elif future_return_pct <= lower_thresh:
            return -1
        else:
            return 0

    def _process_raw_fdata(self, filepath : str, write_filepath : str):
        """
            loads the csv and calculates the indicators
        """
        # debug info
        foundation.log(f"loading csv from {filepath}", self.__str__())
        # read the csv & parse the date as python datetime
        df = foundation.pd.read_csv(filepath, parse_dates = ["Date"])
        # sort by date and ticker, crucial for time-series and lagging
        df = df.sort_values(by = ["Date", "Ticker"]).copy()

        # calculate technical indicators (ema, sma, macd, rsi, daily_return, momemtum, volatality)
        df["ema"] = df.groupby("Ticker")["Close"].transform(
            lambda x: foundation.EMAIndicator(x, window = 14).ema_indicator())
        df["sma"] = df.groupby("Ticker")["Close"].transform(
            lambda x: foundation.SMAIndicator(x, window = 14).sma_indicator())
        df["rsi"] = df.groupby("Ticker")["Close"].transform(
            lambda x: foundation.RSIIndicator(x, window = 14).rsi())
        df["macd"] = df.groupby("Ticker")["Close"].transform(
            lambda x: foundation.MACD(x).macd())
        df["macd_diff"] = df.groupby("Ticker")["Close"].transform(
            lambda x: foundation.MACD(x).macd_diff())
        df["macd_signal"] = df.groupby("Ticker")["Close"].transform(
            lambda x: foundation.MACD(x).macd_signal())
        df["daily_return"] = df.groupby("Ticker")["Close"].pct_change() * 100
        df["momentum"] = df.groupby("Ticker")["Close"].transform(
            lambda x: (x - x.shift(7)) / x.shift(7))
        df["volatility"] = df.groupby("Ticker")["Close"].transform(
            lambda x: x.pct_change().rolling(5).std())

        # calculate future return (1 week)
        df["future_return"] = df.groupby("Ticker")["Close"].transform(
            lambda for_every_close_in_close: (for_every_close_in_close.shift(-7) / for_every_close_in_close) - 1) * 100
        # close lag and volume lag
        df["Close_lag1"] = df.groupby("Ticker")["Close"].shift(-1)
        df["Volume_lag1"] = df.groupby("Ticker")["Volume"].shift(-1)

        # Compute adaptive thresholds
        upper_thresh = df["future_return"].quantile(0.70)
        lower_thresh = df["future_return"].quantile(0.30)
        # labels for classification by RandomForest algorithm
        df["signal_label"] = df["future_return"].apply(
            lambda x: self._label_returns_percentile(x, upper_thresh, lower_thresh))

        # -------------------- STRICTLY FOR RANDOM FOREST CLASSIFIER (BEGIN) --------------------
        # # reset index for one hot encoding
        # df.reset_index()
        # # perform one hot encoding
        # df = foundation.pd.get_dummies(df, columns = ["Ticker"], prefix = "Ticker")
        # -------------------- (END) --------------------

        # write to csv
        df.to_csv(write_filepath)
        # debug info
        foundation.log(f"processed raw data and written to {write_filepath}", self.__str__())
        return df

# -------------------- STRICTLY FOR RANDOM FOREST CLASSIFIER (BEGIN) --------------------
# class rf_classifier(object):

#     def __str__(self):
#         return "rf_classifier"
    
#     def __init__(self, df: foundation.pd.DataFrame):

#         feature_columns = [
#             "Close", "High", "Low", "Volume", 
#             "ema", "sma", "rsi", 
#             "macd", "macd_signal", "macd_diff",
#             "Close_Lag1", "Volume_Lag1", "momentum", "volatality", "daily_return"]
        
#         ticker_columns = [col for col in df.columns if col.startswith("Ticker_")]

#         self.feature_columns = list(set(ticker_columns + feature_columns))
#         self.feature_columns = [f for f in self.feature_columns if f in df.columns]
        
#         self.classifier = foundation.RandomForestClassifier(class_weight = "balanced",
#                                                             n_estimators = 400, 
#                                                             verbose = 2, 
#                                                             max_depth = 32,
#                                                             min_samples_split = 2,
#                                                             min_samples_leaf = 5, n_jobs = -1)
#         self.smote = foundation.SMOTE(random_state = 42)
#         self.x_train = df[self.feature_columns].fillna(0)
#         self.y_train = df["signal_label"].fillna(0)

#         pipeline = foundation.ImbPipeline([
#             ('smote', self.smote),
#             ('classifier', self.classifier)
#         ])

#         param_grid = {
#         'classifier__n_estimators': [100, 200, 300],
#         'classifier__max_depth': [5, 10, 15],
#         'classifier__min_samples_split': [2, 4],
#         'classifier__min_samples_leaf': [1, 3]}

#         grid_search = foundation.GridSearchCV(
#             estimator=pipeline,
#             param_grid=param_grid,
#             scoring="accuracy",  # Or 'accuracy', 'roc_auc_ovr', etc.
#             cv=3,
#             verbose=1,
#             n_jobs=-1
#         )

#         grid_search.fit(self.x_train, self.y_train)
#         print("Best Params:", grid_search.best_params_)
#         print("Best Score:", grid_search.best_score_)

#         # Final classifier
#         self.classifier = grid_search.best_estimator_

        
#         self.x_train_resampled, self.y_train_resampled = self.smote.fit_resample(self.x_train, self.y_train) # type:ignore

#         self.classifier.fit(self.x_train_resampled, self.y_train_resampled)

#     def predict(self, x_test: foundation.pd.DataFrame):
#         """
#         Makes predictions on new data using the trained classifier.
#         Ensures the test data has the same columns as the training data.
#         """
#         # Ensure x_test has the same columns as x_train, adding missing ones with 0 if necessary
#         # and dropping extra columns.
#         missing_cols = set(self.feature_columns) - set(x_test.columns)
#         for c in missing_cols:
#             x_test[c] = 0
        
#         # Ensure the order of columns is the same as during training
#         x_test_processed = x_test[self.feature_columns]
#         y_test = x_test["signal_label"]

#         foundation.log(f"Making predictions on {len(x_test_processed)} samples.", self.__str__())
#         predictions = self.classifier.predict(x_test_processed)
#         foundation.log(f"Predictions made successfully.", self.__str__())
#         print(foundation.classification_report(predictions, y_test))

#         print("ROC-AUC:", foundation.roc_auc_score(y_test, self.classifier.predict_proba(x_test_processed), multi_class='ovo'))
#         print("Confusion Matrix:")
#         print(foundation.confusion_matrix(y_test, predictions))
#         return predictions

#     def generate_prediction_report(self, predictions: foundation.pd.Series):
#         """
#         Generates a sample report for the predictions.
#         """
#         report = f"--- Prediction Report ---\n"
#         report += f"Total predictions: {len(predictions)}\n"
#         report += f"Predicted signal distribution:\n"
        
#         # Calculate value counts for each predicted label
#         signal_counts = foundation.pd.Series(predictions).value_counts().sort_index()
        
#         for label, count in signal_counts.items():
#             if label == 1:
#                 report += f"  - Buy/Up Signal (1): {count} samples\n"
#             elif label == -1:
#                 report += f"  - Sell/Down Signal (-1): {count} samples\n"
#             else: # label == 0
#                 report += f"  - Neutral Signal (0): {count} samples\n"
        
#         foundation.log("Prediction report generated.", self.__str__())
#         return report
 # -------------------- (END) --------------------