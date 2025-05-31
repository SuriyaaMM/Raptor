# Raptor

## Overview

Raptor is a comprehensive pipeline designed for automated financial news extraction, robust web content scraping, and advanced Natural Language Processing (NLP) analysis to Analyze the Market.

## Features

* **Financial News Extraction:** Automated fetching of financial news metadata (titles, URLs, authors) from popular APIs.
    * Integration with **NewsAPI** for general financial news.
    * Integration with **Yahoo Finance News** for company-specific news (using S&P 500 tickers).
* **Concurrent Web Scraping:** A highly efficient `article_scraper` module designed to fetch full article content from extracted URLs in parallel batches.
    * **Robust Error Handling:** Manages common scraping challenges including HTTP errors (e.g., 404, 500), network issues, and request timeouts.
    * **Advanced Content Extraction:** Employs sophisticated HTML parsing strategies using BeautifulSoup to maximize the extraction of meaningful text content from diverse web page structures, handling cases where content is not simply in `<p>` tags.
* **NLP Pipeline:** A core NLP module to process scraped articles for deeper insights.
    * **Sentiment Analysis:** Utilizes a pre-trained Hugging Face Transformer model (`sentiment_analyzer`) to determine the sentiment (Positive, Negative, Neutral) of article content.
        * **Long Text Handling:** Automatically chunks articles that exceed the model's context window using `RecursiveCharacterTextSplitter`, analyzes each chunk, and intelligently aggregates the sentiment results to provide an overall sentiment for the entire article.
    * **Entity Linking (Initial):** Basic functionality for identifying and linking financial entities within the text using a spaCy model (`entity_linker`). This lays the foundation for more targeted analysis of companies, products, or financial concepts mentioned in the news.
* **Modular ML Components:** Dedicated classes for key machine learning tasks:
    * `sentiment_analyzer`: Encapsulates the sentiment classification model.
    * `sentence_embedder`: Generates numerical embeddings for text, also handling long texts by chunking and averaging embeddings to ensure comprehensive representation.
    * `entity_linker`: Handles the identification and linking of named entities.
* **Structured Output:** All processed data and NLP analysis results are stored in clean, easy-to-parse JSONL (JSON Lines) format.
* **Deep Learning Pipeline** Incorporated Transformer based architecture to predict `buy/sell/hold` signals effectively acheive `80% accuracy`

## System Architecture

The Raptor pipeline follows a sequential, modular architecture to ensure clear separation of concerns and efficient data flow.

1.  **Extraction Layer:** News metadata (URL, Title, Author) is fetched from external APIs (NewsAPI, Yahoo Finance).
2.  **Scraping Layer:** The extracted URLs are concurrently processed to scrape the full article content.
3.  **NLP Processing Layer:** Scraped content undergoes sentiment analysis and optional entity linking.
4.  **Reporting Layer:** Final analysis results are stored in structured reports.
5. **Deep Learning Pipeline** Final deep learning predictions and report accuracy and loss
**Detailed System Design:**

![Raptor System Design](raptor_data/Raptor_System_Design.svg)

## Deep Learning Pipeline

The Deep Learning pipeline extends the data processing and feature engineering to incorporate a Transformer-based model for financial time series prediction.

### Data Processing and Feature Engineering (`fdata` module)

The `fdata` module handles the raw financial data, processing it into a format suitable for time-series modeling.

* **Data Loading:** Reads historical stock data (Close, High, Low, Open, Volume) from CSV files.
* **Technical Indicator Calculation:** Calculates a comprehensive set of technical indicators crucial for financial analysis using the `ta` (Technical Analysis) library.
* **Future Return Calculation:** Computes the `future_return` (1-week ahead percentage change) as the target variable.
* **Signal Labeling:** Classifies the `future_return` into discrete signals (`-1`, `0`, `1`) based on adaptive percentile thresholds (e.g., 30th and 70th percentiles). This transforms the regression problem into a 3-class classification task (e.g., "bearish", "neutral", "bullish").
* **Missing Value Handling:** Drops rows with any `NaN` values resulting from indicator calculations or lagging.

### Feature Scaling (Z-score Normalization)

To ensure numerical stability and improve model training performance, the calculated features undergo Z-score normalization.

* **Methodology:** Each feature is transformed to have a mean of 0 and a standard deviation of 1 using the formula: $$X_{\text{standardized}} = \frac{X - \mu}{\sigma}$$
* **Data Leakage Prevention:** Crucially, the mean ($\mu$) and standard deviation ($\sigma$) are calculated *only* from the training dataset. These parameters are then applied consistently to both the training and test datasets after the train-test split, preventing any information leakage from the test set.

### Model Architecture (`ftransformer` in `dl_models` module)

* **Transformer-based Model:** A custom Transformer Encoder model is used, leveraging its ability to capture long-range dependencies in sequential data, which is highly beneficial for time series.
    * It incorporates positional embeddings to account for the order of features within the time window.
    * A linear layer maps the input features to the Transformer's `d_model` dimension.
    * The output of the Transformer encoder is averaged across the sequence dimension and passed through a final linear classifier to predict the 3-class signal.

## Results and Model Selection Rationale

- Initially, traditional machine learning algorithms like `RandomForestClassifier` were explored for this classification task. While Random Forests are powerful, non-linear, and relatively interpretable models, they exhibited significant limitations when applied to financial time series data, leading to the decision to pivot towards a Transformer-based deep learning approach.
- `Transformer` based model increasing the accuracy to `80%` whilist the `RandomForestClassifier` after all finetuning acheive only `40%`
- For Detailed Reports check the `reports` directory

## Current Progress

As of the current development phase, the following core functionalities have been implemented and tested:

* **News Extraction:** Both `newsapi_extractor` and `yfnews_extractor` are functional, successfully fetching news metadata and storing it in JSONL files.
* **Web Scraping:** The `article_scraper` is robust. It performs concurrent HTTP requests, handles various network and HTTP errors, and employs advanced BeautifulSoup logic to extract article content. It processes the entire input file in a single pass (resumability is a planned future enhancement).
* **NLP Pipeline:**
    * The `nlp_pipeline` class is capable of reading processed articles.
    * **Sentiment Analysis** is fully integrated, including the automatic chunking of long texts and aggregation of sentiment results.
    * **Entity Linking** has been set up with a basic spaCy model integration, ready for further refinement and domain-specific entity recognition.
* **Modular Design:** The project adheres to a modular structure, with clear separation between data extraction, scraping, and NLP components, facilitating maintainability and future expansion.
* **Error Handling & Logging:** Comprehensive `try-except` blocks and detailed logging are implemented across all modules to ensure robust operation and easy debugging.
* **Deep Learning Model Training:** The Transformer-based model for financial time series prediction is integrated and shows promising learning progress with Z-score normalized data.

## Getting Started

Follow these steps to set up and run the Raptor pipeline locally.

### Prerequisites

* conda (I used 25.3.1)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SuriyaaMM/Raptor.git](https://github.com/SuriyaaMM/Raptor.git)
    cd Raptor
    ```
2.  **Install environment.yaml:**
    ```bash
    # Assuming you have an environment.yaml for conda
    conda env create -f environment.yaml
    conda activate your_env_name # Replace with your environment name
    ```
3.  **Download spaCy model:**
    The NLP pipeline uses a spaCy model. Download the small English model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

### Configuration (`Environment.json` file)

Create a `Environment.json` file in the root directory of the project and add your API keys and configuration settings.

```json
{
    "NEWSAPI_API_KEY" : "<YOUR_NEWSAPI_API_KEY>",
    
    "MASTER_DB" : {
        "NAME" : "<YOUR_NAME>",
        "USER" : "<USER>",
        "HOST" : "<HOST>",
        "PORT" : "<HOST>"
    }
}