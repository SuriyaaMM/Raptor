import requests
import spacy
import os
import datetime
import json
import psycopg
import faiss

import numpy        as np
import jax.numpy    as jnp
import pandas       as pd

from bs4 import BeautifulSoup
from tqdm import tqdm
from pprint import pprint
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer

class _environment(object):

    def __init__(self):

        self.environment_file = "Environment.json"
        self.environment_vars = None

        with open(self.environment_file, "r") as _file:
            self.environment_vars = json.load(_file)
        
        self.NEWSAPI_API_KEY    =  self.environment_vars["NEWSAPI_API_KEY"]

        self.base_dir   = os.path.dirname(__file__)
        self.cache_dir  = os.path.join(self.base_dir, "raptor_cache")
        self.int_dir    = os.path.join(self.base_dir, "raptor_int")
        self.data_dir   = os.path.join(self.base_dir, "raptor_data")

        os.makedirs(self.int_dir, exist_ok = True)
        os.makedirs(self.cache_dir, exist_ok = True)
        os.makedirs(self.data_dir, exist_ok=True)

        self.s_and_p_500_tickers_path           = os.path.join(self.data_dir, "s_and_p_500_tickcers.csv")
        self.latest_finance_news_filename       = os.path.join(self.int_dir, "latest_finance_news.tmp")
        self.processed_finance_news_filename    = os.path.join(self.int_dir, "processed_finance_news.tmp")

        self.MASTER_DB_INFO     = self.environment_vars["MASTER_DB"]
        self.MASTER_DB_CONNINFO = f"dbname={self.MASTER_DB_INFO["NAME"]} user={self.MASTER_DB_INFO["USER"]} host={self.MASTER_DB_INFO["HOST"]} port={self.MASTER_DB_INFO["PORT"]}"

        self.spacy_model    = "en_core_web_sm"
        self.embedder_model = "sentence-transformers/all-MiniLM-L6-v2"
env = _environment()
# Add this right after env = _environment()
import os

print("=== PATH DEBUG INFO ===")
print(f"__file__ = {__file__}")
print(f"os.path.dirname(__file__) = {os.path.dirname(__file__)}")
print(f"os.getcwd() = {os.getcwd()}")
print(f"base_dir = {env.base_dir}")
print(f"int_dir = {env.int_dir}")
print(f"processed_finance_news_filename = {env.processed_finance_news_filename}")
print(f"File exists? {os.path.exists(env.processed_finance_news_filename)}")
print(f"Is file? {os.path.isfile(env.processed_finance_news_filename)}")
print(f"Absolute path = {os.path.abspath(env.processed_finance_news_filename)}")

# Check if directory exists
print(f"Directory exists? {os.path.exists(env.int_dir)}")
print(f"Directory contents:")
if os.path.exists(env.int_dir):
    for item in os.listdir(env.int_dir):
        full_path = os.path.join(env.int_dir, item)
        print(f"  {item} ({'file' if os.path.isfile(full_path) else 'dir'})")

# Manual check of the expected file
expected_path = "/home/mms/dev/Raptor/raptor_int/processed_finance_news.tmp"
print(f"Manual path check: {os.path.exists(expected_path)}")

print("=== END DEBUG INFO ===")

COLOR_RESET     = "\033[0m"
COLOR_RED       = "\033[91m" 
COLOR_YELLOW    = "\033[93m"    
COLOR_GREEN     = "\033[92m"  

def log(message:str, class_name:str, severity:str | None = None):

    prefix      = f"[Raptor][{class_name}]:"
    colour_code = COLOR_GREEN

    if severity is not None:
        
        if severity.lower() == "warn":
            colour_code = COLOR_YELLOW
        elif severity.lower() == "error":
            colour_code = COLOR_RED
    
    print(f"{colour_code}{prefix} {message}{COLOR_RESET}")

    if severity is not None and severity.lower() == "error":
        os.abort()

