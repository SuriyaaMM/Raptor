from article_analyzer import newsapi_extractor, article_scraper, yfnews_extractor
from natural_language_processer import nlp_pipeline
from fdata import raw_fdata
from dl_models import fdataset, ftransformer, train_model, evaluate_saved_model
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
    #tickers = nlp.process(foundation.env.fnews_processed_yf_filename)

    fdata = raw_fdata()
    #fdata._fetch_raw_fdata(tickers = tickers[:100])

    fdata.process_train_test_data()

    split_index = int(len(fdata.fdata_train) * 0.8)

    train_df = fdata.fdata_train[:split_index].copy()
    test_df = fdata.fdata_train[split_index:].copy()

    train_dataset = fdataset(train_df, train = True) # type:ignore
    test_dataset = fdataset(test_df) #type:ignore

    train_loader = foundation.torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
    test_loader = foundation.torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = True)
    
    device = "cuda" if foundation.torch.cuda.is_available() else "cpu"

    foundation.log(f"torch version : {foundation.torch.__version__}", __name__)
    foundation.log(f"using device : {foundation.torch.device(device)}", __name__)

    model = ftransformer(input_dim = train_dataset[0][0].shape[1], 
                         num_classes = 3)
    
    train_model(model, train_loader, device, epochs = 300)

    evaluate_saved_model(foundation.env.model_transformer_path, 
                         test_loader, 
                         device, 
                         input_dim = train_dataset[0][0].shape[1], 
                         num_classes = 3)