import foundation

# finance transformer class
class ftransformer(foundation.nn.Module):

    def __str__(self):
        return "ftransformer"
    
    def __init__(self, input_dim : int, 
                 classifier_dim : int = 64, 
                 num_heads : int = 8, 
                 num_layers : int = 2, 
                 num_classes : int = 3, 
                 window_length : int = 7):
        """
            initializes model transformer for predicting stock signals

            Args:
                input_dim (int) : dimension to be passed to linear layer for classification
                classifier_dim (int) : hidden dimension of classification layer
                num_heads (int) : number of heads in transformer encoder
                num_layers (int) : num transformer layers
                num_classes (int) : output classification class count
                window_length (int) : window period to consider for future_return etc
        """
        # initialize nn.Module
        super().__init__()
        # positional embedding of classification class
        self.positional_embedding = foundation.nn.Parameter(foundation.torch.randn(1, window_length, classifier_dim))
        # linear layer 1
        self.linear1 = foundation.nn.Linear(input_dim, classifier_dim)
        # transformer encoder layer
        encoder_layer = foundation.nn.TransformerEncoderLayer(d_model = classifier_dim, nhead = num_heads, batch_first = True)
        # transformer itself
        self.transformer = foundation.nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers = num_layers)
        # classifier
        self.classifier = foundation.nn.Linear(classifier_dim, num_classes)

    def forward(self, x):
        # perform forward propagation
        x = self.linear1(x) + self.positional_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim = 1)
        return self.classifier(x)

# finance dataset handler class for compatibility with torch dataset
class fdataset(foundation.torch.utils.data.Dataset):
    
    def __str__(self):
        return "fdataset"
    
    def __init__(self, df: foundation.pd.DataFrame, train : bool = False, window_length : int = 7, zero_tolerance: float = 1e-7):

        self.window_length = window_length
        self.sequences = []
        self.labels = []
        # debug info
        foundation.log(f"processing Dataframe with window length {window_length}", self.__str__())
        # feature columns to consider
        feature_cols = [
            "Close", "Volume", "ema", "sma", "rsi", "macd", "macd_signal",
            "macd_diff", "daily_return", "momentum", "volatility",
            "Close_lag1", "Volume_lag1"
        ]
        # normalization (z - score)
        means = df[feature_cols].mean()
        stds = df[feature_cols].std()
        stds[stds == 0] = zero_tolerance
        df[feature_cols] = (df[feature_cols] - means)/stds
        # debug info
        foundation.log(f"feature columns: {feature_cols}", self.__str__())
        foundation.log(f"normalized given dataframe on above feature columns", self.__str__())
        foundation.log(f"shape of df before dropping: {df.shape}", self.__str__())
        # drop NaN rows
        df = df.dropna().reset_index()
        # debug info
        foundation.log(f"shape of db after dropping & reset indices: {df.shape}", self.__str__())
        # group and slide window for window_length period of data capturing
        for ticker, group in df.groupby("Ticker"):
            # sort by date
            group = group.sort_values("Date").reset_index()
            # x is features
            x = group[feature_cols].values
            # y is target
            y = group["signal_label"].values
            # perform sliding window
            for i in range(len(group) - window_length):
                self.sequences.append(x[i:i+window_length])
                self.labels.append(y[i+window_length - 1])

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # convert to appropriate datatypes before returning
        return(foundation.torch.tensor(self.sequences[idx], dtype=foundation.torch.float32),
               foundation.torch.tensor(self.labels[idx], dtype=foundation.torch.long))
    

def train_model(model : ftransformer, 
                train_loader : foundation.torch.utils.data.DataLoader, 
                device : str, 
                lr : float = 3e-4,
                epochs : int = 50):
    """
        train's the transformer model using ftransformer and fdataset
    """
    __str = "train_model"
    # send model to device
    model = model.to(device)
    # cross entryopy loss for multi-label classification
    criterion = foundation.nn.CrossEntropyLoss()
    # adam optimizer 
    optimizer = foundation.torch.optim.Adam(model.parameters(), lr = lr)
    # perform epoch based training
    for epoch in range(epochs):
        # training mode
        model.train()
        # total loss
        total_loss = 0
        # for accuracy calculation
        correct_predictions = 0
        total_samples = 0
        # perform batchwise training
        for x_batch, y_batch in train_loader:
            # send to gpu IF one is available
            x_batch, y_batch = x_batch.to(device), (y_batch + 1).to(device)
            # zero any accumulated gradients in the chain
            optimizer.zero_grad()
            # calculate output
            outputs = model(x_batch)
            # calculate loss
            loss = criterion(outputs, y_batch)
            # backpropagate through computation chain
            loss.backward()
            # clip gradients preventing gradient explosion
            foundation.torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # accumulate total loss
            total_loss += loss.item()
            # calculate accuracy parameters
            _, predicted = foundation.torch.max(outputs.data, 1)
            total_samples += y_batch.size(0)
            correct_predictions += (predicted == y_batch).sum().item()

        # Calculate average accuracy for the epoch
        train_accuracy = 100 * correct_predictions / total_samples
        # debug info
        foundation.log(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%", "train_model")

    foundation.torch.save(model.state_dict(), foundation.env.model_transformer_path)
    foundation.log(f"model saved to {foundation.env.model_transformer_path}", __str)

def evaluate_saved_model(model_path : str, 
                         test_loader : foundation.torch.utils.data.DataLoader, 
                         device : str, 
                         input_dim : int, 
                         num_classes : int = 3):
    """
        loads a saved model and evaluates it on the test dataset.
    """
    __str = "evaluate_saved_model"
    # load the saved model
    model = ftransformer(input_dim=input_dim, num_classes=num_classes)
    # send to gpu IF one is available
    model.to(device)

    # load state dict
    model.load_state_dict(foundation.torch.load(model_path, map_location=device))
    # debug info
    foundation.log(f"model loaded from {model_path}", __str)
    # evaluation mode
    model.eval()
    # perform evaluation
    criterion = foundation.nn.CrossEntropyLoss()
    test_loss = 0
    correct_test_predictions = 0
    total_test_samples = 0
    # disable gradient accumulation for evaluation
    with foundation.torch.no_grad(): 
        iteration = 1
        # evaluate batchwise
        for x_batch_test, y_batch_test in test_loader:
            # send batches to gpu IF one is available
            x_batch_test, y_batch_test = x_batch_test.to(device), (y_batch_test + 1).to(device)
            # calculate outputs
            test_outputs = model(x_batch_test)
            # calculate loss
            this_test_loss = criterion(test_outputs, y_batch_test).item()
            test_loss += this_test_loss
            # calculate accuracy
            _, predicted_test = foundation.torch.max(test_outputs.data, 1)
            # this test's sample size
            this_test_samples = y_batch_test.size(0)
            # this test's correct predictions
            this_test_correct_predictions = (predicted_test == y_batch_test).sum().item()
            total_test_samples += this_test_samples
            correct_test_predictions += this_test_correct_predictions
            this_test_accuracy = 100 * (this_test_correct_predictions) / this_test_samples
            # debug info
            foundation.log(f"test results on batch {iteration}\n\t accuracy: {this_test_accuracy} \n\t loss: {this_test_loss}", __str)
            iteration += 1
    # average test loss
    avg_test_loss = test_loss / len(test_loader)
    # average test accuracy
    test_accuracy = 100 * correct_test_predictions / total_test_samples

    foundation.log(f"----- evaluation results -----", __str)
    foundation.log(f"net loss: {avg_test_loss:.4f}", __str)
    foundation.log(f"net accuracy: {test_accuracy:.2f}%", __str)
