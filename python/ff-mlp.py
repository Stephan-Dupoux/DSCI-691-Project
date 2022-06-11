

# PRE-PROCESSING    
tweets = pd.read_csv('./data/tweets.tsv', sep='\t', header=None,
                     names=['tweet_id', 'tweet'])
classes = pd.read_csv('./data/class.tsv', sep='\t', header=0)

data = pd.merge(tweets, classes, how='left')

data = data.replace(r'@\w+', '', regex=True)

# remove any emoji from the tweet
data = data.replace(r'[^\w\s]', '', regex=True)

data = data.replace(['NoADE', 'ADE'], [0, 1])


strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=691)
X = data['tweet'].to_numpy()
y = data['label'].to_numpy()

for train_index, test_index in strat_split.split(X, y):
    print(f"Train index: {train_index}", f"Test index: {test_index}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# 4. text representation
# convert tweets to matrix of word counts and remove stop words


countvec = CountVectorizer(stop_words='english')

# normalise count matrix to decrease the effect of word frequencies


tfidf = TfidfTransformer()

# vectorize and transform train and test data
train_transformed = tfidf.fit_transform(countvec.fit_transform(X_train))
test_transformed = tfidf.transform(countvec.transform(X_test))


from sklearn.neural_network import MLPClassifier

activation = 'logistic'
nnet = MLPClassifier(solver='adam', random_state=691, max_iter=300, activation = activation)

# fit
nnet.fit(train_transformed, y_train)
y_pred = nnet.predict(test_transformed)

# print results
print("Multilayer Perceptron: with sigmoid activation")
print(f"AUC: {roc_auc_score(y_test, y_pred)}")  #0.6867425757489182
print(f"Precision: {precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)[0]:.2f}") # 0.53
print(f"Recall: {precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)[1]:.2f}") # 0.40
print(f"F1 Score: {precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label=1)[2]:.2f}") # 0.46