from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_20newsgroups 

categories = ['alt.atheism', 'sci.med'] 

train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True) 

perceptron = Perceptron(max_iter=100) 

from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer()
X_train_counts = cv.fit_transform(train.data)

from sklearn.feature_extraction.text import TfidfTransformer 
tfidf_tf = TfidfTransformer()
X_train_tfidf = tfidf_tf.fit_transform(X_train_counts)

perceptron.fit(X_train_tfidf,train.target) 

test_docs = ['Religion is widespread, even in modern times', 'His kidney failed','The pope is a controversial leader', 'White blood cells fight off infections','The reverend had a heart attack in church'] 

X_test_counts = cv.transform(test_docs)
X_test_tfidf = tfidf_tf.transform(X_test_counts)

pred = perceptron.predict(X_test_tfidf) 

for doc, category in zip(test_docs, pred):
    print('%r => %s' % (doc, train.target_names[category]))

#############################################################
# duplicate with tweet data
import pandas as pd
import numpy as np
tweets = pd.read_csv('../data/training/tweets.tsv', sep='\t', header=None, names=['tweet_id', 'tweet'])
classes = pd.read_csv('../data/training/class.tsv', sep='\t', header=0)
tweets_ls = tweets['tweet'][:17100].tolist()
target_classes = np.array(classes['label'][:17100])

# “Fits the familiar CountVectorizer on our training data”
cv = CountVectorizer()
X_train_counts2 = cv.fit_transform(tweets_ls)
# “Loads, fits, and deploys a TF.IDF transformer from sklearn. It computes TF.IDF representations of our count vectors”
tfidf_tf = TfidfTransformer()
X_train_tfidf2 = tfidf_tf.fit_transform(X_train_counts2)

# trains perceptron on TF.IDF representations of our count vectors
perceptron.fit(X_train_tfidf2,target_classes)

test_tweets = tweets['tweet'][17100:].tolist()
# “Vectorizes the test data: first to count vectors and then to TF.IDF vectors”
X_test_counts2 = cv.transform(test_tweets)
X_test_tfidf2 = tfidf_tf.transform(X_test_counts2)

# apply perceptron to test data
pred2 = perceptron.predict(X_test_tfidf2)

# print results
for nr, tweet in enumerate(test_tweets):
    print('%r => %s' % (test_tweets[nr], pred2[nr]))
# truth: none of the test tweets are ADE
# RESULTS:
# 'Spana symbol! RT @USER_______: What a productive day with my niggas at CIPRO' => NoADE
# "i haven't taken my prozac in over a month and i can't tell if it's affecting me or not" => NoADE
# "@USER________ @USER__ YEP! Let's try! @USER___ are you in?" => NoADE
# "these new class of anticoagulants 'dabigatran, rivaroxaban and apixaban' - sounds like transformer names!  #pharmacy #medicine" => NoADE
# 'These adderall pics got me dyingggg.' => NoADE
# '@USER________ then try to knock you out at night with a little trazadone or seroquel LOL...  so not right' => NoADE
# '@USER__ @USER___________ @USER________ ride/run/cuppa/pint/lozenge... will do! thanks guys.' => NoADE
# 'Thank god for vyvanse' => NoADE
# "@USER_ seroquel, an anti-psychotic, usually for schiz people but i'm on a small dose." => ADE
# '@USER________ and with seroquel i take half of a 25mg pill. and it knocks me out; i can sleep for 12 straight hours' => NoADE
# "went to the doc.  she ordered me more metformin and gave me a blood glucose monitor.  and ordered labs...which i'm going to get done on wed." => NoADE
# 'Lunesta in Hand: Good night To a great weekend' => NoADE
# 'hawkins killing my roll up with his cowardice today. 3-0 up &amp; played like a blind thalidomide ever since. HTTPURL_______________' => NoADE
# '@USER________ i have never tried remicade. just failed humira for the 2nd time.' => NoADE
# 'Still lack of clarity on how to reverse the effect of rivaroxaban & other new oral anticoagulants -ongoing concern for Emergency Dept' => NoADE
# '@USER_________ I just had a look buddy, and my medication (Seroquel) does affect tolerance to the sun.' => NoADE
# "FYI: I've got sore eyes and I'm using this levofloxacin 3x a day. Eye drops, why is this such a challenge? I just kept missing -_-" => NoADE
# 'every time i take vyvanse to be productive i just end up sitting on tumblr and twitter all day.' => NoADE
# "Guess since I'm not retiring yet, then i don't have to get up tomorrow. Beer and Lunesta for a midnight snack." => NoADE
# '@USER______ CHANGE THE RULES... Warehouse 13 should have been sponsored by Kleenex & Prozac  Im such a sad Fangirl.. That was No love letter' => NoADE