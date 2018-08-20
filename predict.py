# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:29:17 2018
@author: Moc
"""

import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import jieba
import numpy as np
import pandas as pd

def _readfile(path):
    with open(path, "r",encoding='gbk',errors='ignore') as fp:
        content = fp.read()
    return content
    
stop_words = _readfile('data/stop_words.txt').splitlines()


string = '做父母一定要有刘墉这样的心态，不断地学习，不断地进步，不断地给自己补充新鲜血液，让自己保持一颗年轻的心。我想，这是他能很好的和孩子沟通的一个重要因素。读刘墉的文章，总能让我看到一个快乐的平易近人的父亲，他始终站在和孩子同样的高度，给孩子创造着一个充满爱和自由的生活环境。很喜欢刘墉在字里行间流露出的做父母的那种小狡黠，让人总是忍俊不禁，父母和子女之间有时候也是一种战斗，武力争斗过于低级了，智力较量才更有趣味。所以，做父母的得加把劲了，老思想老观念注定会一败涂地，生命不息，学习不止。家庭教育，真的是乐在其中。'
words = jieba.cut(string)
words = [word for word in words if word not in stop_words]
segmented_words = ','.join(words)

word_list = []
word_list.append(segmented_words.strip())
word_list_t = pd.Series(word_list)

tfidf_vectorizer_path = 'data/result/tfidf_vectorizer.pkl'

#加载已训练的tfidf_vectorizer
with open(tfidf_vectorizer_path, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
    
X_test_tfidf = tfidf_vectorizer.transform(word_list_t)    

tfidf_path = 'data/result/tfidf.pkl'
print("正在加载已经训练的模型...")
with open(tfidf_path, 'rb') as out_data:
    clf_tfidf = pickle.load(out_data)
    
y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
y_predicted_tfidf_ = clf_tfidf.predict_proba(X_test_tfidf)
print(y_predicted_tfidf[0], y_predicted_tfidf_)