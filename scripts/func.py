import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
from scipy import stats
import statistics
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import math
import nltk
import json


def label_trump_biden(row):
    title = row.title
    trump = (re.search(r"[^\w\d][Tt][rR][uU][mM][pP][^\w\d]", title) != None)
    trump = trump | (re.search(r"^[Tt][rR][uU][mM][pP][^\w\d]", title) != None)
    trump = trump | (re.search(r"[^\w\d][Tt][rR][uU][mM][pP]$", title) != None)
    biden = (re.search(r"[^\w\d][Bb][iI][dD][eE][nN][^\w\d]", title) != None)
    biden = biden | (re.search(r"^[Bb][iI][dD][eE][nN][^\w\d]", title) != None)
    biden = biden | (re.search(r"[^\w\d][Bb][iI][dD][eE][nN]$", title) != None)
    if trump == True and biden == True:
        return 'TB'
    elif trump == True:
        return 'T'
    elif biden == True:
        return 'B'
    else:
        return 'N'
    
def add_candidate(df):
    new_col = df.apply(lambda row: label_trump_biden(row), axis=1).values
    df1 = df.assign(candidate=new_col)
    df1 = df1[df1['candidate']!='N']
    df1.index = range(0,len(df1))
    return df1

def create_dict(df,stop_words,no_stop_words=True,no_trump_biden=False):
    topic_dict = {}
    for idx, row in df.iterrows():
        title = row.title
        topic = row.coding

        title = re.sub(r"[()\[\],-.?!:;#&]", " ", title)
        title = re.split(" ",title)
        title = list(filter(lambda a: a != '', title))
        
        if no_trump_biden == True:
            title = [word.lower() for word in title if word.isalpha() and word.lower() != 'donald' and word.lower() != 'trump' and word.lower() != 'biden' and word.lower() != 'joe']
        else:
            title = [word.lower() for word in title if word.isalpha()]
        
        if no_stop_words==True:
            title = [word for word in title if word not in stop_words]

        if topic not in topic_dict.keys():
            topic_dict[topic] = title
        else:
            topic_dict[topic] += title
    return topic_dict

def create_counts(topic_dict):
    topic_count = {}
    for k, v in topic_dict.items():
        counter = Counter(v)
        topic_count[k] = {x[0]: x[1] for x in counter.most_common() if x[1] >= 5}
    return topic_count


def create_tfidf(all_posts, stop_words, topic_count, all_posts_no_stop_words=True, all_posts_no_trump_biden=False):
    topic_tfidf = {}

    N = 0
    word_freq = {}
    for idx, row in all_posts.iterrows():
        title = row.title
        topic = row.coding

        title = re.sub(r"[()\[\],-.?!:;#&]", " ", title)
        title = re.split(" ",title)
        title = list(filter(lambda a: a != '', title))
        
        if all_posts_no_trump_biden == True:
            title = [word.lower() for word in title if word.isalpha() and word.lower() != 'donald' and word.lower() != 'trump' and word.lower() != 'biden' and word.lower() != 'joe']
        else:
            title = [word.lower() for word in title if word.isalpha()]
        
        if all_posts_no_stop_words==True:
            title = [word for word in title if word not in stop_words]
            
        N += len(title)
        for word in title:
            if word not in word_freq.keys():
                word_freq[word] = 1
            else:
                word_freq[word] += 1
        

    for topic, words in topic_count.items():
        topic_tfidf[topic] = {}
        for word, freq in words.items():
            tf = topic_count[topic][word]
            idf = math.log(N / word_freq[word])
            topic_tfidf[topic][word] = tf*idf

    return topic_tfidf

def load_stop_words(stop_words):
    words = open("../data/stop_words.txt", "r").read().split("\n")
    for word in words:
        stop_words.add(word.strip())
    return stop_words

def plot_word_freq(input_dict,max_words,im_name):
    fig = plt.figure(figsize=(14,12))
    
    plt.subplot(221)
    tmp_dict1 = input_dict['L']
    words_under_topic1 = len(tmp_dict1.keys()) if len(tmp_dict1.keys())<=max_words else max_words
    y_pos1 = np.arange(words_under_topic1)
    plt.bar(y_pos1, sorted(tmp_dict1.values(),reverse=True)[:words_under_topic1], align='center', alpha=0.5)
    plt.xticks(y_pos1, [k1 for k1, v1 in sorted(tmp_dict1.items(), key=lambda item: item[1],reverse=True)[:words_under_topic1]],rotation=70)
    plt.ylabel('Tf-idf score')
    plt.xlabel('Top tokens')
    plt.title("Legitimate")
    
    plt.subplot(222)
    tmp_dict2 = input_dict['N']
    words_under_topic2 = len(tmp_dict2.keys()) if len(tmp_dict2.keys())<=max_words else max_words
    y_pos2 = np.arange(words_under_topic2)
    plt.bar(y_pos2, sorted(tmp_dict2.values(),reverse=True)[:words_under_topic2], align='center', alpha=0.5)
    plt.xticks(y_pos2, [k1 for k1, v1 in sorted(tmp_dict2.items(), key=lambda item: item[1],reverse=True)[:words_under_topic2]],rotation=70)
    plt.ylabel('Tf-idf score')
    plt.xlabel('Top tokens')
    plt.title('Not Legitimate')
    
    plt.subplot(223)
    tmp_dict3 = input_dict['S']
    words_under_topic3 = len(tmp_dict3.keys()) if len(tmp_dict3.keys())<=max_words else max_words
    y_pos3 = np.arange(words_under_topic3)
    plt.bar(y_pos3, sorted(tmp_dict3.values(),reverse=True)[:words_under_topic3], align='center', alpha=0.5)
    plt.xticks(y_pos3, [k1 for k1, v1 in sorted(tmp_dict3.items(), key=lambda item: item[1],reverse=True)[:words_under_topic3]],rotation=70)
    plt.ylabel('Tf-idf score')
    plt.xlabel('Top tokens')
    plt.title("Suspicious")
    
    plt.subplot(224)
    tmp_dict4 = input_dict['I']
    words_under_topic4 = len(tmp_dict4.keys()) if len(tmp_dict4.keys())<=max_words else max_words
    y_pos4 = np.arange(words_under_topic4)
    plt.bar(y_pos4, sorted(tmp_dict4.values(),reverse=True)[:words_under_topic4], align='center', alpha=0.5)
    plt.xticks(y_pos4, [k1 for k1, v1 in sorted(tmp_dict4.items(), key=lambda item: item[1],reverse=True)[:words_under_topic4]],rotation=70)
    plt.ylabel('Tf-idf score')
    plt.xlabel('Top tokens')
    plt.title("Irrelavant")
       
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
                    wspace=0.3)

    fig.savefig(f'../data/output_imgs/{im_name}', format='png', dpi=300)
    
def read_in(file_name):
    with open(file_name,'r') as f:
        content = f.readlines()

    posts = []
    for line in content:
        posts.append(json.loads(line))
    return posts

def plot_tfidf_dist(tfidf_dict):

    fig, axes = plt.subplots(ncols=4)
    fig.set_size_inches(20,6)

    sns.distplot(list(tfidf_dict["L"].values()), ax=axes[0], fit = stats.norm)
    (mu0, sigma0) = stats.norm.fit(list(tfidf_dict["L"].values()))
    axes[0].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu0, sigma0)],loc='best')
    axes[0].set_title("Legitimate Tf-idf Distribution")
    axes[0].set(xlabel="Tf-idf")
    axes[0].axvline(statistics.median(list(tfidf_dict["L"].values())), linestyle='dashed')
    print("median of word frequency: ", statistics.median(list(tfidf_dict["L"].values())))

    sns.distplot(list(tfidf_dict["N"].values()), ax=axes[1], fit = stats.norm)
    (mu0, sigma0) = stats.norm.fit(list(tfidf_dict["N"].values()))
    axes[1].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu0, sigma0)],loc='best')
    axes[1].set_title("Not Legitimate Tf-idf Distribution")
    axes[1].set(xlabel="Tf-idf")
    axes[1].axvline(statistics.median(list(tfidf_dict["N"].values())), linestyle='dashed')
    print("median of word frequency: ", statistics.median(list(tfidf_dict["N"].values())))
    
    sns.distplot(list(tfidf_dict["S"].values()), ax=axes[2], fit = stats.norm)
    (mu0, sigma0) = stats.norm.fit(list(tfidf_dict["S"].values()))
    axes[2].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu0, sigma0)],loc='best')
    axes[2].set_title("Suspicious Tf-idf Distribution")
    axes[2].set(xlabel="Tf-idf")
    axes[2].axvline(statistics.median(list(tfidf_dict["S"].values())), linestyle='dashed')
    print("median of word frequency: ", statistics.median(list(tfidf_dict["S"].values())))
    
    sns.distplot(list(tfidf_dict["I"].values()), ax=axes[3], fit = stats.norm)
    (mu0, sigma0) = stats.norm.fit(list(tfidf_dict["I"].values()))
    axes[3].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu0, sigma0)],loc='best')
    axes[3].set_title("Irrelevant Tf-idf Distribution")
    axes[3].set(xlabel="Tf-idf")
    axes[3].axvline(statistics.median(list(tfidf_dict["I"].values())), linestyle='dashed')
    print("median of word frequency: ", statistics.median(list(tfidf_dict["I"].values())))
    
def extract_entities(text):
    names = []
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                if chunk.label()=='PERSON':
                    names += [' '.join(c[0] for c in chunk.leaves())]
    return names

def plot_pie_chart(dictionary, title):
    labels = dictionary.keys()
    sizes = [v/sum(dictionary.values()) for k,v in dictionary.items()]

    plt.figure(2, figsize=(15, 15/1.6180))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)

    plt.show()
    
    
def plot_top_names(top_names):
    names = [w[0] for w in top_names]
    counts = [w[1] for w in top_names]
    x_pos = np.arange(len(names)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title=f'Most mentioned names')
#     sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, names, rotation='vertical') 
    plt.xlabel('names')
    plt.ylabel('counts')
    plt.show()
    
    
def cooccur_names(top_names2, df_cand):
    with_trump = {}
    with_biden = {}
    with_both = {}
    for idx, row in df_cand.iterrows():
        cand = row.candidate
        title = row.title

        if cand == 'T':
            for top_name in top_names2:
                if top_name in title:
                    if top_name not in with_trump.keys():
                        with_trump[top_name] = 1
                    else:
                        with_trump[top_name] += 1

        if cand == 'B':
            for top_name in top_names2:
                if top_name in title:
                    if top_name not in with_biden.keys():
                        with_biden[top_name] = 1
                    else:
                        with_biden[top_name] += 1

        if cand == 'TB':
            for top_name in top_names2:
                if top_name in title:
                    if top_name not in with_both.keys():
                        with_both[top_name] = 1
                    else:
                        with_both[top_name] += 1
    return with_trump, with_biden, with_both

def name_topic_corr(top_names3, df_cand):
    topic_count_for_top_names = {}
    title_for_top_names = {}

    for idx, row in df_cand.iterrows():
        coding = row.coding
        title = row.title

        for top_name in top_names3:
            if top_name in title:
                if top_name not in title_for_top_names.keys():
                    title_for_top_names[top_name] = {coding: [title]}
                    topic_count_for_top_names[top_name] = {coding: 1}
                else:
                    if coding not in title_for_top_names[top_name].keys():
                        title_for_top_names[top_name][coding] = [title]
                        topic_count_for_top_names[top_name][coding] = 1
                    else:
                        title_for_top_names[top_name][coding] += [title]
                        topic_count_for_top_names[top_name][coding] += 1
    return topic_count_for_top_names, title_for_top_names

def name_one_hot(top_names3, df_cand):
    top_name_oh = {}

    for idx, row in df_cand.iterrows():
        title = row.title

        for top_name in top_names3:
            if top_name in title:
                if top_name not in top_name_oh.keys():
                    top_name_oh[top_name] = [1]
                else:
                    top_name_oh[top_name] += [1]
            else:
                if top_name not in top_name_oh.keys():
                    top_name_oh[top_name] = [0]
                else:
                    top_name_oh[top_name] += [0]
    return top_name_oh


def compute_corr(top_names3, df_cand, topic_oh, top_name_oh):
    top_name_corr = {}

    for top_name in top_names3:
        for topic in df_cand.coding.unique():
            # store the correlation coefficients for topics
            if top_name not in top_name_corr.keys():
                top_name_corr[top_name] = [np.corrcoef(topic_oh[topic], top_name_oh[top_name])[0][1]]
            else:
                top_name_corr[top_name] += [np.corrcoef(topic_oh[topic], top_name_oh[top_name])[0][1]]
    return top_name_corr

def plot_correlation(input_dict,im_name):
    fig = plt.figure(figsize=(16,14))
    
    plt.subplot(421)
    tmp_dict1 = input_dict['Trump']
    plt.bar(tmp_dict1.index,tmp_dict1.values)
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Topic')
    plt.title("Trump-Topic Correlation")
    
    plt.subplot(422)
    tmp_dict2 = input_dict['Biden']
    plt.bar(tmp_dict2.index,tmp_dict2.values)
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Topic')
    plt.title("Biden-Topic Correlation")
    
    plt.subplot(423)
    tmp_dict3 = input_dict['Toomey']
    plt.bar(tmp_dict3.index,tmp_dict3.values)
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Topic')
    plt.title("Toomey-Topic Correlation")
    
    plt.subplot(424)
    tmp_dict4 = input_dict['Powell']
    plt.bar(tmp_dict4.index,tmp_dict4.values)
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Topic')
    plt.title("Powell-Topic Correlation")
    
    plt.subplot(425)
    tmp_dict5 = input_dict['Giuliani']
    plt.bar(tmp_dict5.index,tmp_dict5.values)
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Topic')
    plt.title("Giuliani-Topic Correlation")
    
    plt.subplot(426)
    tmp_dict6 = input_dict['Hogan']
    plt.bar(tmp_dict6.index,tmp_dict6.values)
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Topic')
    plt.title("Hogan-Topic Correlation")
    
    plt.subplot(427)
    tmp_dict7 = input_dict['Chris Christie']
    plt.bar(tmp_dict7.index,tmp_dict7.values)
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Topic')
    plt.title("Chris Christie-Topic Correlation")
       
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
                    wspace=0.3)

    fig.savefig(f'../data/output_imgs/{im_name}', format='png', dpi=300)