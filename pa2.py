#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install nltk
import re
from nltk.stem import PorterStemmer
import nltk
nltk.download("stopwords")
nltk_stopwords = nltk.corpus.stopwords.words("english")
from pathlib import Path
import numpy as np
TOTOAL_FILES = 1095


# In[2]:


#DF dict

data_loc = ".\\data\\"
term_dict = {}
for item in Path(data_loc).iterdir():
    with open(item, mode ="r") as file:
        porter = PorterStemmer()
        text = file.readlines()
        #print(text)
        temp_set = set()
        for i in text:
            sub_text = re.sub(r'[^A-Za-z ]', '', i)
            #print(sub_text)
            sub_text = sub_text.split(" ")
            for term in sub_text:
                word = term.lower()
                word = porter.stem(word)
                if not len(word) or word in nltk_stopwords:
                    continue
                else:
                    temp_set.add(word)
                
        for term in temp_set:
            if not term_dict.get(term):
                term_dict[term] = 1
            else:
                term_dict[term]+=1
        #break
                    


# In[3]:


#做DF_dict排序
sort_DF_dict = {}
idx = 1 # 詞編號從1開始
for key in sorted(term_dict.keys()):
    cur_tuple = (idx,  term_dict[key])
    sort_DF_dict[key] = cur_tuple
    idx+=1


# In[4]:


sort_DF_dict


# In[5]:


#存DF dict

with open(".\\"+"dictionary.txt", mode ="w") as file:
    idx = 1
    file.write("t_index  term  df \n")

    for key in sorted(term_dict.keys()):
        
        cur_num = term_dict[key]
        cur_str = f"{idx+1}  {key}  {cur_num} \n"
        file.write(cur_str)
        idx+=1
    


# In[6]:


#算TF
if not Path(".\\output").exists():
    Path(".\\output").mkdir(parents=True, exist_ok=True)
for item in Path(data_loc).iterdir():
    with open(item, mode ="r") as file:
        porter = PorterStemmer()
        text = file.readlines()
        #print(text)
        TF_dict ={} #紀錄TF
        text_len = 0 #計錄stemmed後文件長度
        for i in text:
            sub_text = re.sub(r'[^A-Za-z ]', '', i)
            
            sub_text = sub_text.split(" ")
            for term in sub_text:
                word = term.lower()
                word = porter.stem(word)
                if not len(word) or word in nltk_stopwords:
                    continue
                else:
                    text_len+=1 #有字的話文件長度+1
                    if TF_dict.get(word):
                        TF_dict[word]+=1
                    else:
                        TF_dict[word] = 1
                        
    #算TFIDF
    with open(".\\output\\doc"+item.name, mode ="w") as new_file:
        temp_list = []
        new_file.write(str(len(TF_dict.keys()))+ "\n")
        new_file.write("t_index  TFIDF \n")
        for key in TF_dict.keys(): 
            TF_dict[key]  = TF_dict[key]/text_len #除以文件長度
            cur_TF = TF_dict[key]
            cur_df = sort_DF_dict[key][1]
            cur_series_num = sort_DF_dict[key][0]
            idf = np.log10(TOTOAL_FILES/cur_df)
            TFIDF = cur_TF*idf
            temp_list.append((cur_series_num, TFIDF))
        
        temp_list.sort(key = lambda x: x[0]) #照編號排序
        new_list = np.array([i[1] for i in temp_list]) #重新排成np array
        new_list = new_list / np.sqrt(np.sum(new_list**2)) #normalize to unit vec
        for i in range(len(temp_list)):
            cur_series_num = temp_list[i][0] #編碼從1開始
            TFIDF = new_list[i]
            new_file.write(f"{cur_series_num}  {TFIDF} \n")
            
            
            
    """
    #這部分沒有normalize成unit vec
        temp_list.sort(key = lambda x: x[0]) #照編號排序
        for i in temp_list:
            cur_series_num = i[0]
            TFIDF = i[1]
            new_file.write(f"{cur_series_num}  {TFIDF} \n")
            
    """


# In[7]:


def cosine_similarity(Dx = 1, Dy = 2, total_term_num = 13485):
    #直接初始化所有字的長度
    zero_array1 = np.zeros(total_term_num)
    zero_array2 = np.zeros(total_term_num)
    #讀入位置跟tfidf
    doc1 = open(".\\output\\"+"doc"+str(Dx)+".txt").readlines()[2:] 
    doc2 = open(".\\output\\"+"doc"+str(Dy)+".txt").readlines()[2:]
    
    for i in doc1:
        temp = i.split(" ") #讀series跟tfidf
        series, tfidf = int(temp[0])-1, float(temp[2])
        assert series>=0 #確保series number不為負數
        zero_array1[series] = tfidf #在對應位子裝tfidf
        
    for k in doc2:
        temp = k.split(" ") #讀series跟tfidf
        series, tfidf = int(temp[0])-1, float(temp[2])
        assert series>=0 #確保series number不為負數
        zero_array2[series] = tfidf #在對應位子裝tfidf
    
    #print(zero_array1, zero_array2)
    dot = np.dot(zero_array1, zero_array2) #內積
    #print(dot.max())
    norm1 = np.sqrt(np.sum(zero_array1**2)) #平方和開根號
    norm2 = np.sqrt(np.sum(zero_array2**2))
    #print(norm1, norm2)
    ans = dot/norm1*norm2
    
    return ans


# In[8]:


sim = -1 #當前的cosine similarity
cur_most_similar = -1 #最像的檔案
for i in range(2, TOTOAL_FILES):
    cur_sim = cosine_similarity(Dx = 1, Dy =i)
    if  cur_sim> sim:
        cur_most_similar = i
        sim = cur_sim
        print(cur_most_similar,sim)


# In[ ]:




