#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install nltk
#nltk.download("stopwords")

import re
import numpy as np
import math
import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from pathlib import Path
from scipy import stats
TOTOAL_FILES = 1095
nltk_stopwords = nltk.corpus.stopwords.words("english")


# In[4]:


with open("training_set.txt") as training_data:
    all_data = training_data.readlines()
    data_dict = {}
    for i in all_data:
        cur_list = list(map(int, i.split()))
        data_dict[cur_list[0]] = cur_list[1:] 
        print(cur_list[0], data_dict[cur_list[0]])


# In[10]:


class MultinomialNaiveBayes(object):
    def __init__(self, if_filt = False, filtered_num = 5, method = "chisquare"):
        self.if_filt = if_filt
        self.filtered_num = filtered_num
        self.method = method
        
        self.data_dict = {}
        self.total_word_num = 0 #總共有多少字
        self.total_word_freq = 0 #字出現次數的總和
        
        self.all_term_dict = {} #訓練集的所有字的freq
        self.class_term_dict = {} #訓練集的各cls的字freq
        self.class_term_present_dict ={} #紀錄該class檔案中出現該字與否的次數
        self.training_data_splited= {}
        
        self.filtered_all_term_dict = {}
        self.filtered_class_term_dict = {}
        
        
    def count_training_terms(self, data_loc = ".\\IRTM\\"):
        porter = PorterStemmer()
        
        for key in self.data_dict.keys():
            cur_class_files = self.data_dict[key]
            self.class_term_dict[key] = {} #每個類別各自的字
            cur_cls_file =[]
            for cur_num in cur_class_files:
                
                #print(f"cur_num {cur_num}")
                #class_term_dict[key][cur_num] = {}
                with open(data_loc+str(cur_num)+".txt", mode ="r") as file:
                    text = file.readlines()
                    #print(text)
                    cls_doc_split_text =[]
                    cur_doc ={} #用dict[term]來表示document
                    for i in text:
                        sub_text = re.sub(r'[^A-Za-z ]', '', i) #去掉非英文字
                        sub_text = sub_text.split()
                        #print(sub_text)
                        for term in sub_text:
                            word = term.lower()
                            word = porter.stem(word)

                            if not len(word) or word in nltk_stopwords:
                                continue #被切完就pass
                            else:
                                cur_doc[word] = 1
                                self.total_word_freq +=1
                                if self.all_term_dict.get(word):
                                    self.all_term_dict[word]+=1
                                    
                                else:
                                    self.all_term_dict[word]=1
                                    self.total_word_num += 1
                                
                                if self.class_term_dict[key].get(word):
                                    self.class_term_dict[key][word]+=1
                                    
                                else:
                                    self.class_term_dict[key][word] = 1
                    #print(cur_doc)
                    #cls_doc_split_text.append(cur_doc)
                cur_cls_file.append(cur_doc)
            self.training_data_splited[key]= cur_cls_file
    
    def build_term_present_freq_dict(self):

        N = self.total_doc_num
        
        cls_list = [key for key in self.data_dict.keys()]
        
        for term in self.all_term_dict.keys():

            self.class_term_present_dict[term] ={}
            for cls in cls_list:
                onTP=0
                offTP= 0
                onTA=0
                offTA= 0
                docs = self.training_data_splited[cls]
                #print(doc)
                for doc in docs:
                    if doc.get(term):
                        onTP+=1
                    else:
                        offTP+=1
                for other_cls in cls_list:
                    if other_cls != cls:
                        other_cls_docs = self.training_data_splited[other_cls]
                        for other_cls_doc in other_cls_docs:
                            if other_cls_doc.get(term):
                                onTA+=1
                            else:
                                offTA+=1
                #print(other_cls_doc)
                self.class_term_present_dict[term][cls] =[onTP, offTP, onTA, offTA]
                #break
                
        
        return self.class_term_present_dict
    
    def get_chisquare(self):
        term_chi = {}
        
        N = 0 # training set len
        for cls in self.data_dict.keys():
            N+= len(self.data_dict[cls]) #training set len = 195
            term_chi[cls] = []
        for term in self.class_term_present_dict.keys():
            maximum_chi = 0
            max_cls = None
            for cls in sorted(list(self.data_dict.keys())):
                #[onTP, offTP, onTA, offTA]
                cur_table =self.class_term_present_dict[term][cls]
                cur_table = np.array(cur_table)
                cur_table = np.reshape(cur_table,(2,2))
                sum_row = np.sum(cur_table, axis=1)
                sum_col = np.sum(cur_table, axis=0)
                class_term_chi =0
                for i in range(len(cur_table)):
                    for j in range(len(cur_table[0])):
                        cur_slot = cur_table[i][j]
                        expect_val = (sum_col[j]*sum_row[i]/sum(sum_row))
                        cur_chi = (cur_slot-expect_val)**2/(expect_val)
                        class_term_chi+=cur_chi
                if class_term_chi>maximum_chi:
                    maximum_chi = class_term_chi
                    max_cls = cls
            term_chi[max_cls].append((term, maximum_chi))
        for cls in self.data_dict.keys():
            term_chi[cls].sort(key = lambda x: x[1], reverse=True)
            
        return term_chi
        
    
    def get_LLR(self):
        term_LLR = {}
        
        N = 0 # training set len
        for cls in self.data_dict.keys():
            N+= len(self.data_dict[cls]) #training set len = 195
            term_LLR[cls] = []
        for term in self.class_term_present_dict.keys():
            maximum_LLR = 0
            max_cls = None
            for cls in sorted(list(self.data_dict.keys())):
                #[onTP, offTP, onTA, offTA]
                cur_table =self.class_term_present_dict[term][cls]
                n11 = cur_table[0]
                n01 = cur_table[1]
                n10 = cur_table[2]
                n00 = cur_table[3]
                class_term_LLR =0
                upper = (((n11+n01)/N)**n11)*((1-(n11+n01)/N)**n10)*((((n11+n01)/N))**n01)*((1-(n11+n01)/N)**n00)
                upper = math.log(upper)
                lower = ((n11/(n11+n10))**n11)*((1-(n11/(n11+n10)))**n10)*((n01/(n01+n00))**n01)* (1-(n01/(n01+n00)))**n00
                lower = math.log(lower)
                CurLLR = -2*(upper-lower)
                
                if CurLLR>maximum_LLR:
                    maximum_LLR = CurLLR
                    max_cls = cls
            term_LLR[max_cls].append((term, maximum_LLR))
        for cls in self.data_dict.keys():
            term_LLR[cls].sort(key = lambda x: x[1], reverse=True)
            
        return term_LLR
    
    def set_filtered_terms(self, ): #using chisquare
        if not self.if_filt:
            self.filtered_all_term_dict = self.all_term_dict
            self.filtered_class_term_dict = self.class_term_dict
        
        else:
            if self.method =="chisquare":
                term_filtered = self.get_chisquare()

                
            if self.method == "LLR":
                term_filtered = self.get_LLR()
                
            for cls in term_filtered.keys():
                self.filtered_class_term_dict[cls] ={}
                for i in range(self.filtered_num):
                    term = term_filtered[cls][i][0]

                    self.filtered_all_term_dict[term] = self.all_term_dict[term] #filtered的all term dict
                    #for cls in range(1, self.categories_num+1):
                        #self.filtered_class_term_dict[cls] = {}
                    if self.class_term_dict[cls].get(term):
                        self.filtered_class_term_dict[cls][term] = self.class_term_dict[cls].get(term) #放進filtered的class dict

            self.all_term_dict = self.filtered_all_term_dict #訓練集的所有字的freq
            self.class_term_dict = self.filtered_class_term_dict
            
            
    def read_training_data(self,loc):
        with open(loc) as training_data:
            self.total_doc_num =0
            all_data = training_data.readlines()
            
            for i in all_data:
                cur_list = list(map(int, i.split()))
                self.data_dict[cur_list[0]] = cur_list[1:]
                self.total_doc_num += len(cur_list[1:])
                print(cur_list[0], self.data_dict[cur_list[0]])
        
        #self.data_dict = data_dict
        
        self.categories_num = len(list(self.data_dict.keys()))
        return self.data_dict
        
    def get_class_prob(self): #P(c)
        #train_data = self.read_training_data()
        #data_dict = train_data[0]
        #total_doc_num = train_data[1]
        
        class_prob = [] #P(c) 的list
        
        for cls in sorted(list(self.data_dict.keys())): #排個序避免順序跑掉
            cur_item_len = len(self.data_dict[cls])
            class_prob.append(math.log(cur_item_len)-math.log(self.total_doc_num)) #取機率log值
            
        return class_prob
    
    
    def get_term_prob(self): #P(X=t|c)
        #term_prob = np.array.zeros((self.total_word_num, len(data_dict.keys()))) #dim:(總共要放幾個字到矩陣 幾個class ex. 500,13)
        term_prob = {}
        all_term_list = list(self.filtered_all_term_dict.keys())
        cls_term_list = sorted(list(self.filtered_class_term_dict.keys()))

        cls_total_freq = []
        for key in self.class_term_dict.keys():
            cur_cls_freq = 0
            for word in self.class_term_dict[key].keys():
                cur_frq = self.class_term_dict[key][word]
                cur_cls_freq += cur_cls_freq

            cls_total_freq.append(cur_cls_freq)

        
        for j in range(len(all_term_list)):
            term = all_term_list[j]
            term_prob[term] = np.zeros(self.categories_num)
            cur_all_freq = self.all_term_dict[term]
            for i in range(len(cls_term_list)):
                cls = cls_term_list[i]
                cur_cls_total_freq = cls_total_freq[i]

                if self.class_term_dict[cls].get(term):
                    cur_class_term_freq = self.class_term_dict[cls][term]+1
                else:
                    cur_class_term_freq = 1

                term_prob[term][i] = math.log(cur_class_term_freq)-math.log(cur_cls_total_freq+self.total_word_num)#term先再放class查比較順
                
        return term_prob
    
    def read_test_data(self, test_data, data_loc = ".\\IRTM\\"):
        porter = PorterStemmer()
        test_data_set = set()
        #for data in test_data:
        with open(data_loc+test_data, mode ="r") as file:
            text = file.readlines()
            for i in text:
                sub_text = re.sub(r'[^A-Za-z ]', '', i) #去掉非英文字
                sub_text = sub_text.split()
                #print(sub_text)
                for term in sub_text:
                    word = term.lower()
                    word = porter.stem(word)

                    if not len(word) or word in nltk_stopwords:
                        continue #被切完就pass
                    else:
                        test_data_set.add(word)
                            
        return test_data_set
    def train(self, loc = "training_set.txt"):
        self.read_training_data(loc)
        self.count_training_terms()
        self.build_term_present_freq_dict()
        
        self.set_filtered_terms()
        self.class_prob = self.get_class_prob()
        self.term_prob = self.get_term_prob()
        
    
    def predict(self, testing_data_list): #給檔案名list去開檔 斷詞

        class_prob = self.get_class_prob()
        ans = []
        #categories_num = len(list(self.data_dict.keys()))
        for data in testing_data_list:
            #print(data)
            cur_data_prob =np.array(class_prob) #init 一個含13個P(c)的list
            cur_test_set = self.read_test_data(data)
            for term in list(cur_test_set):
                #print(self.term_prob.get(term))
                if type(self.term_prob.get(term)) == np.ndarray:
                    #print(self.term_prob.get(term))
                    temp_term_prob = self.term_prob.get(term) #會有該詞的13 classes的機率
                    #print(temp_term_prob)
                    for cls in range(self.categories_num):
                        cur_data_prob[cls]+=temp_term_prob[cls]
            #print(cur_data_prob)
            index = np.argmax(cur_data_prob)
            
            ans.append(index+1)
            
        return ans
                        
                
        
    
    
        


# In[11]:


test_data = []
training_set = []
for k in data_dict.keys():
    training_set.extend(data_dict[k])

for item in Path(".\\IRTM\\").iterdir():
    cur_num = int(item.name[:-4])

    if cur_num not in training_set:
        test_data.append(item.name)
print(len(test_data))


# In[12]:


NB_test =MultinomialNaiveBayes(if_filt=True, filtered_num=15, method ="chisquare")
NB_test.train(loc = "training_set.txt")


# In[13]:


filtered_prediction = NB_test.predict(test_data)


# In[14]:


filtered_prediction


# In[15]:


result_list = {}
for idx in range(len(test_data)):
    ide = test_data[idx][:-4]
    result = filtered_prediction[idx]
    result_list[ide] = result


# In[16]:


test_pd = pd.DataFrame.from_dict(result_list, orient="index", columns=['Value'])
test_pd.index.name = "Id"
test_pd.to_csv("R09725047_劉育志.csv")


# In[ ]:




