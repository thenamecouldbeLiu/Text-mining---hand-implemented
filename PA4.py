#!/usr/bin/env python
# coding: utf-8

# # 導入計算及分詞package

# In[1]:


import re
from nltk.stem import PorterStemmer
import nltk
nltk.download("stopwords")
nltk_stopwords = nltk.corpus.stopwords.words("english")
from pathlib import Path
import numpy as np
import math
TOTAL_FILES = 1095
data_loc = ".\\IRTM\\"


# # MAXHEAP實作

# In[2]:


class MaxHeap(object):
    def __init__(self, heap_list =[]):
        #內存的資料結構為 (cluster, similarity)
        self.pos_dict = {}
        self.heap_list = heap_list
        if len(self.heap_list):
            self.heapify()
            self.root = self.heap_list[0] #init the root
            self.build_pos_dict() #初始化position list

    def get_heaplist(self):
        return self.heap_list
    
    def check_cluster_exist(self, new_cluster):
        #確認是不是有已經存在的cluster
        for cluster in self.heap_list:
            if new_cluster == cluster[0]:
                print(cluser)
                return True
        return False
    def add(self, sim_tuple):
        assert type(sim_tuple) == tuple
        if not self.check_cluster_exist(sim_tuple):
            self.heap_list.append(sim_tuple)
            self.pos_dict[sim_tuple[0]] = len(self.heap_list)-1
            self.heapify()
            
            #print(self.heap_list)
        else:
            print("Cluster already exist")
    def delete_by_item(self, to_be_del):
        for idx in range(len(self.heap_list)):
            item = self.heap_list[idx]
            if item[1] == to_be_del:
                self.heap_list[idx], self.heap_list[-1] = self.heap_list[-1], self.heap_list[idx]
                self.heap_list.pop()
                self.heapify()
                break
        
    def delete_by_idx(self, idx): #對手heap的idx
        #print("before del", self.get_heaplist())
        #print("del",idx)
        
        #print("del", self.heap_list)
        temp = self.heap_list[idx][0]
        
        self.swap_update(idx, -1)
        self.pos_dict.pop(temp)
        #self.heap_list[idx], self.heap_list[-1] = self.heap_list[-1], self.heap_list[idx]
        
        self.heap_list.pop()
        self.heapify()
        #print("after del", self.get_heaplist())
    def get_root(self):
        if len(self.heap_list):
            return self.root
        return None
    def clear_heap(self):
        self.heap_list =[]
        #self.pos_dict ={}
    def build_pos_dict(self):

        for idx in range(len(self.heap_list)):
            cur_node = self.heap_list[idx][0]
            self.pos_dict[cur_node] = idx
            
    def update_pos_list(self,node, idx):
        self.pos_dict[node] = idx
        
    def get_pos_dict(self): #此cluster對應的對手cluster在heap的位子
        return self.pos_dict
    
    def swap_update(self, idx1, idx2):
        self.update_pos_list(self.heap_list[idx1][0], idx2) #交換heap裡面的position
        self.update_pos_list(self.heap_list[idx2][0], idx1)
        self.heap_list[idx1], self.heap_list[idx2] = self.heap_list[idx2], self.heap_list[idx1] #node值交換
    def heapify(self, cur_idx = 0):

        list_len = len(self.heap_list)-1 #list最後一位
        while cur_idx < list_len:
            if 2*cur_idx+1 <=list_len:
                left_idx = 2*cur_idx+1
            else:
                left_idx = None
            if 2*cur_idx+2 <=list_len:
                right_idx = 2*cur_idx+2
            else:
                right_idx = None
            
            if left_idx:
                if right_idx: #有右子樹的話
                    left = self.heap_list[left_idx][1]
                    cur = self.heap_list[cur_idx][1]
                    right = self.heap_list[right_idx][1]
                    
                    if left>right and left > cur:
                        self.swap_update(cur_idx, left_idx)
                        #self.heap_list[cur_idx], self.heap_list[left_idx] = self.heap_list[left_idx], self.heap_list[cur_idx]
                        
                    elif right>=left and right > cur:
                        self.swap_update(cur_idx, right_idx)
                        #self.heap_list[cur_idx], self.heap_list[right_idx] = self.heap_list[right_idx], self.heap_list[cur_idx]
                        
                else: #只有左子樹
                    left = self.heap_list[left_idx][1]
                    cur = self.heap_list[cur_idx][1]

                    if left > cur:
                        self.swap_update(cur_idx, left_idx)
                        #self.heap_list[cur_idx], self.heap_list[left_idx] = self.heap_list[left_idx], self.heap_list[cur_idx]
                        
            #print("child checked")
            parent_tuple = self.get_parent(cur_idx)
            #print(self.heap_list[cur_idx])
            if parent_tuple != None and parent_tuple[0][1]< self.heap_list[cur_idx][1]: #如果parent node比較大的話 往上再做一次heapify
                #print(parent_tuple, self.heap_list[cur_idx], cur_idx)
                parent_idx = parent_tuple[1]
                self.heapify(parent_idx) 
                break
            else:
                cur_idx+=1
            
        #print("heapify", self.heap_list)
        if len(self.heap_list):
            self.root  = self.heap_list[0]
    def pop_max(self):
        temp = self.heap_list[0][0]
        self.heap_list[0], self.heap_list[-1] = self.heap_list[-1], self.heap_list[0]
        cur_max  = self.heap_list.pop()
        self.heapify()
        self.pos_dict.pop(temp) #刪掉在dict中max的位子
        return cur_max
    
    def get_heap_position(self):
        return self.heap_position
    def get_left_child(self, idx):
        return self.heap_list[2*idx+1]
    def get_right_child(self, idx):
        return self.heap_list[2*idx+2]
    def get_parent(self, idx):
        if idx % 2 == 1:
            parent_idx= int((idx-1)/2)
            if parent_idx>=0:
                return self.heap_list[parent_idx], parent_idx #左子樹
        parent_idx= int((idx/2)-1) #右子樹
        if parent_idx>=0:
            return self.heap_list[parent_idx], parent_idx
        return None


# # 測試HEAP

# In[3]:


testL = [(1,3),(3,6), (5,9),(8,20)]
maxheap = MaxHeap(testL)
maxheap.add((8,50))
maxheap.delete_by_idx(3)
print(maxheap.get_pos_dict())
print(maxheap.get_heaplist())


# In[4]:


maxheap.delete_by_idx(2)
print(maxheap.get_pos_dict())
print(maxheap.get_heaplist())


# # 把文檔轉成TFIDF

# In[ ]:


class TFIDF_Transformer(object):
    def __init__(self, TOTAL_FILES = 1095, file_loc = ".\\IRTM\\"):
        self.DF_dict = {} #紀錄DF
        self.TF_dict ={} #紀錄TF
        self.TOTAL_FILES = TOTAL_FILES
        self.file_loc = file_loc
    def get_files_contents(self, file_num):
        with open(self.file_loc+str(file_num)+".txt") as file:
            text = file.readlines()
            return file_num, text
        
    def transfrom_text(self,file_num,text):
        #need doc to present as a list of str

        porter = PorterStemmer()
        temp_set = set()
        text_len = 0 #計錄stemmed後文件長度
        self.TF_dict[file_num] ={}
        
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
                    text_len+=1 #有字的話文件長度+1
                    if self.TF_dict[file_num].get(word):
                        self.TF_dict[file_num][word]+=1
                    else:
                        self.TF_dict[file_num][word] = 1
                        
        for term in self.TF_dict[file_num].keys():
            self.TF_dict[file_num][term] = self.TF_dict[file_num][term]/text_len
        for term in temp_set: #算這個檔案對總體DF的貢獻
            if not self.DF_dict.get(term):
                self.DF_dict[term] = 1
            else:
                self.DF_dict[term]+=1
    
    def get_DF_dict(self):
        return self.DF_dict
    
    def get_TF_dict(self):
        return self.TF_dict
    def fit(self):
        for num in range(1, self.TOTAL_FILES+1):
            file_num, text = self.get_files_contents(num)
            self.transfrom_text(file_num, text)
        for term in self.DF_dict.keys():
            self.DF_dict[term] = math.log(self.TOTAL_FILES/self.DF_dict[term]) #算出IDF
            
    def get_TFIDF_vec(self):
        total_words_len = len(self.DF_dict.keys())
        all_file_vec = np.zeros((self.TOTAL_FILES, total_words_len))
        all_words = list(self.DF_dict.keys())
        for num in range(self.TOTAL_FILES):
            cur_file = self.TF_dict[num+1]
            for word_idx in range(len(all_words)):
                word = all_words[word_idx]
                if cur_file.get(word) != None:
                    all_file_vec[num][word_idx] = cur_file.get(word)*self.DF_dict[word]
                
        all_file_vec = all_file_vec / np.sqrt(np.sum(all_file_vec**2))
        return all_file_vec
                


# In[8]:


transformer = TFIDF_Transformer()
transformer.fit()


# In[9]:


all_file_vec = transformer.get_TFIDF_vec()


# # 用heap實作HAC (以group number分類 僅實作single link)

# In[23]:


class Efficient_HAC_with_GROUP_NUM(object):
    def __init__(self, all_file_vec, sim_type ="single", GROUP_NUM = 8):
        length = len(all_file_vec) # sim matrix 的邊長
        #self.MIN_SIM = MIN_SIM #最小多少停止分
        self.all_file_vec = all_file_vec #檔案的向量
        self.sim_type = sim_type #用什麼當距離
        self.heap_list = []
        self.sim_matrix = [[0 for col in range(length)] for row in range(length)]
        self.availible_list = np.ones(len(all_file_vec))
        self.build_sim_matrix() #初始化similarity matrix
        self.cluster_dict = {}
        for init_clust in range(len(all_file_vec)):
            self.cluster_dict[init_clust] = [] #初始化每個都是自己一組
        self.GROUP_NUM = GROUP_NUM
        #self.cluster_list =[]
    def cal_cosine_similarity(self, x, y):
        dot = np.dot(x, y) #內積

        norm1 = np.sqrt(np.sum(x**2)) #平方和開根號
        norm2 = np.sqrt(np.sum(y**2))

        ans = dot/(norm1*norm2)
        return ans
    
    def cal_single_link(self, cluster1, cluster2):
        min_distance = 100000
        for c1_node in cluster1:
            for c2_node in cluster2:
                similarity = self.cal_cosine_similarity(c1_node, c2_node)
                if similarity < min_distance:
                    min_distance = similarity
                    
        return min_distance
    def cal_complete_link(self, cluster1, cluster2):
        max_distance = -100000
        for c1_node in cluster1:
            for c2_node in cluster2:
                similarity = self.cal_cosine_similarity(c1_node, c2_node)
                if similarity > max_distance:
                    max_distance = similarity
                    
        return max_distance
    
    def build_sim_matrix(self, cur_cluster = None):
        if cur_cluster == None:
            cur_cluster = self.all_file_vec

        length = len(cur_cluster)
        
        for i in range(length):
            cluster1 = cur_cluster[i]
            #print("here", cluster1)
            heap_position = [0 for k in range(length-1)] #因為要pop自己 所以只有L-1個position要維護
            if self.sim_type == "single": #如果是single link 距離為正值 才可以直接用maxheap
                for j in range(length):
                    if i == j:
                        self.sim_matrix[i][j] = (j,10) #不計算直接填10 以免數值錯誤
                    else: #計算sim
                        cluster2 = cur_cluster[j]
                        self.sim_matrix[i][j] = (j, self.cal_cosine_similarity(cluster1, cluster2))

            elif self.sim_type == "complete":
                for j in range(length):
                    if i == j:
                        self.sim_matrix[i][j] = (j,10) #不計算直接填10 以免數值錯誤
                    else: #計算sim
                        cluster2 = cur_cluster[j]
                        self.sim_matrix[i][j] = (j, -self.cal_cosine_similarity(cluster1, cluster2))


            new_sim_row = self.sim_matrix[i]
            cur_heap = self.build_heap(new_sim_row)
            selfnode = cur_heap.pop_max() #去掉自己
            cur_heap.build_pos_dict() #建一個heapPosition的dict去取用item在heap的位子
            

    def build_heap(self, vec):
        cur_heap = MaxHeap(vec)
        self.heap_list.append(cur_heap)
        return cur_heap
    def get_sim_matrix(self):
        return self.sim_matrix

    def cluster(self):
        

        while True:
            #print("cur_clusters", self.cluster_dict)
            global_max = -100
            global_max_idx = -1
            for i in range(len(self.sim_matrix)):
                if not self.availible_list[i]:
                    continue

                else:
                    cur_max = self.heap_list[i].get_root()
                    if cur_max != None:
                        max_idx = i
                        max_val = cur_max[1]
                        if global_max<max_val:
                            global_max = max_val
                            global_max_idx = max_idx
                        
            if not global_max or len(self.cluster_dict.keys()) == self.GROUP_NUM: #即剩下的都不滿足合併標準
                print("stop")
                break
            
            max_idx = global_max_idx
            cur_heap = self.heap_list[max_idx]
            print("max_root",max_idx,cur_heap.get_root())
            #print("cur_merger_heap", cur_heap.get_heaplist())
            #print("cur_merger_heap pos", cur_heap.get_pos_dict())
            cur_merged_idx = cur_heap.pop_max()[0] #要被合併的heap的idx
            
            
            merged_list = self.cluster_dict[cur_merged_idx]
            #self.cluster_list.append((max_idx, cur_merged_idx))
            
            if len(merged_list):
                self.cluster_dict[max_idx].append(cur_merged_idx)
                self.cluster_dict[max_idx].extend(merged_list)
                self.cluster_dict.pop(cur_merged_idx)
            else:
                self.cluster_dict[max_idx].append(cur_merged_idx)                
                self.cluster_dict.pop(cur_merged_idx)
            
            
            #self.cluster_list.append([max_idx, cur_merged_idx])
            cur_heap.clear_heap() #清掉現在要merge別人的heap
            self.availible_list[cur_merged_idx] = 0
            #print("cur_merger_idx: ", max_idx,"merged_idx: ", cur_merged_idx)
            
            for available_idx in range(len(self.availible_list)):
                if global_max_idx!=available_idx and self.availible_list[available_idx] ==1:
                    #print("avail idx", available_idx)
                    other_heap = self.heap_list[available_idx]
                    #print(other_heap.get_heaplist())
                    #print(other_heap.get_pos_dict())
                    #print(other_heap.get_pos_dict())
                    cur_heap_pos = other_heap.get_pos_dict()[max_idx]
                    merger_sim = other_heap.get_heaplist()[cur_heap_pos][1]
                    other_heap.delete_by_idx(cur_heap_pos) #刪掉
                    
                    
                    #print("curheap pos",cur_heap_pos, merged_heap_pos)
                    
                    
                    merged_heap_pos = other_heap.get_pos_dict()[cur_merged_idx]
                    merged_sim = other_heap.get_heaplist()[merged_heap_pos][1]
                    new_sim = max(merger_sim, merged_sim)
                    
                    #print(other_heap.get_pos_dict())
                    #self.sim_matrix[available_idx][max_idx] = new_sim #更新sim matrix
                    #self.sim_matrix[max_idx][available_idx] = new_sim
                    
                    
                    other_heap.delete_by_idx(merged_heap_pos)
                    
                    other_heap.add((max_idx, new_sim))
                    cur_heap.add((available_idx, new_sim))
            
        return self.cluster_dict


# In[30]:


#小批次測試
testL = all_file_vec[:100]
#testL = np.array(testL)
HAC_test = Efficient_HAC_with_GROUP_NUM(testL, GROUP_NUM=8)
#sim_matrix = HAC_test.get_sim_matrix()
#HAC_test.cluster_dict
clusters = HAC_test.cluster()


# In[37]:



work_list = [8, 12, 20]
for cluster_len in work_list:
    testL = all_file_vec
    HAC_test = Efficient_HAC_with_GROUP_NUM(testL, GROUP_NUM=cluster_len)

    clusters = HAC_test.cluster()
    with open(f"{cluster_len}.txt", "w") as cluster_file:
        for key in clusters.keys():
            cluster_file.write(str(key+1)+"\n")
            for item in sorted(clusters[key]):
                cluster_file.write(str(item+1)+"\n")
            cluster_file.write("\n")


# In[31]:


to = 0
for key in clusters.keys():
    to+= len(clusters[key])


# In[36]:


cluster_len


# In[38]:


len(clusters.keys())


# In[20]:


class efficient_HAC_with_MIN_SIM(object):
    def __init__(self, all_file_vec, sim_type ="single", GROUP_NUM = 8):
        length = len(all_file_vec) # sim matrix 的邊長
        #self.MIN_SIM = MIN_SIM #最小多少停止分
        self.all_file_vec = all_file_vec #檔案的向量
        self.sim_type = sim_type #用什麼當距離
        self.heap_list = []
        self.sim_matrix = [[0 for col in range(length)] for row in range(length)]
        self.availible_list = np.ones(len(all_file_vec))
        self.build_sim_matrix() #初始化similarity matrix
        self.cluster_dict = {}
        for init_clust in range(len(all_file_vec)):
            self.cluster_dict[init_clust] = [] #初始化每個都是自己一組
        self.GROUP_NUM =GROUP_NUM
    def cal_cosine_similarity(self, x, y):
        dot = np.dot(x, y) #內積

        norm1 = np.sqrt(np.sum(x**2)) #平方和開根號
        norm2 = np.sqrt(np.sum(y**2))

        ans = dot/(norm1*norm2)
        return ans
    
    def cal_single_link(self, cluster1, cluster2):
        min_distance = 100000
        for c1_node in cluster1:
            for c2_node in cluster2:
                similarity = self.cal_cosine_similarity(c1_node, c2_node)
                if similarity < min_distance:
                    min_distance = similarity
                    
        return min_distance
    def cal_complete_link(self, cluster1, cluster2):
        max_distance = -100000
        for c1_node in cluster1:
            for c2_node in cluster2:
                similarity = self.cal_cosine_similarity(c1_node, c2_node)
                if similarity > max_distance:
                    max_distance = similarity
                    
        return max_distance
    
    def build_sim_matrix(self, cur_cluster = None):
        if cur_cluster == None:
            cur_cluster = self.all_file_vec

        length = len(cur_cluster)
        
        for i in range(length):
            cluster1 = cur_cluster[i]
            #print("here", cluster1)
            heap_position = [0 for k in range(length-1)] #因為要pop自己 所以只有L-1個position要維護
            if self.sim_type == "single": #如果是single link 距離為正值 才可以直接用maxheap
                for j in range(length):
                    if i == j:
                        self.sim_matrix[i][j] = (j,10) #不計算直接填10 以免數值錯誤
                    else: #計算sim
                        cluster2 = cur_cluster[j]
                        self.sim_matrix[i][j] = (j, self.cal_cosine_similarity(cluster1, cluster2))

            elif self.sim_type == "complete":
                for j in range(length):
                    if i == j:
                        self.sim_matrix[i][j] = (j,10) #不計算直接填10 以免數值錯誤
                    else: #計算sim
                        cluster2 = cur_cluster[j]
                        self.sim_matrix[i][j] = (j, -self.cal_cosine_similarity(cluster1, cluster2))


            new_sim_row = self.sim_matrix[i]
            cur_heap = self.build_heap(new_sim_row)
            selfnode = cur_heap.pop_max() #去掉自己
            cur_heap.build_pos_dict() #建一個heapPosition的dict去取用item在heap的位子
            

    def build_heap(self, vec):
        cur_heap = MaxHeap(vec)
        self.heap_list.append(cur_heap)
        return cur_heap
    def get_sim_matrix(self):
        return self.sim_matrix

    def cluster(self):
        

        while True:
            #print("cur_clusters", self.cluster_dict)
            global_max = -100
            global_max_idx = -1
            for i in range(len(self.sim_matrix)):
                if not self.availible_list[i]:
                    continue

                else:
                    cur_max = self.heap_list[i].get_root()
                    if cur_max != None:
                        max_idx = i
                        max_val = cur_max[1]
                        if global_max<max_val:
                            global_max = max_val
                            global_max_idx = max_idx
                        
            if not global_max or global_max<self.MIN_SIM: #即剩下的都不滿足合併標準
                print("stop")
                break
            
            max_idx = global_max_idx
            cur_heap = self.heap_list[max_idx]
            print("max_root",max_idx,cur_heap.get_root())
            #print("cur_merger_heap", cur_heap.get_heaplist())
            #print("cur_merger_heap pos", cur_heap.get_pos_dict())
            cur_merged_idx = cur_heap.pop_max()[0] #要被合併的heap的idx
            
            
            merged_list = self.cluster_dict[cur_merged_idx]
            #self.cluster_list.append((max_idx, cur_merged_idx))
            
            if len(merged_list):
                self.cluster_dict[max_idx].append(cur_merged_idx)
                self.cluster_dict[max_idx].extend(merged_list)
                self.cluster_dict.pop(cur_merged_idx)
            else:
                self.cluster_dict[max_idx].append(cur_merged_idx)                
                self.cluster_dict.pop(cur_merged_idx)
            
            
            #self.cluster_list.append([max_idx, cur_merged_idx])
            cur_heap.clear_heap() #清掉現在要merge別人的heap
            self.availible_list[cur_merged_idx] = 0
            #print("cur_merger_idx: ", max_idx,"merged_idx: ", cur_merged_idx)
            
            for available_idx in range(len(self.availible_list)):
                if global_max_idx!=available_idx and self.availible_list[available_idx] ==1:
                    #print("avail idx", available_idx)
                    other_heap = self.heap_list[available_idx]
                    #print(other_heap.get_heaplist())
                    #print(other_heap.get_pos_dict())
                    #print(other_heap.get_pos_dict())
                    cur_heap_pos = other_heap.get_pos_dict()[max_idx]
                    merger_sim = other_heap.get_heaplist()[cur_heap_pos][1]
                    other_heap.delete_by_idx(cur_heap_pos) #刪掉
                    
                    
                    #print("curheap pos",cur_heap_pos, merged_heap_pos)
                    
                    
                    merged_heap_pos = other_heap.get_pos_dict()[cur_merged_idx]
                    merged_sim = other_heap.get_heaplist()[merged_heap_pos][1]
                    new_sim = max(merger_sim, merged_sim)
                    
                    #print(other_heap.get_pos_dict())
                    #self.sim_matrix[available_idx][max_idx] = new_sim #更新sim matrix
                    #self.sim_matrix[max_idx][available_idx] = new_sim
                    
                    
                    other_heap.delete_by_idx(merged_heap_pos)
                    
                    other_heap.add((max_idx, new_sim))
                    cur_heap.add((available_idx, new_sim))
            
        return self.cluster_dict


# In[ ]:


class MinHeap(object):
    def __init__(self, heap_list =[]):
        #內存的資料結構為 (cluster, similarity)
        self.heap_list = heap_list
        if len(self.heap_list):
            self.heapify()
            self.root = self.heap_list[0] #init the root
    def get_heaplist(self):
        return self.heap_list
    
    def check_cluster_exist(self, new_cluster):
        #確認是不是有已經存在的cluster
        for cluster in self.heap_list:
            if new_cluster == cluster[0]:
                print(cluser)
                return True
        return False
    def add(self, num):
        assert type(num) == tuple
        if not self.check_cluster_exist(num):
            self.heap_list.append(num)
            self.heapify()
            print(self.heap_list)
        else:
            print("Cluster already exist")
    def delete(self, idx):
        self.heap_list[idx], self.heap_list[-1] = self.heap_list[-1], self.heap_list[idx]
        self.heap_list.pop()
        self.heapify()
        print(self.heap_list)
    def get_root(self):
        if len(self.heap_list):
            return self.root
        return None
    def heapify(self, cur_idx = 0):
        
        list_len = len(self.heap_list)-1 #list最後一位
        while cur_idx < list_len:
            if 2*cur_idx+1 <=list_len:
                left_idx = 2*cur_idx+1
            else:
                left_idx = None
            if 2*cur_idx+2 <=list_len:
                right_idx = 2*cur_idx+2
            else:
                right_idx = None
            
            if left_idx:
                if right_idx: #有右子樹的話
                    left = self.heap_list[left_idx][1]
                    cur = self.heap_list[cur_idx][1]
                    right = self.heap_list[right_idx][1]
                    
                    if left<=right and left < cur:
                        self.heap_list[cur_idx], self.heap_list[left_idx] = self.heap_list[left_idx], self.heap_list[cur_idx]
                        
                    elif right<left and right < cur:
                        self.heap_list[cur_idx], self.heap_list[right_idx] = self.heap_list[right_idx], self.heap_list[cur_idx]
                        
                else: #只有左子樹
                    left = self.heap_list[left_idx][1]
                    cur = self.heap_list[cur_idx][1]

                    if left < cur:
                        self.heap_list[cur_idx], self.heap_list[left_idx] = self.heap_list[left_idx], self.heap_list[cur_idx]
                        
            #print("child checked")
            parent_tuple = self.get_parent(cur_idx)
            #print(self.heap_list[cur_idx])
            if parent_tuple != None and parent_tuple[0][1]> self.heap_list[cur_idx][1]: #如果parent node比較大的話 往上再做一次heapify
                #print(parent_tuple, self.heap_list[cur_idx], cur_idx)
                parent_idx = parent_tuple[1]
                #print("reheapifu")
                self.heapify(parent_idx) 
                #print("break")
                break
            else:
                cur_idx+=1
            
            #print("heapify", self.heap_list)
        if len(self.heap_list):
            self.root  = self.heap_list[0]
    def pop_min(self):
        #cur_min = self.root
        self.heap_list[0], self.heap_list[-1] = self.heap_list[-1], self.heap_list[0]
        cur_min  = self.heap_list.pop()
        self.heapify()
        return cur_min
    
    def get_left_child(self, idx):
        return self.heap_list[2*idx+1]
    def get_right_child(self, idx):
        return self.heap_list[2*idx+2]
    def get_parent(self, idx):
        if idx % 2 == 1:
            parent_idx= int((idx-1)/2)
            if parent_idx>=0:
                return self.heap_list[parent_idx], parent_idx #左子樹
        parent_idx= int((idx/2)-1) #右子樹
        if parent_idx>=0:
            return self.heap_list[parent_idx], parent_idx
        return None


# In[ ]:




