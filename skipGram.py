# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:46:57 2020

@author: veeru
"""

from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
import collections
import _pickle as pickle



def text2sentences(path):
    punctuations = {'!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',' '}
    sentences = []
    with open(path) as f:
        for l in f:
            words = [''.join(ch for ch in word if ch not in punctuations) for word in l.lower().split()]
            if len(words) > 1:      # we have removed the sentences that have only one word.
                sentences.append( words )
    return sentences

def loadPairs(path):
        data = pd.read_csv(path, delimiter='\t')
        data = pd.DataFrame (data, columns = ['word1','word2','similarity'])
        pairs = zip(data['word1'],data['word2'],data['similarity'])
        return pairs
    
class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        
        # initializing the class variables
       
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount= minCount
        self.nEmbed = nEmbed
        self.lr = 0.01
        
        self.word_counts = collections.defaultdict(int)
        for row in sentences:
            for word in row:
                self.word_counts[word] += 1
        ## How many unique words in vocab
        self.v_count = len(self.word_counts.keys())
        # Generate Lookup Dictionaries (vocab)
        self.vocab = list(self.word_counts.keys())
        
        # Generate word:index
        self.w2id = dict((word, i) for i, word in enumerate(self.vocab))
        
        # Generate index:word
        self.id2w = dict((i, word) for i, word in enumerate(self.vocab))
        
        self.trainset = sentences
        
        # initializing random weight matrices 
        self.w1 = np.random.randn(self.v_count, nEmbed)  
        self.w2 = np.random.randn(nEmbed,self.v_count)   
        
        
        #onehotencoding
    def word2onehot(self, word_index):
        # word_vec - initialise a blank vector
        word_vec = [0 for i in range(0, self.v_count)] # Alternative - np.zeros(self.v_count)
        # Change value from 0 to 1 according to ID of the word
        word_vec[word_index] = 1
        word_vec= np.array(word_vec)
        return word_vec
        
    #noise distribution for negative sampling
    def generate_nd(self,word_count1):
        sum_tc = sum(word_count1.values())
        unig_dist = {key: val/sum_tc for key, val in word_count1.items()}
        alpha      = 3 / 4
        noise_dist = {key: val ** alpha for key, val in unig_dist.items()}
        Z = sum(noise_dist.values())
        word_with_probs = {key: val / Z for key, val in noise_dist.items()}
        return word_with_probs
    
    # defining softmax function
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
                
    #negative sampling
    def sample(self, omit): 
        omit = list(omit)
        widx=self.id2w.get(omit[0])
        cidx=self.id2w.get(omit[1])
        word_count1={k:v for k,v in self.word_counts.items() if k not in omit}

        word_with_probs = self.generate_nd(word_count1) 
        
        # here we chose the negative samples randomly
        sampling = np.random.choice(list(word_with_probs.keys()), size= self.negativeRate,replace = False,p=list(word_with_probs.values()))
       
        neg_sam=[]
        for x in list(sampling):
            neg_sam.append(self.w2id.get(x))
        return(neg_sam)

        
            
    def train(self):
        
        for counter, sentence in enumerate(sentences):
            #sentence = filter(lambda word: word in self.vocab, sentence)
            
            for wpos, word in enumerate(sentence):
                wIdx = self.w2id[word]
                start = max(0, wpos - self.winSize)
                end = min(wpos + self.winSize + 1, len(sentence))

                for context_word in sentence[start:end]:
                    ctxtId = self.w2id[context_word]
                    if ctxtId == wIdx: continue
                    negativeIds = self.sample({wIdx, ctxtId})
                   
                    self.trainWord(wIdx, ctxtId, negativeIds)

                    
                    
    def trainWord(self, wordId, contextId, negativeIds): 
        word_vec=self.word2onehot(wordId)
        
        #calculating the hidden layer
        h= np.dot(np.transpose(self.w1),word_vec)
        
        # multiplying w2 with hidden layer and passing it to softmax
        sig_out_layer= self.softmax(np.dot(np.transpose(self.w2),h))
        
        # caluclating Error
        pred_error= sig_out_layer- word_vec
        
        # calculating the gradients
        grad_to_be_summed= pred_error[contextId] * self.w2[:,(contextId)] #gradiant of contextId
        for i in negativeIds:
            grad_neg = pred_error[i] * self.w2[:,i]                # gradients of negative IDs
            grad_to_be_summed +=grad_neg
        
        grad_output= np.dot(h[:,None],pred_error[None,:])       #gradient of output layer
         
        #updating weights
        self.w1[wordId,] = self.w1[wordId,] - self.lr * grad_to_be_summed # updating input layer
        self.w2= self.w2 - self.lr * grad_output                          #updating output layer
        
    def save(self, path):
    	with open(path, 'wb') as f:
    		pickle.dump(self, f)
               
       
    def similarity(self, word1, word2):
        if (word1 in self.vocab and word2 in self.vocab):
            word1_emb= self.w1[self.w2id[word1],:]         # word embedding for the word1
            word2_emb=self.w1[self.w2id[word2],:]          # word embedding for the word2
        else: 
            return 0.1               # if the word doesnt exist in vocab then we return 0.1
        
        # we use the cosine similarity function to compute the similarity score

        score = np.abs(np.dot(word1_emb, word2_emb) / (np.linalg.norm(word1_emb) * np.linalg.norm(word2_emb)))

        # if a word doesn't exist in the dictionary, as the embedding is set to zeros, the similarity will be Nan
        # here we check if it's Nan, we will generate a random number in[0,1)
        if (np.isnan(score)):
            return np.random.random()
        else:
            return score 
        raise NotImplementedError('implement it!')
        
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            sg = pickle.load(f)
        return sg
    	
    
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg= SkipGram(sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(sg.similarity(a,b))
