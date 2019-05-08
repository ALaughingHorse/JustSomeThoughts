import pandas as pd
import numpy as np
import gensim

class review_analyzer:
    """
    Usage:
    
    #Initiate the analyzer with text, stem and lemmatization flags as well as stop words:
    
    sa = review_analyzer(text)
    
    # Get the lemmatized and stemmed version of the text:
    sa.lemmatize_stemming()
    
    # Tokenize the text 
    sa.txt_pre_pros()
    
    # If the input text is a sequence/list of text, use txt_pre_pros_all
    
    # Get popular nouns and adjectives
    sa.get_Nouns_Adjs()
    
    # Get text summary with related term search:
    sa.get_review_summary()  
    """
    
    def __init__(self,text,stem_flag = True, lemma_flag = False,stopwords = []):
        self.text = text
        self.stem_flag = stem_flag
        self.lemma_flag = lemma_flag
        self.stopwords = stopwords
    
    def flatten(listOfLists):
        from itertools import chain
        "Flatten one level of nesting"
        return chain.from_iterable(listOfLists)

    def lemmatize_stemming(self):
        
        text = self.text
        lemma = self.lemma_flag
        stem = self.stem_flag
        
        from nltk.stem import WordNetLemmatizer, SnowballStemmer
        stemmer = SnowballStemmer("english")

        if lemma:
            text = WordNetLemmatizer().lemmatize(text, pos='v')

        if stem:
            text = stemmer.stem(text)

        return text
 
    def txt_pre_pros(self,lemmatize_stemming = lemmatize_stemming):
        """
        Perform pre-processing for text contents

        ------Parameters
        text: the text input

        ------Output
        the stemmed and lemmatized 
        """
        
        processed_text = lemmatize_stemming(self)
        
        stopwords = self.stopwords
        result = []
        import gensim
        for token in gensim.utils.simple_preprocess(processed_text):
            if token not in stopwords and len(token):
                result.append(token)
        return result     
    
    def txt_pre_pros_all(self):
        """
        Process collection of documents
        """
        docs = pd.Series(self.text).apply(lambda x: review_analyzer(x).txt_pre_pros())
        return(docs)  
    
    def get_Nouns_Adjs(self,add_stop = [],flatten = flatten):
        """
        Take in a list/Series of documents, output the most popular Nouns and Adjtives and write into csv files

        product_name should also be provided as inputs for stop-word removal later
        """
        import string
        import pandas as pd
        import nltk
        import numpy as np
        import gensim
        from gensim.utils import simple_preprocess
        from gensim.parsing.preprocessing import STOPWORDS
        
        # Frist pre process the documents. Tokenize without stopwords, lemmatization or stemming.
        processed_doc = self.txt_pre_pros_all()
        
        # Tag each word within a list of tokenized documents
        tagged_doc = processed_doc.apply(lambda x: nltk.pos_tag(x))

        # Extract all the tags
        tags = tagged_doc.apply(lambda x: [tup[1] for tup in x])

        # Remove stopwords
        all_stops = list(gensim.parsing.preprocessing.STOPWORDS) + add_stop
        tagged_doc_cleaned = tagged_doc.apply(lambda x: [tup for tup in x if review_analyzer(tup[0]).lemmatize_stemming() \
                                                         not in all_stops])

        # Need to do the same for processed documents
        processed_doc_cleaned = processed_doc.apply(lambda x: [token for token in x if \
                                                               review_analyzer(token).lemmatize_stemming() not in all_stops])
        
        # Remove the nested structure of the tuples
        all_tups = list(flatten(tagged_doc_cleaned))
        all_terms = list(flatten(processed_doc_cleaned))

        # Find all the Nouns
        idx_n = [(tup[1] in ['NN','NNS','NNP','NNPS']) for tup in all_tups]
        all_nouns = (pd.Series(all_terms)[idx_n])

        # Find all the Adjs
        idx_a = [(tup[1] in ['JJ','JJS','JJR']) for tup in all_tups]
        all_adjs = (pd.Series(all_terms)[idx_a])

        # Construct the Noun Table
        all_noun = pd.DataFrame({'Nouns':all_nouns})
        all_noun_agg = pd.DataFrame(all_noun.groupby('Nouns').size().sort_values(ascending = False)).reset_index()
        all_noun_agg.columns = ['Terms','Count']

        # Constuct the Adj Table
        all_adj = pd.DataFrame({'Adjs':all_adjs})
        all_adj_agg = pd.DataFrame(all_adj.groupby('Adjs').size().sort_values(ascending = False)).reset_index()
        all_adj_agg.columns = ['Terms','Count']

        # Write the files
        #all_noun_agg.to_csv('pythonOutputs/all_nouns_tb.csv')
        #all_adj_agg.to_csv('pythonOutputs/all_adj_tb.csv')

        class output:
            all_adj  = all_adj_agg
            all_noun = all_noun_agg
            tagged_terms = tagged_doc_cleaned

        return(output)
    
    def get_related_terms(word,target_type, boundary_type, tagged_terms):
  
        """ For a specific word/term, find the related terms of specific type 

        ------Parameters:
        word: the word/term
        target_type: the type of related terms that we are looking for
        boundary_type: the type of boundary terms. Usually the same as the type of the input word
        tagged_terms: A POS tagged document

        ------Return:
        related terms

        """
        import numpy as np
        if len(tagged_terms) == 0:
            return []

        all_terms = np.array([x[0] for x in tagged_terms])
        all_tags  = np.array([x[1] for x in tagged_terms])

        if word not in all_terms:
            return []

        word_location = np.array(range(len(tagged_terms)))[[all_terms == word]]
        boundary_location = np.array(range(len(tagged_terms)))[[(all_tags[i] in boundary_type) for i in range(len(all_tags))]]
        related_terms = []


        for word_loc in word_location: 
            if ((len(boundary_location) < 3)):
                left_bound = 0
            elif (word_loc <= min(boundary_location)) :
                left_bound = 0
            else:
                left_bound = max(boundary_location[boundary_location < word_loc])

            if ((len(boundary_location) < 3)):
                right_bound = word_loc + 1
            elif (word_loc >= max(boundary_location)):
                right_bound = word_loc + 1
            else:
                right_bound = min(boundary_location[boundary_location > word_loc])

            related_terms = related_terms + ([all_terms[i] for i in range(len(all_terms)) if (i > left_bound and i < right_bound and all_tags[i] in target_type)])

        return related_terms

    def get_all_related_terms(word, target_type, boundary_type,tagged_terms,n = 15,get_related_terms = get_related_terms):
        """
        A Wrapper of get_related_terms
        """
        tagged_doc_cleaned = tagged_terms
        # Loop over all the terms
        related = []
        import numpy as np
        for i in range(len(tagged_doc_cleaned)):
            related = related + get_related_terms(word = word, target_type = target_type,boundary_type = boundary_type,\
                                                  tagged_terms = tagged_doc_cleaned[i])

        res = pd.DataFrame({"terms":related})
        res_tb = res.groupby('terms').size().sort_values(ascending = False).reset_index()
        res_tb.columns = ["related_terms","Count"]

        return res_tb.iloc[:n,:]
    
    def get_review_summary(self, n = 5, add_stop = [],get_related_terms = get_related_terms, \
                         get_all_related_terms = get_all_related_terms):
     
        """
        Input an array of text, output the summary table.
        """
        get_terms = self.get_Nouns_Adjs(add_stop = add_stop)

        top_adj = get_terms.all_adj.iloc[:15,:]
        top_noun = get_terms.all_noun.iloc[:15,:]

        """
        Find related terms for all nouns
        """
        related_adjs = []

        for x in top_noun['Terms']:
            temp_df = get_all_related_terms(x,['JJ'],['NN','NNS'],get_terms.tagged_terms,n = n)

            related = ", ".join(temp_df['related_terms'])
            related_adjs.append(related)
        top_noun['related_terms'] = related_adjs

        """
        Find related nouns for all adjs
        """

        related_noun = []

        for x in top_adj['Terms']:
            temp_df = get_all_related_terms(x,['NN','NNS'],['JJ'],get_terms.tagged_terms,n = n)
            related = ", ".join(temp_df['related_terms'])
            related_noun.append(related)

        top_adj['related_terms'] = related_noun

        all_keywords = top_adj.append(top_noun).sort_values('Count',ascending = False).reset_index(drop= True)

        return all_keywords