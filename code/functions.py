import gensim
import string
import pandas as pd
import nltk
import numpy as np
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import pickle

# nltk.download('averaged_perceptron_tagger')

def lemmatize_stemming(text,stem = True, lemma = True):
    from nltk.stem import WordNetLemmatizer, SnowballStemmer
    stemmer = SnowballStemmer("english")
    
    if lemma:
      text = WordNetLemmatizer().lemmatize(text, pos='v')
      
    if stem:
      text = stemmer.stem(text)
    
    return text


def txt_pre_pros(text,lemma_flag = True, stem_flag = True, stopwords = list(gensim.parsing.preprocessing.STOPWORDS)):
    """
    Perform pre-processing for text contents
    
    ------Parameters
    text: the text input
    
    ------Output
    the stemmed and lemmatized 
    """
    result = []
    import gensim
    for token in gensim.utils.simple_preprocess(text):
        if token not in stopwords and len(token):
            result.append(lemmatize_stemming(token,lemma = lemma_flag, stem = stem_flag))
    return result

def flatten(listOfLists):
    from itertools import chain
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

def get_Nouns_Adjs(docs,product_name = [],add_stop = []):
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

    docs = pd.Series(docs)

    # Frist pre process the documents. Tokenize without stopwords, lemmatization or stemming.
    processed_doc = docs.apply(lambda x: txt_pre_pros(x,stopwords = [], lemma_flag = False, stem_flag = False))

    # Tag each word within a list of tokenized documents
    tagged_doc = processed_doc.apply(lambda x: nltk.pos_tag(x))

    # Extract all the tags
    tags = tagged_doc.apply(lambda x: [tup[1] for tup in x])

    # Remove stopwords
    all_stops = list(gensim.parsing.preprocessing.STOPWORDS) + add_stop
    tagged_doc_cleaned = tagged_doc.apply(lambda x: [tup for tup in x if tup[0] not in all_stops])

    # Need to do the same for processed documents
    processed_doc_cleaned = processed_doc.apply(lambda x: [token for token in x if token not in all_stops])

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

def get_all_related_terms(word, target_type, boundary_type,tagged_terms,n = 15):
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
    
def get_text_summary(text ,n = 5, add_stop = []):
    
    """
    Input an array of text, output the summary table
    """
    get_terms = get_Nouns_Adjs(text,add_stop = add_stop)

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

    all_keywords = top_adj.append(top_noun).reset_index(drop= True)
    
    return all_keywords