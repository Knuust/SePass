# SePass: Semantic Password Guessing using k-nn Similarity Search in Word Embeddings

We introduce SePass, a novel password guessing method that utilizes word embeddings to discover and exploit semantic correlations in password lists. 
Our tool here is made for research purposes and is intended to be used for further research. It is therefore still a work in progress and not designed for usability at the moment. 
 
 # Overview
Commonly used tools for password guessing work with passwords leaks and use these lists for candidate generation based on handcrafted or inferred rules. These methods are often limited in their capability of producing entirely novel passwords, based on vocabulary not included in the given password lists. SePass, is a novel tool that utilizes word embeddings to discover and exploit semantic correlations in order to guess novel base words for passwords deliberately. 
 
 # License
 
 
 
 # Installation
 
 
 
 ```
 pip install -r requirements.txt
 ```
 
In order to run our method pretrained word embeddings are needed, that are then loaded and used by [gensim](). We suggest using the fasttext models for 157 langugages in order to choose which languages should be . The models can be downloaded [here](https://fasttext.cc/docs/en/crawl-vectors.html).
 
 # Reproducibility
 
 If you are looking to reproduce the results from our corresponding paper (unpublished, in review) you can find detailed instructions and jupyter notebooks in [the evaluation folder](/evaluation/) 
 
 # Usage
 
```
 SePass.py [-h] [--debug] [-len LIST_LENGTH]
               [--models MODELS [MODELS ...]] [-mwl MIN_WORD_LENGTH]
               [-o OUT] [-rr RELEVANT_RULESET_RATIO] [-sr SEMANTIC_RATING]
               [-rv RESTRICT_VOCAB] [--mode MODE]
               pwlist
               
positional arguments:
  pwlist                path of password list to be analyzed

optional arguments:
  -h, --help            show this help message and exit
  --debug               print debug messages
  -len LIST_LENGTH, --list_length LIST_LENGTH
                        number of password suggestions to be created.
                        Default=1000000
  --models MODELS [MODELS ...]
                        name of the fasttext word embeddings models to use
                        (stored in models/ folder)
  -mwl MIN_WORD_LENGTH, --min_word_length MIN_WORD_LENGTH
                        length of the smallest word that should be searched
                        for. Default=4.
  -o OUT, --out OUT     Output path
  -rr RELEVANT_RULESET_RATIO, --relevant_ruleset_ratio RELEVANT_RULESET_RATIO
                        this parameter will determine the percentage of
                        (different) rules to be taken into account.
                        Default=0.1
  -sr SEMANTIC_RATING, --semantic_rating SEMANTIC_RATING
                        higher semantic rating will increase the influence of
                        word semantic in sorting process
  -rv RESTRICT_VOCAB, --restrict_vocab RESTRICT_VOCAB
                        this parameter restricts the number of words of each
                        model to the most frequent k
  --mode MODE           this parameter defines the mode for semantic
                        expansion. Default = k-NN
```
