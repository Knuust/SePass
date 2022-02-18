# SePass: Semantic Password Guessing using k-nn Similarity Search in Word Embeddings

 We introduce SePass, a novel password guessing method that utilizes word embeddings to discover and exploit semantic correlations in password lists. 
 
 # Overview
 
 # License
 
 # Installation
 
 
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
