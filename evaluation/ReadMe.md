# Evaluation

## Training and Generation
We trained and tested SePass on the train and test data found in the [data folder](../data/).

We ran our method using the following arguments, assuming you downloaded the embedding models:

```
python SePass.py ./data/train.txt --models cc.de.300.vec.gz cc.en.300.vec.gz cc.fr.300.vec.gz cc.es.300.vec.gz cc.it.300.vec.gz cc.tr.300.vec.gz cc.fi.300.vec.gz cc.nl.300.vec.gz cc.pt.300.vec.gz cc.pl.300.vec.gz -rr 0.01 -rv 1000000 -len 50000000  -o ./suggestions/spg_suggestion_list.txt
```

As you can see, we generated 50 million password candidates using half of the vocabulary of each word embedding (1 million), 10 langugages (German, English, French, Spanish, Italian, Finish, Dutch, Polish, Portugese and Turkish). We set the `rr` parameter to 0.01 to only use 1% of the generated rules in order to focus our method more on generating new base words from the word embeddings instead of mostly applying word mangling rules. 

## Evaluating the Hit Percentages

## Evaluating the Novel Vocabularies Found
