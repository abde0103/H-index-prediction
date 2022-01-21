# H-index-prediction


Note : Local Paths need to be changed.

DATA :
Embeddings for train : https://drive.google.com/file/d/1BZY5Fp3X5xtSEx-sa7rlGqTXD2MAN_Ll/view?usp=sharing
Embeddings for test : https://drive.google.com/file/d/1--FabNwPe5w_4_M5tGMHk_5RRSi_WxdM/view?usp=sharing
Coauthorship graph : https://drive.google.com/file/d/1sHYaDh8PKQByqq0RA8AA_As5R6pXsYoA/view?usp=sharing
Papers texts : https://drive.google.com/file/d/1gT3XZcWvs5EWdcviVwkO_1Pn3hkrSZ0j/view?usp=sharing

PreprocessingAndGetIndices.py :Used to preporcess text data and to associate to each word in our text data an index.

embedding.py : Used to Train an embedding used the Skip-Gram model and negative simpling

ShowEmbedding.py: Used to visualize the embedding

CalculateEmbedding.py : Used to calculate the Embedding of each author


Excel.py: Used to add a column from a dictionnaire to a dataframe. Was used to add the embedding to ohther graph features.

Classifior+regressor.py : Used to try to train a classifior then a regressor. "Did not improve the result"

AuthorToOneHotEncoding.py/TrainEncodingHindex.py : Used to train the encoding while traing to predict the Hindex : Took too long to run. "Did not comtribute to the final result"

Find_results.ipynb : extract features from graph and test different models on features extracted.
