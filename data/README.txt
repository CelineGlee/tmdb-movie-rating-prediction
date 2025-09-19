================= Data Files =================

1. TMDB_train.csv
-----------------
This file contains the movie features and labels for training instances.
Number of instances: 100,000
Number of columns: 44

The columns are (the column names are in the first row):
id, title, release_year, overview, tagline, runtime, budget, revenue, adult, original_language, popularity, production_companies, genre_Action, genre_Adventure, genre_Animation, genre_Comedy, genre_Crime, genre_Documentary, genre_Drama, genre_Family, genre_Fantasy, genre_History, genre_Horror, genre_Music, genre_Mystery, genre_Romance, genre_Science Fiction, genre_TV Movie, genre_Thriller, genre_War, genre_Western, product_of_Canada, product_of_France, product_of_Germany, product_of_India, product_of_Italy, product_of_Japan, product_of_Spain, product_of_UK, product_of_USA, product_of_other_countries, vote_count, rate_category, average_rate

The columns title, overview, tagline and production_companies contain the raw text data of these features.

The class label is in the last two columns: rate_category and average_rate. rate_category has 6 possible levels: 0, 1, 2, 3, 4 or 5.

2. TMDB_eval.csv
-----------------
This file contains the movie features and labels for evaluation instances.
Number of instances: 20,000
Number of columns: 44

The columns in this dataset are similar to the training instances. You are going to use these instances to check the performance of your models.

2. TMDB_test.csv
----------------
This file contains the movie features for test instances.
Number of instances: 20,000
Number of columns: 42

The columns are (the column names are in the first row):
id, title, release_year, overview, tagline, runtime, budget, revenue, adult, original_language, popularity, production_companies, genre_Action, genre_Adventure , genre_Animation, genre_Comedy, genre_Crime, genre_Documentary, genre_Drama, genre_Family, genre_Fantasy, genre_History, genre_Horror, genre_Music, genre_Mystery, genre_Romance, genre_Science Fiction, genre_TV Movie, genre_Thriller, genre_War, genre_Western, product_of_Canada, product_of_France, product_of_Germany, product_of_India, product_of_Italy, product_of_Japan, product_of_Spain, product_of_UK, product_of_USA, product_of_other_countries, vote_count

You are going to use your developed model to predict the label for these instances and upload your results to the Kaggle Inclass competition.

3. TMDB_unlabelled.csv
----------------------
This file contains the movie features for unsupervised or semi-supervised training.
Number of instances: 254,701 
Number of columns: 42

The columns in this dataset are similar to the test dataset.

4. TMDB_text_features_*.zip: 
-----------------------------
pre-processed text features for training, evaluation, test and unlabelled sets, 1 zipped file for each text encoding method

4.1 TMDB_text_features_bow.zip
------------------------------
CountVectorizer converts the text features to a matrix of token counts. 

There are 25 files in this folder.

**(1) title_vocab_bow.pkl
This file contains tokens that the CountVectorizer extracted using the movie "title" text in the training set.

To load the file in Python:
	title_vocab = pickle.load(open("train_title_vocab.pkl", "rb"))
	
To access the list of vocabulary (this will give you a dict):
	title_vocab_dict = title_vocab.vocabulary_
	
More about how to use the CountVectorizer can be found: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

**(2,3,4) overview_vocab_bow.pkl, tagline_vocab_bow.pkl, production_companies_vocab_bow.pkl
These files contain tokens that the CountVectorizer extracted using the movie "overview",  tagline , and  production_companies  text in the training set respectively.

**(5) concat_vocab_bow.pkl
This file contains tokens that the CountVectorizer extracted using all four text features connected using a command such as the following:
train['concatenated_text'] = train['title'] + ' ' + train['overview'] + ' ' + train['tagline'] + ' ' + train['production_companies']

**(6) train_title_bow.npz
This file contains a sparse matrix of the CountVectorizer (Bag-of-Word) representation of the movie titles for training data.

The dense version of this matrix should be [100,000 * size of vocabulary], and the element (i,j) in the matrix is the count of each vocabulary term j in instance i. The vocabulary corresponds to the vocabulary_ attribute of vocab (which can be checked as detailed in (1))
 
As many elements in this matrix are zeros, it has been compressed to a sparse matrix. After loading, the sparse matrix can be used as a normal matrix for training or testing.

To load the sparse matrix:
	import scipy
	scipy.sparse.load_npz('train_title_bow.npz')

**(7, 8, 9, 10) train_overview_bow.npz, train_tagline_bow.npz, train_production_companies_bow.npz, train_concat_bow.npz are the sparse matrices for "overview",  tagline ,  production_companies  and the concatenated features in the training dataset, respectively.

**(11, 12, 13, 14, 15) eval_title_bow.npz, eval_overview_bow.npz, eval_tagline_bow.npz, eval_production_companies_bow.npz, eval_concat_bow.npz are the sparse matrices for  title , "overview",  tagline ,  production_companies  and the concatenated features in the training dataset, respectively.

**(16, 17, 18, 19, 20) test_title_bow.npz, test_overview_bow.npz, test_tagline_bow.npz, test_production_companies_bow.npz, test_concat_bow.npz are the sparse matrices for  title , "overview",  tagline ,  production_companies  and the concatenated features in the training dataset, respectively.

**(21, 22, 23, 24, 25) unlabelled_title_bow.npz, unlabelled_overview_bow.npz, unlabelled_tagline_bow.npz, unlabelled_production_companies_bow.npz, unlabelled_concat_bow_npz are the sparse matrices for  title , "overview",  tagline ,  production_companies  and the concatenated features in the training dataset, respectively.

4.2 movie_text_features_tfidf.zip
---------------------------------
TfidfVectorizer converts the text features to a vector of values that measure their importance using the TFIDF formula. 

There are 25 files in this folder.

**(1, 2, 3, 4, 5) title_vocab_tfidf.pkl, overview_vocab_tfidf.pkl, tagline_vocab_tfidf.pkl, production_companies_vocab_tfidf.pkl, concat_vocab_tfidf.pkl
These files contain tokens that the TfidfVectorizer extracted using the movie  title , "overview",  tagline ,  production_companies  and the concatenated features, respectively.

**(6, 7, 8, 9, 10) train_title_tfidf.npz, train_overview_tfidf.npz, train_tagline_tfidf.npz, train_production_companies_tfidf.npz, train_concat_tfidf.npz are the sparse importance matrices for  title , "overview",  tagline , and  production_companies  features in the training dataset, respectively.

**(11, 12, 13, 14, 15) eval_title_tfidf.npz, eval_overview_tfidf.npz, eval_tagline_tfidf.npz, eval_production_companies_tfidf.npz, eval_concat_tfidf.npz are the sparse importance matrices for  title , "overview",  tagline , and  production_companies  features in the evaluation dataset, respectively.

**(16, 17, 18, 19, 20) test_title_tfidf.npz, test_overview_tfidf.npz, test_tagline_tfidf.npz, test_production_companies_tfidf.npz, test_concat_tfidf.npz are the sparse importance matrices for  title , "overview",  tagline , and  production_companies  features in the test dataset, respectively.

**(21, 22, 23, 24, 25) unlabelled_title_tfidf.npz, unlabelled_overview_tfidf.npz, unlabelled_tagline_tfidf.npz, unlabelled_production_companies_tfidf.npz, unlabelled_concat_tfidf.npz are the sparse importance matrices for  title , "overview",  tagline , and  production_companies  features in the unlabelled dataset, respectively.


