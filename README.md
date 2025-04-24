# DIS ML FINAL PROJECT: Bourdain or Bogus?
## Using ML and NLP to Discover the Truth Behind Reviews
### Dylan Epstein-Gross and Hang Pham

This file contains the code and datasets associated with our final project for DIS's Machine Learning B class.

Explanations of the relevance and use of the code/data can be found in our presentation at https://docs.google.com/presentation/d/1Lc7Lr90mz6tZhgsE4idphYng0tYwPI6ypP-JzzUdJcU/edit?usp=sharing.

In summary, sample_reviews.csv contains the 40000 sample product reviews we used to train our model, and oos_reviews.csv contains the 110 out-of-sample restaurant reviews
we tried to generalize to. The file download_data.ipynb loads the data from Kaggle, and the file exploratory_data_analysis_final.ipynb performs EDA on it.
The files classical_features.ipynb, bow_features.ipynb, and embedding_features.ipynb pertain to the feature selection and model training associated with each
of the three approaches discussed in the presentation. They also contain evaluations of the best models on the out-of-sample dataset. The file train_embedding.py
separately finetunes the ultimate MiniLM embedding model, which is stored in the folder finetuned-embedding-model for use in embedding_features.ipynb.