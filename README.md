# amazon_reviews_nlp
### Top modeling and Sentiment analysis of Amazon "online music" product reviews and metadata from 1996 to 2014  

Author : Srini Ananthakrishnan  
Date   : 12/15/2016  

Data Source:  
J. McAuley and J. Leskovec  

## Topic modelling - Section 1
Pipeline is constructed using NLTK, GraphLab and pyLDAvis packages  
### Modelling Pipeline:  
Step 1: Data Exploration  
Understand data --> Class balance --> Visualize and Explore  

Step 2: Pre-processing  
Tokenize --> Remove stopwords --> Lemmetize  

Step 3: Build N-Gram models  
Unigram --> Bigram --> Trigram  

Step 4: Apply LDA (Latent Dirchilet Allocation)  
GraphLab LDA to identify topics  

Step 5: Visualize Topics  
Visualize Top 5 most relevant words for each topic  

## Sentiment Analysis - Section 2  
Pipeline is constructed using NLTK, GraphLab and pyLDAvis packages  
### Modelling Pipeline:  
Step 1: Data Exploration  
Understand data --> Binarize class --> Explore  

Step 2: Pre-processing  
Split data(Test/Train) --> TF-IDF (Vectorize => Tokenize,Remove stopwords)  

Step 3: Fit Models (with Hyper-param tunning) and evaluate  
Fit different classification models (with tunned hyper-parameters) and evaluate  

Step 4: Fit Models using Cross Validation  
Fit different classification models using cross validation and evaluate  

## Appendix:   
Files (.py) associated with this notebook  
- pre_processing.py  
- sentiment_analysis.py  
## Data  
music_reviews.jason  




