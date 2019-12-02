
Connection - R script to collect tweets from twitter. Requires authorization tokens sepcified in script from API. 

Clean Tweets - R script to clean tweets extracted from API. Cleaning steps include 
		1. removing special characters, punctuations, numbers, hashtags, url links, extra white spaces, stopwords 
		2. normalization 
		3. stemming
	       R script to generate emotion and sentiment scores using EmoLex
	        

Topic - Python script to 
	1. generate topics using Latent Dirichlet Allocation
	2. finalize number of topics in dataset
	2. get top 200 words from each topic

Ngram generator - R script for
		1. generating tokens (uni-grams, bi-grams and both) 
		2. creating document-term matrix
		3. applying term frequency-inverse document frequency to document-term matrix
		4. applying latent sematic analysis to get reduced feature space
		5. apply different machine learning techniques
		5. adding emotions, sentiments and topics in combination to get new feature set
		7. adding cosine distance as a similarity measure for tweets
		8. test the results on new data

Word cloud - R script to generate word cloud for topics.

LDAVis - R script to get LDA visualization plot. 




