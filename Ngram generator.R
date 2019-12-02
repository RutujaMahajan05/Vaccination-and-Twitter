memory.limit()
library(caret)
library(tm)

tweet_preprocessed_data_1 <- read.csv("Tweets_Emotion_Sentiment_Topic.csv", stringsAsFactors = FALSE)

# Convert the output labels to factor
levels(tweet_preprocessed_data_1$Pro.Anti.Score) <- make.names(levels(factor(tweet_preprocessed_data_1$Pro.Anti.Score)))
# Check the number of tweets as Pro-vaxer, Anti-vaxer and Neutral
table(tweet_preprocessed_data_1$Pro.Anti.Score)

# Convert emotions, sentiments and topic to numeric from character
tweet_preprocessed_data_1$anticipation <- as.numeric(as.character(tweet_preprocessed_data_1$anticipation))
tweet_preprocessed_data_1$disgust <- as.numeric(as.character(tweet_preprocessed_data_1$disgust))
tweet_preprocessed_data_1$fear <- as.numeric(as.character(tweet_preprocessed_data_1$fear))
tweet_preprocessed_data_1$joy <- as.numeric(as.character(tweet_preprocessed_data_1$joy))
tweet_preprocessed_data_1$sadness <- as.numeric(as.character(tweet_preprocessed_data_1$sadness))
tweet_preprocessed_data_1$surprise <- as.numeric(as.character(tweet_preprocessed_data_1$surprise))
tweet_preprocessed_data_1$trust <- as.numeric(as.character(tweet_preprocessed_data_1$trust))
tweet_preprocessed_data_1$negative <- as.numeric(as.character(tweet_preprocessed_data_1$negative))
tweet_preprocessed_data_1$positive <- as.numeric(as.character(tweet_preprocessed_data_1$positive))
tweet_preprocessed_data_1$topics <- as.numeric(as.character(tweet_preprocessed_data_1$topics))

# 75-25 split to get only row numbers of data
# Stratified split
indexes <- createDataPartition(tweet_preprocessed_data_1$Pro.Anti.Score, times = 1,
                               p = 0.75, list = FALSE)

# Filter row by indexes
train <- tweet_preprocessed_data_1[indexes,]
test <- tweet_preprocessed_data_1[-indexes,]

# Verify equal proportions
prop.table(table(train$Pro.Anti.Score))
prop.table(table(test$Pro.Anti.Score))

# Create tokens
library(quanteda)
train_tokens <- tokens(train$Tweet, what = "word")
train_tokens[[300]] # Check tokens for 300th tweet

train_tokens <- tokens_wordstem(train_tokens, language = "english")

# Create bag-of-words model
train_tokens_dfm <- dfm(train_tokens)
train_tokens_dfm

# Transform to a matrix and inspect
train_tokens_matrix <- as.matrix(train_tokens_dfm)
# View(train_tokens_matrix[1:20, 1:50])
dim(train_tokens_matrix)

# Add tokens with labels to create a dataframe
train_tokens_df <- cbind(Pro.Anti.Score = train$Pro.Anti.Score, as.data.frame(train_tokens_dfm))
# Additional pre-processing 
names(train_tokens_df) <- make.names(names(train_tokens_df), unique = TRUE)

# 10 fold cross validation for 3 times
cv.folds <- createMultiFolds(train$Pro.Anti.Score, k = 10, times = 3)
cv.cntrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, index = cv.folds,savePredictions = "final",
                         summaryFunction = multiClassSummary, classProbs = TRUE, verbose = FALSE)

library(doSNOW)
library(e1071)

# Execute code for Support Vector Machine model
start.time <- Sys.time()

# Create cluster to work on 3 logical/parallel cores (Increase if execution takes more time)
# Increases CPU percentage by running different instances of Rstudio in the background.
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# SVM algorithm for our model
rpart.cv.1 <- train(Pro.Anti.Score ~ ., data = train_tokens_df, method = "svmRadial", 
                    trControl = cv.cntrl, tuneLength = 7)

# Processing is done, stop cluster
stopCluster(cl)

# Total time of execution
total.time <- Sys.time() - start.time ; total.time

# Check the results
rpart.cv.1


#                    --------------------------------------------------
#                    --------------------------------------------------


# TF-IDF

#Function to calculate term frequency (TF):
term_frequency <- function(row) {
  row / sum(row)
}

#Function to calculate inverse term document (IDF)
inverse_doc_freq <- function(col) {
  corpus_size <- length(col)
  doc_count <- length(which(col > 0))
  
  log10(corpus_size / doc_count)
}

#Function to calculate TF-IDF
tf_idf <- function(tf, idf) {
  tf * idf
}


#Step 1 : Normalize all documents using TF
# 1 represents rows while 2 represents columns
train_tokens_df_tf <- apply(train_tokens_matrix, 1, term_frequency)
dim(train_tokens_df_tf)
#dim(train_tokens_matrix)
#Check the above code to see that rows and columns are swapped 
#i.e transposed the matrix
#View(train_tokens_df_tf[1:20,1:20])

#Step 2: Calculate IDF that we will be used for both training and test data
#We will use these same idf values to test the new data.
train_tokens_idf <- apply(train_tokens_matrix, 2, inverse_doc_freq)
str(train_tokens_idf)
#Output specifies idf values for corresponding terms

#Step 3: Calculate TF_IDF for training set
train_tokens_tfidf <- apply(train_tokens_df_tf, 2, tf_idf, idf = train_tokens_idf)
dim(train_tokens_tfidf)
#View(train_tokens_tfidf[1:20,1:20])

# Transpose the matrix again
train_tokens_tfidf <- t(train_tokens_tfidf)
dim(train_tokens_tfidf)
#View(train_tokens_tfidf[1:20,1:20])

#Check for incomplete cases
incomplete.cases <- which(!complete.cases(train_tokens_tfidf))
train$Tweet[incomplete.cases]

#Fix incomplete cases
train_tokens_tfidf[incomplete.cases,] <- rep(0.0, ncol(train_tokens_tfidf))
dim(train_tokens_tfidf)
sum(which(!complete.cases(train_tokens_tfidf)))

#clean up the data frame names 
train.token_tfidf_df <- cbind(Pro.Anti.Score = train$Pro.Anti.Score, data.frame(train_tokens_tfidf))
names(train.token_tfidf_df) <- make.names(names(train.token_tfidf_df))

# Execute code with new dataset of tf-idf
start.time <- Sys.time()

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)
rpart.cv.2 <- train(Pro.Anti.Score ~ ., data = train.token_tfidf_df, method = "svmRadial", 
                    trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
total.time <- Sys.time() - start.time ; total.time
rpart.cv.2


#                    --------------------------------------------------
#                    --------------------------------------------------


# Adding n-grams, unigrams and bigrams both to the training data and TF-IDF
# Transform the expanded feature matrix to see if accuracy improves

# Generating unigrams (n = 1), bigrams (n = 2) and unified-grams (n = 1:2)

# N- grams or unified grams generation
train_tokens[[35]]
train_tokens_ngram <- tokens_ngrams(train_tokens, n = 1:2)
train_tokens_ngram[[35]]

# Transform to dfm and then a matrix
train_tokens_ngram_dfm <- dfm(train_tokens_ngram)
train_tokens_ngram_matrix <- as.matrix(train_tokens_ngram_dfm)
train_tokens_ngram_dfm
# Check number of features (columns)

# Normalize all documents via TF
train_tokens_ngram_df <- apply(train_tokens_ngram_matrix, 1, term_frequency)

# Calculate IDF vector that will be used for train and test data
train_tokens_ngram_idf <- apply(train_tokens_ngram_matrix, 2, inverse_doc_freq)

# Calculate TF_IDF for training set
train_tokens_ngram_tfidf <- apply(train_tokens_ngram_df, 2, tf_idf,
                                  idf = train_tokens_ngram_idf)

# Transpose the matrix 
train_tokens_ngram_tfidf <- t(train_tokens_ngram_tfidf)

# Fix incomplete cases
incomplete.cases <- which(!complete.cases(train_tokens_ngram_tfidf))
train_tokens_ngram_tfidf[incomplete.cases,] <- rep(0.0, ncol(train_tokens_ngram_tfidf))

# Make clean data frame
train_tokens_ngram_tfidf_df <- cbind(Pro.Anti.Score = train$Pro.Anti.Score , data.frame(train_tokens_ngram_tfidf))
names(train_tokens_ngram_tfidf_df) <- make.names(names(train_tokens_ngram_tfidf_df))



# Adding only bi-grams to training 
train_tokens_bigram <- tokens_ngrams(train_tokens, n = 2)
train_tokens_bigram[[35]]

train_tokens_bigram_dfm <- dfm(train_tokens_bigram)
train_tokens_bigram_matrix <- as.matrix(train_tokens_bigram_dfm)
train_tokens_bigram_dfm

train_tokens_bigram_df <- apply(train_tokens_bigram_matrix, 1, term_frequency)
train_tokens_bigram_idf <- apply(train_tokens_bigram_matrix, 2, inverse_doc_freq)
train_tokens_bigram_tfidf <- apply(train_tokens_bigram_df, 2, tf_idf,
                                   idf = train_tokens_bigram_idf)
train_tokens_bigram_tfidf <- t(train_tokens_bigram_tfidf)
incomplete.cases <- which(!complete.cases(train_tokens_bigram_tfidf))
train_tokens_bigram_tfidf[incomplete.cases,] <- rep(0.0, ncol(train_tokens_bigram_tfidf))
train_tokens_bigram_tfidf_df <- cbind(Pro.Anti.Score = train$Pro.Anti.Score, data.frame(train_tokens_bigram_tfidf))
names(train_tokens_bigram_tfidf_df) <- make.names(names(train_tokens_bigram_tfidf_df))


# Adding only uni-grams to training 
train_tokens_unigram <- tokens_ngrams(train_tokens, n = 1)
train_tokens_unigram[[35]]
train_tokens_unigram_dfm <- dfm(train_tokens_unigram)
train_tokens_unigram_matrix <- as.matrix(train_tokens_unigram_dfm)
train_tokens_unigram_dfm
train_tokens_unigram_df <- apply(train_tokens_unigram_matrix, 1, term_frequency)
train_tokens_unigram_idf <- apply(train_tokens_unigram_matrix, 2, inverse_doc_freq)
train_tokens_unigram_tfidf <- apply(train_tokens_unigram_df, 2, tf_idf,
                                    idf = train_tokens_unigram_idf)
train_tokens_unigram_tfidf <- t(train_tokens_unigram_tfidf)
incomplete.cases <- which(!complete.cases(train_tokens_unigram_tfidf))
train_tokens_unigram_tfidf[incomplete.cases,] <- rep(0.0, ncol(train_tokens_unigram_tfidf))
train_tokens_unigram_tfidf_df <- cbind(Pro.Anti.Score = train$Pro.Anti.Score, data.frame(train_tokens_unigram_tfidf))
names(train_tokens_unigram_tfidf_df) <- make.names(names(train_tokens_unigram_tfidf_df))


#                    --------------------------------------------------
#                    --------------------------------------------------


# irlba package for SVD. The package allows us to spcify number of most
# importaant singular vectors we wish to calculate and retain for features.
# Provides truncated SVD
library(irlba)
# LSA on n-grams
start.time <- Sys.time()

# Perform the SVD. Reduce dimensionality to 300 columns for LSA
train.irlba_ngram <- irlba(t(train_tokens_ngram_tfidf), nv = 300, maxit = 600)
# Right singular vector means documents or rows and left singular vectore means terms or columns. 

# Total time of execution
total.time <- Sys.time() - start.time
total.time
# Took around 10 mins

# Check new features. Check values in ?irlba
View(train.irlba_ngram$v)


# LSA on bi-grams
start.time <- Sys.time()
train.irlba_bigram <- irlba(t(train_tokens_bigram_tfidf), nv = 300, maxit = 600)
total.time <- Sys.time() - start.time
total.time                         # Took around 7 minutes
View(train.irlba_bigram$v)

# LSA on unigrams 
start.time <- Sys.time()
train.irlba_unigram <- irlba(t(train_tokens_unigram_tfidf), nv = 300, maxit = 600)
total.time <- Sys.time() - start.time
total.time                        #Took around 3 mins
View(train.irlba_unigram$v)

# PROJECT NEW DATA IN TRAINING SET BEFORE RESULTS

# As with TF-IDF, we will need to project new data (eg. test data) into SVD semantic space. Following code achieves this 
# by using a row of training data that has already been transformed by TF-IDF as per the mathematics

# N- grams
sigma_inverse_ngram <- 1 / train.irlba_ngram$d   
u_transpose_ngram <- t(train.irlba_ngram$u)
document_ngram <- train_tokens_ngram_tfidf[1,]
document_hat_ngram <- sigma_inverse_ngram * u_transpose_ngram %*% document_ngram

# Look at first 10 components of projected document & corresponding row
# in our document semantic space (i.e the V matrix)
document_hat_ngram[1:10]
train.irlba_ngram$v[1, 1:10]
# Check if they both are same. 

# Bi-grams
sigma_inverse_bigram <- 1 / train.irlba_bigram$d 
u_transpose_bigram <- t(train.irlba_bigram$u)
document_bigram <- train_tokens_bigram_tfidf[1,]
document_hat_bigram <- sigma_inverse_bigram * u_transpose_bigram %*% document_bigram
document_hat_bigram[1:10]
train.irlba_bigram$v[1, 1:10]

# Uni-grams
sigma_inverse_unigram <- 1 / train.irlba_unigram$d 
u_transpose_unigram <- t(train.irlba_unigram$u)
document_unigram <- train_tokens_unigram_tfidf[1,]
document_hat_unigram <- sigma_inverse_unigram * u_transpose_unigram %*% document_unigram
document_hat_unigram[1:10]
train.irlba_unigram$v[1, 1:10]

# Data frame for new features of n-grams, bi-grams and uni-grams
train_ngrams <- data.frame(Pro.Anti.Score = train$Pro.Anti.Score, train.irlba_ngram$v)
train_bigrams <- data.frame(Pro.Anti.Score = train$Pro.Anti.Score, train.irlba_bigram$v)
train_unigrams <- data.frame(Pro.Anti.Score = train$Pro.Anti.Score, train.irlba_unigram$v)

# Add emotions, sentiments and topics to the new feature set for n-grams model
train_ngrams$anger <- train$anger
train_ngrams$anticipation <- train$anticipation
train_ngrams$disgust <- train$disgust
train_ngrams$fear <- train$fear
train_ngrams$joy <- train$joy
train_ngrams$sadness <- train$sadness
train_ngrams$surprise <- train$surprise
train_ngrams$trust <- train$trust
train_ngrams$negative <- train$negative
train_ngrams$positive <- train$positive
train_ngrams$topics <- train$topics

# Add to bi-gram model
train_bigrams$anger <- train$anger
train_bigrams$anticipation <- train$anticipation
train_bigrams$disgust <- train$disgust
train_bigrams$fear <- train$fear
train_bigrams$joy <- train$joy
train_bigrams$sadness <- train$sadness
train_bigrams$surprise <- train$surprise
train_bigrams$trust <- train$trust
train_bigrams$negative <- train$negative
train_bigrams$positive <- train$positive
train_bigrams$topics <- train$topics

# Add to uni-grams model
train_unigrams$anger <- train$anger
train_unigrams$anticipation <- train$anticipation
train_unigrams$disgust <- train$disgust
train_unigrams$fear <- train$fear
train_unigrams$joy <- train$joy
train_unigrams$sadness <- train$sadness
train_unigrams$surprise <- train$surprise
train_unigrams$trust <- train$trust
train_unigrams$negative <- train$negative
train_unigrams$positive <- train$positive

# N-grams model to see the results
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)
start.time <- Sys.time()

# Re-run the training process with the additional feature.
n_grams_new_features <- train(Pro.Anti.Score ~ ., data = train_ngrams, method = "rf",
                          trControl = cv.cntrl, tuneLength = 7, importance = TRUE)

stopCluster(cl)
total.time <- Sys.time() - start.time
total.time
n_grams_new_features
# Run bi-grams and uni-grams model 

#                    --------------------------------------------------
#                    --------------------------------------------------

library(lsa)

# Calculate cosine distance for tweets
# No need to calculate cosine similarity of emotions, sentiments as tweets as it is outside of our vector representation

train_ngrams[1:10, c(1,302:311)]

train_similarities_ngrams <- cosine(t(as.matrix(train_ngrams[,-c(1,302:311)])))
dim(train_similarities_ngrams)
train_similarities_ngrams[1:10,1:10]

train_similarities_bigrams <- cosine(t(as.matrix(train_bigrams[,-c(1,302:311)])))
dim(train_similarities_bigrams)

train_similarities_unigrams <- cosine(t(as.matrix(train_unigrams[,-c(1,302:311)])))
dim(train_similarities_unigrams)

# The hypothesis being same labeled tweets must be similar or clustered together.

# Find indexes of Anti-vaxer and Pro-vaxer tweets
Anti_indexes <- which(train$Pro.Anti.Score == "X2")  # X2 is label for Anti-vaxer tweets
Pro_indexes <- which(train$Pro.Anti.Score == "X1")   # X1 is label for Pro-vaxer tweets

# Cosine similarity for n-grams
train_ngrams$Anti <- rep(0.0, nrow(train_ngrams))
for (i in 1:nrow(train_ngrams)) {
  train_ngrams$Anti[i] <- mean(train_similarities_ngrams[i, Anti_indexes])
  
}
train_ngrams$Pro <- rep(0.0, nrow(train_ngrams))
for (i in 1:nrow(train_ngrams)) {
  train_ngrams$Pro[i] <- mean(train_similarities_ngrams[i, Pro_indexes])
  
}
# Cosine similarity for bi-grams
train_bigrams$Anti <- rep(0.0, nrow(train_bigrams))
for (i in 1:nrow(train_bigrams)) {
  train_bigrams$Anti[i] <- mean(train_similarities_bigrams[i, Anti_indexes])
  
}
train_bigrams$Pro <- rep(0.0, nrow(train_bigrams))
for (i in 1:nrow(train_bigrams)) {
  train_bigrams$Pro[i] <- mean(train_similarities_bigrams[i, Pro_indexes])
  
}
# Cosine similarity for uni-grams
train_unigrams$Anti <- rep(0.0, nrow(train_unigrams))
for (i in 1:nrow(train_unigrams)) {
  train_unigrams$Anti[i] <- mean(train_similarities_unigrams[i, Anti_indexes])
  
}
train_unigrams$Pro <- rep(0.0, nrow(train_unigrams))
for (i in 1:nrow(train_unigrams)) {
  train_unigrams$Pro[i] <- mean(train_similarities_unigrams[i, Pro_indexes])
  
}

# Visualize data grouped together as Pro-vaxer, Anti-vaxer and Neutral
library(ggplot2)

# N-gram model
ggplot(train_ngrams, aes(x = c(Anti), fill = Pro.Anti.Score)) + 
  theme_bw() + 
  geom_histogram(binwidth = 0.05) + 
  labs(y = "Tweet Count",
       x = "Mean Anti Tweet Cosine Similarity",
       title = "Distribution of Pro Anti Neutral tweets using Anti Cosine Similarity on n-grams model")

# Check performance of new dataset 
# N-gram model
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)
start.time <- Sys.time()
n_grams_SVM_cosine <- train(Pro.Anti.Score ~ ., data = train_ngrams, method = "svmRadial",
                                    trControl = cv.cntrl, tuneLength = 7)
stopCluster(cl)
total.time <- Sys.time() - start.time
total.time
n_grams_SVM_cosine

#                    --------------------------------------------------
#                    --------------------------------------------------

#                                         Test Data

# Convert emotions, sentiments and topic to numeric from character
test$anger <- as.numeric(as.character(test$anger))
test$anticipation <- as.numeric(as.character(test$anticipation))
test$disgust <- as.numeric(as.character(test$disgust))
test$fear <- as.numeric(as.character(test$fear))
test$joy <- as.numeric(as.character(test$joy))
test$sadness <- as.numeric(as.character(test$sadness))
test$surprise <- as.numeric(as.character(test$surprise))
test$trust <- as.numeric(as.character(test$trust))
test$negative <- as.numeric(as.character(test$negative))
test$positive <- as.numeric(as.character(test$positive))

test_tokens <- tokens(test$Tweet, what = "word")

test_tokens <- tokens_wordstem(test_tokens, language = "english")

# Add n-grams or unified grams
test_tokens_ngrams <- tokens_ngrams(test_tokens, n = 1:2)

#Convert n-grams to dtm matrix
test_tokens_ngrams_dfm <- dfm(test_tokens_ngrams)

# Explore the train and test dfm objects
train_tokens_ngram_dfm
test_tokens_ngrams_dfm
# The difference between number of features and documents in train and test set
# Make sure that in your test dfm you are providing features atleast as much as 
# there are in training set. Since our data is split in 75-25, you might already
# have more features in test set

# Add bi-grams
test_tokens_bigrams <- tokens_ngrams(test_tokens, n = 2)
test_tokens_bigrams_dfm <- dfm(test_tokens_bigrams)
train_tokens_bigram_dfm
test_tokens_bigrams_dfm

# Add uni-grams
test_tokens_unigrams <- tokens_ngrams(test_tokens, n = 1)
test_tokens_unigrams_dfm <- dfm(test_tokens_unigrams)
train_tokens_unigram_dfm
test_tokens_unigrams_dfm

# Ensure that test dfm has the same n-grams as the training dfm
test_tokens_ngrams_dfm <- dfm_select(test_tokens_ngrams_dfm, pattern = train_tokens_ngram_dfm, selection = "keep")
test_tokens_ngrams_matrix <- as.matrix(test_tokens_ngrams_dfm)
test_tokens_ngrams_dfm

# Ensure that test dfm has the same bi-grams as the training dfm
test_tokens_bigrams_dfm <- dfm_select(test_tokens_bigrams_dfm, pattern = train_tokens_bigram_dfm, selection = "keep")
test_tokens_bigrams_matrix <- as.matrix(test_tokens_bigrams_dfm)
test_tokens_bigrams_dfm

# Ensure that test dfm has the same uni-grams as the training dfm
test_tokens_unigrams_dfm <- dfm_select(test_tokens_unigrams_dfm, pattern = train_tokens_unigram_dfm, selection = "keep")
test_tokens_unigrams_matrix <- as.matrix(test_tokens_unigrams_dfm)
test_tokens_unigrams_dfm


#                    --------------------------------------------------
#                    --------------------------------------------------


# With the raw test features in place next up is the projecting the term
# counts for the unigrams into the same TF-IDF vector space as our training
# data. The high level process is as follows:
#      1 - Normalize each document (i.e, each row)
#      2 - Perform IDF multiplication using training IDF values


# N-grams
# Normalize all documents via TF.
test_tokens_ngrams_df <- apply(test_tokens_ngrams_matrix, 1, term_frequency)
str(test_tokens_ngrams_df)

# Lastly, calculate TF-IDF for our training corpus.
test_tokens_ngrams_tfidf <- apply(test_tokens_ngrams_df, 2, tf_idf, idf = train_tokens_ngram_idf)
dim(test_tokens_ngrams_tfidf)
#View(test_tokens_ngrams_tfidf[1:20,1:20])

# Transpose the matrix
test_tokens_ngrams_tfidf <- t(test_tokens_ngrams_tfidf)

# Fix incomplete cases
summary(test_tokens_ngrams_tfidf[1,])
test_tokens_ngrams_tfidf[is.na(test_tokens_ngrams_tfidf)] <- 0.0
summary(test_tokens_ngrams_tfidf[1,])

# Do the same for bi-grams and uni-grams

# With the test data projected into the TF-IDF vector space of the training data we can now to the final projection 
# into the training LSA semantic space (i.e. the SVD matrix factorization).
test_svd_raw_ngrams <- t(sigma_inverse_ngram * u_transpose_ngram %*% t(test_tokens_ngrams_tfidf))
dim(test_svd_raw_ngrams)

# Lastly, we can now build the test data frame to feed into our trained machine learning model for predictions. 
# First up, add output labels to the data
test_svd_ngrams <- data.frame(Pro.Anti.Score = test$`Pro-Anti Score`, test_svd_raw_ngrams)

# Next step, calculate Anti and Pro Similarity for all the test documents. 
test_similarities_ngram_Anti <- rbind(test_svd_raw_ngrams, train.irlba_ngram$v[Anti_indexes,])
test_similarities_ngram_Anti <- cosine(t(test_similarities_ngram))
dim(test_similarities_ngram_Anti)
test_similarities_ngram_Pro <- rbind(test_svd_raw_ngrams, train.irlba_ngram$v[Pro_indexes,])
test_similarities_ngram_Pro <- cosine(t(test_similarities_ngram))
dim(test_similarities_ngram_Pro)

test_svd_ngrams$Anti <- rep(0.0, nrow(test_svd_ngrams))
Anti_cols <- (nrow(test_svd_ngrams) + 1): ncol(test_similarities_ngram_Anti)
for ( i in 1:nrow(test_svd_ngrams)) {
  test_svd_ngrams$Anti <- mean(test_similarities_ngram[i, Anti_cols])
}
test_svd_ngrams$Pro <- rep(0.0, nrow(test_svd_ngrams))
Pro_cols <- (nrow(test_svd_ngrams) + 1): ncol(test_similarities_ngram_Pro)
for ( i in 1:nrow(test_svd_ngrams)) {
  test_svd_ngrams$Pro <- mean(test_similarities_ngram[i, Pro_cols])
}

# Some tweets become empty as a result of stopword and special character removal. 
# This results in spam similarity measures of 0.
test_svd_ngrams$Anti[!is.finite(test_svd_ngrams$Anti)] <- 0
test_svd_ngrams$Pro[!is.finite(test_svd_ngrams$Pro)] <- 0

# Add emotions, sentiments and topics to test data
test_svd_ngrams$anger <- test$anger
test_svd_ngrams$anger <- test$anger
test_svd_ngrams$anticipation <- test$anticipation
test_svd_ngrams$disgust <- test$disgust
test_svd_ngrams$fear <- test$fear
test_svd_ngrams$joy <- test$joy
test_svd_ngrams$sadness <- test$sadness
test_svd_ngrams$surprise <- test$surprise
test_svd_ngrams$trust <- test$trust
test_svd_ngrams$negative <- test$negative
test_svd_ngrams$positive <- test$positive
test_svd_ngrams$topics <- test$topics

# Make predictions on test data set
SVM_pred_ngram <- predict(n_grams_SVM_cosine, test_svd_ngrams)
confusionMatrix(SVM_pred_ngram, test_svd_ngrams$Pro.Anti.Score)



