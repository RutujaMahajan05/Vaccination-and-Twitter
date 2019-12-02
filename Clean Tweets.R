library(tm)
library(SnowballC)
library(RSentiment)
library(stringr)
library(NLP)

tweet_file <- read.csv("D2.csv", header = T, stringsAsFactors = F)

eachtweet <- tweet_file$ttext

TextPreprocessing <- lapply(eachtweet, function(x) {
  
  x = gsub('http\\S+\\s*', '', x) ## Remove URLs
  
  #x = gsub('\\b+RT', '', x) ## Remove RT
  
  #x = gsub('#\\S+', '', x) ## Remove Hashtags and words with them
  
  x = gsub('@\\S+', '', x) ## Remove Mentions
  
  x = gsub('[[:cntrl:]]', '', x) ## Remove Controls and special characters
  
  #x = gsub("\\d", '', x) ## Remove Controls and special characters
  
  x = gsub('[[:punct:]]', '', x) ## Remove Punctuations
  
  x = tolower(x)

  # x = removeWords(x, c(stopwords("english"),"#", "the", "and", "for", "amp", "its", "had", "from", "was", "about", "his", "you",
  #                      "can", "could", "should", "she", "did", "will", "they", "has", "her", "our", "many", "this", "that", 
  #                      "does", "did", "some", "has", "our", "&lt", "&gt") )
  
  # x = removeWords(x, c(stopwords("english"),"vaccine", "vaccines", "vaccination", "hpv", "shot", "immune", "vaccin", "rabi", "fluseason",
  #                      "cdc", "hepat", "flushot", "flu", "immun", "immunis", "influenza", "hepat", "autism", 
  #                      "vaccin", "immun", "antivaccin", "vaccin", "amp"))
  
  x = gsub("^[[:space:]]*","",x) ## Remove leading whitespaces
  
  x = gsub("[[:space:]]*$","",x) ## Remove trailing whitespaces
  
  x = gsub(' +',' ',x) ## Remove extra whitespaces
  
})

#Convert list to matrix
cleaned_output <- matrix(unlist(TextPreprocessing), byrow = TRUE)  # Unlist cleaned tweets and save it as matrix
output <- data.frame(cleaned_output)                               # Convert the matrix into dataframe
colnames(output) <- c("Tweet")                                     # Label the dataframe
write.table(output, file = "Cleaned_Tweets.txt", sep = "\n",
            row.names = FALSE)                                     # Save output to text file
write.csv(output, file = "Cleaned_Tweets.csv")                     # Save output to csv file



#             ---------------------------------------------------
#             ---------------------------------------------------



# Stemming 
tweets_source <- VectorSource(cleaned_output)
tweets_corpus <- VCorpus(tweets_source)   # Convert tweet to corpus
tweets_corpus   # Check the dimensions of corpus
#tweets_corpus[[15]]
#tweets_corpus[[15]][1]
#str(tweets_corpus[[15]])

tweets_corpus <- tm_map(tweets_corpus,stemDocument)        # Stemming words in tweets
tweets_corpus <- tm_map(tweets_corpus, stripWhitespace)    # Remove extra white spaces generated after stemming words
#writeLines(as.character(tweets_corpus[15]))               # Check 15th tweet

dataframe<-data.frame(text=unlist(sapply(tweets_corpus, `[`, "content")), stringsAsFactors=F)   # Convert corpus to dataframe

write.csv(output, file = "Stemmed_Tweets.csv")    # Save output in csv file



#             ---------------------------------------------------
#             ---------------------------------------------------


# Emotion and Sentiment scores 
dataset_ES <- tweet_file$ttext                

Emotion_Analysis <- get_nrc_sentiment(dataset_ES)                         # Get emotion and sentiment scores for dataset
Emotion_Analysis_Scores<-data.frame(colSums(Emotion_Analysis[,]))      # Save scores in dataframe

write.csv(Emotion_Analysis, "Emotion_Analysis.csv")

names(Emotion_Analysis_Scores)<-"Score"
Emotion_Analysis_Scores<-cbind("Emotion"=rownames(Emotion_Analysis_Scores),Emotion_Analysis_Scores)
rownames(Emotion_Analysis_Scores)<-NULL

#plotting the emotions with scores
ggplot(data=Emotion_Analysis_Scores,aes(x=Emotion,y=Score))+geom_bar(aes(fill=Emotion),stat = "identity")+
  theme(legend.position="none")+
  xlab("Emotions")+ylab("scores")+ggtitle("Emotion Valence")


#             ---------------------------------------------------
#             ---------------------------------------------------


#average length of each tweet
Sample_tweets$Text_length <- nchar(Sample_tweets$ttext)
summary(Sample_tweets$Text_length)

Count_of_tweets <- Sample_tweets$Relevancy
Count_of_tweets <- as.factor(Count_of_tweets)
#Sample_tweets$Relevancy <- as.factor(Sample_tweets$Relevancy)

my_plot <- ggplot(Sample_tweets, aes(x = Text_length, fill = Relevancy)) +
  theme_bw() +
  geom_histogram(binwidth = 5) +
  labs(y = "Text Count", x = "Length of Text", 
       title = "Distribution of Text Length with Class Labels")  


my_plot
my_plot + scale_fill_discrete(name = " ", breaks = c(0,1), labels = c("Irrelevant", "Relevant") ) 
#guides(fill = guide_legend(title = Label), breaks = c(0,1), labels = c("Irrelevant", "Relevant"))

my_plot + scale_fill_manual(values=c("deepskyblue3", "gray62"), 
                            name = " ", 
                            breaks = c(0,1), 
                            labels = c("Irrelevant", "Relevant") )
