datafile <- read.csv("Terms in topic 1 and 2 Pyfile.csv")

colnames(datafile_new)

datafile_new <- datafile[1:100,]
#library(wordcloud)

wordcloud::wordcloud(words = datafile_new$T1_Word, freq = datafile_new$ï..T1_Importance, maxwords = 100,
                     scale = c(3.9,0.4), 
                     colors = brewer.pal(8, "Dark2"), random.order = F, rot.per = 0.2 )

wordcloud::wordcloud(words = datafile_new$T2_Word, freq = datafile_new$T2_Importance, maxwords = 100,
                     scale = c(4,0.35), 
                     colors = brewer.pal(8, "Dark2"), random.order = F, rot.per = 0.2 )


