
library(devtools)
install_github("cpsievert/LDAvis")
library(LDAvis)
library(tm)
library(servr)

filenames <- list.files(getwd(),pattern="*.txt")
files <- lapply(filenames,readLines)

# Convert tweets to corpus 
docs <- Corpus(VectorSource(files)) 

# Save an object to a file
#saveRDS(docs_new, file = "all_tweet_corpus.rds")
# Restore the object
docs <- readRDS(file = "all_tweet_corpus.rds")

docs <- tm_map(docs, removeWords, c("rt","RT"))

# tokenize
doc.list <- strsplit(docs[[2]]$content, "[[:space:]]+")  # Split the corpus as per tweets

#table of terms
term.table <- table(unlist(doc.list))  # Create frequency table for words
term.table <- sort(term.table, decreasing = TRUE)  # Sort the table in decreasing order

#remove terms that occur fewer than 5 times:
term.table <- term.table[term.table > 5]
vocab <- names(term.table) ; vocab # Convert the terms to character 

# Format to lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms) 

# LDA model stats
D <- length(documents)   # length of corpus content
W <- length(vocab)       # words in corpus
doc.length <- sapply(documents, function(x) sum(x[2, ])) 
N <- sum(doc.length)
term.frequency <- as.integer(term.table).

# Model tuning parameters:
K <- 2
#G <- 5000
G <- 100
alpha <- 0.02
eta <- 0.02

library(doSNOW)
library(e1071)
library(lda)

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

# Fit the model:
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                     num.iterations = G, alpha = alpha, 
                                     eta = eta, initial = NULL, burnin = 0,
                                     compute.log.likelihood = TRUE)
stopCluster(cl)
t2 <- Sys.time()
t2 - t1  

# fit
#log_likelihood <- fit$log.likelihoods
#write.csv(log_likelihood, file = "log_likelihoodvalues.csv")

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x))) 

tweets.list <- list(phi = phi,
                    theta = theta,
                    doc.length = doc.length,
                    vocab = vocab,
                    term.frequency = term.frequency) 

library(LDAvis)


library("tsne")
svd_tsne <- function(x) tsne(svd(x)$u)

json_tsne <- createJSON(phi = tweets.list$phi, 
                        theta = tweets.list$theta, 
                        doc.length = tweets.list$doc.length, 
                        vocab = tweets.list$vocab,
                        term.frequency = tweets.list$term.frequency,
                        mds.method = svd_tsne, 
                        plot.opts = list(xlab="", ylab="")
)

#serVis(json_tsne, out.dir = 'vis', open.browser = FALSE)
serVis(json_tsne, out.dir = 'vis', open.browser = interactive(), as.gist = FALSE)

servr::daemon_stop(3)
