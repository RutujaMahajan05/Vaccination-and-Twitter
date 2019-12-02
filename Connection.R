library(twitteR)
library(httr)
library(RCurl)
library(ROAuth)

#Establish connection 

ckey =""                  # consumer or API key
csecret=""                # consumer or API secret

atoken=""                 # access token
asecret=""                # access token secret

twitteR:::setup_twitter_oauth(ckey, csecret, atoken, asecret)

keywords = "vaccine, vaccination, immunize, immunization, immunise, immunisation, antivaccine, antivaccination, 
        antivax, antivaxx, antivaxxer, antivaxer, vaxxed, saynotovaccine, autism+vaccine, 
        autism+vaccination, autism+immunize, autism+immunization, autism+immunise, autism+immunisation, exvaxxer, 
        stopmandatoryvaccine, vaccinedangers, vaccineskill, vaccineinjury, educatebeforeyouvaccinate, 
        vaccinesharmactually, vaccineinjuryawareness, antivak, antivaksin, antivac, vaccineingredients, 
        vaccinescauseautism, vaccinescausedisease, vaccinecorruption, vaccinessaveslives, justvaccinate, 
        vaccinesmaimandkill, vaccinescampaign, goodluckwithyourvaccines, vaccinesrevealed, vaccinedamage, 
        vaccinetruth, antiflushot, flushot, pro-vaccine, fluseason, vaccineevolution, HPV, measles, MMR, TDAP, 
        anthrax vaccine, Hepatitis A, Hepatitis B, tetanus, rabies, smallpox+vaccine, chickenpox+vaccine, 
        Meningitis, pro-safe, thimerosal, cdc+vaccine, who+vaccine, Donald Trump+vaccine, 
        Australian Vaccination skeptics Network, Boko Haram+vaccine, Christian Science, GreenMedInfo, 
        Infowars+vaccine, NaturalNews+vaccine, flu+vaccine, flu+vaccination, influenza+vaccine, 
        influenza+vaccination, anti+flu+shot"
keywords <- unlist(strsplit(keywords,","))

tweets = twitteR:::searchTwitter("vaccine", n = 100, lang = "en", since = "2019-10-26", until = "2019-10-27", tweet_)
options(max.print = 10000000)

tweetdate = lapply(tweets, function(x) x$getCreated())

tweettext = sapply(tweets, function(x) x$getText())
#print(tweettext)
username = sapply(tweets, function(x) x$getScreenName())

tweettime = sapply(tweetdate, function(x) strftime(x, format = "%Y-%m-%d %H:%M:%S"))

#location = sapply(tweets, function(x) x$getLocation)

#followers = sapply(tweets1, function(x) x$getFollowersCount())

#friends = sapply(tweets, function(x) x$getFriendsCount())

isretweet = sapply(tweets, function(x) x$getIsRetweet())

retweetcount = sapply(tweets, function(x) x$getRetweetCount())

# Cleaning Stage 1 
# Convert all characters to ASCII 
tweettext = lapply(tweettext, function(x) iconv(x, "latin1", "ASCII", sub = ""))
tweettext = lapply(tweettext, function(x) gsub("\n", '', x))
tweettext = unlist(tweettext)

#Create Data Frame
data = as.data.frame(cbind(ttext=tweettext,
                           date=tweetdate,
                           username=username,
                           time=tweettime,
                           isretweet=isretweet,
                           retweetcount=retweetcount))
library(xlsx)
dirtext<-c("")                                       # Path for file to be created
file <- paste(dirtext,"/Tweets.xlsx", sep="")        # Name of file
res<- write.xlsx(data, file, row.names=TRUE)        

