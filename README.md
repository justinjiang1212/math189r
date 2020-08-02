# Github Repo for Math 189r - Mathematics of Big Data with Professor Weiqing Gu.

This repo is for the homework submissions for Math 189r, which I took in the summer of 2020 with a final course grade of an A.

This repo also contains the final project for the class, which Andy Liu and I worked on. The final project used topic modeling and natural language processing to predict the approval rating of President Trump during a 3 day period.

We built a custom web scraper to continuously download news articles from the [Google News topic about Trump](https://news.google.com/topics/CAAqIggKIhxDQkFTRHdvSkwyMHZNR054ZERrd0VnSmxiaWdBUAE?hl=en-US&gl=US&ceid=US%3Aen). Our scraper has minute resolution, such that every 60 seconds, it scans the provided HTML for updates. In three days of running the scraper, we were able to get ~2,600 articles, which we then classified using topic modeling and assigned a sentiment score using natural language processing.

We then compared these live sentiment scores to a predicted sentiment score derived from a Long Short Term Memory(LSTM) network, with Fivethirtyeight approval data as an input. Our writeup can be found [here](https://github.com/justinjiang1212/math189r/blob/master/final_project/writeup/Math_189R_Final_Writeup.pdf). 
