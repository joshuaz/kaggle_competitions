# Natural Language Processing with Disaster Tweets
Predict which Tweets are about real disasters and which ones are not

## Competition Description
Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster. 

## Dataset Description
You'll need `train.csv`, `test.csv` and `sample_submission.csv`.

### Files
`train.csv` - the training set<br>
`test.csv` - the test set

### Columns
`id` - a unique identifier for each tweet<br>
`text` - the text of the tweet<br>
`location` - the location the tweet was sent from (may be blank)<br>
`keyword` - a particular keyword from the tweet (may be blank)<br>
`target` - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)<br>

### What am I predicting?
You are predicting whether a given tweet is about a real disaster or not. If so, predict a `1`. If not, predict a `0`.
