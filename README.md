## Chinese Customer Sentiment

### Proejct Overview

This dataframe is made up of columns like customer reviews, review score, survery time, store code and trade zone.
Generally, it is like fast food app gathering feedback from customer to see what store can do more to keep customers. As
a result, NLP techniques are applied on this dataframe. This project gives me more practical experience on handling the
words (nlp) for a data analyst.

Speaking of text data, there is always a major challenge in working with them because they are always unlabeled and
unstructured. It is hard to utilise traditional supervised learning to handle. Thus topic modeling has become a solution
to this problem, which allows for the analysis of massive text collection. Latent Direichlet Allocation (LDA) is one of
most popular topic modeling algorithm, which is why it is picked up to solve this problem. Furthermore, topic modelling
could be used to build recommendation systems based on mutual content or content similarities.

<p align="center">
  <img src="flow diagram.png" />
</p>

<h4 align = 'center'>Figure 1: Project Flow Diagram</h4>

### Resources Used

* Pycharm
* Python 3
* draw.io
* Packages needed: See requirements.txt

### Project Structure

* data : where data is placed
* exploration: description.py is used to explore the dataframe
* models: different models are written e.g. word2vec, lda, lstm
* src: data-preprocess before feed into the model

P.S. If you have any problem, please feel free to reach out to me.


