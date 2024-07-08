from exploration.description import *
from src.preprocess import *
from models.LDA import *
import torch.optim as optim

import warnings


# Reference:
# 7. https://zhuanlan.zhihu.com/p/86579316
# 11. https://docs.python.org/3/library/multiprocessing.html
# 12. https://medium.com/kuzok/news-documents-clustering-using-python-latent-semantic-analysis-b95c7b68861c
# 13. https://spotintelligence.com/2023/11/07/linear-discriminant-analysis-lda/
# 14. https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/


def warn(*args, **kwargs):
    pass

'''
def topic_modeling(num_topics, vocab, num_epochs,X):
    # Prepare your data here, such as tokenizing your Chinese reviews and converting them to tensors

    # Instantiate and train your LDA model
    model = LDA(num_topics=num_topics, vocab_size=len(vocab))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        theta, phi, doc_topic_dist, topic_word_dist = model(X)

        # Calculate loss here, such as calculating the negative log likelihood of the data given the model
        loss = calculate_loss(X, doc_topic_dist, topic_word_dist)

        loss.backward()
        optimizer.step()

    # Once you have trained your model, you can use it to generate topic distributions for new Chinese reviews

    # For fine-tuning, you can adjust the optimizer parameters, learning rate, or even the model architecture
'''
def file_export(df, text):
    df.to_csv('test.csv')

    with open(r'words.txt','w') as fp:
        fp.write("\n".join(str(word) for word in text))




if __name__ == '__main__':
    # input your data with relative path
    warnings.warn = warn()
    customer_feedback = pd.read_excel('data/CUSTOMER_FEEDBACK.xlsx', sheet_name='Sheet')

    # Column Adjustment
    customer_feedback = data_process(customer_feedback,'SURVEY_TIME','CONTENT_TX')
    # visulaisation(customer_feedback)  # generate visualisatoin output to folder

    customer_feedback = EDA(customer_feedback)

    # word process -- > add cleaned reviews columns
    customer_feedback,words_list = data_preprocess(customer_feedback, 'CONTENT_TX')

    # Export file to view its output
    file_export(customer_feedback, words_list)

    # Call LDA model
    lda_model = LDA(words_list)









