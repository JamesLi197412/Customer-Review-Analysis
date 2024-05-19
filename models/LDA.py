import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LDA(nn.Module):
    def __init__(self, num_topics, vocab_size):
        super(LDA, self).__init__()
        self.alpha = nn.Parameter(torch.rand(num_topics))
        self.beta = nn.Parameter(torch.rand(num_topics, vocab_size))

    def forward(self, X):
        theta = F.softmax(self.alpha)
        phi = F.softmax(self.beta, dim=1)

        doc_topic_dist = torch.mm(X, phi.t())
        topic_word_dist = torch.mm(doc_topic_dist, phi)

        return theta, phi, doc_topic_dist, topic_word_dist


