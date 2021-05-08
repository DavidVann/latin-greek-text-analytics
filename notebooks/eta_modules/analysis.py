"""analysis.py: Collection of classes for performing analysis on Corpus"""
# David Vann (dv6bq@virginia.edu)
# DS 5001
# 6 May 2021

from os import remove
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.cluster.hierarchy as sch
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer

from eta_modules.preprocessing import Corpus

class HierarchicalClusterAnalysis:
    def __init__(self, max_features=5000, tfidf_method='max', OHCO_level=['work_id', 'chapter_id']):
        self.max_features = max_features
        self.tfidf_method = tfidf_method
        self.OHCO_level = OHCO_level

        self.vocab = None
        self.metrics = None

    def fit(self, corpus: Corpus, metrics=['cosine']):
        # Copy corpus over to prevent undesired modifications
        self.corpus = corpus.copy()
        self.metrics = metrics

        self.bow = self.corpus.bow
        self.vocab = self.corpus.vocab

        # If original TFIDF bag or method doesn't match, recalculate
        # Otherwise, already have good TFIDF values to use
        if (corpus.tfidf_OHCO != self.OHCO_level) or (f"tfidf_{self.tfidf_method}_sum" not in self.vocab):
            self.corpus.compute_tfidf(OHCO_level=self.OHCO_level, methods=[self.tfidf_method])
            # Reassign objects just to be safe
            self.bow = self.corpus.bow
            self.vocab = self.corpus.vocab

        self.vocab['dfidf'] = self.vocab['df'] * self.vocab['idf']

        # Filter VOCAB to `max_features` words using DF-IDF; use that to filter BOW TFIDF values
        self.vocab = self.vocab.sort_values('dfidf', ascending=False).head(self.max_features)
        self.tfidf = self.bow[f"tfidf_{self.tfidf_method}"].unstack(fill_value=0)
        self.tfidf = self.tfidf[self.vocab.index]

        # Collapse tfidf to book level means
        self.tfidf = self.tfidf.groupby(['work_id']).mean()

        ## Create DataFrame to hold pairwise distances
        # Multindex -- combinations of indices; e.g., (0, 1), (0, 2), etc.
        work_ids = self.corpus.lib.index.tolist()
        self.pdists = pd.DataFrame(index=pd.MultiIndex.from_product([work_ids, work_ids])).reset_index()
        # Remove self-combinations in index; e.g., (0, 0), (1, 1), etc.
        self.pdists = self.pdists[self.pdists['level_0'] < self.pdists['level_1']].set_index(['level_0', 'level_1'])
        self.pdists.index.names = ['doc_a', 'doc_b']

        for metric in self.metrics:
            if metric in ['jaccard', 'dice']:
                L0 = self.tfidf.astype('bool').astype('int') # Binary
                self.pdists[metric] = pdist(L0, metric)
            elif metric in ['jensenshannon']:
                L1 = self.tfidf.apply(lambda x: x / x.sum(), 1)
                self.pdists[metric] = pdist(L1, metric)
            else:
                self.pdists[metric] = pdist(self.tfidf, metric)

    def plot_dendrogram(self, linkage='complete', color_thresh=0.3, figsize=(8, 10)):
        for metric in self.metrics:
            tree = sch.linkage(self.pdists[metric], method=linkage)
            labels = (self.corpus.lib['author'] + ': ' + self.corpus.lib['title']).values
            plt.figure(figsize=figsize)
            # fig, axes = plt.subplots(figsize=figsize)
            # dendrogram = sch.dendrogram(tree,
            #                             labels=labels,
            #                             orientation="left",
            #                             count_sort=True,
            #                             distance_sort=True,
            #                             above_threshold_color='0.75',
            #                             color_threshold=color_thresh)
            sch.dendrogram(tree,
                                        labels=labels,
                                        orientation="left",
                                        count_sort=True,
                                        distance_sort=True,
                                        above_threshold_color='0.75',
                                        color_threshold=color_thresh)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.title(f"Metric: {metric}")

class PCA:
    def __init__(self, max_features=5000, tfidf_method='max', OHCO_level=['work_id', 'chapter_id']):
        self.max_features = max_features
        self.tfidf_method = tfidf_method
        self.OHCO_level = OHCO_level

        self.vocab = None

    def fit(self, corpus, n_components=10):
        # Copy corpus over to prevent undesired modifications
        self.corpus = corpus.copy()

        # Modify token to add in author label (affects output from computing TFIDF)
        self.corpus.token = self.corpus.token.join(self.corpus.lib).reset_index().set_index(['author'] + self.corpus.OHCO)

        # Since we want to include author groupings, must recalculate TFIDF regardless of original values
        self.corpus.compute_tfidf(OHCO_level=(['author'] + self.OHCO_level), methods=[self.tfidf_method])
        self.bow = self.corpus.bow
        self.vocab = self.corpus.vocab

        self.vocab['dfidf'] = self.vocab['df'] * self.vocab['idf']
        
        # Filter VOCAB to `max_features` words using DF-IDF; use that to filter BOW TFIDF values
        self.vocab = self.vocab.sort_values('dfidf', ascending=False).head(self.max_features)

        # Form TFIDF matrix by taking mean based on author + work + chapter group means of terms
        self.tfidf = self.bow.groupby(['author'] + self.corpus.OHCO[:2] + ['term_str'])[[f"tfidf_{self.tfidf_method}"]].mean().unstack(fill_value=0)
        
        # Column index is currently a multi-index, with the top level being one element (the tfidf method used, e.g., "tfidf_max")
        # Drop this level in the column index so we can index columns more easily and remove undesired words based on `max_features`
        self.tfidf.columns = self.tfidf.columns.droplevel(0)
        self.tfidf = self.tfidf[self.vocab.index] # filter words based on DF-IDF

        # Apply L2 normalization to TFIDF rows (e.g., normalize values for words across a chapter)
        self.tfidf = self.tfidf.apply(lambda x: x / np.sqrt(np.square(x).sum()), axis=1)
        self.tfidf = self.tfidf - self.tfidf.mean() # center word vectors

        ## PCA calculations
        cov = self.tfidf.cov() # covariance matrix
        eig_val, eig_vec = eigh(cov) # eigendecomposition of covariance matrix (dim: max_features x max_features)

        self.eig_vec = pd.DataFrame(eig_vec, index=cov.index, columns=cov.index) # (dim: max_features x max_features) 
        self.eig_val = pd.DataFrame(eig_val, index=cov.index, columns=['eig_val']) # (dim: max_features x 1)
        
        self.eig_pairs = self.eig_val.join(self.eig_vec.T)
        self.eig_pairs['exp_var'] = np.round((self.eig_pairs['eig_val'] / self.eig_pairs['eig_val'].sum()) * 100, 2)

        # Get top n components by explained variance
        self.pc = self.eig_pairs.sort_values('exp_var', ascending=False).head(n_components).reset_index(drop=True)
        self.pc.index.name = 'comp_id'
        self.pc.index = [f"PC{i}" for i in self.pc.index.tolist()]
        self.pc.index.name = 'pc_id'

        # Project TFIDF using components (document-component matrix) and get loadings
        self.dcm = self.tfidf.dot(self.pc[cov.index].T)
        self.loadings = self.pc[cov.index].T
        self.loadings.index.name = 'term_str'

    def plot_2d(self, comp_id_1=0, comp_id_2=1):
        dcm_plot = self.dcm.reset_index().copy()
        dcm_plot = dcm_plot.merge((self.corpus.lib['author'] + '-' + self.corpus.lib['title']).to_frame('doc').reset_index())
        dcm_plot['doc'] = dcm_plot['doc'] + '-' + dcm_plot['chapter_id'].astype('str')

        fig = px.scatter(dcm_plot, f"PC{comp_id_1}", f"PC{comp_id_2}", color='author', hover_name='doc',
                         marginal_x='box', height=800)
        fig.show()


class TopicModel:
    def __init__(self, remove_proper_nouns=True, OHCO_level=['work_id', 'chapter_id'], max_features=5000, n_topics=40, n_topic_terms=10, 
                    ngram_range=[1, 2], max_iter=20, random_state=None):
        self.remove_proper_nouns = remove_proper_nouns
        self.bag = OHCO_level
        self.max_features=max_features
        self.n_topics = n_topics
        self.n_topic_terms = n_topic_terms
        self.ngram_range = ngram_range
        self.max_iter = max_iter
        self.random_state=random_state

    def fit(self, corpus):
        # Copy corpus over to prevent undesired modifications
        self.corpus = corpus.copy()
        
        # Create a list of more complete document strings to work with scikit-learn's modules
        self.corpus.token.term_str = self.corpus.token.term_str.astype('str')

        if self.remove_proper_nouns:
            regex_expr = r'^NNS?$'
        else:
            regex_expr = r'^NNP?S?$'
        self.doc = (self.corpus.token[self.corpus.token.pos.str.match(regex_expr)]
                    .groupby(self.bag).term_str
                    .apply(lambda x: ' '.join(x))
                    .to_frame('doc_str'))

        vectorizer = CountVectorizer(max_features=self.max_features, ngram_range=self.ngram_range, stop_words='english')
        self.counts = vectorizer.fit_transform(self.doc.doc_str)
        self.term = vectorizer.get_feature_names()

        lda = LDA(n_components=self.n_topics, max_iter=self.max_iter, learning_offset=50., random_state=self.random_state)
        
        # Theta table -- documents vs. topics
        self.theta = pd.DataFrame(lda.fit_transform(self.counts), index=self.doc.index)
        self.theta.columns.name = 'topic_id'
        
        # Phi table -- terms vs. topics
        self.phi = pd.DataFrame(lda.components_, columns=self.term)
        self.phi.index.name = 'topic_id'
        self.phi.columns.name = 'term_str'

        # Topic table
        self.topic = (self.phi.stack().to_frame('topic_weight')
                        .groupby('topic_id')
                        .apply(lambda x: x.sort_values('topic_weight', ascending=False)
                            .head(self.n_topic_terms)
                            .reset_index()
                            .drop('topic_id', 1)['term_str']
                            )
                        )
        self.topic['label'] = self.topic.apply(lambda x: str(x.name) + ' ' + ', '.join(x[:self.n_topic_terms]), 1)
        self.topic['doc_weight_sum'] = self.theta.sum()

    def plot_topic_weights(self):
        self.topic.sort_values('doc_weight_sum', ascending=True).plot.barh(y='doc_weight_sum', x='label', figsize=(5, self.n_topics/2))
