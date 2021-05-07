"""analysis.py: Collection of classes for performing analysis on Corpus"""
# David Vann (dv6bq@virginia.edu)
# DS 5001
# 6 May 2021

import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

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

    def fit(self, corpus):
        # Copy corpus over to prevent undesired modifications
        self.corpus = corpus.copy()

        self.bow = self.corpus.bow
        self.vocab = self.corpus.vocab


    def plot_2d(self):
        pass