#!/usr/bin/env python3
# coding: utf-8
# File: doc2Vec.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-24
import collections
import glob
from itertools import chain
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Doc2Vec:
    def __init__(self):
        self.data_index = 0

    #生成训练数据
    def build_dataset(self, file_name, min_count):
        sentences = open(file_name).read().split('\n')
        words = ''.join(sentences).split()
        count = [['UNK', -1]]
        count.extend([item for item in collections.Counter(words).most_common() if item[1] >= min_count])
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
            
        unk_count = 0
        sent_data = []
        for sentence in sentences:
            data = []
            for word in sentence.split():
                if word in dictionary:
                    index = dictionary[word]
                else:
                    index = 0  # dictionary['UNK']
                    unk_count = unk_count + 1
                data.append(index)
            sent_data.append(data)

        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return sent_data, count, dictionary, reverse_dictionary

    #创建文档doc训练数据
    def build_instance(self, vocabulary_size, data):
        window_size = 3
        instances = 0
        for i in range(len(data)):
            data[i] = [vocabulary_size]*window_size+data[i]+[vocabulary_size]*window_size

        for sentence  in data:
            instances += len(sentence)-2*window_size
        context = np.zeros((instances, window_size * 2 + 1), dtype=np.int32)
        labels = np.zeros((instances, 1), dtype=np.int32)
        doc = np.zeros((instances, 1), dtype=np.int32)

        k = 0
        for doc_id, sentence in enumerate(data):
            for i in range(window_size, len(sentence) - window_size):
                context[k] = sentence[i - window_size:i + window_size + 1]  # Get surrounding words
                labels[k] = sentence[i]  # Get target variable
                doc[k] = doc_id
                k += 1

        context = np.delete(context, window_size, 1)  # delete the middle word

        shuffle_idx = np.random.permutation(k)
        labels = labels[shuffle_idx]
        doc = doc[shuffle_idx]
        context = context[shuffle_idx]

        return labels, doc, context, instances

    #生成训练样本
    def generate_batch(self, batch_size, instances, labels, context, doc):

        if self.data_index + batch_size<instances:
            batch_labels = labels[self.data_index : self.data_index + batch_size]
            batch_doc_data = doc[self.data_index : self.data_index + batch_size]
            batch_word_data = context[self.data_index : self.data_index + batch_size]
            self.data_index += batch_size
        else:
            overlay = batch_size - (instances - self.data_index)
            batch_labels = np.vstack([labels[self.data_index : instances],labels[:overlay]])
            batch_doc_data = np.vstack([doc[self.data_index : instances],doc[:overlay]])
            batch_word_data = np.vstack([context[self.data_index : instances],context[:overlay]])
            self.data_index = overlay

        batch_word_data = np.reshape(batch_word_data,(-1,1))

        return batch_labels, batch_word_data, batch_doc_data

    #训练模型
    def train_model(self, vocabulary_size, window_size, data, instances, labels, context, doc):
        batch_size = 256
        context_window = 2 * window_size
        embedding_size = 50  # Dimension of the embedding vector.
        softmax_width = embedding_size  # +embedding_size2+embedding_size3
        num_sampled = 5  # Number of negative examples to sample.
        sum_ids = np.repeat(np.arange(batch_size), context_window)
        len_docs = len(data)


        #定义训练网络结构
        graph = tf.Graph()

        with graph.as_default():
            train_word_dataset = tf.placeholder(tf.int32, shape=[batch_size * context_window])
            train_doc_dataset = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            segment_ids = tf.constant(sum_ids, dtype=tf.int32)

            word_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            word_embeddings = tf.concat([word_embeddings, tf.zeros((1, embedding_size))], 0)
            doc_embeddings = tf.Variable(tf.random_uniform([len_docs, embedding_size], -1.0, 1.0))

            softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, softmax_width],
                                                              stddev=1.0 / np.sqrt(embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))


            embed_words = tf.segment_mean(tf.nn.embedding_lookup(word_embeddings, train_word_dataset), segment_ids)
            embed_docs = tf.nn.embedding_lookup(doc_embeddings, train_doc_dataset)
            embed = (embed_words + embed_docs) / 2.0

            loss = tf.reduce_mean(tf.nn.nce_loss(softmax_weights, softmax_biases, train_labels,
                                                 embed, num_sampled, vocabulary_size))

            optimizer = tf.train.AdagradOptimizer(0.5).minimize(loss)

            norm = tf.sqrt(tf.reduce_sum(tf.square(doc_embeddings), 1, keep_dims=True))
            normalized_doc_embeddings = doc_embeddings / norm

        num_steps = 10000
        step_delta = int(num_steps/20)

        #训练网络
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            average_loss = 0
            for step in range(num_steps):
                batch_labels, batch_word_data, batch_doc_data = self.generate_batch(batch_size, instances, labels, context, doc)
                feed_dict = {train_word_dataset : np.squeeze(batch_word_data),
                             train_doc_dataset : np.squeeze(batch_doc_data),
                             train_labels : batch_labels}
                _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += l
                if step % step_delta == 0:
                    if step > 0:
                        average_loss = average_loss / step_delta
                    print('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0

            final_word_embeddings = word_embeddings.eval()
            final_word_embeddings_out = softmax_weights.eval()
            final_doc_embeddings = normalized_doc_embeddings.eval()

        return final_doc_embeddings, final_word_embeddings, final_word_embeddings_out

    #训练主控函数
    def train_main(self):
        filename = './data/data.txt'
        model_path = './model/doc2vec.bin'
        cluster_imgpath = './img/doc_cluster.png'
        window_size = 3  # 窗口大小
        min_count = 5 #最低词频
        data, count, dictionary, reverse_dictionary = self.build_dataset(filename, min_count)
        vocabulary_size = len(count)
        labels, doc, context, instances = self.build_instance(vocabulary_size, data)
        print('start training models.......')
        final_doc_embeddings, final_word_embeddings, final_word_embeddings_out = self.train_model(vocabulary_size, window_size, data, instances, labels, context, doc)
        print('save doc models.......')
        self.save_docembedding(final_doc_embeddings, model_path)
        print('save clustering images.......')
        self.cluster_show(final_doc_embeddings, cluster_imgpath)

    # 定义可视化Word2Vec效果的函数
    def plot_with_labels(self, low_dim_embs, labels, imagename):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize=(18, 18))
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.savefig(imagename)

    # 聚类展示
    def cluster_show(self, final_embeddings, imagename):
        # 使用tsne进行降维
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [i for i in range(plot_only)]
        self.plot_with_labels(low_dim_embs, labels, imagename)


    # 保存doc-enbedding文件
    def save_docembedding(self, final_embeddings, modelpath):
        f = open(modelpath, 'w+')
        for index, item in enumerate(final_embeddings):
            f.write(str(index) + '\t' + ' '.join([str(vec) for vec in item]) + '\n')
        f.close()

doc_trainer = Doc2Vec()
doc_trainer.train_main()
