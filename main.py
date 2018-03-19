#!/usr/bin/python
# -*- encoding: utf8 -*-

import optparse
import argparse
import sys
import tensorflow as tf
import tempfile
import numpy as np

from ds import loader
from ds import mp
from model.mp2vec_s_modified import MP2Vec

__author__ = 'djc5627'


def main(graph_fname, node_vec_fname, path_vec_fname, options):
    #DONE: Get init parameters as arguements to program
    #DONE: num nodes, get length of graph
    #DONE:  num rel get from len node_vocab/path_vocab, didnt work, just use node# for now

    #TODO(NEWEST) get working with batches and fix shape of objective function to work with them, pick a good cross entropy function, sigmoid?
    #TODO(NEWEST) get p_ into same format as mnist labels [0,1],[1,0] change p_ shape to [None, 2]

    print('Load a HIN...')
    g = loader.load_a_HIN(graph_fname)

    NUM_NODES = g.node_count()
    NUM_REL = g.node_count()

    print('Generate random walks...')
    _, tmp_walk_fname = tempfile.mkstemp()
    print('DEBUG: Saving random walks to file...')
    randomWalkFile = open('randWalks.txt', 'w')
    print(tmp_walk_fname)
    with open(tmp_walk_fname, 'w') as f:
        for walk in g.random_walks(options.walk_num, options.walk_length):
            f.write('%s\n' % ' '.join(map(str, walk)))
            randomWalkFile.write('%s\n' % ' '.join(map(str, walk)))


    _, tmp_node_vec_fname = tempfile.mkstemp()
    _, tmp_path_vec_fname = tempfile.mkstemp()

    # node_vocab = mp.NodeVocab.load_from_file(tmp_walk_fname)
    # path_vocab = mp.PathVocab.load_from_file(tmp_walk_fname,options.window)

    model = MP2Vec(size=options.dim,
                   window=options.window,
                   neg=options.neg,
                   num_processes=options.num_processes,
                   #                  iterations=i,
                   alpha=options.alpha,
                   same_w=True,
                   normed=False,
                   is_no_circle_path=False,
                   )

    neighbors = None
    if options.correct_neg:
        for id_ in g.graph:
            g._get_k_hop_neighborhood(id_, options.window)
        neighbors = g.k_hop_neighbors[options.window]

    model.train(g,
                tmp_walk_fname,
                g.class_nodes,
                k_hop_neighbors=neighbors,
                )

    #TODO: store neg/pos data without using file
    #DONE: format neg data for tensorflow


    # Load training samples from file into list
    x_data = []
    y_data = []
    r_data = []
    p_data = []

    #TODO: Fix the append causes errors randomly, there is problem with generation of test data to file
    with open("pos_data.txt", "r") as pos:
        with open("neg_data.txt", "r") as neg:
            for l in pos:
                temp_pos = l.strip().split(",")
                # Read pos training data
                if (len(temp_pos) == 3):
                    if temp_pos[0] is not None and temp_pos[1] is not None and temp_pos[2] is not None:
                        x_data.append(int(temp_pos[0]))
                        y_data.append(int(temp_pos[1]))
                        r_data.append(int(temp_pos[2]))
                        p_data.append(int(1))

                        # Read neg training data
                        temp_neg = neg.readline().strip().split(",")
                        for j in temp_neg:
                            if (j is not None and j != ""):
                                x_data.append(int(temp_pos[0]))
                                y_data.append(int(j))
                                r_data.append(int(temp_pos[2]))
                                p_data.append(int(0))

    # Debugging
    test = open("read_data.txt", "w+")
    for i in range(len(x_data)):
        test.write(str(x_data[i]) + "," + str(y_data[i]) + "," + str(r_data[i]) + "," + str(p_data[i]) + "\n")



    #for i in range(0, len(x_data)):
    #    print (x_data[i], y_data[i], r_data[i])


    # Convert the data to numpy array
    x_np = np.array(x_data)
    y_np = np.array(y_data)
    r_np = np.array(r_data)
    p_np = np.array(p_data)

    # Convert the x,y,r into 1-hot numpy array
    x_onehot = np.zeros((len(x_data), NUM_NODES))
    x_onehot[np.arange(len(x_data)), x_np] = 1

    y_onehot = np.zeros((len(y_data), NUM_NODES))
    y_onehot[np.arange(len(y_data)), y_np] = 1

    r_onehot = np.zeros((len(r_data), NUM_NODES))
    r_onehot[np.arange(len(r_data)), r_np] = 1



    #DONE: Prepare positive training data to get form (x,y,r,L(x,y,r))
    #DONE: Prepare negative samples (x'',y'',r'') for  each positive entry
    #DONE: Generate bathes from training set
    #DONE: Get g_ val from training data

    # Input
    #x = tf.placeholder(tf.float32, [None, NUM_NODES])
    #y = tf.placeholder(tf.float32, [None, NUM_NODES])
    #r = tf.placeholder(tf.float32, [None, NUM_REL])
    p_ = tf.placeholder(tf.float32, [None])

    x_id = tf.placeholder(tf.int32, [None])
    y_id = tf.placeholder(tf.int32, [None])
    r_id = tf.placeholder(tf.int32, [None])



    #DONE: instantiate learned weights with rand uniform dist (mp2vec_s.py)
    #TODO: Case where Wx = Wy
    # Learned weights


    Wx = tf.Variable(np.random.uniform(low=-0.5/options.dim,
                                high=0.5/options.dim,
                                size=(NUM_NODES, options.dim)).astype(np.float32))
    Wy = tf.Variable(np.random.uniform(low=-0.5 / options.dim,
                                       high=0.5 / options.dim,
                                       size=(NUM_NODES, options.dim)).astype(np.float32))
    Wr = tf.Variable(np.random.uniform(low=0.0/options.dim,
                                high=1.0/options.dim,
                                       size=(NUM_NODES, options.dim)).astype(np.float32))

    # Aggregate Vectors
    #Wx_x = tf.matmul(x, Wx)
    #Wy_y = tf.matmul(y, Wy)

    Wx_x_id = tf.nn.embedding_lookup(Wx, x_id)
    Wy_y_id = tf.nn.embedding_lookup(Wx, y_id)
    Wr_r_id = tf.round(tf.nn.sigmoid(tf.nn.embedding_lookup(Wr, r_id)))

    #TODO: Make binary step default and option to use sigmoid
    # Regularization of Wr (Binary step by rounding sigmoid)
    #Wr_r = tf.round(tf.nn.sigmoid(tf.matmul(r, Wr)))

    # Regularization of Wr (just sigmoid)
    #Wr_r = tf.nn.sigmoid(tf.multiply(tWr, r))

    #DONE: Find a way to do element-wise mult with diff size tensors (Wr_r is diff size), make one-hot
    # Hidden Layer (element-wise mult)
    #h = tf.multiply(tf.multiply(Wx_x, Wy_y), Wr_r)
    h = tf.multiply(tf.multiply(Wx_x_id, Wy_y_id), Wr_r_id)


    # Output (sigmoid of element summation), reduce the sum along dim and keep [None] dim
    #p = tf.nn.sigmoid(tf.reduce_sum(h, 1))
    p = tf.reduce_sum(h, 1, keep_dims=False)


    #TODO: Implement obj version for binary step
    # Objective Function
    #objective = tf.cond(tf.equal(p_,  1), lambda: p, lambda: 1-p)
    objective = tf.nn.sigmoid_cross_entropy_with_logits(labels=p_, logits=p)
    #objective = tf.nn.softmax_cross_entropy_with_logits(labels=p_, logits=p)
    #objective = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=p_, logits=p)



    #DONE: Take negative of obj or not?
    # Train step (want to minimize negative of objective)
    train_step = tf.train.GradientDescentOptimizer(options.alpha).minimize(objective)


    #DONE: calculate accuracy correctly
    # Accuracy
    correct_prediction = tf.equal(tf.round(tf.sigmoid(p)), p_)


    u = tf.cast(correct_prediction,tf.float32)
    accuracy = tf.reduce_mean(u)

    # Training
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()



    #TODO: integreate -i iterations option
    #TODO: train all data one at a time? or train random data points at a time?

    step = 50
    for j in xrange(0,1):
        print ("epoch " + str(j) + " -----------------")
        for i in xrange(0, len(x_data), step):
            #x_val = x_onehot[i:i+step]
            #y_val = y_onehot[i:i+step]
            #r_val = r_onehot[i:i+step]
            p__val = p_np[i:i+step]

            x_id_val = x_data[i:i+step]
            y_id_val = y_data[i:i+step]
            r_id_val = r_data[i:i+step]

            #print(str(x_id_val) + str(y_id_val) + str(r_id_val) + str(p__val))
            #raw_input()

            '''
            Wx_row = sess.run(Wx_x_id, feed_dict={p_: p__val,
                                                  x_id: x_id_val, y_id: y_id_val, r_id: r_id_val})

            Wy_row = sess.run(Wy_y_id, feed_dict={p_: p__val,
                                                  x_id: x_id_val, y_id: y_id_val, r_id: r_id_val})

            Wr_row = sess.run(Wr_r_id, feed_dict={p_: p__val,
                                                  x_id: x_id_val, y_id: y_id_val, r_id: r_id_val})

            p_res = sess.run(p, feed_dict={p_: p__val,
                                                 x_id: x_id_val, y_id: y_id_val, r_id: r_id_val})

            print(Wx_row)
            print(Wy_row)
            print(Wr_row)
            print(p_res)
            raw_input()
            '''

            sess.run(train_step, feed_dict={p_: p__val,
                                            x_id: x_id_val, y_id: y_id_val, r_id: r_id_val})

            train_accuracy = sess.run(accuracy, feed_dict={p_: p__val,
                                            x_id: x_id_val, y_id: y_id_val, r_id: r_id_val})

            print('step %d, training accuracy %f' % (i, train_accuracy))


            Wxx = sess.run(Wx, feed_dict={p_: p__val,
                                        x_id: x_id_val, y_id: y_id_val, r_id: r_id_val})


            np.savetxt('node_vec_temp.txt', Wxx, delimiter=' ')




            '''
            Wx_row = sess.run(Wx_x_id, feed_dict={p_: p__val,
                                                  x_id: x_id_val, y_id: y_id_val, r_id: r_id_val})

            Wy_row = sess.run(Wy_y_id, feed_dict={p_: p__val,
                                                  x_id: x_id_val, y_id: y_id_val, r_id: r_id_val})

            Wr_row = sess.run(Wr_r_id, feed_dict={p_: p__val,
                                                  x_id: x_id_val, y_id: y_id_val, r_id: r_id_val})

            p_res = sess.run(p, feed_dict={p_: p__val,
                                                 x_id: x_id_val, y_id: y_id_val, r_id: r_id_val})

            print(Wx_row)
            print(Wy_row)
            print(Wr_row)
            print(p_res)
            raw_input()
            '''

    #Output vectors to file
    lines = []
    lines.append(str(NUM_NODES) + " " + str(options.dim) + "\n")

    tmpLines = []
    with open("node_vec_temp.txt", "r") as f:
        for l in f:
            tmpLines.append(l)

    id2name = dict([(id_, name) for name, id_ in g.node2id.items()])
    for i in xrange(0, NUM_NODES):
        lines.append(str(id2name[i]) + " " + tmpLines[i])

    with open("node_vec.txt", "w+") as out:
        for l in lines:
            out.write(l)




    #TODO: Output node and path vectors to file


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=main.__doc__)
    parser.add_option('-l', '--walk-length', action='store',
                      dest='walk_length', default=100, type='int',
                      help=('The length of each random walk '
                            '(default: 100)'))
    parser.add_option('-k', '--walk-num', action='store',
                      dest='walk_num', default=10, type='int',
                      help=('The number of random walks starting from '
                            'each node (default: 10)'))
    parser.add_option('-n', '--negative', action='store', dest='neg',
                      default=5, type='int',
                      help=('Number of negative examples (>0) for '
                            'negative sampling, 0 for hierarchical '
                            'softmax (default: 5)'))
    parser.add_option('-d', '--dim', action='store', dest='dim',
                      default=100, type='int',
                      help=('Dimensionality of word embeddings '
                            '(default: 100)'))
    parser.add_option('-a', '--alpha', action='store', dest='alpha',
                      default=0.025, type='float',
                      help='Starting learning rate (default: 0.025)')
    parser.add_option('-w', '--window', action='store', dest='window',
                      default=3, type='int',
                      help='Max window length (default: 3)')
    parser.add_option('-p', '--num_processes', action='store',
                      dest='num_processes', default=1, type='int',
                      help='Number of processes (default: 1)')
    parser.add_option('-i', '--iter', action='store', dest='iter',
                      default=1, type='int',
                      help='Training iterations (default: 1)')
    parser.add_option('-s', '--same-matrix', action='store_true',
                      dest='same_w', default=False,
                      help=('Same matrix for nodes and context nodes '
                            '(Default: False)'))
    parser.add_option('-c', '--allow-circle', action='store_true',
                      dest='allow_circle', default=False,
                      help=('Set to all circles in relationships between '
                            'nodes (Default: not allow)'))
    parser.add_option('-r', '--sigmoid_regularization',
                      action='store_true', dest='sigmoid_reg',
                      default=False,
                      help=('Use sigmoid function for regularization '
                            'for meta-path vectors '
                            '(Default: binary-step function)'))
    parser.add_option('-t', '--correct-negs',
                      action='store_true', dest='correct_neg',
                      default=False,
                      help=('Select correct negative data '
                            '(Default: false)'))
    options, args = parser.parse_args()

    if len(args) != 3:
        parser.print_help()
        sys.exit()

    sys.exit(main(args[0], args[1], args[2], options))




