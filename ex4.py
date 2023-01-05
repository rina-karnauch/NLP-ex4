import nltk
import numpy as np

# from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx
from networkx import DiGraph, maximum_spanning_arborescence

nltk.download('dependency_treebank')
from nltk.corpus import dependency_treebank


class Arc:
    def __init__(self, head, tail, weight):
        self.head = head
        self.tail = tail
        self.weight = weight


def import_data():
    all_sentences = dependency_treebank.parsed_sents()

    all_tagged_words = np.array([w_t for w_t in dependency_treebank.tagged_words()] + [("ROOT", "ROOT")])
    all_words = all_tagged_words[:, 0]
    number_of_words = len(all_words)
    all_tags = all_tagged_words[:, 1]
    d = build_tuple_index_map(set(all_words[:100]), set(all_tags[:100]))

    s_len = len(all_sentences)
    train_percent_index = int(0.9 * s_len)
    train_set, test_set = all_sentences[:train_percent_index], all_sentences[train_percent_index:]
    return train_set, test_set, d, number_of_words


def build_tuple_index_map(all_words, all_tags):
    d = dict()
    d_size = 0
    for w_i in all_words:
        for w_j in all_words:
            word_tuple = (w_i, w_j)
            if word_tuple not in d.keys():
                d[word_tuple] = d_size
                d_size += 1
    for t_i in all_tags:
        for t_j in all_tags:
            tag_tuple = (t_i, t_j)
            if tag_tuple not in d.keys():
                d[tag_tuple] = d_size
                d_size += 1
    return d


def feature_function(u, v, d):
    u_word, v_word = u[0], v[0]
    u_tag, v_tag = u[1], v[1]

    word_bigram_index = d.get((u_word, v_word), -1)
    tag_bigram_index = d.get((u_tag, v_tag), -1)

    return word_bigram_index, tag_bigram_index


def subtract_maps(current_tree_map, max_tree_map):
    for k, v in max_tree_map.items():
        if k in current_tree_map.keys():
            max_tree_map[k] -= current_tree_map[k]
        # else:
        #     max_tree_map[k] = v
    for k, v in current_tree_map.items():
        if k not in max_tree_map.keys():
            max_tree_map[k] = -v
    return max_tree_map


def get_gradient(max_tree, arcs, d):
    current_tree_map = dict()
    max_tree_map = dict()

    for a in arcs:
        word_bigram_index, tag_bigram_index = feature_function(a.head, a.tail, d)
        if word_bigram_index not in current_tree_map.keys():
            current_tree_map[word_bigram_index] = 0
        current_tree_map[word_bigram_index] += 1
        if tag_bigram_index not in current_tree_map.keys():
            current_tree_map[tag_bigram_index] = 0
        current_tree_map[tag_bigram_index] += 1

    for v in max_tree.values():
        word_bigram_index, tag_bigram_index = feature_function(v.head, v.tail, d)
        if word_bigram_index not in max_tree_map.keys():
            max_tree_map[word_bigram_index] = 0
        max_tree_map[word_bigram_index] += 1
        if tag_bigram_index not in max_tree_map.keys():
            max_tree_map[tag_bigram_index] = 0
        max_tree_map[tag_bigram_index] += 1

    return subtract_maps(current_tree_map, max_tree_map)


def create_all_arcs(sentence, theta, d):
    arcs = []
    for head, rel, tail in sentence.triples():
        word_bigram_index, tag_bigram_index = feature_function(head, tail, d)
        weight = theta[word_bigram_index] + theta[tag_bigram_index]
        current_arc = Arc(head, tail, weight)
        arcs.append(current_arc)
    return arcs


def max_spanning_arborescence_nx(arcs, sink):
    """
    Wrapper for the networkX min_spanning_tree to follow the original API
    :param arcs: list of Arc tuples
    :param sink: unused argument. We assume that 0 is the only possible root over the set of edges given to
     the algorithm.
    """
    G = DiGraph()
    for arc in arcs:
        G.add_edge(arc.head, arc.tail, weight=arc.weight)
    ARB = maximum_spanning_arborescence(G)
    result = {}
    headtail2arc = {(a.head, a.tail): a for a in arcs}
    for edge in ARB.edges:
        tail = edge[1]
        result[tail] = headtail2arc[(edge[0], edge[1])]
    return result


def update_weight(weight_vector, learning_rate, gradient):
    for k, v in gradient.items():
        weight_vector[k] += v * learning_rate
    return weight_vector


def perceptron_algorithm(iterations, sentences, d, learning_rate):
    N = len(sentences)
    weight_vector = np.zeros(len(d))
    weights_sum = weight_vector
    for r in range(iterations):
        for i_s, s in enumerate(sentences):
            arcs = create_all_arcs(s, weight_vector, d)
            # result is dictionary, keys: node object, values: object, that contains arc defined by this tail
            if len(arcs) > 0:
                max_tree = max_spanning_arborescence_nx(arcs, 0)
                gradient = get_gradient(max_tree, arcs, d)
                weight_vector = np.array(update_weight(weight_vector, learning_rate, gradient))
                weights_sum += weight_vector
            print(f"Epoch: {r + 1}, Iteration: {i_s + 1}")
    return weights_sum / (iterations * N)


def get_sentence_tuples(s):
    sentence_tuples = []
    for head, rel, dep in s.triples():
        sentence_tuples.append((head[0], dep[0]))
    return sentence_tuples


def evaluation(test_set, theta, d, number_of_words):
    correct_edges_counter = 0
    for s in test_set:
        arcs = create_all_arcs(s, theta, d)
        if len(arcs) > 0:
            mst = max_spanning_arborescence_nx(arcs, 0)
            current_word_tuples = get_sentence_tuples(s)
            # result is dictionary, keys: node object, values: object, that contains arc defined by this tail
            for obj in mst.values():
                head_word, tail_word = obj.head[0], obj.tail[0]
                if (head_word, tail_word) in current_word_tuples:
                    correct_edges_counter += 1
    return correct_edges_counter / number_of_words


train_set, test_set, d, number_of_words = import_data()
theta = perceptron_algorithm(2, train_set, d, 1)
accuracy = evaluation(test_set, theta, d, number_of_words)
print(accuracy)
