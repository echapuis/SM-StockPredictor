from os.path import join
import os
import pickle


def create_wordCount_dictionary():
    directories = ['data/reddit/finance-trained/no_stem_no_stop',
                   'data/reddit/finance-trained/with_stem_no_stop',
                   'data/twitter/finance-trained/no_stem_no_stop',
                   'data/twitter/finance-trained/with_stem_no_stop']
    subdirectory = 'TC_saveFolder'

    dictionary = {}
    i = 0
    for directory in directories:
        clusterpath = join(directory, subdirectory, 'cluster_map.dict')
        wordlistpath = join(directory, subdirectory, 'wordlists.list')
        i += 1

        cluster_map = pickle.load(open(clusterpath, 'rb'))
        wordlist = pickle.load(open(wordlistpath, 'rb'))

        keys = list(cluster_map.keys())

        dictionary[i] = {}
        for key in keys:
            # newkey = key + str(i)
            dictionary[i][key] = [cluster_map[key], 0]

        for text in wordlist:
            for word in text:  # text is list of list of words
                # word = word + str(i)
                if word in dictionary[i]:
                    dictionary[i][word][1] += 1

    pickle.dump(dictionary, open("alldata_word_clusterCount_map.dict", 'wb'))


def sort_cluster(folder):
    dict = pickle.load(open('alldata_word_clusterCount_map.dict', 'rb'))

    items = list([item for item in dict[folder].items()])

    # print(items)

    tuples = list([(i[1][1], i[0], i[1][0]) for i in items])

    sort = sorted(tuples)

    file = open('cluster_output{}.tsv'.format(folder), 'w')
    file.write('weight\tword\tcluster\n')
    for item in sort:
        file.write(str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2]) + '\n')

    file.close()


if __name__ == '__main__':
    create_wordCount_dictionary()
    sort_cluster(1)
    sort_cluster(2)
    sort_cluster(3)
    sort_cluster(4)

