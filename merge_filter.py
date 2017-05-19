import argparse
import numpy as np
import pandas as pd
import os
import pickle
import langdetect as lang
import time
from datetime import datetime
import json

directory = 'data/twitter'
outfile = 'output.csv'
verbose = False

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, default = directory)
    parser.add_argument('outfile', type=str, default=outfile)
    args = parser.parse_args()
    return args

def DF_to_Dict(df, saveAs=''):
    dict = {}

    try:
        dict['date'] = np.array(df['date'].tolist())
        dict['retweets'] = np.array(df['retweets'].tolist())
        dict['favorites'] = np.array(df['favorites'].tolist())
        dict['text'] = np.array(df['text'].tolist())
        dict['hashtags'] = np.array(df['hashtags'].tolist())
        dict['id'] = np.array(df['id'].tolist())
        dict['permalink'] = np.array(df['permalink'].tolist())
    except:
        dict['id'] = np.array(df['id'].tolist())
        dict['date'] = np.array(df['date'].tolist())
        dict['text'] = np.array(df['text'].tolist())
        dict['upvotes'] = np.array(df['upvotes'].tolist())

    if saveAs != '':
        pickle.dump(dict, open(saveAs, 'wb'))

    return dict

def get_files_of_type(type, directory):
    if type[0] != '.': type = '.' + type
    return [os.path.join(directory, f) for f in os.listdir(directory) if
     os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1] == type]

def json_to_csv(jsonfile, saveAs='output.csv', index=0, opt='w'):
    with open(jsonfile) as file:
        data = json.load(file)

    csv = open(saveAs, opt)
    if opt == 'w': csv.write('id;date;text;upvotes')
    for submission in data:
        d = str(datetime.fromtimestamp(submission['SubmissionTime']))
        t = '"' + str(submission['SubmissionTitle']) + '"'
        u = str(submission['SubmitUpvotes'])
        csv.write('\n' + ";".join([str(index), d, t, u]))
        index += 1
        for comment in submission['Comments']:
            d = str(datetime.fromtimestamp(comment['CommentTime']))
            t = '"' + str(comment['CommentText']) + '"'
            u = str(comment['CommentUpvotes'])
            csv.write('\n' + ";".join([str(index), d, t, u]))
            index += 1

    csv.close()
    return index

def json_to_Dict(jsonfile, dict={}, saveAs=''):
    with open(jsonfile) as file:
        data = json.load(file)

    if len(list(dict.keys())) == 0:
        dict['id'] = np.array([])
        dict['date'] = np.array([])
        dict['text'] = np.array([])
        dict['upvotes'] = np.array([])

    index = len(dict['id'])

    newids = []
    newdates = []
    newtext = []
    newupvotes = []

    for submission in data:
        # newids += [index]
        # newdates += [datetime.fromtimestamp(submission['SubmissionTime'])]
        # newtext += [submission['SubmissionTitle']]
        # newupvotes += [submission['SubmitUpvotes']]
        # index += 1
        for comment in submission['Comments']:
            newids += [int(index)]
            newdates += [float(comment['CommentTime'])]
            newtext += [comment['CommentText']]
            newupvotes += [int(comment['CommentUpvotes'])]
            index += 1
            if index > 100: break

    dict['id'] = np.concatenate((dict['id'], np.array(newids)))
    dict['date'] = np.concatenate((dict['date'], np.array(newdates)))
    dict['text'] = np.concatenate((dict['text'], np.array(newtext)))
    dict['upvotes'] = np.concatenate((dict['upvotes'], np.array(newupvotes)))

    if saveAs != '':
        pickle.dump(dict, open(saveAs, 'wb'))

    return dict

def merge(directory, fileType='csv', saveAs=''):
    if verbose: print("Merging files...", flush=True)
    files = get_files_of_type(fileType, directory)

    if 'csv' in fileType.lower():
        split = os.path.split(saveAs)
        csvFile = os.path.join(split[0], split[1].split('.')[0]) + '.csv'
        df = pd.read_csv(files[0], header=0, sep=";", quoting=1, quotechar='"', error_bad_lines=False, warn_bad_lines=False)

        for i in range(1, len(files)):
            add = pd.read_csv(files[i], header=0, delimiter=";", quoting=1, quotechar='"', error_bad_lines=False, warn_bad_lines=False)
            df = pd.concat([df, add])

        df.to_csv(csvFile, sep=';', index=False, quoting=2)
        if verbose: print("Successfully merged {} files with {} lines.".format(len(files), df.shape[0]), flush=True)
        dict = DF_to_Dict(df)

    if 'json' in fileType.lower():
        split = os.path.split(saveAs)
        csvFile = os.path.join(split[0], split[1].split('.')[0]) + '.csv'
        index = json_to_csv(files[0], saveAs=csvFile)
        dict = json_to_Dict(files[0])

        for i in range(1, len(files)):
            index = json_to_csv(files[i], saveAs=csvFile, index=index, opt='a')
            dict = json_to_Dict(files[i], dict=dict)



    if 'dict' in fileType.lower():
        dict = pickle.load(open(files[0], 'rb'))
        keys = list(dict.keys())
        for i in range(1, len(files)):
            add = pickle.load(open(files[i], 'rb'))
            for key in keys:
                dict[key] = np.concatenate((dict[key], add[key]))

    if saveAs != '':
        pickle.dump(dict, open(saveAs, 'wb'))

    return dict

def filter(dict, with_words=[], without_words=[], language = 'en', saveAs='', startFrame=-1, endFrame=-1):
    d = dict.copy()
    keys = list(d.keys())

    if startFrame != -1 and endFrame != -1:
        for key in keys:
            d[key] = d[key][startFrame:]

    text = d['text']
    remove_indices = np.zeros(len(text), dtype=bool)
    start = len(text)
    if verbose: print("Filtering file with {} entries...".format(start), flush=True)
    # if verbose: print("Time estimated to filter: {:.0f} minutes.".format(start*.011//60+1), flush=True)

    language_filter = []
    i = 0
    z = 0
    startTime = time.time()
    for t in text:
        try:
            language_filter.append(lang.detect(t) != language)
        except:
            language_filter.append(True)
        i += 1

        if verbose and (time.time()-startTime)//60 > z:
            z += 1
            print("{:.2f}% of text filtered after {} minutes. Estimated {:.0f} minutes remaining.".format(i/start*100, z, (start-i)/i * z+1), flush=True)



    remove_indices += language_filter

    if len(with_words) != 0:
        for word in with_words:
            remove_indices += [word not in t for t in text]

    if len(without_words) != 0:
        for word in without_words:
            remove_indices += [word in t for t in text]


    for key in keys:
        d[key] = d[key][~remove_indices]

    if saveAs != '':
        pickle.dump(d, open(saveAs, 'wb'))

    end = len(d[keys[0]])
    if verbose: print("Successfully filtered file from {} entries to {}.".format(start,end), flush=True)
    return d

def merge_stocks(directory, saveAs=''):
    files = get_files_of_type('csv', directory)

    split = os.path.split(saveAs)
    csvFile = os.path.join(split[0], split[1].split('.')[0]) + '.csv'

    cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = pd.read_csv(files[0], index_col=0, header=None, sep=',')
    df.columns = cols
    df['volume'] *= ((df['close'] - df['open']) / 2 + df['open'])

    for i in range(1, len(files)):
        add = pd.read_csv(files[i], index_col=0, header=None, sep=',')
        add.columns = cols
        add['volume'] *= ((add['close'] - add['open']) / 2 + add['open'])
        df = df.add(add, fill_value=0)

    df.to_csv(csvFile, sep=',', index=True,index_label='date', quoting=3)

    if saveAs != '':
        pickle.dump(df, open(saveAs, 'wb'))

    return df

def stock_to_DF(stockCSVFile, saveAs=''):
    df = pd.read_csv(stockCSVFile, header=0, sep=',')
    df['Date'] = [int(d.replace('-','')) for d in df['Date']]
    df = df.set_index(df['Date']).filter(['Close'])

    if saveAs != '':
        pickle.dump(df, open(saveAs, 'wb'))

    return df



if __name__ == "__main__":
    stock_to_DF('data/indexes/sp500.csv', saveAs='data/indexes/sp500.df')


    # dict = pickle.load(open('data/reddit/raw_finance.dict', 'rb'))
    # verbose = True
    # filter(dict, saveAs='data/reddit/filtered_finance.dict')

    # keys = list(dict.keys())
    # print(dict['text'])
    # print(len(dict[keys[0]]), len(keys))

    # json_to_Dict('data/reddit/apple-.json', saveAs='data/reddit/raw_AAPL.dict')

    # merge('data/reddit/finance_folder', fileType='json', saveAs='data/reddit/raw_finance.dict')
    # json_to_Dict('data/reddit/apple-.json', saveAs='data/reddit/raw_AAPL.dict')
    # json_to_csv('data/reddit/apple-.json', saveAs='data/reddit/raw_AAPL.csv')
    # dict = pickle.load(open('data/reddit/raw_finance.dict', 'rb'))


    # verbose = False
    # dir = 'data/twitter/merge_folder'
    # df = merge(dir, fileType='dict', saveAs='data/twitter/filtered_finance.dict')
    # outfile = 'testCSV.csv'
    # index = json_to_csv('data/reddit/market-.json', saveAs=outfile)
    # df = pd.read_csv('data/reddit/raw_finance.csv', header=0, sep=";", quoting=1, quotechar='"', error_bad_lines=False, warn_bad_lines=False)
    # print(len(df.index))
    # json_to_csv('data/reddit/market-.json', saveAs=outfile, index=index, opt='a')
    # df = pd.read_csv(outfile, header=0, sep=";", quoting=1, quotechar='"', error_bad_lines=False, warn_bad_lines=False)
    # print(len(df.index))
    # dir = 'data/reddit/merge_folder'
    # df = merge(dir, fileType='json')




    # dict = DF_to_Dict(df, saveAs='data/twitter/raw_twitter.dict')
    # startFrame = 400000
    # endFrame = 500000
    # dict = pickle.load(open('data/twitter/raw_twitter.dict', 'rb'))
    # # print("Finished loading dict")
    # filtered = filter(dict,saveAs=os.path.join('data/twitter','filtered_AAPL.dict'))


