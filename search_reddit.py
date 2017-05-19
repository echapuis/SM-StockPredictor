import praw
import json
import ujson
import numpy as np
import collections
import datetime
import pandas as pd
import csv
import argparse


# Helper Functions
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('subreddit', type=str)
    parser.add_argument('keyword', type=str)
    args = parser.parse_args()
    return args


# search reddit for keywords
def search(reddit, subreddit, keyword, ltime, rtime, sorttype):

    results = reddit.subreddit(subreddit).search('(and timestamp:{}..{} title:{})'.format(ltime, rtime, repr(keyword)), sort=sorttype, syntax='cloudsearch', limit=10000)
    # print('(and timestamp:{}..{} title:{})'.format(ltime, rtime, repr(keyword)))

    dict_names = ["SubmissionTime", "SubmissionTitle", "SubmitAuthorKarma", "SubmitUpvotes", "SubmitScore", "NumComments", "Comments", "CommentTime", "CommentText", "CommentUpvotes", "CommentScore", "CommentAuthorKarma"]
    dict = []

    for submission in results:

        submission.comments.replace_more(limit=0)
        # print(submission.selftext)
        # submission author karma
        try:
            karma1 = str(submission.author.link_karma)
        except:
            karma1 = 'null'

        sub = {dict_names[0]: int(submission.created_utc), dict_names[1]: submission.title,
               dict_names[2]: karma1, dict_names[3]: submission.ups, dict_names[4]: submission.score,
               dict_names[5]: submission.num_comments}

        allcomments = []
        for comment in submission.comments.list():

            # comment author karma
            try:
                karma2 = str(comment.author.link_karma)
            except:
                karma2 = 'null'

            # write line to json file
            try:
                scomment = comment.body.replace("\n", " ")
                slimcomment = scomment.replace("> ", "")
                allcomments.append({dict_names[7]: int(comment.created_utc), dict_names[8]: slimcomment,
                                    dict_names[9]: comment.ups, dict_names[10]: comment.score, dict_names[11]: karma2})
            except UnicodeEncodeError:
                pass

        sub[dict_names[6]] = allcomments
        dict.append(sub)

    with open(subreddit + "-" + keyword + ".json", 'w') as file:
        ujson.dump(dict, file)
    file.close()

# merge any json file
def merge_json(dict1, dict2, name1, name2):

    print("currently merging: " + name1 + " & " + name2)

    md = dict1
    # md = merge_dict.append(dict1)

    listSubmissions = []
    for d1 in md:
        listSubmissions.append(d1.get("SubmissionTitle"))

    for d2 in dict2:
        title = d2.get("SubmissionTitle")
        if title not in listSubmissions:
            md.append(d2)
            listSubmissions.append(title)

    return md

# merge reddit india data
def merge_india():

    bjp = ujson.loads(open('india-bjp.json').read())
    aap = ujson.loads(open('india-aap.json').read())
    ele = ujson.loads(open('india-election.json').read())
    mod = ujson.loads(open('india-modi.json').read())
    con = ujson.loads(open('india-congress.json').read())
    los = ujson.loads(open('india-lok sabha.json').read())
    nam = ujson.loads(open('india-namo.json').read())

    print(len(bjp))
    print(len(aap))
    print(len(ele))
    print(len(mod))
    print(len(con))
    print(len(los))
    print(len(nam))

    print("merged files")

    bjp_aap = merge_json(bjp, aap, 'bjp', 'aap')
    print(len(bjp_aap))

    bjp_aap_ele = merge_json(bjp_aap, ele, 'bjp_aap', 'ele')
    print(len(bjp_aap_ele))

    bjp_aap_ele_mod = merge_json(bjp_aap_ele, mod, 'bjp_aap_ele', 'mod')
    print(len(bjp_aap_ele_mod))

    bjp_aap_ele_mod_con = merge_json(bjp_aap_ele_mod, con, 'bjp_aap_ele_mod', 'con')
    print(len(bjp_aap_ele_mod_con))

    los_nam = merge_json(los, nam, 'los', 'nam')
    print(len(los_nam))

    all = merge_json(bjp_aap_ele_mod_con, los_nam, 'bjp_aap_ele_mod_con', 'los_nam')

    with open("india-final.json", 'w') as file:
        json.dump(all, file)
    file.close()

# merge reddit UK data
def merge_uk():

    eb = ujson.loads(open('europe-brexit.json').read())
    ukb = ujson.loads(open('ukpolitics-brexit.json').read())
    ukr = ujson.loads(open('ukpolitics-remain.json').read())
    ukl = ujson.loads(open('ukpolitics-leave.json').read())
    arb = ujson.loads(open('askreddit-brexit.json').read())

    print(len(eb))
    print(len(ukb))
    print(len(ukr))
    print(len(ukl))
    print(len(arb))

    print("merged files")

    eb_ukb = merge_json(eb,ukb, 'eb', 'ukb')
    print(len(eb_ukb))

    eb_ukb_ukr = merge_json(eb_ukb, ukr, 'eb_ukb', 'ukr')
    print(len(eb_ukb_ukr))

    eb_ukb_ukr_ukl = merge_json(eb_ukb_ukr, ukl, 'eb_ukb_ukr', 'ukl')
    print(len(eb_ukb_ukr_ukl))

    eb_ukb_ukr_ukl_arb = merge_json(eb_ukb_ukr_ukl, arb, 'eb_ukb_ukr_ukl', 'arb')
    print(len(eb_ukb_ukr_ukl_arb))

    with open("brexit-final.json", 'w') as file:
        json.dump(eb_ukb_ukr_ukl_arb, file)
    file.close()

# turns json to bag of words
def bagofwords(data):

    # ' '.join(list)
    words = []
    for s in data:
        title = s.get("SubmissionTitle")
        tlist = title.split()
        words.extend(tlist)
        # print(tlist)
        for c in s.get("Comments"):
            com = c.get("CommentText")
            try:
                comlist = com.split()
                words.extend(comlist)
            except:
                pass

    return words

# counts frequency of words
def countfreq(data):

    counter = collections.Counter(data)
    print(counter.most_common(100))

# broken
# should plot comment words over time
def timeplot(data):

    dates = []
    for sub in data:
        subtime = convertunixtodate(sub.get("SubmissionTime"))
        dates.append(subtime)
        for c in sub.get("Comments"):
            com1 = c.get("CommentTime")
            try:
                com = convertunixtodate(com1)
                dates.append(com)
            except:
                pass

    # print(dates_com)

    count = dict(collections.Counter(dates))
    # countcom = dict(collections.Counter(dates_com))

    with open('dict2.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in count.items():
            writer.writerow([key, value])

    csv_file.close()

# broken
# attempt 2 at plotting words over time
def plotCounts(data):

    dates_sub = []
    dates_com = []
    for sub in data:
        subtime = convertunixtodate(sub.get("SubmissionTime"))
        dates_sub.append(subtime)
        for c in sub:
            try:
                com = convertunixtodate(c.get("CommentTime"))
                dates_com.append(com)
            except:
                pass

    print(date_sub)
    countsub = collections.Counter(dates_sub)
    countcom = collections.Counter(dates_com)


# converts unix date-time to date
def convertunixtodate(unix_time):

    dte = (datetime.datetime.fromtimestamp(int(unix_time)).strftime('%m-%d-%Y'))
    return(dte)


# updates json
def update_reddit(data, filename):

    cid = 1
    sid = 1
    for d1 in data:
        generate_sub_id = 'rs' + str(sid)
        d1.update({'SubmissionID': generate_sub_id})

        com = d1.get("Comments")
        for d2 in com:
            generate_com_id = generate_sub_id + '-' + 'rc' + str(cid)
            d2.update({'CommentID': generate_com_id})
            cid += 1

        sid += 1

    with open(filename + "-ID.json", 'w') as file:
        ujson.dump(data, file)
    file.close()

    print("completed writing " + filename + ".json")

# main function / connects to reddit etc.
def main(args):

    reddit = praw.Reddit(client_id='UwThPYLE8xEJXA',
                         client_secret='RGnEyP6r_nKbcRWe7RBvjbPxhNY',
                         user_agent='cseproject by /u/joyofbubbles')
    # print(convertunixtodate(1262304000))
    # print(convertunixtodate(1357100000))
    # 1262304000 = 12-31-2009
    # 1357100000 = 01-01-2013
    """
    Search Criteria

    # SUBREDDIT
    INDIA: india
    BREXIT: ukpolitics, europe, AskReddit

    # KEYWORDS
    INDIA: election, modi, bjp, congress, aap, namo, lok sabha
    BREXIT: brexit, remain, leave

    SORT_TYPE = 'relevance' or 'new'

    # TIME: Brexit Referendum
    ltime = 1450828800 # December 23, 2015 00:00
    rtime = 1466640000 # June 23, 2016 00:00

    # TIME: India Elections
    ltime = 1383782400 # November 7, 2013 00:00
    rtime = 1396828800 # April 7, 2014 00:00

    # reddit, subreddit, keyword, ltime, ritime, sort
    """

    search(reddit, args.subreddit, args.keyword, 1357100000, 1492980706, 'relevance')



    # search(reddit, 'india', 'namo', ltime, rtime, 'relevance')

    # merge_india()
    # merge_uk()

    # india = ujson.loads(open('../data/reddit/india_reddit.json').read())
    # brexit = ujson.loads(open('../data/reddit/brexit_reddit.json').read())
    #
    # update_reddit(india, "india")
    # update_reddit(brexit, "brexit")

    # timeplot(india)
    # timeplot(brexit)

    # words_india = bagofwords(india)
    # words_brexit = bagofwords(brexit)
    # countfreq(words_india)
    # countfreq(words_brexit)



if __name__ == "__main__":
    args = parse_arguments()


    main(args)
