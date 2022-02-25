#!/usr/bin/env python
# SePass  -  Semantic Password Guessing
#
# VERSION 1.0.1
#
# Copyright (C) 2021-2022 Levin Schäfer
# All rights reserved.
#
# This tool uses rulegen as part of PACK (Password Analysis and Cracking Kit)
# by Peter Kacherginsky
#
# Please see the attached LICENSE file for additional licensing information.

import argparse
import os
import pathlib
import progressbar
import rulegen
import shutil
import sys
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from itertools import chain
import numpy as np
import time
import datetime
import gensim

# Replacement rules to clean new word suggestions. It's a list of tuples (A, b),
# where every occurrence of a letter in A will be replaced by b.
CHARSET_GERMAN = [("áàâãåæāăąǎ", "a"), ("ÀÁÂÃÅÆĀĂĄǍ", "A"), ("ç", "c"), ("ÇĆĈĊČćĉċč", "C"), ("ðďđ", "d"),
                  ("ÐĎ", "D"), ("èéêëēĕėęě", "e"), ("ÈÉÊËĒĔĖĘĚ", "E"), ("ĝğġģ", "g"), ("ĜĞĠĢ", "G"),
                  ("ìíîïīĩĭįıǐ", "i"), ("ÌÍÎÏĨĪĬĮİǏ", "I"), ("ñńņňŉŋ", "n"), ("ÑŃŅŇ", "N"), ("òóôõøǒ", "o"),
                  ("ÒÓÔÕØŌŎŐǑ", "O"), ("ùúûōŏőœũūŭůűųǔ", "u"), ("ÙÚÛŨŪŬŮŰŲǓ", "U"), ("ýÿ", "y"), ("Ý", "Y")]

# The set of valid symbols that can be found in new basewords. Every other symbol will be deleted
VALID_SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜß'


# This function generates for a word and a vocab all letter
# sequences of the word that are contained in the vocab.
def get_known_substrings(word, min_len, vocab):
    substrings = [word[i: j].lower() for i in range(len(word))
                  for j in range(i + min_len, len(word) + 1)]
    substrings = list(filter(lambda x: x.isalpha(), substrings))
    substrings.sort(key=len, reverse=True)
    sub_strings = list(filter(lambda x: x in vocab, substrings))

    return sub_strings


# This function calculates the best decomposition of the word into the subwords
# from a given word and a set of subwords. It returns a triple ([subwords], [splits], k),
# where k is the number of symbols that couldn't be matched to a subword.
def dissolve(word, sub_words):
    lower_word = word.lower()
    sub_words = list(set(sub_words))
    sub_words.sort(key=len, reverse=True)
    if word == '':
        return [], [], 0
    if lower_word in sub_words:
        return [lower_word], [word], 0
    else:
        sub_words = [w for w in sub_words if w in lower_word]
        sub_words.sort(key=lambda x: lower_word.find(x))
        if sub_words is not None and len(sub_words) > 0:
            # Word can be decomposed and its not last split
            possible_results = []
            for s in sub_words:
                split_index = lower_word.find(s) + len(s)
                cw, cs, v = dissolve(word[split_index:], sub_words)
                if v == len(word[split_index:]):
                    # end of the word needs to be concatenated to last split
                    cs = [word]
                else:
                    cs.insert(0, word[:split_index])
                cw.insert(0, s)
                v = v + lower_word.find(s)
                current_solution = (cw, cs, v)
                possible_results.append(current_solution)
            possible_results.sort(key=lambda x: x[2])
            best_value = possible_results[0][2]
            best_results = list(filter(lambda x: x[2] == best_value, possible_results))
            best_results.sort(key=lambda x: len(x[1]))
            return best_results[0]
        else:
            # Word cannot be decomposed
            return [], [word], len(word)


# This function cleans a list of given words. Unwanted symbols will be deleted/changed
# and words that are too long will be deleted.
def fit_words_to_alphabet(words, rule_set, max_length):
    cleaned_words = []
    for w in words:
        word = w[0]
        for (l, r) in rule_set:
            # change letters
            for x in l:
                word = word.replace(x, r)
        # delete symbols
        cleaned_word = "".join(m for m in word if m in VALID_SYMBOLS)
        # ignore words that are too long
        if 0 < len(cleaned_word) <= max_length:
            cleaned_words.append((cleaned_word, w[1]))
    return cleaned_words


# Given a vocabulary and a word, this function finds the most likely subwords of the word from the vocabulary.
# It will return a tuple of [subwords], [splits]
def get_composite_words(word, min_length, vocab):
    if len(word) > 1:
        sub_strings = get_known_substrings(word, min_length, vocab)
    else:
        return [], word
    subwords_splits = dissolve(word, sub_strings)[:2]  # ([word], [split])
    return subwords_splits


# With a given word embedding model, this function will use the given mode of semantic expansion
# to find neighbours for the given word.
def find_neighbours(base_word, num_neighbours, w_e_model, mode, rv=300000, eps=None):
    new_words_and_simils = None
    if mode == 'k-NN':
        base_word = base_word.lower()
        if base_word not in w_e_model.vocab:
            return []
        new_words_and_simils = w_e_model.most_similar(base_word, topn=num_neighbours, restrict_vocab=rv)
    elif mode == 'epsilon':
        if args.debug:
            print("DEBUG: get neighbours in epsilon-environment")
        base_word = base_word.lower()
        if base_word not in w_e_model.vocab:
            return []
        candidates = w_e_model.most_similar(base_word, topn=num_neighbours * 10, restrict_vocab=rv)
        new_words_and_simils = [x for x in candidates if x[1] >= eps]
    else:
        print('ERROR: there is no function '+mode)
        exit(1)
    return new_words_and_simils


# Simple function to read a list of words from a file.
def read_list_from_file(path, encoding='utf8'):
    words = []
    try:
        with open(path, 'r', encoding=encoding) as file:
            for line in file:
                words.append(line[:-1])
        return words
    except Exception as e:
        print('Can\'t open the file %s', path)
        print(e)
        exit(1)


# Simple function to write a list to file.
def write_list_to_file(words, path):
    file = open(path, 'w')
    for line in words:
        file.write(line + "\n")
    file.close()


# This function calculates the password score of a new password candidate from all given parameters.
def calculate_password_score(rule_score, word_score, mrm, sr):
    score = rule_score * word_score ** (1 / sr)
    score = score ** (1/mrm)
    return score


# This function scales the list of given numbers to interval [0,1].
def scale_data(data):
    res = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(res)
    res = scaler.transform(res)
    scaled_data = list(chain.from_iterable(res))
    return scaled_data


# This function gets a list of (rule, #occurences) and returns a list of (rule, rulescore).
def get_scored_rules(rules_and_numbers):
    rs, counts = list(zip(*rules_and_numbers))
    scored_rules = list(zip(rs, scale_data(counts)))
    if args.debug:
        print("DEBUG: calculated rules score: "+str(scored_rules[:10])+' ('+str(len(scored_rules))+')')
    return scored_rules


# This function takes a dict of baseword:[similarities] and returns a list of (baseword, wordscore) sorted in descending
# order by wordscore.
def score_and_sort_words(word_similarities):
    words = word_similarities.keys()
    similarities = []
    for key in words:
        s = sum(word_similarities[key])
        similarities.append(s)
    scaled_similarities = scale_data(similarities)
    sorted_words = list(zip(words, scaled_similarities))
    sorted_words.sort(key=lambda x: x[1], reverse=True)
    if args.debug:
        print("DEBUG: calculated similarity scores: "+str(sorted_words[:10])+'...')
    return sorted_words


# This function calculates the character position based on hashcat position expression
def get_hashcat_position(pos):
    if pos in '0123456789':
        return int(pos)
    else:
        s = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.find(pos)
        if s == -1:
            raise ValueError('position must be one of 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        else:
            return s + 10


# This function calculates the password candidate from a baseword and a hashcat rule.
# In earlier versions, this process was handled directly by hashcat. However, this is
# inefficient for short password lists.
def create_password_from_hashcat_rule(base_word, rule):
    while len(rule) > 0:
        command = rule[0]
        if command == ':':
            break
        elif command == 'i':  # insert
            position = get_hashcat_position(rule[1])
            if position < len(base_word):
                base_word = base_word[:position] + rule[2] + base_word[position:]
            rule = rule[4:]
        elif command == '$':  # append char
            base_word += rule[1]
            rule = rule[3:]
        elif command == '^':  # prepend char
            base_word = rule[1] + base_word
            rule = rule[3:]
        elif command == ']':  # trunkate right
            base_word = base_word[:-1]
            rule = rule[2:]
        elif command == '[':  # trunkate left
            base_word = base_word[1:]
            rule = rule[2:]
        elif command == 's':  # replace
            base_word = base_word.replace(rule[1], rule[2])
            rule = rule[4:]
        elif command == 'o':  # overwrite
            position = get_hashcat_position(rule[1])
            if position < len(base_word):
                base_word = base_word[:position] + rule[2] + base_word[position + 1:]
            rule = rule[4:]
        elif command == ',':  # replace char before
            position = get_hashcat_position(rule[1])
            if position < len(base_word):
                base_word = base_word[:position] + base_word[position - 1] + base_word[position + 1:]
            rule = rule[3:]
        elif command == '.':  # replace char after
            position = get_hashcat_position(rule[1])
            if position + 1 < len(base_word):
                base_word = base_word[:position] + base_word[position + 1] + base_word[position + 1:]
            rule = rule[3:]
        elif command == '*':  # swap chars
            p1 = get_hashcat_position(rule[1])
            p2 = get_hashcat_position(rule[2])
            if p1 > p2:
                t = p1
                p1 = p2
                p2 = t
            if p2 < len(base_word):
                base_word = base_word[:p1] + base_word[p2] + base_word[p1 + 1:p2] + base_word[p1] + base_word[p2 + 1:]
            rule = rule[4:]
        elif command == 'D':  # delete at pos
            position = get_hashcat_position(rule[1])
            base_word = base_word[:position] + base_word[position + 1:]
            rule = rule[3:]
        elif command == '-':  # decrement char by ascii
            position = get_hashcat_position(rule[1])
            if position < len(base_word):
                base_word = base_word[:position] + chr(ord(base_word[position]) - 1) + base_word[position + 1:]
            rule = rule[3:]
        elif command == '+':  # increment char by ascii
            position = get_hashcat_position(rule[1])
            if position < len(base_word):
                base_word = base_word[:position] + chr(ord(base_word[position]) + 1) + base_word[position + 1:]
            rule = rule[3:]
        elif command == 'T':  # toggle case at position
            position = get_hashcat_position(rule[1])
            if position < len(base_word):
                base_word = base_word[:position] + str(base_word[position]).swapcase() + base_word[position + 1:]
            rule = rule[3:]
        elif command == 'c':  # capital first letter, lower rest
            base_word = str.title(base_word)
            rule = rule[2:]
        elif command == 'u':  # uppercase
            base_word = base_word.upper()
            rule = rule[2:]
        elif command == 'R':  # Bit shifting right
            position = get_hashcat_position(rule[1])
            if position < len(base_word):
                base_word = base_word[:position] + chr(ord(base_word[position]) >> 1) + base_word[position + 1:]
            rule = rule[3:]
        elif command == 'k':  # swap first two chars
            if len(base_word) > 1:
                base_word = base_word[1] + base_word[0] + base_word[2:]
            rule = rule[2:]
        elif command == 'K':  # swap last two chars
            if len(base_word) > 1:
                base_word = base_word[:-2] + base_word[-1] + base_word[-2]
            rule = rule[2:]
        elif command == 't':  # swap case
            base_word = base_word.swapcase()
            rule = rule[2:]
        elif command == 'L':  # Bit shifting left
            position = get_hashcat_position(rule[1])
            if position < len(base_word):
                base_word = base_word[:position] + chr(ord(base_word[position]) << 1) + base_word[position + 1:]
            rule = rule[3:]
        elif command == 'l':  # lowercase
            base_word = base_word.lower()
            rule = rule[2:]
        else:
            print("WARNING: unknown rule: " + rule)
            break
    return base_word


# This is the main function for semantic expansion. For all basewords,
# it will create similar words using a word embedding.
def create_word_suggestions(counted_base_words, w_e_model, num_neighbours, mode, rv, m_word_len, epsi=None):
    suggestion_word_dict = {}
    with progressbar.ProgressBar(max_value=len(counted_base_words) + 1) as bar2:
        h = 0
        for originWord, multiplier in counted_base_words:
            bar2.update(h)
            h += 1
            suggestions_and_scores = fit_words_to_alphabet(find_neighbours(originWord, num_neighbours, w_e_model, mode,
                                                                           rv=rv, eps=epsi),
                                                           CHARSET_GERMAN, m_word_len)
            for (s, d) in suggestions_and_scores:
                if s not in suggestion_word_dict:
                    suggestion_word_dict[s] = [d] * multiplier
                else:
                    suggestion_word_dict[s].extend([d] * multiplier)
            if originWord not in suggestion_word_dict:
                suggestion_word_dict[originWord] = [1.0] * multiplier
            else:
                suggestion_word_dict[originWord].extend([1.0] * multiplier)
    return suggestion_word_dict


# This function calculates epsilon as the average of the similarity of k-NN.
def get_epsilon_threshold(k, w_e_model, words, rv):
    similarities = []
    for c, _ in words:
        word = c.lower()
        if word in w_e_model.vocab:
            neighbours = w_e_model.most_similar(word, topn=k, restrict_vocab=rv)
            similarities.append(neighbours[-1][1])
    thresh = np.mean(similarities)
    if args.debug:
        print('DEBUG: ε: '+str(thresh))
    return thresh


# This function will analyze a list of passwords with a word embedding model. As result it will return a list of
# baseword, rules, segment_count. segment_count are the frequencies of segments with n basewords.
# Segment count was used to create multi-segment-passwords. However this was not successful so far...
def analyze_pws_with_model(pw_list, w_e_model):
    base_words = []
    segments = []
    segments_not_containing_words = []
    ruls = []
    print("finding base words in passwords")
    with progressbar.ProgressBar(max_value=len(pw_list)) as bar3:
        for r in range(len(pw_list)):
            bar3.update(r)
            if len(pw_list[r]) > 20:
                # dont get stuck with too long words...
                continue
            if pw_list[r] == '':
                continue
            bw, seg = get_composite_words(pw_list[r], args.min_word_length, w_e_model.vocab)
            if not bw:
                # no basewords found...
                segments_not_containing_words.append(pw_list[r])
            else:
                base_words.extend(bw)
                segments.extend(seg)
    if args.debug:
        print("DEBUG: before digging deeper:")
        print("DEBUG: baseWords: " + str(base_words[:10]) + "("+str(len(base_words)) + ")")
        print("DEBUG: segments: " + str(segments[:10]) + "("+str(len(segments)) + ")")
        print("DEBUG: didn't find any base words in: " + str(segments_not_containing_words[:10])
              + "(" + str(len(segments_not_containing_words)) + ")")
    # Try to find baseWord in unknown words first
    if segments_not_containing_words:
        print("trying to find difficult base words")
        filename = "tmp/vocab.txt"
        write_list_to_file(list(w_e_model.vocab), filename)
        detected_words, r, unknown_passwords = rulegen.main(segments_not_containing_words, wordlist=filename,
                                                            maxrulelen=10, maxwords=1, maxrules=5)
        if args.debug:
            print("DEBUG: Found "+str(detected_words[:10]) + " ("+str(len(detected_words))+") when digging deeper")
        ruls.extend(r)
        base_words.extend(detected_words)
    filename = "tmp/baseWords.txt"
    write_list_to_file(base_words, filename)
    print("generating rules")
    _, r, u = rulegen.main(segments, wordlist=filename,
                           maxrulelen=5, maxwords=1, maxrules=5)
    ruls.extend(r)
    if args.debug:
        print("DEBUG: Rules: "+str(ruls[:10]) + "("+str(len(ruls))+")")
        print("DEBUG: final unknown: "+str(u[:10]) + "("+str(len(u))+")")
    return base_words, ruls


window_width = shutil.get_terminal_size().columns
print("####################################################################################".center(window_width))
print("#                                                                                  #".center(window_width))
print("#                                     SePass                                       #".center(window_width))
print("#                           Semantic Password Guessing                             #".center(window_width))
print("#                                                                                  #".center(window_width))
print("#                                   version 1.0.1                                  #".center(window_width))
print("#                                                                                  #".center(window_width))
print("#                      Copyright (C) 2021 -2022 Levin Schäfer                      #".center(window_width))
print("#                             levin.schaefer@posteo.de                             #".center(window_width))
print("#                                                                                  #".center(window_width))
print("#                                All rights reserved.                              #".center(window_width))
print("#                                                                                  #".center(window_width))
print("#   This tool uses rulegen as Part of PACK (Password Analysis and Cracking Kit)    #".center(window_width))
print("#    Please see the attached LICENSE file for additional licensing information     #".center(window_width))
print("#                                                                                  #".center(window_width))
print("####################################################################################".center(window_width))
print("".center(window_width))


# Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('pwlist', help='path of password list to be analyzed')
parser.add_argument('--debug', action='store_true', help='print debug messages')
parser.add_argument('-len', '--list_length', help='number of password suggestions to be created. Default=1000000',
                    type=int, default=10000000)
parser.add_argument('--models', help='name of the fasttext word embeddings models to use (stored in models/ folder)',
                    default=['cc.en.300.vec.gz'], nargs='+')
parser.add_argument('-mwl', '--min_word_length', help='length of the smallest word that should be searched for. '
                                                      'Default=4.',
                    type=int, default=4)
parser.add_argument('-o', '--out', help='Output path', default='suggestions.txt')
parser.add_argument('-rr', '--relevant_ruleset_ratio',
                    help='this parameter will determine the percentage of (different) rules to be taken into account'
                         '. Default=0.1', default=0.1, type=float)
parser.add_argument('-sr', '--semantic_rating', help='higher semantic rating will increase the influence of word '
                                                     'semantic in sorting process', default=0.65, type=float)
parser.add_argument('-rv', '--restrict_vocab', help='this parameter restricts the number of words ov each model to the'
                                                    'most frequent k', default=300000, type=int)
parser.add_argument('--mode', help='this parameter defines the mode for semantic expansion. Default = k-NN',
                    default='k-NN')

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
args = parser.parse_args()
args.pwlist = str(pathlib.Path().absolute()) + '/' + args.pwlist
base_models_path = 'models/'

# parameter validation
if not os.path.exists(args.pwlist):
    print('File '+args.pwlist+' does not exist.')
    exit(1)
if args.list_length < 0:
    print('Resulting dictionary length must be positive.')
    exit(1)
for m_name in args.models:
    if not os.path.exists(base_models_path+m_name):
        print("ERROR: model " + m_name + " does not exist.")
        exit(1)
if args.min_word_length < 1:
    print('Minimum word length must be at least 1.')
    exit(1)
if os.path.split(args.out)[0] != '':
    if not os.path.exists(os.path.split(args.out)[0]):
        print('Output path is not valid.')
        exit(1)
if not 0 < args.relevant_ruleset_ratio < 1:
    print('Relevant ruleset ratio must be between 0 and 1.')
    exit(1)
if args.semantic_rating < 0:
    print('Semantic rating must be positive.')
    exit(1)
if args.restrict_vocab < 1:
    print('Restrict vocab must be positive.')
    exit(1)
if args.mode not in ['k-NN', 'epsilon']:
    print('function must be k-NN or epsilon.')
    exit(1)

# parameter recommendations
if args.min_word_length != 4:
    print("It is recommended to search for words with minimal length of 4. Are you sure you want to continue?")
    answer = input("yes/no (y)")
    if answer.lower() == 'no' or answer.lower() == 'n':
        exit(0)
if not 0.05 < args.relevant_ruleset_ratio < 0.5:
    print("It is recommended to use a relevant ruleset ratio between 0.05 and 0.5. Are you sure you want to continue?")
    answer = input("yes/no (y)")
    if answer.lower() == 'no' or answer.lower() == 'n':
        exit(0)

start_time = time.time()

shutil.rmtree('tmp', ignore_errors=True)
os.mkdir("tmp")

if args.debug:
    print("Running in debug mode")

print("loading password list...", end='')
password_list = read_list_from_file(args.pwlist)
print("done.")
password_suggestions = []
model_0_baseWord_count = None

for model_name in args.models:
    print("loading model " + model_name)
    model = gensim.models.KeyedVectors.load_word2vec_format(base_models_path + model_name, binary=False, )
    all_base_words, rules = analyze_pws_with_model(password_list, model)
    if model_0_baseWord_count is None:
        model_0_baseWord_count = len(all_base_words)
        model_relevance_multiplier = 1
    else:
        model_relevance_multiplier = len(all_base_words) / model_0_baseWord_count
    all_rules_and_counts = list(Counter(rules).most_common())
    all_base_words_and_counts = list(Counter(all_base_words).most_common())
    if args.debug:
        print("DEBUG: I found " + str(len(all_base_words)) + " ("
              + str(len(all_base_words_and_counts)) + " individual) basewords")
    final_rules_and_counts = all_rules_and_counts[:int(len(all_rules_and_counts) * args.relevant_ruleset_ratio) + 1]
    # count_basewords: number of different basewords in total
    total_needed_basewords = int(args.list_length / len(final_rules_and_counts)) + 1
    if args.debug:
        print("DEBUG: from total " + str(len(all_rules_and_counts)) + " individual rules, I will use the most common "
              + str(len(final_rules_and_counts)))
    new_basewords_per_word = int((int(total_needed_basewords / len(all_base_words_and_counts)) * 7) / len(args.models))
    longest_baseword_seen = max([len(x) for x in all_base_words])
    if args.debug:
        print("DEBUG: I need " + str(total_needed_basewords) + ", so each bw needs to find "
              + str(new_basewords_per_word) + ' more.')
    print("creating new base word suggestions")
    epsilon = None
    if args.mode == 'epsilon':
        epsilon = get_epsilon_threshold(int(new_basewords_per_word), model,
                                        all_base_words_and_counts, args.restrict_vocab)
    expanded_base_words_and_counts = create_word_suggestions(all_base_words_and_counts, model, new_basewords_per_word,
                                                             args.mode, args.restrict_vocab, longest_baseword_seen,
                                                             epsi=epsilon)
    expanded_base_words_scored = score_and_sort_words(expanded_base_words_and_counts)
    f_base_word_suggestions_scored = expanded_base_words_scored[:int(total_needed_basewords * 1.2)]

    # fürs paper: speichere die Basiswörter mit wert.
    with open('basewords.txt', 'a') as file:
        for tuple in f_base_word_suggestions_scored:
            file.write(tuple[0]+', '+str(tuple[1])+'\n')

    if args.debug:
        print("DEBUG: After cropping, there are " + str(len(f_base_word_suggestions_scored)) + " base words left.")

    f_rules_scored = get_scored_rules(final_rules_and_counts)

    print("constructing new password suggestions...")
    with progressbar.ProgressBar(max_value=len(f_rules_scored)) as bar:
        # iterate over all rules
        for i in range(len(f_rules_scored)):
            bar.update(i)
            for w_suggestion_w_score in f_base_word_suggestions_scored:
                pw_score = calculate_password_score(f_rules_scored[i][1], w_suggestion_w_score[1],
                                                    model_relevance_multiplier, float(args.semantic_rating))
                candidate = create_password_from_hashcat_rule(w_suggestion_w_score[0], f_rules_scored[i][0])
                suggestion_and_score = (candidate, pw_score)
                password_suggestions.append(suggestion_and_score)

password_suggestions.sort(key=lambda x: x[1], reverse=True)
# delete multiple candidates
containing_doubles = password_suggestions
tmp = set()
password_suggestions = []
for p in containing_doubles:
    if not p[0] in tmp:
        tmp.add(p[0])
        password_suggestions.append(p)

if args.debug:
    print("DEBUG: I found new password suggestions: " + str(password_suggestions[:10])
          + " (" + str(len(password_suggestions)) + ")")

password_suggestions = password_suggestions[:args.list_length]
print("writing results to file "+str(args.out)+"...")
result = list(zip(*password_suggestions))[0]
write_list_to_file(result, args.out)
print("done in "+str(datetime.timedelta(seconds=round(time.time()-start_time))))
os.remove('analysis.word')
os.remove('analysis.rule')
shutil.rmtree('tmp', ignore_errors=True)
