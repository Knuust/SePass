from tqdm import tqdm 
import pickle

train_test_list_path = 'data/'
password_list_path = 'suggestions/'
test_file = 'test.txt'
pass_list_files = []

def load_password_list(file_name):
    file = open(file_name, "r", encoding='latin1')
    password_list = [line.rstrip() for line in file.readlines()]
    return password_list

def guess_percentage(test_list, suggestion_list):
    return len(set(suggestion_list).intersection(set(test_list))) / len(set(test_list))

def compare_differences(test_list, suggestion_list_1, suggestion_list_2):
    intersection1 = set(suggestion_list_1).intersection(set(test_list))
    intersection2 = set(suggestion_list_2).intersection(set(test_list))
    difference1 = intersection1.difference(intersection2)
    difference2 = intersection2.difference(intersection1)
    return difference1, difference2, len(difference1), len(difference2)

def evaluate_hits(suggestions, test):
    hits = []
    hits_sum = 0
    test = set(test)
    for i in tqdm(range(len(suggestions))):
        candidate = suggestions[i]
        if candidate in test:
            hits_sum += 1
        hits.append(hits_sum / len(test))
    return hits

def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

test_set = load_password_list(train_test_list_path + test_file)

hit_percentages = []
hit_percentages_over_guesses = []

for p in range(len(pass_list_files)):
    print('loading Password Suggestion List')
    print(pass_list_files[p])
    suggestions_file = load_password_list(password_list_path + pass_list_files[p])
    print(len(suggestions_file))
        
    print('Removing Duplicates')
    suggestions_list = remove_duplicates(suggestions_file)
    print(len(suggestions_list))

    print('Cutting Lists')
    if len(suggestions_list) > 50000000: 
        suggestions_list = suggestions_list[:50000000]

    hit_percentage = guess_percentage(test_set, suggestions_list)
    hit_list = evaluate_hits(suggestions_list, test_set)

    pickle.dump(hit_percentage, open(pass_list_files[p][:-4] + '_hitp.pkl', 'wb'))
    pickle.dump(hit_list, open(pass_list_files[p][:-4] + '_hitlist.pkl', 'wb'))

    hit_percentages.append(hit_percentage)
    hit_percentages_over_guesses.append(hit_list)