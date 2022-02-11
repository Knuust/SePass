import pickle
from tqdm import tqdm

def load_password_list(file_name):
    file = open(file_name, "r", encoding='latin1')
    password_list = [line.rstrip() for line in file.readlines()]
    return password_list

def guess_percentage(test_list, suggestion_list):
    return len(set(suggestion_list).intersection(set(test_list))) / len(set(test_list))

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

def flatten(t):
    return [item for sublist in t for item in sublist]

train_test_list_path = 'wörterbücher/'
password_list_path = 'generated_password_lists/'

pass_list_files = ['spg_with_numbers_50M.txt', 'pcfg_50M.txt']

test_set = load_password_list(train_test_list_path + 'test.txt')
spg_suggestions = load_password_list(password_list_path + pass_list_files[0])
pcfg_suggestions = remove_duplicates(load_password_list(password_list_path + pass_list_files[1]))

print('Loaded')
combined_list = list(zip(pcfg_suggestions[:50000000], spg_suggestions))
print('Zipped')
full_combined_list = flatten(combined_list)
print('flattened')
reduced_combined_list = remove_duplicates(full_combined_list)
print('reduced')
combined_hits = evaluate_hits(reduced_combined_list[:50000000], test_set)

pickle.dump(guess_percentage(test_set, reduced_combined_list[:50000000]), open('combined_percentage.pkl', 'wb'))
pickle.dump(combined_hits, open('combined_hitlist.pkl', 'wb'))
