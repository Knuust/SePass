test_list_path = 'final/wörterbücher/'
password_list_path = 'final/generated_password_lists/'

def load_password_list(file_name):
    file = open(file_name + '.txt', "r", encoding='latin1')
    password_list = [line.rstrip() for line in file.readlines()]
    return password_list

test_set = load_password_list(test_list_path + 'test')
spg_suggestions = load_password_list(password_list_path + 'spg_with_numbers_50M')
pcfg_suggestions = load_password_list(password_list_path + 'pcfg_50M')

methods = {}
methods['spg'] = set(spg_suggestions).intersection(test_set)
methods['pcfg'] = set(pcfg_suggestions).intersection(test_set)

results = {}
for key in methods.keys():
    results[key] = len(methods[key]) / len(test_set)

def find_exclusive_passwords(method):
     return len(set(methods[method]).difference(set().union(*[value for key, value in methods.items() if key != method]))) / len(test_set)

exclusive = {}

for key in methods.keys():
    exclusive[key] = find_exclusive_passwords(key)

import matplotlib.pyplot as plt

plt.bar(results.keys(), results.values(), edgecolor=['orange', 'b', 'm', 'y', 'g', 'c'], color='None')
plt.bar(results.keys(), exclusive.values(), color=['orange', 'b', 'm', 'y', 'g', 'c'])

plt.xlabel('Methods')
plt.ylabel('% of testset guessed')
plt.savefig('exclusive_words_real.png')