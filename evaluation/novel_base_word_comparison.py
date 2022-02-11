import pickle
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

test_list_path = 'final/wörterbücher/'
password_list_path = 'final/generated_password_lists/'

def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def load_password_list(file_name):
    file = open(file_name + '.txt', "r", encoding='latin1')
    password_list = [line.rstrip() for line in file.readlines()]
    return password_list

test_set = load_password_list(test_list_path + 'test')
train_words = pickle.load(open('train_words_real.pkl', 'rb'))
test_words = pickle.load(open('test_words_real.pkl', 'rb'))

novel_test_words = set(test_words).difference(train_words)

test_novel_passwords = []

low_test_words = [word.lower() for word in novel_test_words]

for password in test_set:
    flag = False
    for word in low_test_words:
        if word in password.lower():
            flag = True
            break
    if flag:
        test_novel_passwords.append(password)  

spg_suggestions = load_password_list(password_list_path + 'spg_with_numbers_50M')
pcfg_suggestions = remove_duplicates(load_password_list(password_list_path + 'pcfg_50M'))[:50000000]

methods = {}
methods['spg'] = set(spg_suggestions).intersection(test_novel_passwords)
methods['pcfg'] = set(pcfg_suggestions).intersection(test_novel_passwords)

results = {}
for key in methods.keys():
    results[key] = len(methods[key]) 

print(results)

def find_exclusive_passwords(method):
     return len(set(methods[method]).difference(set().union(*[value for key, value in methods.items() if key != method]))) 
exclusive = {}

for key in methods.keys():
    exclusive[key] = find_exclusive_passwords(key)

print(exclusive)

set1 = set(methods['spg'])
set2 = set(methods['pcfg'])
set3 = set(test_novel_passwords)

venn3([set1, set2, set3], ('SeePass', 'PCFG', 'Test'))

plt.savefig('sepass_pcfg_test_venn_real.png')

