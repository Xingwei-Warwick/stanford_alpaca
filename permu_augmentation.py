from itertools import permutations
import random
import json


def add_permutation(train_list, upper_limit=5):
    # maximum permutation is the upper_limit
    new_train_list = []
    #tk_len_list = []
    for doc_dict in train_list:
        count = 0
        origin_target = doc_dict['output']
        #tk_len_list.append(len(nltk.word_tokenize(origin_target)))
        edge_list = origin_target[14:-1].split('\n')
        for permu in permutations(edge_list):
            if count >= upper_limit:
                break
            new_train_list.append({
                "instruction": doc_dict['instruction'],
                "output": "strict graph {\n" + '\n'.join(permu) + "\n}",
                "input": doc_dict['input']
            })
            count += 1

    return new_train_list


if __name__ == "__main__":
    with open("data/NYT_des_train_alpaca.json", 'r') as f:
        data = json.loads(f.read())

    new_data = add_permutation(data, upper_limit=5)
    print(len(new_data), len(data), len(new_data)/len(data))
    with open("data/NYT_des_train_alpaca_permu5.json", 'w') as f:
        f.write(json.dumps(new_data, indent=4))

