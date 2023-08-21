import json
from transformers import AutoTokenizer
from argparse import ArgumentParser


INSTRUCTION = "Generate an event temporal graph in DOT format based on the given document."


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="/mnt/Data2/shared/alpaca_weights", help="model name")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.path)

    with open("data/NYT_des_test.json", 'r') as f:
        data = json.loads(f.read())

    output_list = []
    for doc_id in data:
        document = data[doc_id]['document']
        if document[:len("LEAD : ")] == "LEAD : ":
            document = document[len("LEAD : "):]

        tk_list = tokenizer(f"{INSTRUCTION} \n{document}", truncation=False)
        if len(tk_list['input_ids']) > 2048:
            print(doc_id, len(tk_list['input_ids']))
            continue
        else:
            output_list.append(
                {
                    "instruction": INSTRUCTION,
                    "input": document,
                    "output": data[doc_id]['target']
                }
            )

    print(len(output_list), len(data), len(output_list)/len(data))
    with open("data/NYT_des_test_alpaca.json", 'w') as f:
        f.write(json.dumps(output_list, indent=4))
    

    with open("data/NYT_des_train.json", 'r') as f:
        data = json.loads(f.read())

    output_list = []
    for doc_id in data:
        document = data[doc_id]['document']
        if document[:len("LEAD : ")] == "LEAD : ":
            document = document[len("LEAD : "):]

        tk_list = tokenizer(f"{INSTRUCTION} \n{document}", truncation=False)
        if len(tk_list['input_ids']) > 2048:
            print(doc_id, len(tk_list['input_ids']))
            continue
        else:
            output_list.append(
                {
                    "instruction": INSTRUCTION,
                    "input": document,
                    "output": data[doc_id]['target']
                }
            )

    print(len(output_list), len(data), len(output_list)/len(data))
    with open("data/NYT_des_train_alpaca.json", 'w') as f:
        f.write(json.dumps(output_list, indent=4))