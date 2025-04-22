import torch
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score
from collections import defaultdict
import os
import sys

#best prompt
instruction_template = """
You are an experienced software engineer. Your task is to check the functional correctness of code for the given problem statement. Generate 'yes' if the code is functionally correct (i.e., code meets the problem's requirements) otherwise generate 'no'. 
Refer to the examples below on functional correctness for code evaluation.

{examples}

Evaluate the following code snippet.

Problem Description: 
{problem}

Code Snippet: 
{code}

Functionally Correct:

"""


def load_codebert():
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = RobertaModel.from_pretrained('microsoft/codebert-base')
    return tokenizer, model

def load_codellama(model_name="codellama/CodeLlama-7b-Instruct-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return tokenizer, model, device

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line.strip()) for line in file]
    return pd.DataFrame(data)

class ProblemInstance:
    def __init__(self, task_id, prompt, code_key, code, result, prompt_embedding, code_embedding):
        self.task_id = task_id
        self.prompt = prompt
        self.code_key = code_key
        self.code = code
        self.result = result
        self.prompt_embedding = prompt_embedding
        self.code_embedding = code_embedding


def compute_embeddings(df, tokenizer, model):
    prompt_embeddings = {}
    code_embeddings = {}

    for idx, row in df.iterrows():
        prompt = row['prompt']
        if prompt not in prompt_embeddings:
            prompt_embeddings[prompt] = get_cls_embedding(prompt, tokenizer, model)
        for i in range(1, 11):
            code_key = f'code_{i}'
            if code_key in row and not pd.isna(row[code_key]):
                code = row[code_key]
                if code not in code_embeddings:
                    code_embeddings[code] = get_cls_embedding(code, tokenizer, model)
    return prompt_embeddings, code_embeddings


def get_cls_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding

def transform_data(df, prompt_embeddings, code_embeddings):
    instances = []
    for idx, row in df.iterrows():
        task_id = row['task_id']
        prompt = row['prompt']
        for i in range(1, 11):
            code_key = f'code_{i}'
            if code_key in row and not pd.isna(row[code_key]):
                code = row[code_key]
                instances.append(ProblemInstance(task_id, prompt, code_key, code, row['results'][i - 1], prompt_embeddings[prompt], code_embeddings[code]))
    return instances

def save_instances_by_result(instances, output_dir):
    correct_instances = [inst for inst in instances if inst.result == 1]
    incorrect_instances = [inst for inst in instances if inst.result == 0]

    correct_file = os.path.join(output_dir, 'correct_instances.jsonl')
    incorrect_file = os.path.join(output_dir, 'incorrect_instances.jsonl')

    save_instances_to_file(correct_instances, correct_file)
    save_instances_to_file(incorrect_instances, incorrect_file)

def save_instances_to_file(instances, file_path):
    with open(file_path, 'w') as f:
        for inst in instances:
            json.dump(inst.__dict__, f, default=convert_ndarray_to_list)
            f.write('\n')

def calculate_top_k_similarities(test_instance, train_instances, k, p, filter_unique=False):
    test_prompt_embedding = test_instance.prompt_embedding.reshape(1, -1)
    test_code_embedding = test_instance.code_embedding.reshape(1, -1)

    prompt_similarities = cosine_similarity(test_prompt_embedding, np.array([inst.prompt_embedding for inst in train_instances])).flatten()
    code_similarities = cosine_similarity(test_code_embedding, np.array([inst.code_embedding for inst in train_instances])).flatten()

    total_similarities = p * prompt_similarities + (1 - p) * code_similarities

    similarities = list(zip(total_similarities, train_instances))
    sorted_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)

    if filter_unique:
        unique_prob_statements = set()
        top_k_unique = []
        
        for similarity, instance in sorted_similarities:
            if instance.prompt not in unique_prob_statements:
                top_k_unique.append((similarity, instance))
                unique_prob_statements.add(instance.prompt)
            if len(top_k_unique) >= k:
                break

        return top_k_unique
    else:
        return sorted_similarities[:k]

def calculate_similarities(test_instance, train_instances, k, p, filter_unique, balanced):
    if balanced:
        correct_instances = [inst for inst in train_instances if inst.result == 1]
        incorrect_instances = [inst for inst in train_instances if inst.result == 0]
        top_k_correct = calculate_top_k_similarities(test_instance, correct_instances, k, p, filter_unique)
        top_k_incorrect = calculate_top_k_similarities(test_instance, incorrect_instances, k, p, filter_unique)
        return top_k_correct, top_k_incorrect
    else:
        top_2k_instances = calculate_top_k_similarities(test_instance, train_instances, 2 * k, p, filter_unique)
        return top_2k_instances, []  # Return top 2k examples directly

def get_examples(test_task_id, test_code_key, top_k_correct, top_k_incorrect):
    examples = ""
    for i, (similarity, train_instance) in enumerate(top_k_correct + top_k_incorrect, 1):
        correctness = "yes" if train_instance.result == 1 else "no"
        examples += f"Example {i}:\n Problem Description: {train_instance.prompt}\n Code snippet: \n {train_instance.code}\n  \n Functionally Correct: {correctness}\n\n"
    return examples

def get_first_token_logits(problem, code, examples, tokenizer, model, device):
    prompt = instruction_template.format(problem=problem, code=code, examples=examples)
    print(prompt)
    pos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("yes"))[0]
    neg_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("no"))[0]
    
    tokenized_input = tokenizer(prompt, return_tensors='pt').to(device)
    try:
        outputs = model.generate(**tokenized_input, max_new_tokens=1,  return_dict_in_generate=True, output_scores=True)
        probs = torch.nn.functional.softmax(outputs['scores'][0].detach().cpu(), dim=-1)
        return probs[0, pos_token_id].item(), probs[0, neg_token_id].item()
    
    except torch.cuda.OutOfMemoryError:
        print(f"Skipping instance due to CUDA OOM error.")
        torch.cuda.empty_cache()
        return None, None  # Return None if OOM error occurs
    
def logits_to_probabilities(logits_yes, logits_no):
    if logits_yes is None or logits_no is None:
        return 0.0 
    score = np.exp(logits_yes) / (np.exp(logits_yes) + np.exp(logits_no))
    return score

def convert_ndarray_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data



def split_train_dev(training_data, split_ratio=0.1):
    unique_task_ids = training_data['task_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_task_ids)
    split_idx = int(len(unique_task_ids) * (1 - split_ratio))
    train_task_ids = set(unique_task_ids[:split_idx])
    dev_task_ids = set(unique_task_ids[split_idx:])
    train_data = training_data[training_data['task_id'].isin(train_task_ids)]
    dev_data = training_data[training_data['task_id'].isin(dev_task_ids)]
    return train_data, dev_data



def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    codebert_tokenizer, codebert_model = load_codebert()
    llama_tokenizer, llama_model, device = load_codellama()

    training_data = load_data(config['input_train_file'])
    training_data, _ = split_train_dev(training_data)
    testing_data = load_data(config['input_test_file'])

    prompt_embeddings, code_embeddings = compute_embeddings(pd.concat([training_data, testing_data]), codebert_tokenizer, codebert_model)

    training_instances = transform_data(training_data, prompt_embeddings, code_embeddings)
    testing_instances = transform_data(testing_data, prompt_embeddings, code_embeddings)

    # balanced data if specified
    if config['balanced']:
        save_instances_by_result(training_instances, config['output_dir'])

    ndcg_results = {}
    filter_unique = config.get('filter', True)

    for k in config['k_values']:
        # Reinitialize lists for each value of k
        global_relevance = []
        global_predicted_scores = []
        task_local_ndcgs = defaultdict(list)

        predicted_results = []

        for test_instance in testing_instances:
            task_id = test_instance.task_id
            top_k_correct, top_k_incorrect = calculate_similarities(test_instance, training_instances, k, config['p'], filter_unique, config['balanced'])
            examples = get_examples(test_instance.task_id, test_instance.code_key, top_k_correct, top_k_incorrect)

            logits_yes, logits_no = get_first_token_logits(test_instance.prompt, test_instance.code, examples, llama_tokenizer, llama_model, device)
            score = logits_to_probabilities(logits_yes, logits_no)
            print('score', score)

            if score is not None:
                predicted_results.append({
                    'task_id': task_id,
                    'code_key': test_instance.code_key,
                    'logits_yes': logits_yes,
                    'logits_no': logits_no,
                    'score': score
                })
            else:
                predicted_results.append({
                    'task_id': task_id,
                    'code_key': test_instance.code_key,
                    'logits_yes': None,
                    'logits_no': None,
                    'score': 0.0
                })

            global_relevance.append(test_instance.result)
            global_predicted_scores.append(score if score is not None else 0.0)

            task_local_ndcgs[task_id].append((test_instance.result, score if score is not None else 0.0))

        # Truncate global relevance to match predicted score length
        if len(global_relevance) > len(global_predicted_scores):
            global_relevance = global_relevance[:len(global_predicted_scores)]

        save_similarity_results(predicted_results, config['output_dir'], k)
        avg_task_scores = calculate_average_scores(predicted_results)
        save_average_scores(avg_task_scores, config['output_dir'], k)
        global_ndcg = ndcg_score([global_relevance], [global_predicted_scores])

        local_ndcg_values = []
        for task_id, relevance_scores_and_scores in task_local_ndcgs.items():
            relevance_scores, predicted_scores = zip(*relevance_scores_and_scores)

            # Truncate relevance scores to match predicted scores
            relevance_scores = relevance_scores[:len(predicted_scores)]

            if len(relevance_scores) > 1:
                local_ndcg = ndcg_score([relevance_scores], [predicted_scores])
            else:
                print(f"Task ID with less than two items: {task_id}")
                local_ndcg = 0.0

            local_ndcg_values.append(local_ndcg)

        local_avg_ndcg = np.mean(local_ndcg_values) if local_ndcg_values else 0.0

        ndcg_results[k] = {
            'global_nDCG': global_ndcg,
            'local_avg_nDCG': local_avg_ndcg,
        }

    display_results(ndcg_results)


def save_similarity_results(predicted_results, output_dir, k):
    similarity_output_file = os.path.join(output_dir, f'similarity_results_k_{k}.jsonl')
    with open(similarity_output_file, 'w') as f:
        for result in predicted_results:
            result = {key: convert_ndarray_to_list(value) for key, value in result.items()}
            json.dump(result, f)
            f.write('\n')

def calculate_average_scores(predicted_results):
    task_scores = defaultdict(list)
    for result in predicted_results:
        task_scores[result['task_id']].append(result['score'])
    return {task_id: np.mean(scores) for task_id, scores in task_scores.items()}

def save_average_scores(avg_task_scores, output_dir, k):
    avg_scores_output_file = os.path.join(output_dir, f'avg_scores_k_{k}.jsonl')
    with open(avg_scores_output_file, 'w') as f:
        for task_id, score in avg_task_scores.items():
            output_dict = {'task_id': task_id, 'score': convert_ndarray_to_list(score)}
            json.dump(output_dict, f)
            f.write('\n')

def display_results(ndcg_results):
    for k, results in ndcg_results.items():
        print(f"Results for k={k}:")
        print(f"Global nDCG: {results['global_nDCG']}")
        print(f"Local Average nDCG@10: {results['local_avg_nDCG']}")
        print("\n------------------------\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    main(config_file)
