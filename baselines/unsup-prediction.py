import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, RobertaModel
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import cosine_similarity
import os
import configparser
from sklearn.metrics import ndcg_score

tokenizer_codebert = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model_codebert = RobertaModel.from_pretrained("microsoft/codebert-base")

class Instance:
    def __init__(self, task_id, problem_text, solution_list):
        self.task_id = task_id
        self.problem_text = problem_text
        self.solution_list = solution_list
        self.problem_emb = None
        self.list_solution_emb = []
        self.score = []
        self.individual_scores = []  # To store individual predicted scores with code_id
        # self.ap = ap
        # self.pass_at_3 = pass_at_3

    def get_embeddings(self, text):
        if not text:
            raise ValueError(f"The text input is empty. Task ID: {self.task_id}. Please provide valid text for tokenization.")
        inputs = tokenizer_codebert(text, return_tensors='pt', padding=True, truncation=True)
        outputs = model_codebert(**inputs)
        embeddings = outputs.last_hidden_state.squeeze().detach().numpy()
        return embeddings

    def compute_embeddings(self):
        self.problem_emb = self.get_embeddings(self.problem_text)
        for solution in self.solution_list:
            try:
                solution_emb = self.get_embeddings(solution)
                self.list_solution_emb.append(solution_emb)
            except ValueError as e:
                print(e)

    def calc_tls_sim(self):
        '''
        Calculate TLS similarity scores between each solution and all others,
        and then average those scores to get a score for each solution.
        '''
        solution_scores = [[] for _ in range(len(self.list_solution_emb))]
        for i in range(len(self.list_solution_emb)):
            for j in range(len(self.list_solution_emb)):
                if i != j:
                    solution_embeddings_1 = self.list_solution_emb[i]
                    solution_embeddings_2 = self.list_solution_emb[j]
                    sim_matrix = cosine_similarity(solution_embeddings_1[1:], solution_embeddings_2[1:])
                    max_similarities_1 = np.max(sim_matrix, axis=1)
                    max_similarities_2 = np.max(sim_matrix, axis=0)
                    sum_max_similarities_1 = np.sum(max_similarities_1)
                    sum_max_similarities_2 = np.sum(max_similarities_2)
                    score_1 = sum_max_similarities_1 / solution_embeddings_1.shape[0]
                    score_2 = sum_max_similarities_2 / solution_embeddings_2.shape[0]
                    f1_score = (2 * score_1 * score_2) / (score_1 + score_2)
                    solution_scores[i].append(f1_score)
        # Average the scores for each solution
        self.score = [np.mean(scores) for scores in solution_scores]

    def calc_els_sim(self):
        '''
        Calculate CLS similarity scores between each solution and all others,
        and then average those scores to get a score for each solution.
        '''
        solution_scores = [[] for _ in range(len(self.list_solution_emb))]
        for i in range(len(self.list_solution_emb)):
            for j in range(len(self.list_solution_emb)):
                if i != j:
                    cls_embedding1 = self.list_solution_emb[i][0]
                    cls_embedding2 = self.list_solution_emb[j][0]
                    cls_similarity = cosine_similarity([cls_embedding1], [cls_embedding2])
                    solution_scores[i].append(cls_similarity[0][0])
        # Average the scores for each solution
        self.score = [np.mean(scores) for scores in solution_scores]

def initialize_llm_model():
    model_name = "codellama/CodeLlama-7b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return tokenizer, model, device

def get_first_token_logits(problem, code, tokenizer, model, device, instruction_template):
    prompt = instruction_template + f"\nProblem Description: \n{problem}\nCode snippet: \n {code}\n \nFunctionally Correct:"
    pos_token = tokenizer.tokenize("yes")
    pos_token_to_id = tokenizer.convert_tokens_to_ids(pos_token[0])
    neg_token = tokenizer.tokenize("no")
    neg_token_to_id = tokenizer.convert_tokens_to_ids(neg_token[0])

    tokenized_input = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(**tokenized_input, max_new_tokens=1, temperature=0.1, return_dict_in_generate=True, output_scores=True)
    logits = torch.nn.functional.softmax(outputs['scores'][0].detach().cpu(), dim=-1)
    logits_yes = logits[0, pos_token_to_id].item()
    logits_no = logits[0, neg_token_to_id].item()
    score = np.exp(logits_yes) / (np.exp(logits_yes) + np.exp(logits_no))
    return score, logits_yes, logits_no

def llm_prediction(instances, instruction_template):
    tokenizer, model, device = initialize_llm_model()
    log_file = open("./output/llm_logits_output.txt", "w")
    for inst in instances:
        problem = inst.problem_text
        for idx, solution in enumerate(inst.solution_list):
            score, logits_yes, logits_no = get_first_token_logits(problem, solution, tokenizer, model, device, instruction_template)
            inst.score.append(score)
            inst.individual_scores.append({
                "task_id": inst.task_id,
                "code_id": f"code_{idx + 1}",  # Store code_id as code_1, code_2, etc.
                "score": score
            })
            log_file.write(f"Task ID: {inst.task_id}, Logits Yes: {logits_yes}, Logits No: {logits_no}\n")
    log_file.close()

def load_data(input_file):
    instances = []
    with open(input_file, "r") as file:
        data = [json.loads(line) for line in file]

    for item in data:
        task_id = item["task_id"]
        prompt = item["prompt"]
        solutions = [item[f"code_{i+1}"] for i in range(10) if f"code_{i+1}" in item]
        if len(solutions) < 2:
            print(f"Skipping Task ID {task_id} due to insufficient number of solutions.")
            continue
        # ap = item['nDCG@10']
        # pass_at_3 = item['pass@3']
        # instance = Instance(task_id, prompt, solutions, ap, pass_at_3)
        instance = Instance(task_id, prompt, solutions)
        instances.append(instance)
    return instances

def calc_score(instances, task, instruction_template=None):
    if task == "llm":
        llm_prediction(instances, instruction_template)
    else:
        for inst in instances:
            inst.compute_embeddings()
            if task == "tls":
                inst.calc_tls_sim()
            elif task == "els":
                inst.calc_els_sim()

def calc_avg_or_var(values, aggr):
    if aggr == 'avg':
        return [np.mean(val) for val in values]
    elif aggr == 'var':
        return [1 - np.var(val) for val in values]

def eval_corr(gt_metrics, aggr_score, instances):
    gt_metrics = np.array(gt_metrics)
    aggr_score = np.array(aggr_score)
    
    if np.isnan(gt_metrics).any() or np.isnan(aggr_score).any():
        nan_tasks = [inst.task_id for inst, score in zip(instances, aggr_score) if np.isnan(score).any()]
        raise ValueError(f"Input contains NaN values. Please check the input data. Tasks with NaN values: {nan_tasks}")

    tau, _ = kendalltau(gt_metrics, aggr_score)
    rank_score1 = compute_ndcg(gt_metrics, aggr_score)
    rank_score2 = compute_ndcg(aggr_score, gt_metrics)
    ndcg = (rank_score1 + rank_score2) / 2
    return tau, ndcg

def compute_relevance_values(ranks):
    N = len(ranks)
    relevance_values = N - np.array(ranks) + 1
    return relevance_values

def compute_ndcg(predicted_scores, ground_truth_scores, k=None):
    ground_truth_ranks = np.argsort(np.argsort(ground_truth_scores)[::-1]) + 1
    ground_truth_relevance = compute_relevance_values(ground_truth_ranks).reshape(1, -1)
    predicted_scores = np.array(predicted_scores).reshape(1, -1)
    ndcg = ndcg_score(ground_truth_relevance, predicted_scores)
    return ndcg

def main(task, aggr, input_file, instruction_template_file=None):
    instances = load_data(input_file)

    instruction_template = ""
    if task == "llm" and instruction_template_file:
        with open(instruction_template_file, 'r') as file:
            instruction_template = file.read()

    calc_score(instances, task, instruction_template)
    values = [inst.score for inst in instances]

    input_file_name = os.path.basename(input_file).replace('.jsonl', '')
    # individual_scores_file = f"./output/{input_file_name}-{task}-{aggr}-individual-scores.jsonl"
    # output_file = f"./output/{input_file_name}-{task}-{aggr}.jsonl"
    individual_scores_file = os.path.join(output_dir, f"{input_file_name}-{task}-{aggr}-individual-scores.jsonl")
    output_file = os.path.join(output_dir, f"{input_file_name}-{task}-{aggr}.jsonl")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write individual scores to JSONL file before averaging
    with open(individual_scores_file, "w") as ind_file:
        for inst in instances:
            for idx, score in enumerate(inst.score):
                entry = {
                    "task_id": inst.task_id,
                    "code_id": f"code_{idx + 1}",
                    "score": float(score)  # Ensure score is serialized as a float
                }
                ind_file.write(json.dumps(entry) + "\n")

    # Calculate aggregated scores (e.g., average or variance)
    aggr_score = calc_avg_or_var(values, aggr)
    print('aggr score', aggr_score)

    # avg_precision_metrics = [inst.ap for inst in instances]
    # print('avg_precision_metrics', avg_precision_metrics)
    # pass_at_3_metrics = [inst.pass_at_3 for inst in instances]

    # tau_avg_precision, ndcg = eval_corr(avg_precision_metrics, aggr_score, instances)

    # Write aggregated scores to JSONL file
    with open(output_file, "w") as out_file:
        out_file.write(f"task_id\t{aggr}\n")
        for inst, score in zip(instances, aggr_score):
            task_id = inst.task_id
            out_file.write(f"{task_id}\t{score}\n")

    # print(f"\nKendall's Tau with nDCG@10: {tau_avg_precision}\n")
    # print(f" new nDCG average: {ndcg} \n")
    print(f"Results written to {output_file}")
    print(f"Individual scores written to {individual_scores_file}")

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.properties')

    task = config['TASK']['task']
    aggr = config['AGGREGATOR']['aggr']
    input_file = config['DATA']['input_file']
    output_dir = config['DATA']['output_dir']
    instruction_template_file = config['DATA'].get('instruction_template_file', None)

    if task == "llm" and not instruction_template_file:
        raise ValueError("instruction_template_file is required when task is 'llm'")

    main(task, aggr, input_file, instruction_template_file)
