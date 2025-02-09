from deepcase_copy.preprocessing.preprocessor import Preprocessor
import os
import torch
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepcase_copy.context_builder.loss import LabelSmoothing
from deepcase_copy.context_builder.context_builder import ContextBuilder
from deepcase_copy.interpreter.interpreter import Interpreter
from deepcase_copy.context_builder.embedding import EmbeddingOneHot
from sklearn.model_selection import StratifiedKFold

def disable_dropout(m):
    if isinstance(m, torch.nn.Dropout):
        m.p = 0.0

def to_cuda(item):
    if torch.cuda.is_available():
        return item.to('cuda')
    return item

def to_one_hot(t):
    return to_cuda(one_hot_encoder(t).clone().detach())

def to_cuda_tensor(item):
    return to_cuda(torch.tensor(item))

def get_unique_indices_per_row(context_chosen, events_chosen, indices_chosen):
    indices_list = []
    indices_list_set = set()
    for row in tqdm(indices_chosen):
        curr = context_chosen[row].tolist()
        curr.append(events_chosen[row].item())
        curr = tuple(curr)
        if curr in indices_list_set:
            continue
        indices_list.append(row)
        indices_list_set.add(curr)
    return indices_list

def get_unique_indices_per_row_or_file(context_chosen, events_chosen, indices_chosen, f_name):
    if os.path.exists(f_name):
        return torch.load(f_name)
    t = get_unique_indices_per_row(context_chosen, events_chosen, indices_chosen)
    torch.save(t, f_name)
    return t

def get_data_split(context_chosen, events_chosen, labels_chosen):
    train_test_split_list = []
    skf = StratifiedKFold(n_splits=5)
    stratification = None
    if DATASET == "ided":
        stratification = [f"{l.item()}_{e.item()}" for l, e in zip(labels_chosen, events_chosen)]
    else:
        stratification = events_chosen
    for train_index_chosen, test_index_chosen in skf.split(range(len(stratification)), stratification):
        train_test_split_list.append(
            (train_index_chosen, get_unique_indices_per_row(context_chosen, events_chosen, test_index_chosen)))
    return train_test_split_list

def get_context_builder_interpreter(context_chosen, events_chosen, labels_chosen):
    context_builder_chosen = ContextBuilder(
        input_size=FEATURES,  # Number of input features to expect
        output_size=FEATURES,  # Same as input size
        hidden_size=128,  # Number of nodes in hidden layer, in paper we set this to 128
        max_length=SEQ_LEN,  # Length of the context, should be same as context in Preprocessor,
    )
    context_builder_chosen.fit(
        X=context_chosen,  # Context to train with
        y=events_chosen.reshape(-1, 1),  # Events to train with, note that these should be of shape=(n_events, 1)
        epochs=10,  # Number of epochs to train with
        batch_size=128,  # Number of samples in each training batch, in paper this was 128
        learning_rate=0.01,  # Learning rate to train with, in paper this was 0.01
        verbose=True,  # If True, prints progress,
        already_embedded=False
    )
    context_builder_chosen = to_cuda(context_builder_chosen)
    interpreter_chosen = Interpreter(
        context_builder=context_builder_chosen,  # ContextBuilder used to fit data
        features=FEATURES,  # Number of input features to expect, should be same as ContextBuilder
        eps=0.1,  # Epsilon value to use for DBSCAN clustering, in paper this was 0.1
        min_samples=5,  # Minimum number of samples to use for DBSCAN clustering, in paper this was SEQ_LEN=5
        threshold=PREDICT_THRESHOLD,
        # Confidence threshold used for determining if attention from the ContextBuilder can be used, in paper this was 0.2
    )
    interpreter_chosen.cluster(
        X=context_chosen,  # Context to train with
        y=events_chosen.reshape(-1, 1),  # Events to train with, note that these should be of shape=(n_events, 1)
        iterations=100,  # Number of iterations to use for attention query, in paper this was 100
        batch_size=1024,  # Batch size to use for attention query, used to limit CUDA memory usage
        verbose=True,  # If True, prints progress/
        already_embedded=False
    )
    if DATASET == "ided":
        scores = interpreter_chosen.score_clusters(
            scores=labels_chosen.cpu(),
            # Labels used to compute score (either as loaded by Preprocessor, or put your own labels here)
            strategy="max",  # Strategy to use for scoring (one of "max", "min", "avg")
            NO_SCORE=-1,  # Any sequence with this score will be ignored in the strategy.
            # If assigned a cluster, the sequence will inherit the cluster score.
            # If the sequence is not present in a cluster, it will receive a score of NO_SCORE.
        )
        interpreter_chosen.score(
            scores=scores,  # Scores to assign to sequences
            verbose=True,  # If True, prints progress
        )
    context_builder_chosen.apply(disable_dropout)
    return context_builder_chosen, interpreter_chosen

SEQ_LEN = 10
MAX_ITER = 20
DATASET = "ided"
PICKING = "rand"
BATCH_SIZE = 100
PREDICT_THRESHOLD = 0.2
FEATURES = 30 if DATASET == "hdfs" else 100
one_hot_encoder = EmbeddingOneHot(FEATURES)
# PICKING = "rand" if DATASET == "hdfs" else "first"

def to_trace(o):
    return torch.argmax(o, dim=-1).tolist()[0]

def max_to_one_first(tensor):
    max_indices = torch.argmax(tensor, dim=-1, keepdim=True)
    result = torch.zeros_like(tensor)
    result.scatter_(-1, max_indices, 1.0)
    return result

def max_to_one_rand(tensor):
    max_values = torch.max(tensor.squeeze(0), dim=1).values
    comparison = tensor == max_values.unsqueeze(1)
    tensor_indices = torch.nonzero(comparison, as_tuple=True)
    max_indices = [[] for _ in range(tensor.size(1))]
    for row, col in zip(tensor_indices[1].tolist(), tensor_indices[2].tolist()):
        max_indices[row].append(col)
    random_max_indices = to_cuda_tensor([random.choice(sublist) for sublist in max_indices]).unsqueeze(0)
    return to_one_hot(random_max_indices)

def max_to_one(tensor):
    return max_to_one_rand(tensor) if PICKING == "rand" else max_to_one_first(tensor)

def get_correct_prediction_for_list(interpreter_chosen, context_chosen, events_chosen, attention_query=False):
    _, mask = interpreter_chosen.attended_context(
        X=to_one_hot(context_chosen),
        y=to_cuda(events_chosen).reshape(-1, 1),
        iterations=100 if attention_query else 0
    )
    return torch.where(~mask)[0], torch.where(mask)[0]

def get_changes_list(start, final):
    perturbations_made = []
    for i, (s, f) in enumerate(zip(start, final)):
        if s != f:
            if isinstance(f, int):
                perturbations_made.append((i, f))
            else:
                perturbations_made.append((i, f.item()))
    return perturbations_made

def get_perturbations(context_builder_chosen, interpreter_chosen, context_chosen, events_chosen, focus, attention_query=False):
    get_shortcuts_func = get_minimum_change_for_perturbation_aq if attention_query else get_minimum_change_for_perturbation_no_aq
    _, pred_true = get_correct_prediction_for_list(interpreter_chosen, context_chosen, events_chosen, attention_query=attention_query)
    focused_pred_true = [x for x in pred_true if x in focus]
    pick_list = []
    perturbed_indices_main = []
    perturbed_iterations_main = []
    states = [len(context_chosen) - len(focused_pred_true), 0, 0]
    for current_trace_num in tqdm(focused_pred_true):
        con, e = context_chosen[[current_trace_num]].unsqueeze(0), events_chosen[[current_trace_num]].unsqueeze(0)
        perturbed_result, perturb_iterations = bim_attack(context_builder_chosen, interpreter_chosen, con, e, attention_query=attention_query)
        if perturbed_result is not None:
            states[1] += 1
            pick_list.append(get_shortcuts_func(interpreter_chosen, perturbed_result, con, e, attention_query=attention_query))
            perturbed_indices_main.append(current_trace_num)
            perturbed_iterations_main.append(perturb_iterations)
        else:
            states[2] += 1
    print(f"{states=}")
    pick_list = [t if t.dim() == 1 else t.squeeze(0) for t in pick_list]
    return to_cuda(torch.stack(pick_list)), to_cuda_tensor(perturbed_indices_main), to_cuda_tensor(states), to_cuda_tensor(perturbed_iterations_main)

def get_possible_combinations(perturbations_made):
    subsets = []
    for r_index in range(1, len(perturbations_made)):
        subsets.append(itertools.combinations(perturbations_made, r_index))
    return [list(subset) for subset in subsets]

def get_minimum_change_for_perturbation_aq(interpreter_chosen, perturbed_chosen, context_chosen, events_chosen, attention_query=False):
    changes_list = get_changes_list(context_chosen[0], perturbed_chosen[0])
    if len(changes_list) > 1:
        combinations = get_possible_combinations(changes_list)
        left, right, curr = 0, len(combinations), None
        while left < right:
            mid = (left + right) // 2
            trace_combinations = []
            for combination in combinations[mid]:
                copy = context_chosen[0].clone().detach()
                for index_of_change, value_of_change in combination:
                    copy[index_of_change] = value_of_change
                trace_combinations.append(copy)
            for i in range(0, len(trace_combinations), BATCH_SIZE):
                trace_combinations_batched = trace_combinations[i:i + BATCH_SIZE]
                mask_indices = get_correct_prediction_for_list(
                    interpreter_chosen,
                    to_cuda(torch.stack(trace_combinations_batched)),
                    to_cuda(torch.full((len(trace_combinations_batched), 1), events_chosen.item())),
                    attention_query
                )[0]
                if len(mask_indices) > 0:
                    curr = trace_combinations_batched[mask_indices[0]]
                    right = mid
                    break
            else:
                left = mid + 1
        if curr is None:
            return perturbed_chosen
        return curr
    return perturbed_chosen

def get_minimum_change_for_perturbation_no_aq(interpreter_chosen, perturbed_chosen, context_chosen, events_chosen, attention_query=False):
    changes_list = get_changes_list(context_chosen[0], perturbed_chosen[0])
    if len(changes_list) > 1:
        for group in get_possible_combinations(changes_list):
            trace_combinations = []
            for combination in group:
                copy = context_chosen[0].clone().detach()
                for index_of_change, value_of_change in combination:
                    copy[index_of_change] = value_of_change
                trace_combinations.append(copy)
            mask_indices = get_correct_prediction_for_list(
                interpreter_chosen,
                to_cuda(torch.stack(trace_combinations)),
                to_cuda(torch.full((len(trace_combinations), 1), events_chosen.item())),
                attention_query
            )[0]
            if len(mask_indices) > 0:
                return trace_combinations[mask_indices[0]]
    return perturbed_chosen

def get_matrix_perturb(context_chosen, perturb_chosen):
    return [len(get_changes_list(c, p)) for c, p in zip(context_chosen, perturb_chosen)]

def interpret_context(interpreter_chosen, context_passed, events_passed, attention_query=False):
    c = to_one_hot(context_passed)
    e = events_passed.reshape(-1, 1)
    return interpreter_chosen.predict(X=c, y=e, iterations=100 if attention_query else 0)

def get_combined(interpreter_chosen, context_chosen, events_chosen, perturbed_chosen, perturbed_indices_chosen, attention_query=False):
    context_test_copy = context_chosen.clone().detach()
    context_test_copy[perturbed_indices_chosen] = perturbed_chosen
    return interpret_context(interpreter_chosen, context_test_copy, events_chosen, attention_query=attention_query)

def bim_attack(context_builder_chosen, interpreter_chosen, context_chosen, event_chosen, attention_query=False):
    context_processed = to_one_hot(context_chosen)
    criterion = LabelSmoothing(context_builder_chosen.decoder_event.out.out_features, 0.1)
    for iteration in range(MAX_ITER):
        context_processed.requires_grad_(True)
        output = context_builder_chosen.predict(context_processed)
        if len(get_correct_prediction_for_list(interpreter_chosen, to_cuda_tensor(to_trace(context_processed)).unsqueeze(0), event_chosen, attention_query=attention_query)[1]) == 0:
            return to_cuda_tensor(to_trace(context_processed)).unsqueeze(0), iteration
        loss = criterion(output[0][0], event_chosen)
        context_processed.retain_grad()
        loss.backward(retain_graph=True)
        context_processed = max_to_one(context_processed + context_processed.grad.sign())
    return None, -1

def format_series(series):
    return pd.Series(series).value_counts().sort_index()

def format_confusion_matrix(y_true, y_pred):
    y_pred[y_pred == -3] = y_true[y_pred == -3]
    return format_series(y_pred)

def format_interpret_combined(interpreter_chosen, context_chosen, events_chosen, labels_chosen):
    df_combined = None
    if DATASET == "ided":
        df0 = format_series(labels_chosen.cpu())
        df1 = format_confusion_matrix(labels_chosen.cpu(), interpret_context(interpreter_chosen, context_chosen, events_chosen, attention_query=False))
        df2 = format_confusion_matrix(labels_chosen.cpu(), interpret_context(interpreter_chosen, context_chosen, events_chosen, attention_query=True))
        df_combined = pd.concat([df0, df1, df2], axis=1, join="outer").fillna(0)
        df_combined.columns = ["Base", "No AQ", "AQ"]
    else:
        df1 = format_series(interpret_context(interpreter_chosen, context_chosen, events_chosen, attention_query=False))
        df2 = format_series(interpret_context(interpreter_chosen, context_chosen, events_chosen, attention_query=True))
        df_combined = pd.concat([df1, df2], axis=1, join="outer").fillna(0)
        df_combined.columns = ["No AQ", "AQ"]
    df_combined = df_combined.astype(int)
    df_combined = df_combined.sort_index(ascending=True)
    return df_combined

def get_events_dist(interpreter_chosen, context_chosen, events_chosen, labels_chosen, mapping_chosen, indices_chosen, attention_query=False):
    _, mask = interpreter_chosen.attended_context(
        X=to_one_hot(context_chosen),
        y=events_chosen.unsqueeze(1),
        iterations=100 if attention_query else 0
    )
    targeted_indices = [i for i in range(len(context_chosen)) if i not in torch.where(~mask)[0].tolist()]
    events_series = pd.Series(events_chosen[indices_chosen].cpu()).value_counts().sort_values(ascending=False)
    events_df = events_series.reset_index()
    events_df.columns = ['label', 'count']
    if labels_chosen is not None:
        events_df["level"] = events_df["label"].apply(lambda label: list(set(labels_chosen[np.where(events_chosen.cpu() == label)[0]].tolist()))[0])
    events_df['print'] = events_df.apply(lambda row: f"\\colorcellpercentamount{{{row['count']}}}{{{torch.sum(events_chosen[targeted_indices] == row['label'])}}}", axis=1)
    events_df.drop(columns=['count'], inplace=True)
    if labels_chosen is not None:
        events_df["mapping"] = events_df["label"].apply(lambda label: mapping_chosen[label])
    events_df["label"] = events_df["label"].apply(lambda label: f"\\textbf{{{label}}}")
    return events_df

def format_events_dist(interpreter_chosen, context_chosen, events_chosen, labels_chosen, mapping_chosen, indices_chosen,indices_chosen_query):
    df1 = get_events_dist(interpreter_chosen, context_chosen, events_chosen, labels_chosen, mapping_chosen, indices_chosen, attention_query=False)
    df2 = get_events_dist(interpreter_chosen, context_chosen, events_chosen, labels_chosen, mapping_chosen, indices_chosen_query, attention_query=True)
    df_combined = pd.concat([df1, df2], axis=1, join="outer").fillna(0)
    return df_combined

def get_all_info(seq_chosen, threshold_chosen):
    global SEQ_LEN, PREDICT_THRESHOLD
    SEQ_LEN, PREDICT_THRESHOLD = seq_chosen, threshold_chosen
    print(f"Processing {SEQ_LEN=} {PREDICT_THRESHOLD=}")
    preprocessor = Preprocessor(length=SEQ_LEN, timeout=86400)
    context, events, labels, mapping = preprocessor.csv(f'alerts.csv', verbose=False) if DATASET == "ided" else preprocessor.text(f'hdfs_test_normal', verbose=True)
    for split_n, (train_indices, test_indices) in enumerate(get_data_split(context, events, labels)):
        print(len(train_indices), len(test_indices))
        context_test = to_cuda(context[test_indices])
        events_test = to_cuda(events[test_indices])
        labels_test = to_cuda(labels[test_indices]) if labels is not None else None
        print("Training")
        context_builder, interpreter = get_context_builder_interpreter(context_test, events_test, labels_test)
        with open(f"{DATASET}/{SEQ_LEN=}/{PREDICT_THRESHOLD=}/results-{split_n}.txt", "a+") as file:
            file.write(f"{format_interpret_combined(interpreter, context_test, events_test, labels_test)}\n")
            print("Perturbing No AQ")
            perturb_main, indices_main, result_main, iterations_main = get_perturbations(context_builder, interpreter, context_test, events_test, range(len(context_test)), attention_query=False)
            print("Perturbing AQ")
            perturb_main_q, indices_main_q, result_main_q, iterations_main_q = get_perturbations(context_builder, interpreter, context_test, events_test, indices_main, attention_query=True)
            print("Getting Results")
            file.write(f"{result_main}, {result_main_q}")
            print(format_combined_results(interpreter, context_test, events_test, labels_test, perturb_main, indices_main, perturb_main_q, indices_main_q))
            file.write(f"{format_combined_results(interpreter, context_test, events_test, labels_test, perturb_main, indices_main, perturb_main_q, indices_main_q)}\n")
            file.write(f"{format_iter_count(iterations_main.cpu(), iterations_main_q.cpu())}\n")
            file.write(f"{format_iter_count(get_matrix_perturb(context_test[indices_main], perturb_main), get_matrix_perturb(context_test[indices_main_q], perturb_main_q))}\n")
            file.write(f"{format_events_dist(interpreter, context_test, events_test, labels_test, mapping, indices_main, indices_main_q)}\n")
        print("Results Done")
    print("All jobs done")

def format_combined_results(interpreter_chosen, context_chosen, events_chosen, labels_chosen, perturbed_chosen, indices_chosen, perturbed_chosen_query, indices_chosen_query):
    if DATASET == "ided":
        df1 = format_confusion_matrix(labels_chosen.cpu(), get_combined(interpreter_chosen, context_chosen, events_chosen, perturbed_chosen, indices_chosen))
        df2 = format_confusion_matrix(labels_chosen.cpu(), get_combined(interpreter_chosen, context_chosen, events_chosen, perturbed_chosen, indices_chosen, attention_query=True))
        df3 = format_confusion_matrix(labels_chosen.cpu(), get_combined(interpreter_chosen, context_chosen, events_chosen, perturbed_chosen_query, indices_chosen_query, attention_query=True))
        df_combined = pd.concat([df1, df2, df3], axis=1, join="outer").fillna(0)
        df_combined = df_combined.astype(int)
        df_combined = df_combined.sort_index(ascending=True)
        df_combined.columns = ["No AQ", "No AQ on AQ", "AQ"]
        return df_combined
    else:
        return ""

def format_iter_count(count_no_aq, count_aq):
    df1 = format_series(count_no_aq)
    df2 = format_series(count_aq)
    df1_avg = pd.DataFrame(count_no_aq).mean().iloc[0]
    df2_avg = pd.DataFrame(count_aq).mean().iloc[0]
    df_combined = pd.concat([df1, df2], axis=1, join="outer").fillna(0)
    df_combined = df_combined.astype(int)
    df_combined.columns = ["No AQ", "AQ"]
    return df_combined, df1_avg.round(2), df2_avg.round(2)

if __name__ == "__main__":
    get_all_info(10, 0.2)
    # get_all_info(10, 0.6)
    # get_all_info(5, 0.2)
    # get_all_info(5, 0.6)
    # get_all_info(20, 0.2)
    # get_all_info(20, 0.6)
