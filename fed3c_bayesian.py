import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from scipy.io import arff
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import logging
from joblib import Parallel, delayed
from scipy import stats

args = type('', (), {})()
args.num_clients = 10
args.max_centroids = None
args.n_calls = 50
args.k = 1
args.experiment_name = "k1_rice_reference_experiment"

if args.experiment_name is None:
    exp_name = f"clients_{args.num_clients}_opt_{args.n_calls}_k_{args.k}"
    if args.max_centroids is not None:
        exp_name += f"_maxcentroid_{args.max_centroids}"
    args.experiment_name = exp_name

RESULTS_ROOT = "results"
RESULTS_DIR = os.path.join(RESULTS_ROOT, args.experiment_name)

os.makedirs(RESULTS_DIR, exist_ok=True)

log_file_path = os.path.join(RESULTS_DIR, 'rice_bayesian.log')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])

logging.info(f"===== EXPERIMENT CONFIGURATION =====")
logging.info(f"Experiment Name: {args.experiment_name}")
logging.info(f"Number of Clients: {args.num_clients}")
logging.info(f"Optimization Calls: {args.n_calls}")
logging.info(f"KNN n_neighbors (k): {args.k}")
logging.info(f"Max Centroids: {'Half of training data' if args.max_centroids is None else args.max_centroids}")
logging.info(f"Results Directory: {RESULTS_DIR}")

data_path = 'rice/Rice_Cammeo_Osmancik.arff'
data, meta = arff.loadarff(data_path)
data_df = pd.DataFrame(data)

label_encoder = LabelEncoder()
data_df['Class'] = label_encoder.fit_transform(data_df['Class'].astype(str))

X = data_df.drop(columns=['Class'])
y = data_df['Class']

kf = KFold(n_splits=5, shuffle=True, random_state=42)
total_data = len(data_df)
train_data_per_fold = total_data * (4/5)

if args.max_centroids is not None:
    max_centroids = args.max_centroids
else:
    max_centroids = int(train_data_per_fold / 2)

logging.info(f"Calculated Max Centroids: {max_centroids}")

space = [Integer(10, max_centroids, name='total_centroids')]

best_accuracy = float('inf')
best_centroids = None
best_accuracies = []
best_precisions = []
best_recalls = []
best_f1s = []
best_times = []
best_client_times = []
best_comm_costs = []
best_y_test = None
best_predictions = None

all_centroids = []
all_accuracies = []
all_precisions = []
all_recalls = []
all_f1s = []
all_times = []
all_client_times = []
all_comm_costs = []

def log_details(split_data, centroids_per_split):
    for i, (data, centroids) in enumerate(zip(split_data, centroids_per_split)):
        logging.info(f"Client {i+1}: Data count = {len(data)}, Number of assigned centroids = {int(centroids)}")

def distribute_centroids(n_data, total_centroids):
    proportions = n_data / np.sum(n_data)
    floored_centroids = np.floor(proportions * total_centroids).astype(int)
    remainder = total_centroids - np.sum(floored_centroids)

    while remainder > 0:
        idx = np.argmax(proportions - floored_centroids / total_centroids)
        floored_centroids[idx] += 1
        remainder -= 1

    return floored_centroids

def calculate_communication_cost(centroids_per_split, feature_dim):
    bytes_per_centroid = feature_dim * 8
    total_bytes = sum(centroids_per_split) * bytes_per_centroid
    return total_bytes / 1024

def process_fold(train_index, test_index, total_centroids, X, y):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X.iloc[train_index])
    X_test_scaled = scaler.transform(X.iloc[test_index])
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    start_time = time.time()

    num_splits = args.num_clients
    split_train_data = np.array_split(X_train_scaled, num_splits)
    split_train_labels = np.array_split(y_train, num_splits)
    data_counts = np.array([len(data) for data in split_train_data])
    centroids_per_split = distribute_centroids(data_counts, total_centroids)

    center_labels = {}
    for split_data, split_labels, centroids_count in zip(split_train_data, split_train_labels, centroids_per_split):
        unique_labels, counts = np.unique(split_labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts / counts.sum()))
        centroids_per_class = {label: max(1, int(count * centroids_count))
                              for label, count in label_counts.items()}

        total_class_centroids = sum(centroids_per_class.values())
        if total_class_centroids > centroids_count:
            scale_factor = centroids_count / total_class_centroids
            centroids_per_class = {label: max(1, int(count * scale_factor))
                                  for label, count in centroids_per_class.items()}

        for label in unique_labels:
            class_data = split_data[split_labels == label]
            if centroids_per_class[label] > 0 and len(class_data) > 0:
                kmeans = KMeans(n_clusters=min(len(class_data), centroids_per_class[label]), random_state=42)
                kmeans.fit(class_data)
                if label in center_labels:
                    center_labels[label].extend(kmeans.cluster_centers_)
                else:
                    center_labels[label] = list(kmeans.cluster_centers_)

    final_centers = [center for sublist in center_labels.values() for center in sublist]
    final_labels = [label for label, centers in center_labels.items() for center in centers]

    knn = KNeighborsClassifier(n_neighbors=args.k)
    knn.fit(final_centers, final_labels)
    predictions = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    end_time = time.time()
    exec_time = end_time - start_time

    feature_dim = X.shape[1]
    comm_cost = calculate_communication_cost(centroids_per_split, feature_dim)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time': exec_time,
        'comm_cost': comm_cost,
        'y_test': y_test,
        'predictions': predictions
    }

@use_named_args(space)
def objective_sequential(total_centroids):
    logging.info(f"Total number of centroids tried = {total_centroids}")

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_times = []
    fold_client_times = []
    fold_comm_costs = []

    for train_index, test_index in kf.split(X):
        fold_start_time = time.time()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X.iloc[train_index])
        X_test_scaled = scaler.transform(X.iloc[test_index])
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        num_splits = args.num_clients
        split_train_data = np.array_split(X_train_scaled, num_splits)
        split_train_labels = np.array_split(y_train, num_splits)
        data_counts = np.array([len(data) for data in split_train_data])
        centroids_per_split = distribute_centroids(data_counts, total_centroids)

        center_labels = {}
        client_times = []

        for split_data, split_labels, centroids_count in zip(split_train_data, split_train_labels, centroids_per_split):
            client_start_time = time.time()

            unique_labels, counts = np.unique(split_labels, return_counts=True)
            label_counts = dict(zip(unique_labels, counts / counts.sum()))
            centroids_per_class = {label: max(1, int(count * centroids_count))
                                  for label, count in label_counts.items()}

            total_class_centroids = sum(centroids_per_class.values())
            if total_class_centroids > centroids_count:
                scale_factor = centroids_count / total_class_centroids
                centroids_per_class = {label: max(1, int(count * scale_factor))
                                      for label, count in centroids_per_class.items()}

            for label in unique_labels:
                class_data = split_data[split_labels == label]
                if centroids_per_class[label] > 0 and len(class_data) > 0:
                    kmeans = KMeans(n_clusters=min(len(class_data), centroids_per_class[label]), random_state=42)
                    kmeans.fit(class_data)
                    if label in center_labels:
                        center_labels[label].extend(kmeans.cluster_centers_)
                    else:
                        center_labels[label] = list(kmeans.cluster_centers_)

            client_end_time = time.time()
            client_times.append(client_end_time - client_start_time)

        final_centers = [center for sublist in center_labels.values() for center in sublist]
        final_labels = [label for label, centers in center_labels.items() for center in centers]

        knn = KNeighborsClassifier(n_neighbors=args.k)
        knn.fit(final_centers, final_labels)
        predictions = knn.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')

        fold_end_time = time.time()
        fold_exec_time = fold_end_time - fold_start_time

        feature_dim = X.shape[1]
        comm_cost = calculate_communication_cost(centroids_per_split, feature_dim)

        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)
        fold_times.append(fold_exec_time)
        fold_client_times.append(client_times)
        fold_comm_costs.append(comm_cost)

    mean_accuracy = np.mean(fold_accuracies)
    mean_precision = np.mean(fold_precisions)
    mean_recall = np.mean(fold_recalls)
    mean_f1 = np.mean(fold_f1s)
    mean_time = np.mean(fold_times)
    mean_client_time = np.mean([np.mean(client_times) for client_times in fold_client_times])
    mean_comm_cost = np.mean(fold_comm_costs)

    global best_accuracy, best_centroids, best_accuracies, best_precisions, best_recalls, best_f1s, best_times, best_client_times, best_comm_costs, best_y_test, best_predictions

    all_centroids.append(total_centroids)
    all_accuracies.append(mean_accuracy)
    all_precisions.append(mean_precision)
    all_recalls.append(mean_recall)
    all_f1s.append(mean_f1)
    all_times.append(mean_time)
    all_client_times.append(mean_client_time)
    all_comm_costs.append(mean_comm_cost)

    if -mean_accuracy < best_accuracy:
        best_accuracy = -mean_accuracy
        best_centroids = total_centroids
        best_accuracies = fold_accuracies
        best_precisions = fold_precisions
        best_recalls = fold_recalls
        best_f1s = fold_f1s
        best_times = fold_times
        best_client_times = [np.mean(client_times) for client_times in fold_client_times]
        best_comm_costs = fold_comm_costs

        # Test
        if len(fold_accuracies) > 0:
            best_fold_idx = np.argmax(fold_accuracies)
            best_fold_indices = list(kf.split(X))[best_fold_idx]
            test_indices = best_fold_indices[1]
            best_y_test = y.iloc[test_indices]

            best_X_test = X.iloc[test_indices]
            best_X_test_scaled = scaler.transform(best_X_test)
            best_predictions = knn.predict(best_X_test_scaled)

    logging.info(f"Centroids: {total_centroids}, Acc: {mean_accuracy:.7f}, Prec: {mean_precision:.7f}, " +
                 f"Rec: {mean_recall:.7f}, F1: {mean_f1:.7f}, Time (fold): {mean_time:.2f}s, " +
                 f"Time (client): {mean_client_time:.7f}s, Comm: {mean_comm_cost:.2f}KB")

    return -mean_accuracy

def evaluate_robustness(best_model, X_test, y_test, noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5], n_bootstrap=100):
    results = []

    n_samples = len(y_test)

    original_pred = best_model.predict(X_test)
    original_acc = accuracy_score(y_test, original_pred)
    original_f1 = f1_score(y_test, original_pred, average='weighted')
    results.append(('No Noise', original_acc, original_f1))

    bootstrap_original_acc = []
    bootstrap_original_f1 = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_test[indices]
        y_boot = y_test.iloc[indices] if hasattr(y_test, 'iloc') else y_test[indices]

        boot_pred = best_model.predict(X_boot)
        bootstrap_original_acc.append(accuracy_score(y_boot, boot_pred))
        bootstrap_original_f1.append(f1_score(y_boot, boot_pred, average='weighted'))

    p_values_acc = []
    p_values_f1 = []
    bootstrap_acc_all = []
    bootstrap_f1_all = []

    for noise in noise_levels:
        bootstrap_acc = []
        bootstrap_f1 = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_test[indices].copy()
            y_boot = y_test.iloc[indices] if hasattr(y_test, 'iloc') else y_test[indices]

            noise_matrix = np.random.normal(0, noise, X_boot.shape)
            X_boot += noise_matrix

            noisy_pred = best_model.predict(X_boot)
            noisy_acc = accuracy_score(y_boot, noisy_pred)
            noisy_f1 = f1_score(y_boot, noisy_pred, average='weighted')

            bootstrap_acc.append(noisy_acc)
            bootstrap_f1.append(noisy_f1)

        mean_acc = np.mean(bootstrap_acc)
        mean_f1 = np.mean(bootstrap_f1)

        t_stat_acc, p_val_acc = stats.ttest_rel(bootstrap_original_acc, bootstrap_acc)
        t_stat_f1, p_val_f1 = stats.ttest_rel(bootstrap_original_f1, bootstrap_f1)

        results.append((f'Noise {noise:.2f}', mean_acc, mean_f1))
        p_values_acc.append(p_val_acc)
        p_values_f1.append(p_val_f1)
        bootstrap_acc_all.append(bootstrap_acc)
        bootstrap_f1_all.append(bootstrap_f1)

    plt.figure(figsize=(12, 7))
    noise_labels = [r[0] for r in results]
    acc_values = [r[1] for r in results]
    f1_values = [r[2] for r in results]

    x = np.arange(len(noise_labels))
    width = 0.35

    plt.subplot(1, 2, 1)
    bars_acc = plt.bar(x - width/2, acc_values, width, label='Accuracy')
    bars_f1 = plt.bar(x + width/2, f1_values, width, label='F1 Score')
    plt.xlabel('Noise Level')
    plt.ylabel('Score')
    plt.title('Model Robustness to Noise')
    plt.xticks(x, noise_labels, rotation=45)
    plt.legend()

    for i in range(len(p_values_acc)):
        if i > 0:
            if p_values_acc[i-1] < 0.001:
                plt.text(i - width/2, acc_values[i] - 0.02, '***', ha='center')
            elif p_values_acc[i-1] < 0.01:
                plt.text(i - width/2, acc_values[i] - 0.02, '**', ha='center')
            elif p_values_acc[i-1] < 0.05:
                plt.text(i - width/2, acc_values[i] - 0.02, '*', ha='center')

            if p_values_f1[i-1] < 0.001:
                plt.text(i + width/2, f1_values[i] - 0.02, '***', ha='center')
            elif p_values_f1[i-1] < 0.01:
                plt.text(i + width/2, f1_values[i] - 0.02, '**', ha='center')
            elif p_values_f1[i-1] < 0.05:
                plt.text(i + width/2, f1_values[i] - 0.02, '*', ha='center')

    plt.subplot(1, 2, 2)
    bars_acc = plt.bar(x - width/2, acc_values, width, label='Accuracy')
    bars_f1 = plt.bar(x + width/2, f1_values, width, label='F1 Score')
    plt.xlabel('Noise Level')
    plt.ylabel('Score')
    plt.title('Zoomed View (0.85-1.0 range) with p-values')
    plt.xticks(x, noise_labels, rotation=45)
    plt.ylim(0.85, 1.0)

    for i in range(len(p_values_acc)):
        if i > 0:
            if p_values_acc[i-1] < 0.001:
                plt.text(i - width/2, acc_values[i] + 0.005, '***', ha='center', fontsize=10)
            elif p_values_acc[i-1] < 0.01:
                plt.text(i - width/2, acc_values[i] + 0.005, '**', ha='center', fontsize=10)
            elif p_values_acc[i-1] < 0.05:
                plt.text(i - width/2, acc_values[i] + 0.005, '*', ha='center', fontsize=10)

            if p_values_f1[i-1] < 0.001:
                plt.text(i + width/2, f1_values[i] + 0.005, '***', ha='center', fontsize=10)
            elif p_values_f1[i-1] < 0.01:
                plt.text(i + width/2, f1_values[i] + 0.005, '**', ha='center', fontsize=10)
            elif p_values_f1[i-1] < 0.05:
                plt.text(i + width/2, f1_values[i] + 0.005, '*', ha='center', fontsize=10)

    for i, (acc, f1) in enumerate(zip(acc_values, f1_values)):
        plt.text(i - width/2, acc - 0.01, f'{acc:.7f}', ha='center', va='bottom', fontsize=8, rotation=90)
        plt.text(i + width/2, f1 - 0.01, f'{f1:.7f}', ha='center', va='bottom', fontsize=8, rotation=90)

    plt.figtext(0.5, 0.01, "* p<0.05, ** p<0.01, *** p<0.001", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'rice_noise_robustness.png'))
    plt.close()

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    boxplot_data_acc = [bootstrap_original_acc] + bootstrap_acc_all
    plt.boxplot(boxplot_data_acc, labels=noise_labels, showfliers=False)
    plt.title('Distribution of Accuracy Scores Across Noise Levels')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(2, 1, 2)
    boxplot_data_f1 = [bootstrap_original_f1] + bootstrap_f1_all
    plt.boxplot(boxplot_data_f1, labels=noise_labels, showfliers=False)
    plt.title('Distribution of F1 Scores Across Noise Levels')
    plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'rice_noise_distribution.png'))
    plt.close()

    logging.info("\n===== NOISE ROBUSTNESS RESULTS WITH P-VALUES =====")
    logging.info(f"{'Noise Level':<12} {'Accuracy':<15} {'F1 Score':<15} {'p-val (Acc)':<12} {'p-val (F1)':<12}")

    logging.info(f"{noise_labels[0]:<12} {acc_values[0]:.7f}      {f1_values[0]:.7f}      -            -")

    for i in range(1, len(noise_labels)):
        logging.info(f"{noise_labels[i]:<12} {acc_values[i]:.7f}      {f1_values[i]:.7f}      {p_values_acc[i-1]:.2e}     {p_values_f1[i-1]:.2e}")

    return results, p_values_acc, p_values_f1

def analyze_results():
    acc_mean, acc_std = np.mean(best_accuracies), np.std(best_accuracies)
    prec_mean, prec_std = np.mean(best_precisions), np.std(best_precisions)
    rec_mean, rec_std = np.mean(best_recalls), np.std(best_recalls)
    f1_mean, f1_std = np.mean(best_f1s), np.std(best_f1s)
    time_mean, time_std = np.mean(best_times), np.std(best_times)
    client_time_mean, client_time_std = np.mean(best_client_times), np.std(best_client_times)
    comm_cost_mean, comm_cost_std = np.mean(best_comm_costs), np.std(best_comm_costs)

    logging.info("\n===== BEST CONFIGURATION RESULTS (CLASS-BALANCED CENTROID ALLOCATION) =====")
    logging.info(f"Best number of centroids: {best_centroids}")
    logging.info(f"Number of clients: {args.num_clients}")
    logging.info(f"KNN n_neighbors (k): {args.k}")
    logging.info(f"Centroid allocation strategy: Class-balanced (Allocation to each class proportional to its data size)")
    logging.info(f"Accuracy: {acc_mean:.7f} ± {acc_std:.7f}")
    logging.info(f"Precision: {prec_mean:.7f} ± {prec_std:.7f}")
    logging.info(f"Recall: {rec_mean:.7f} ± {rec_std:.7f}")
    logging.info(f"F1 Score: {f1_mean:.7f} ± {f1_std:.7f}")
    logging.info(f"Average time per fold: {time_mean:.7f}s ± {time_std:.7f}s")
    logging.info(f"Average time per client: {client_time_mean:.7f}s ± {client_time_std:.7f}s")
    logging.info(f"Communication cost: {comm_cost_mean:.7f}KB ± {comm_cost_std:.7f}KB")

    if best_y_test is not None:
        unique_labels, counts = np.unique(best_y_test, return_counts=True)
        class_ratios = dict(zip(unique_labels, counts / counts.sum()))

        for label, ratio in class_ratios.items():
            class_name = label_encoder.inverse_transform([label])[0] if hasattr(label_encoder, 'inverse_transform') else f"Class {label}"
            logging.info(f"Class ratio ({class_name}): {ratio:.7f}")

    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [acc_mean, prec_mean, rec_mean, f1_mean]
    errors = [acc_std, prec_std, rec_std, f1_std]

    bars = plt.bar(metrics, values, yerr=errors, capsize=10)
    plt.title(f'Performance Metrics with Error Bars (Centroids: {best_centroids}, Clients: {args.num_clients})')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)

    for bar, val, err in zip(bars, values, errors):
        plt.text(bar.get_x() + bar.get_width()/2., val + err + 0.02,
                f'{val:.7f}±{err:.7f}', ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'rice_performance_metrics.png'))
    plt.close()

    if best_y_test is not None and best_predictions is not None:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(best_y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix (Centroids: {best_centroids}, Clients: {args.num_clients})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(RESULTS_DIR, 'rice_confusion_matrix.png'))
        plt.close()

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.plot(range(len(all_centroids)), all_accuracies, 'b-', label='Accuracy')
    plt.plot(range(len(all_centroids)), all_f1s, 'g-', label='F1 Score')
    plt.axvline(x=all_centroids.index(best_centroids), color='r', linestyle='--', label=f'Best Centroids={best_centroids}')
    plt.title('Accuracy and F1 Score vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(2, 2, 2)
    plt.plot(range(len(all_centroids)), all_precisions, 'm-', label='Precision')
    plt.plot(range(len(all_centroids)), all_recalls, 'c-', label='Recall')
    plt.axvline(x=all_centroids.index(best_centroids), color='r', linestyle='--', label=f'Best Centroids={best_centroids}')
    plt.title('Precision and Recall vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(2, 2, 3)
    plt.plot(range(len(all_centroids)), all_times, 'k-', label='Fold Time')
    plt.plot(range(len(all_centroids)), all_client_times, 'y-', label='Client Time')
    plt.axvline(x=all_centroids.index(best_centroids), color='r', linestyle='--', label=f'Best Centroids={best_centroids}')
    plt.title('Execution Time vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(2, 2, 4)
    plt.plot(range(len(all_centroids)), all_comm_costs, 'r-', label='Comm Cost')
    plt.axvline(x=all_centroids.index(best_centroids), color='b', linestyle='--', label=f'Best Centroids={best_centroids}')
    plt.title('Communication Cost vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Communication Cost (KB)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'rice_optimization_progress.png'))
    plt.close()

    return {
        'acc_mean': acc_mean, 'acc_std': acc_std,
        'prec_mean': prec_mean, 'prec_std': prec_std,
        'rec_mean': rec_mean, 'rec_std': rec_std,
        'f1_mean': f1_mean, 'f1_std': f1_std,
        'time_mean': time_mean, 'time_std': time_std,
        'client_time_mean': client_time_mean, 'client_time_std': client_time_std,
        'comm_cost_mean': comm_cost_mean, 'comm_cost_std': comm_cost_std
    }

def main():
    start_time = time.time()

    logging.info(f"Bayesian optimization starting (without parallelization, with class-balanced centroid allocation)...")
    logging.info(f"Parameter configuration: Number of Clients={args.num_clients}, Max Centroid={max_centroids}, Optimization Iterations={args.n_calls}")

    res_gp = gp_minimize(objective_sequential, space, n_calls=args.n_calls, random_state=0)

    total_time = time.time() - start_time
    logging.info(f"Optimization completed. Total time: {total_time:.7f} seconds")

    metrics = analyze_results()

    logging.info("Running noise robustness test...")

    best_fold_idx = np.argmax(best_accuracies)
    train_indices = list(kf.split(X))[best_fold_idx][0]
    test_indices = list(kf.split(X))[best_fold_idx][1]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X.iloc[train_indices])
    X_test_scaled = scaler.transform(X.iloc[test_indices])
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    client_start_time = time.time()

    num_splits = args.num_clients
    split_train_data = np.array_split(X_train_scaled, num_splits)
    split_train_labels = np.array_split(y_train, num_splits)
    data_counts = np.array([len(data) for data in split_train_data])
    centroids_per_split = distribute_centroids(data_counts, best_centroids)

    client_times = []

    center_labels = {}
    for split_data, split_labels, centroids_count in zip(split_train_data, split_train_labels, centroids_per_split):
        client_iter_start = time.time()

        unique_labels, counts = np.unique(split_labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts / counts.sum()))
        centroids_per_class = {label: max(1, int(count * centroids_count))
                              for label, count in label_counts.items()}

        total_class_centroids = sum(centroids_per_class.values())
        if total_class_centroids > centroids_count:
            scale_factor = centroids_count / total_class_centroids
            centroids_per_class = {label: max(1, int(count * scale_factor))
                                   for label, count in centroids_per_class.items()}

        logging.info(f"Centroid distribution: {centroids_per_class}")

        for label in unique_labels:
            class_data = split_data[split_labels == label]
            if centroids_per_class[label] > 0 and len(class_data) > 0:
                kmeans = KMeans(n_clusters=min(len(class_data), centroids_per_class[label]), random_state=42)
                kmeans.fit(class_data)
                if label in center_labels:
                    center_labels[label].extend(kmeans.cluster_centers_)
                else:
                    center_labels[label] = list(kmeans.cluster_centers_)

        client_iter_end = time.time()
        client_times.append(client_iter_end - client_iter_start)

    client_end_time = time.time()
    client_total_time = client_end_time - client_start_time
    client_avg_time = np.mean(client_times)

    feature_dim = X.shape[1]
    comm_cost = calculate_communication_cost(centroids_per_split, feature_dim)

    logging.info(f"Client training times: Total={client_total_time:.7f}s, Average={client_avg_time:.7f}s")
    logging.info(f"Per client communication cost: {comm_cost/num_splits:.7f}KB")

    final_centers = [center for sublist in center_labels.values() for center in sublist]
    final_labels = [label for label, centers in center_labels.items() for center in centers]

    best_model = KNeighborsClassifier(n_neighbors=args.k)
    best_model.fit(final_centers, final_labels)

    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    n_bootstrap = 100

    noise_results, p_values_acc, p_values_f1 = evaluate_robustness(
        best_model, X_test_scaled, y_test, noise_levels, n_bootstrap)

    summary_file = os.path.join(RESULTS_DIR, 'experiment_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"===== EXPERIMENT SUMMARY =====\n")
        f.write(f"Experiment Name: {args.experiment_name}\n")
        f.write(f"Number of Clients: {args.num_clients}\n")
        f.write(f"KNN n_neighbors (k): {args.k}\n")
        f.write(f"Max Centroids: {max_centroids}\n")
        f.write(f"Optimization Iterations: {args.n_calls}\n")
        f.write(f"Total Execution Time: {total_time:.2f} seconds\n\n")

        f.write(f"===== BEST RESULTS =====\n")
        f.write(f"Best Centroids: {best_centroids}\n")
        f.write(f"Accuracy: {metrics['acc_mean']:.7f} ± {metrics['acc_std']:.7f}\n")
        f.write(f"Precision: {metrics['prec_mean']:.7f} ± {metrics['prec_std']:.7f}\n")
        f.write(f"Recall: {metrics['rec_mean']:.7f} ± {metrics['rec_std']:.7f}\n")
        f.write(f"F1 Score: {metrics['f1_mean']:.7f} ± {metrics['f1_std']:.7f}\n")
        f.write(f"Client Time: {metrics['client_time_mean']:.7f}s ± {metrics['client_time_std']:.7f}s\n")
        f.write(f"Communication Cost: {metrics['comm_cost_mean']:.7f}KB ± {metrics['comm_cost_std']:.7f}KB\n")

    logging.info(f"Process completed. Results were saved to the {RESULTS_DIR} folder.")
    logging.info(f"Experiment summary: {summary_file}")

if __name__ == "__main__":
    main()