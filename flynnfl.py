import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from scipy.io import arff
import logging
from joblib import Parallel, delayed
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('rice_flynn.log'), logging.StreamHandler()])


class FlyNN:
    def __init__(self, m=2000, s=10, rho=20, gamma=0.5, random_state=42):

        self.m = m
        self.s = s
        self.rho = rho
        self.gamma = gamma
        self.random_state = random_state
        self.M = None
        self.class_fbf = {}

    def _generate_lifting_matrix(self, d):
        np.random.seed(self.random_state)
        M = np.zeros((self.m, d), dtype=np.bool_)

        actual_s = min(self.s, d)

        for i in range(self.m):
            nonzero_indices = np.random.choice(d, actual_s, replace=False)
            M[i, nonzero_indices] = True

        return M

    def _flyhash(self, x):
        if self.M is None:
            raise ValueError("Lifting matrix M has not been initialized")

        if hasattr(x, 'values'):
            x = x.values

        projections = self.M.dot(x)

        actual_rho = min(self.rho, self.m)

        h = np.zeros(self.m, dtype=np.bool_)
        top_indices = np.argsort(projections)[-actual_rho:]
        h[top_indices] = True

        return h

    def fit(self, X, y):
        d = X.shape[1]
        self.M = self._generate_lifting_matrix(d)

        unique_classes = np.unique(y)

        for cls in unique_classes:
            self.class_fbf[cls] = np.ones(self.m, dtype=np.float32)

        for i in range(len(X)):
            if isinstance(X, (pd.DataFrame, pd.Series)):
                x = X.iloc[i]
            else:
                x = X[i]

            if isinstance(y, (pd.DataFrame, pd.Series)):
                label = y.iloc[i]
            else:
                label = y[i]

            h = self._flyhash(x)

            self.class_fbf[label][h] *= self.gamma

        return self

    def predict(self, X):
        if not self.class_fbf:
            raise ValueError("Model has not been trained")

        predictions = np.zeros(len(X), dtype=int)

        for i in range(len(X)):
            if isinstance(X, (pd.DataFrame, pd.Series)):
                x = X.iloc[i]
            else:
                x = X[i]

            h = self._flyhash(x)

            scores = {}
            for cls, fbf in self.class_fbf.items():
                scores[cls] = np.sum(fbf[h])

            predictions[i] = min(scores, key=scores.get)

        return predictions

def train_client_flynn(X_train, y_train, m, s, rho, gamma, random_state):
    client_model = FlyNN(m=m, s=s, rho=rho, gamma=gamma, random_state=random_state)
    client_model.fit(X_train, y_train)
    return client_model.M, client_model.class_fbf

def aggregate_flynn_models(client_models, gamma):
    M = client_models[0][0]

    all_classes = set()
    for _, class_fbf in client_models:
        all_classes.update(class_fbf.keys())

    aggregated_fbf = {}
    for cls in all_classes:
        log_sum = 0
        count = 0

        for _, class_fbf in client_models:
            if cls in class_fbf:
                log_sum += np.log(class_fbf[cls]) / np.log(gamma)
                count += 1

        if count > 0:
            aggregated_fbf[cls] = np.power(gamma, log_sum / count)

    return M, aggregated_fbf

def apply_differential_privacy(class_fbf, epsilon, samples, gamma):
    log_fbf = {}
    for cls, fbf in class_fbf.items():
        log_fbf[cls] = np.log(fbf) / np.log(gamma)

    dp_log_fbf = {}
    for cls, log_counts in log_fbf.items():
        dp_log_fbf[cls] = np.zeros_like(log_counts)

        probabilities = np.exp(log_counts / (4 * samples))
        probabilities = probabilities / np.sum(probabilities)

        selected_indices = np.random.choice(
            len(log_counts),
            size=min(samples, len(log_counts)),
            replace=False,
            p=probabilities
        )

        noise = np.random.laplace(0, 2 * samples / epsilon, size=len(selected_indices))
        for i, idx in enumerate(selected_indices):
            dp_log_fbf[cls][idx] = max(0, log_counts[idx] + noise[i])

    dp_fbf = {}
    for cls, dp_log_counts in dp_log_fbf.items():
        dp_fbf[cls] = np.power(gamma, dp_log_counts)

    return dp_fbf

def federated_train_flynn(split_data, split_labels, m, s, rho, gamma, random_state, is_dp=False, epsilon=1.0, dp_samples=100):
    client_models = []

    for client_data, client_labels in zip(split_data, split_labels):
        M, class_fbf = train_client_flynn(client_data, client_labels, m, s, rho, gamma, random_state)

        if is_dp:
            class_fbf = apply_differential_privacy(class_fbf, epsilon, dp_samples, gamma)

        client_models.append((M, class_fbf))

    M, aggregated_fbf = aggregate_flynn_models(client_models, gamma)

    model = FlyNN(m=m, s=s, rho=rho, gamma=gamma, random_state=random_state)
    model.M = M
    model.class_fbf = aggregated_fbf

    return model

def evaluate_flynn_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1, y_test, y_pred

def calculate_communication_cost(m, num_classes, feature_dim, clients):
    bytes_per_fbf = m * 4
    bytes_per_client = num_classes * bytes_per_fbf
    total_bytes = clients * bytes_per_client

    return total_bytes / 1024

space = [
    Integer(2000, 10000, name='m'),
    Integer(5, 50, name='s'),
    Integer(10, 100, name='rho'),
]

@use_named_args(space)
def objective_flynn(m, s, rho):
    logging.info(f"Evaluating FlyNN hyperparameters: m={m}, s={s}, rho={rho}")

    gamma = 0.5
    num_splits = 10

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
        y_train, y_test = y[train_index], y[test_index]

        X_train_scaled = np.array(X_train_scaled)
        X_test_scaled = np.array(X_test_scaled)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        split_train_data = np.array_split(X_train_scaled, num_splits)
        split_train_labels = np.array_split(y_train, num_splits)

        client_times = []

        client_start_time = time.time()

        flynn_model = federated_train_flynn(
            split_train_data,
            split_train_labels,
            m, s, rho, gamma,
            random_state=42,
            is_dp=False
        )

        client_end_time = time.time()
        client_times.append(client_end_time - client_start_time)

        accuracy, precision, recall, f1, _, _ = evaluate_flynn_model(flynn_model, X_test_scaled, y_test)

        fold_end_time = time.time()
        fold_exec_time = fold_end_time - fold_start_time

        num_classes = len(np.unique(y_train))
        feature_dim = X.shape[1]
        comm_cost = calculate_communication_cost(m, num_classes, feature_dim, num_splits)

        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)
        fold_times.append(fold_exec_time)
        fold_client_times.append(np.mean(client_times))
        fold_comm_costs.append(comm_cost)

    mean_accuracy = np.mean(fold_accuracies)
    mean_precision = np.mean(fold_precisions)
    mean_recall = np.mean(fold_recalls)
    mean_f1 = np.mean(fold_f1s)
    mean_time = np.mean(fold_times)
    mean_client_time = np.mean(fold_client_times)
    mean_comm_cost = np.mean(fold_comm_costs)

    all_hyperparams.append((m, s, rho))
    all_accuracies.append(mean_accuracy)
    all_precisions.append(mean_precision)
    all_recalls.append(mean_recall)
    all_f1s.append(mean_f1)
    all_times.append(mean_time)
    all_client_times.append(mean_client_time)
    all_comm_costs.append(mean_comm_cost)

    global best_accuracy, best_params, best_accuracies, best_precisions, best_recalls
    global best_f1s, best_times, best_client_times, best_comm_costs, best_y_test, best_predictions

    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_params = (m, s, rho)
        best_accuracies = fold_accuracies
        best_precisions = fold_precisions
        best_recalls = fold_recalls
        best_f1s = fold_f1s
        best_times = fold_times
        best_client_times = fold_client_times
        best_comm_costs = fold_comm_costs

    logging.info(f"FlyNN: m={m}, s={s}, rho={rho}, Acc: {mean_accuracy:.7f}, Prec: {mean_precision:.7f}, " +
                f"Rec: {mean_recall:.7f}, F1: {mean_f1:.7f}, Time: {mean_time:.2f}s, " +
                f"Client Time: {mean_client_time:.7f}s, Comm: {mean_comm_cost:.2f}KB")

    return -mean_accuracy


def evaluate_flynn_robustness(best_model, X_test, y_test, noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5], n_bootstrap=100):
    results = []

    X_test = np.array(X_test)
    y_test = np.array(y_test)

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
        y_boot = y_test[indices]

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
            y_boot = y_test[indices]

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

    visualize_robustness_results(results, p_values_acc, p_values_f1, bootstrap_original_acc, bootstrap_original_f1, bootstrap_acc_all, bootstrap_f1_all)

    return results, p_values_acc, p_values_f1

def visualize_robustness_results(results, p_values_acc, p_values_f1, bootstrap_original_acc, bootstrap_original_f1, bootstrap_acc_all, bootstrap_f1_all):
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
    plt.title('FlyNN Model Robustness to Noise')
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
    plt.savefig('flynn_noise_robustness.png')
    plt.close()

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    boxplot_data_acc = [bootstrap_original_acc] + bootstrap_acc_all
    plt.boxplot(boxplot_data_acc, labels=noise_labels, showfliers=False)
    plt.title('Distribution of Accuracy Scores Across Noise Levels (FlyNN)')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(2, 1, 2)
    boxplot_data_f1 = [bootstrap_original_f1] + bootstrap_f1_all
    plt.boxplot(boxplot_data_f1, labels=noise_labels, showfliers=False)
    plt.title('Distribution of F1 Scores Across Noise Levels (FlyNN)')
    plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('flynn_noise_distribution.png')
    plt.close()

def analyze_flynn_results():
    acc_mean, acc_std = np.mean(best_accuracies), np.std(best_accuracies)
    prec_mean, prec_std = np.mean(best_precisions), np.std(best_precisions)
    rec_mean, rec_std = np.mean(best_recalls), np.std(best_recalls)
    f1_mean, f1_std = np.mean(best_f1s), np.std(best_f1s)
    time_mean, time_std = np.mean(best_times), np.std(best_times)
    client_time_mean, client_time_std = np.mean(best_client_times), np.std(best_client_times)
    comm_cost_mean, comm_cost_std = np.mean(best_comm_costs), np.std(best_comm_costs)

    logging.info("\n===== FLYNN: BEST CONFIGURATION RESULTS =====")
    logging.info(f"Best parameters: m={best_params[0]}, s={best_params[1]}, rho={best_params[2]}")
    logging.info(f"Accuracy: {acc_mean:.7f} ± {acc_std:.7f}")
    logging.info(f"Precision: {prec_mean:.7f} ± {prec_std:.7f}")
    logging.info(f"Recall: {rec_mean:.7f} ± {rec_std:.7f}")
    logging.info(f"F1 Score: {f1_mean:.7f} ± {f1_std:.7f}")
    logging.info(f"Average time per fold: {time_mean:.7f}s ± {time_std:.7f}s")
    logging.info(f"Average time per client: {client_time_mean:.7f}s ± {client_time_std:.7f}s")
    logging.info(f"Communication cost: {comm_cost_mean:.2f}KB ± {comm_cost_std:.2f}KB")

    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [acc_mean, prec_mean, rec_mean, f1_mean]
    errors = [acc_std, prec_std, rec_std, f1_std]

    bars = plt.bar(metrics, values, yerr=errors, capsize=10)
    plt.title(f'FlyNN Performance Metrics (m={best_params[0]}, s={best_params[1]}, rho={best_params[2]})')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)

    for bar, val, err in zip(bars, values, errors):
        plt.text(bar.get_x() + bar.get_width()/2., val + err + 0.02,
                f'{val:.7f}±{err:.7f}', ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig('flynn_performance_metrics.png')
    plt.close()

    if best_y_test is not None and best_predictions is not None:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(best_y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for Best FlyNN Configuration')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('flynn_confusion_matrix.png')
        plt.close()

def main_flynn():
    global X, y, kf
    global best_accuracy, best_params, best_accuracies, best_precisions, best_recalls
    global best_f1s, best_times, best_client_times, best_comm_costs, best_y_test, best_predictions
    global all_hyperparams, all_accuracies, all_precisions, all_recalls, all_f1s, all_times, all_client_times, all_comm_costs

    data_path = 'rice/Rice_Cammeo_Osmancik.arff'
    data, meta = arff.loadarff(data_path)
    data_df = pd.DataFrame(data)

    data_df['Class'] = data_df['Class'].apply(lambda x: x.decode('utf-8'))

    label_encoder = LabelEncoder()
    data_df['Class'] = label_encoder.fit_transform(data_df['Class'])

    X = data_df.drop(columns=['Class'])
    y = data_df['Class']

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_hyperparams = []
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_times = []
    all_client_times = []
    all_comm_costs = []

    best_accuracy = 0
    best_params = None
    best_accuracies = []
    best_precisions = []
    best_recalls = []
    best_f1s = []
    best_times = []
    best_client_times = []
    best_comm_costs = []
    best_y_test = None
    best_predictions = None

    start_time = time.time()

    logging.info("FlyNN Bayesian optimization starting...")
    res_gp = gp_minimize(objective_flynn, space, n_calls=50, random_state=0)

    total_time = time.time() - start_time
    logging.info(f"FlyNN optimization completed. Total time: {total_time:.2f} seconds")

    m_best, s_best, rho_best = best_params
    logging.info(f"Best FlyNN parameters: m={m_best}, s={s_best}, rho={rho_best}")

    analyze_flynn_results()

    logging.info("FlyNN: Running noise robustness test...")

    best_fold_idx = np.argmax(best_accuracies)
    train_indices = list(kf.split(X))[best_fold_idx][0]
    test_indices = list(kf.split(X))[best_fold_idx][1]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X.iloc[train_indices])
    X_test_scaled = scaler.transform(X.iloc[test_indices])
    y_train, y_test = y[train_indices], y[test_indices]

    X_train_scaled = np.array(X_train_scaled)
    X_test_scaled = np.array(X_test_scaled)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    num_splits = 10
    split_train_data = np.array_split(X_train_scaled, num_splits)
    split_train_labels = np.array_split(y_train, num_splits)

    gamma = 0.5
    best_flynn_model = federated_train_flynn(
        split_train_data,
        split_train_labels,
        m_best, s_best, rho_best, gamma,
        random_state=42,
        is_dp=False
    )

    best_y_test = y_test
    best_predictions = best_flynn_model.predict(X_test_scaled)

    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    n_bootstrap = 100

    noise_results, p_values_acc, p_values_f1 = evaluate_flynn_robustness(
        best_flynn_model, X_test_scaled, y_test, noise_levels, n_bootstrap)

    logging.info("===== NOISE ROBUSTNESS RESULTS WITH P-VALUES =====")
    logging.info(f"{'Noise Level':<12} {'Accuracy':<12} {'F1 Score':<12} {'p-val (Acc)':<12} {'p-val (F1)':<12}")

    logging.info(f"{noise_results[0][0]:<12} {noise_results[0][1]:.7f}      {noise_results[0][2]:.7f}      -            -")

    for i in range(1, len(noise_results)):
        logging.info(f"{noise_results[i][0]:<12} {noise_results[i][1]:.7f}      {noise_results[i][2]:.7f}      {p_values_acc[i-1]:.2e}     {p_values_f1[i-1]:.2e}")

    logging.info("FlyNN process completed.")

    return best_params, best_accuracies, best_precisions, best_recalls, best_f1s, best_times, best_client_times, best_comm_costs

if __name__ == "__main__":
    main_flynn()