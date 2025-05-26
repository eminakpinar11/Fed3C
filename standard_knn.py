import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.io import arff
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('k1_rice_standard_knn.log'), logging.StreamHandler()])

data_path = 'rice/Rice_Cammeo_Osmancik.arff'
data, meta = arff.loadarff(data_path)
data_df = pd.DataFrame(data)

data_df['Class'] = data_df['Class'].apply(lambda x: x.decode('utf-8'))

label_encoder = LabelEncoder()
data_df['Class'] = label_encoder.fit_transform(data_df['Class'])

X = data_df.drop(columns=['Class'])
y = data_df['Class']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

k_value = 1

accuracies = []
precisions = []
recalls = []
f1_scores = []
times = []
best_y_test = None
best_predictions = None

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
    plt.savefig(f'standard_knn_k{k_value}_noise_robustness.png')
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
    plt.savefig(f'standard_knn_k{k_value}_noise_distribution.png')
    plt.close()

    logging.info("\n===== NOISE ROBUSTNESS RESULTS WITH P-VALUES =====")
    logging.info(f"{'Noise Level':<12} {'Accuracy':<12} {'F1 Score':<12} {'p-val (Acc)':<12} {'p-val (F1)':<12}")

    logging.info(f"{noise_labels[0]:<12} {acc_values[0]:.7f}      {f1_values[0]:.7f}      -            -")

    for i in range(1, len(noise_labels)):
        logging.info(f"{noise_labels[i]:<12} {acc_values[i]:.7f}      {f1_values[i]:.7f}      {p_values_acc[i-1]:.2e}     {p_values_f1[i-1]:.2e}")

    return results, p_values_acc, p_values_f1

def analyze_results():

    acc_mean, acc_std = np.mean(accuracies), np.std(accuracies)
    prec_mean, prec_std = np.mean(precisions), np.std(precisions)
    rec_mean, rec_std = np.mean(recalls), np.std(recalls)
    f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
    time_mean, time_std = np.mean(times), np.std(times)

    logging.info("\n===== RESULTS (Standard KNN) =====")
    logging.info(f"k value (n_neighbors): {k_value}")
    logging.info(f"Accuracy: {acc_mean:.7f} ± {acc_std:.7f}")
    logging.info(f"Precision: {prec_mean:.7f} ± {prec_std:.7f}")
    logging.info(f"Recall: {rec_mean:.7f} ± {rec_std:.7f}")
    logging.info(f"F1 Score: {f1_mean:.7f} ± {f1_std:.7f}")
    logging.info(f"Average execution time: {time_mean:.7f}s ± {time_std:.7f}s")

    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [acc_mean, prec_mean, rec_mean, f1_mean]
    errors = [acc_std, prec_std, rec_std, f1_std]

    bars = plt.bar(metrics, values, yerr=errors, capsize=10)
    plt.title(f'Performance Metrics with Error Bars (Standard KNN, n_neighbors={k_value})')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)

    for bar, val, err in zip(bars, values, errors):
        plt.text(bar.get_x() + bar.get_width()/2., val + err + 0.02,
                f'{val:.7f}±{err:.7f}', ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig(f'standard_knn_k{k_value}_performance_metrics.png')
    plt.close()

    if best_y_test is not None and best_predictions is not None:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(best_y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for Standard KNN (k={k_value})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'standard_knn_k{k_value}_confusion_matrix.png')
        plt.close()

def main():
    global accuracies, precisions, recalls, f1_scores, times, best_y_test, best_predictions

    start_time = time.time()

    logging.info(f"Standard KNN test started (k={k_value})...")

    best_acc = -float('inf')

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        fold_start_time = time.time()

        knn = KNeighborsClassifier(n_neighbors=k_value)
        knn.fit(X_train_scaled, y_train)

        predictions = knn.predict(X_test_scaled)

        fold_end_time = time.time()
        exec_time = fold_end_time - fold_start_time

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        times.append(exec_time)

        logging.info(f"Fold, Acc: {accuracy:.7f}, Prec: {precision:.7f}, Rec: {recall:.7f}, F1: {f1:.7f}, Time: {exec_time:.7f}s")

        if accuracy > best_acc:
            best_acc = accuracy
            best_y_test = y_test
            best_predictions = predictions

    total_time = time.time() - start_time
    logging.info(f"Test completed. Total time: {total_time:.2f} second")

    analyze_results()

    logging.info("Noise resistance test started...")

    best_fold_idx = np.argmax(accuracies)
    train_indices = list(kf.split(X))[best_fold_idx][0]
    test_indices = list(kf.split(X))[best_fold_idx][1]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_model = KNeighborsClassifier(n_neighbors=k_value)
    best_model.fit(X_train_scaled, y_train)

    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    n_bootstrap = 100

    noise_results, p_values_acc, p_values_f1 = evaluate_robustness(
        best_model, X_test_scaled, y_test, noise_levels, n_bootstrap)

    logging.info("Code snippet completed...")

if __name__ == "__main__":
    main()