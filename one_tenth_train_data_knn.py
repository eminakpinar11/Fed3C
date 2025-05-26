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
                    handlers=[logging.FileHandler('k1_rice_one_tenth_train_data_knn.log'), logging.StreamHandler()])

data_path = 'rice/Rice_Cammeo_Osmancik.arff'
data, meta = arff.loadarff(data_path)
data_df = pd.DataFrame(data)

data_df['Class'] = data_df['Class'].apply(lambda x: x.decode('utf-8'))

label_encoder = LabelEncoder()
data_df['Class'] = label_encoder.fit_transform(data_df['Class'])
logging.info(f"Class Labels: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")


X = data_df.drop(columns=['Class'])
y = data_df['Class']

N_OUTER_SPLITS = 5
kf = KFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=42)

k_value = 1
N_INNER_SPLITS = 10

all_individual_accuracies = []
all_individual_precisions = []
all_individual_recalls = []
all_individual_f1_scores = []
all_individual_times = []

all_inner_model_details = []


def evaluate_robustness(model_to_test, X_test, y_test, noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5], n_bootstrap=100, model_desc="Test Model"):
    results = []
    logging.info(f"evaluate_robustness for {model_desc} - X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    n_samples = len(y_test)
    if n_samples == 0:
        logging.warning(f"evaluate_robustness ({model_desc}): Test data is empty, bootstrap cannot be done.")
        for noise in noise_levels: results.append((f'Noise {noise:.2f}', np.nan, np.nan))
        return results, [np.nan]*len(noise_levels), [np.nan]*len(noise_levels)

    original_pred = model_to_test.predict(X_test)
    original_acc = accuracy_score(y_test, original_pred)
    original_f1 = f1_score(y_test, original_pred, average='weighted', zero_division=0)
    results.append(('No Noise', original_acc, original_f1))
    logging.info(f"evaluate_robustness ({model_desc}) - No Noise: Acc={original_acc:.7f}, F1={original_f1:.7f}")

    bootstrap_original_acc = []
    bootstrap_original_f1 = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot, y_boot = X_test[indices], (y_test.iloc[indices] if hasattr(y_test, 'iloc') else y_test[indices])
        boot_pred = model_to_test.predict(X_boot)
        bootstrap_original_acc.append(accuracy_score(y_boot, boot_pred))
        bootstrap_original_f1.append(f1_score(y_boot, boot_pred, average='weighted', zero_division=0))

    p_values_acc, p_values_f1, bootstrap_acc_all, bootstrap_f1_all = [], [], [], []

    for noise in noise_levels:
        bootstrap_acc, bootstrap_f1 = [], []
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X_test[indices].copy(), (y_test.iloc[indices] if hasattr(y_test, 'iloc') else y_test[indices])
            noise_matrix = np.random.normal(0, noise, X_boot.shape)
            X_boot += noise_matrix
            noisy_pred = model_to_test.predict(X_boot)
            bootstrap_acc.append(accuracy_score(y_boot, noisy_pred))
            bootstrap_f1.append(f1_score(y_boot, noisy_pred, average='weighted', zero_division=0))

        mean_acc, mean_f1 = (np.mean(bootstrap_acc) if bootstrap_acc else np.nan), (np.mean(bootstrap_f1) if bootstrap_f1 else np.nan)
        p_val_acc, p_val_f1 = (stats.ttest_rel(bootstrap_original_acc, bootstrap_acc)[1] if bootstrap_original_acc and bootstrap_acc else np.nan), \
                              (stats.ttest_rel(bootstrap_original_f1, bootstrap_f1)[1] if bootstrap_original_f1 and bootstrap_f1 else np.nan)

        results.append((f'Noise {noise:.2f}', mean_acc, mean_f1))
        p_values_acc.append(p_val_acc); p_values_f1.append(p_val_f1)
        bootstrap_acc_all.append(bootstrap_acc); bootstrap_f1_all.append(bootstrap_f1)

    plt.figure(figsize=(13, 7))
    noise_labels = [r[0] for r in results]
    acc_values = [r[1] for r in results]
    f1_values_plot = [r[2] for r in results]
    x_indices = np.arange(len(noise_labels))
    width = 0.35

    plt.subplot(1, 2, 1)
    plt.bar(x_indices - width/2, acc_values, width, label='Accuracy')
    plt.bar(x_indices + width/2, f1_values_plot, width, label='F1 Score')
    plt.xlabel('Noise Level'); plt.ylabel('Score'); plt.title(f'Model Robustness to Noise ({model_desc})')
    plt.xticks(x_indices, noise_labels, rotation=45); plt.legend()
    for i in range(len(p_values_acc)):
        y_star_acc = acc_values[i+1] - 0.02 if not np.isnan(acc_values[i+1]) else 0
        y_star_f1 = f1_values_plot[i+1] - 0.02 if not np.isnan(f1_values_plot[i+1]) else 0
        if not np.isnan(p_values_acc[i]):
            if p_values_acc[i] < 0.001: plt.text(x_indices[i+1] - width/2, y_star_acc, '***', ha='center')
            elif p_values_acc[i] < 0.01: plt.text(x_indices[i+1] - width/2, y_star_acc, '**', ha='center')
            elif p_values_acc[i] < 0.05: plt.text(x_indices[i+1] - width/2, y_star_acc, '*', ha='center')
        if not np.isnan(p_values_f1[i]):
            if p_values_f1[i] < 0.001: plt.text(x_indices[i+1] + width/2, y_star_f1, '***', ha='center')
            elif p_values_f1[i] < 0.01: plt.text(x_indices[i+1] + width/2, y_star_f1, '**', ha='center')
            elif p_values_f1[i] < 0.05: plt.text(x_indices[i+1] + width/2, y_star_f1, '*', ha='center')

    plt.subplot(1, 2, 2)
    valid_scores = [s for s in acc_values + f1_values_plot if not np.isnan(s)]
    zoom_min_val = max(0.0, min(valid_scores) - 0.05) if valid_scores else 0.0
    zoom_max_val = min(1.0, max(valid_scores) + 0.05) if valid_scores else 1.0
    if zoom_min_val >= zoom_max_val: zoom_min_val, zoom_max_val = 0.0, 1.0

    plt.bar(x_indices - width/2, acc_values, width, label='Accuracy')
    plt.bar(x_indices + width/2, f1_values_plot, width, label='F1 Score')
    plt.xlabel('Noise Level'); plt.ylabel('Score'); plt.title(f'Zoomed View ({zoom_min_val:.2f}-{zoom_max_val:.2f})')
    plt.xticks(x_indices, noise_labels, rotation=45); plt.ylim(zoom_min_val, zoom_max_val)
    for i in range(len(p_values_acc)):
        if not np.isnan(acc_values[i+1]):
            y_text_acc = acc_values[i+1] + 0.005 if acc_values[i+1] < zoom_max_val - 0.01 else acc_values[i+1] - 0.01
            plt.text(x_indices[i+1] - width/2, acc_values[i+1] - (zoom_max_val-zoom_min_val)*0.05 if acc_values[i+1] > zoom_min_val + (zoom_max_val-zoom_min_val)*0.1 else acc_values[i+1] + (zoom_max_val-zoom_min_val)*0.02, f'{acc_values[i+1]:.3f}', ha='center', va='bottom' if acc_values[i+1] > zoom_min_val + (zoom_max_val-zoom_min_val)*0.1 else 'top', fontsize=7, rotation=90)
            if not np.isnan(p_values_acc[i]):
                if p_values_acc[i] < 0.001: plt.text(x_indices[i+1] - width/2, y_text_acc, '***', ha='center', fontsize=10)
                elif p_values_acc[i] < 0.01: plt.text(x_indices[i+1] - width/2, y_text_acc, '**', ha='center', fontsize=10)
                elif p_values_acc[i] < 0.05: plt.text(x_indices[i+1] - width/2, y_text_acc, '*', ha='center', fontsize=10)
        if not np.isnan(f1_values_plot[i+1]):
            y_text_f1 = f1_values_plot[i+1] + 0.005 if f1_values_plot[i+1] < zoom_max_val - 0.01 else f1_values_plot[i+1] - 0.01
            plt.text(x_indices[i+1] + width/2, f1_values_plot[i+1] - (zoom_max_val-zoom_min_val)*0.05 if f1_values_plot[i+1] > zoom_min_val + (zoom_max_val-zoom_min_val)*0.1 else f1_values_plot[i+1] + (zoom_max_val-zoom_min_val)*0.02, f'{f1_values_plot[i+1]:.3f}', ha='center', va='bottom' if f1_values_plot[i+1] > zoom_min_val + (zoom_max_val-zoom_min_val)*0.1 else 'top', fontsize=7, rotation=90)
            if not np.isnan(p_values_f1[i]):
                if p_values_f1[i] < 0.001: plt.text(x_indices[i+1] + width/2, y_text_f1, '***', ha='center', fontsize=10)
                elif p_values_f1[i] < 0.01: plt.text(x_indices[i+1] + width/2, y_text_f1, '**', ha='center', fontsize=10)
                elif p_values_f1[i] < 0.05: plt.text(x_indices[i+1] + width/2, y_text_f1, '*', ha='center', fontsize=10)

    plt.figtext(0.5, 0.01, "* p<0.05, ** p<0.01, *** p<0.001 (vs No Noise)", ha="center", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Model Robustness ({model_desc}, k={k_value})', fontsize=14)
    plt.savefig(f'modified_knn_strategy3_k{k_value}_{model_desc.replace(" ","_")}_robustness.png')
    plt.close()

    plt.figure(figsize=(14, 8))
    boxplot_labels = ['No Noise'] + [f'Noise {n:.2f}' for n in noise_levels]
    plt.subplot(2, 1, 1);
    valid_boxplot_acc = [d for d in ([bootstrap_original_acc] + bootstrap_acc_all) if d]
    if valid_boxplot_acc: plt.boxplot(valid_boxplot_acc, labels=boxplot_labels[:len(valid_boxplot_acc)], showfliers=False)
    plt.title(f'Distribution of Accuracy Scores ({model_desc})'); plt.ylabel('Accuracy'); plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(2, 1, 2);
    valid_boxplot_f1 = [d for d in ([bootstrap_original_f1] + bootstrap_f1_all) if d]
    if valid_boxplot_f1: plt.boxplot(valid_boxplot_f1, labels=boxplot_labels[:len(valid_boxplot_f1)], showfliers=False)
    plt.title(f'Distribution of F1 Scores ({model_desc})'); plt.ylabel('F1 Score'); plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'modified_knn_strategy3_k{k_value}_{model_desc.replace(" ","_")}_noise_distribution.png')
    plt.close()

    logging.info(f"\n===== NOISE ROBUSTNESS RESULTS ({model_desc}) =====")
    logging.info(f"{'Noise Level':<12} {'Mean Acc':<12} {'Mean F1':<12} {'p-val (Acc)':<12} {'p-val (F1)':<12}")
    acc_val_str = f"{acc_values[0]:.7f}" if not np.isnan(acc_values[0]) else "NaN"
    f1_val_str = f"{f1_values_plot[0]:.7f}" if not np.isnan(f1_values_plot[0]) else "NaN"
    logging.info(f"{noise_labels[0]:<12} {acc_val_str:<12} {f1_val_str:<12} -            -")
    for i in range(len(noise_levels)):
        p_acc_str = f"{p_values_acc[i]:.2e}" if not np.isnan(p_values_acc[i]) else "N/A"
        p_f1_str = f"{p_values_f1[i]:.2e}" if not np.isnan(p_values_f1[i]) else "N/A"
        acc_val_str = f"{acc_values[i+1]:.7f}" if not np.isnan(acc_values[i+1]) else "NaN"
        f1_val_str = f"{f1_values_plot[i+1]:.7f}" if not np.isnan(f1_values_plot[i+1]) else "NaN"
        logging.info(f"{noise_labels[i+1]:<12} {acc_val_str:<12} {f1_val_str:<12} {p_acc_str:<12} {p_f1_str:<12}")
    return results, p_values_acc, p_values_f1

def analyze_results(model_for_cm=None, X_cm=None, y_cm=None, model_desc="Test Model"):
    if not all_individual_accuracies:
        logging.warning("analyze_results: No model results found to calculate.")
        acc_mean, acc_std, prec_mean, prec_std, rec_mean, rec_std, f1_mean, f1_std, time_mean, time_std = [np.nan]*10
    else:
        acc_mean, acc_std = np.mean(all_individual_accuracies), np.std(all_individual_accuracies)
        prec_mean, prec_std = np.mean(all_individual_precisions), np.std(all_individual_precisions)
        rec_mean, rec_std = np.mean(all_individual_recalls), np.std(all_individual_recalls)
        f1_mean, f1_std = np.mean(all_individual_f1_scores), np.std(all_individual_f1_scores)
        time_mean, time_std = np.mean(all_individual_times), np.std(all_individual_times)

    logging.info("\n===== OVERALL RESULTS (Averaged over all N_OUTER_SPLITS * N_INNER_SPLITS models) =====")
    logging.info(f"k value (n_neighbors): {k_value}")
    logging.info(f"Number of inner splits per outer fold: {N_INNER_SPLITS}")
    logging.info(f"Total individual models evaluated: {len(all_individual_accuracies)}")
    logging.info(f"Accuracy: {acc_mean:.7f} ± {acc_std:.7f}")
    logging.info(f"Precision: {prec_mean:.7f} ± {prec_std:.7f}")
    logging.info(f"Recall: {rec_mean:.7f} ± {rec_std:.7f}")
    logging.info(f"F1 Score: {f1_mean:.7f} ± {f1_std:.7f}")
    logging.info(f"Average execution time per inner model: {time_mean:.7f}s ± {time_std:.7f}s")

    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [acc_mean, prec_mean, rec_mean, f1_mean]
    errors = [acc_std, prec_std, rec_std, f1_std]
    bars = plt.bar(metrics, values, yerr=errors, capsize=10)
    plt.title(f'Overall Performance (Avg. of Inner Models, k={k_value}, Inner Splits={N_INNER_SPLITS})')
    plt.ylabel('Score'); plt.ylim(0, 1.1)
    for bar, val, err in zip(bars, values, errors):
        if not np.isnan(val) and not np.isnan(err):
            plt.text(bar.get_x() + bar.get_width()/2., val + err + 0.02 if val + err < 1.05 else 1.05,
                    f'{val:.7f}±{err:.7f}', ha='center', va='bottom', rotation=0)
    plt.tight_layout()
    plt.savefig(f'modified_knn_strategy3_k{k_value}_n{N_INNER_SPLITS}_overall_metrics.png')
    plt.close()

    if model_for_cm and X_cm is not None and y_cm is not None and len(y_cm) > 0:
        logging.info(f"analyze_results CM for {model_desc}: X_cm shape: {X_cm.shape}, y_cm shape: {y_cm.shape}")
        cm_predictions = model_for_cm.predict(X_cm)
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_cm, cm_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title(f'Confusion Matrix ({model_desc}, k={k_value})')
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.savefig(f'modified_knn_strategy3_k{k_value}_n{N_INNER_SPLITS}_{model_desc.replace(" ","_")}_CM.png')
        plt.close()
    elif not model_for_cm: logging.warning(f"analyze_results: Model ({model_desc}) not found for CM.")
    else: logging.warning(f"analyze_results: X_cm or y_cm for CM ({model_desc}) is missing/empty.")


def main():
    global all_individual_accuracies, all_individual_precisions, all_individual_recalls
    global all_individual_f1_scores, all_individual_times, all_inner_model_details

    all_individual_accuracies.clear(); all_individual_precisions.clear(); all_individual_recalls.clear()
    all_individual_f1_scores.clear(); all_individual_times.clear(); all_inner_model_details.clear()

    overall_start_time = time.time()
    logging.info(f"Modified KNN testing begins (Strateji 3: Strategy 3: Typical Internal Model) (k={k_value}, N_INNER_SPLITS={N_INNER_SPLITS})...")

    fold_counter = 0
    for train_index, test_index in kf.split(X, y):
        fold_counter += 1
        logging.info(f"\n--- Outer Fold {fold_counter}/{N_OUTER_SPLITS} ---")
        X_train_full_orig, X_test_orig_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_full_orig, y_test_orig_fold = y.iloc[train_index], y.iloc[test_index]

        scaler = StandardScaler()
        X_train_full_scaled = scaler.fit_transform(X_train_full_orig)
        X_test_scaled_fold = scaler.transform(X_test_orig_fold)
        current_outer_fold_accuracies = []
        num_samples_in_full_train = len(X_train_full_scaled)
        actual_inner_splits = N_INNER_SPLITS
        subset_size = num_samples_in_full_train // actual_inner_splits

        if num_samples_in_full_train < N_INNER_SPLITS:
            logging.warning(f"Outer Fold {fold_counter}: Training data size ({num_samples_in_full_train}) is smaller than N_INNER_SPLITS ({N_INNER_SPLITS}). Reducing N_INNER_SPLITS to {num_samples_in_full_train}(each sample is a split).")
            actual_inner_splits = num_samples_in_full_train if num_samples_in_full_train > 0 else 1
            subset_size = 1

        if subset_size == 0 and actual_inner_splits > 0 and num_samples_in_full_train > 0 :
             logging.warning(f"Outer Fold {fold_counter}: Subset size is 0, num_samples_in_full_train={num_samples_in_full_train}, actual_inner_splits={actual_inner_splits}. Setting number of splits to 1.")
             actual_inner_splits = 1
             subset_size = num_samples_in_full_train
        elif num_samples_in_full_train == 0:
            logging.warning(f"Outer Fold {fold_counter}: Full training set is empty. Skipping this fold.")
            continue


        for i in range(actual_inner_splits):
            start_index = i * subset_size
            end_index = (i + 1) * subset_size if i < actual_inner_splits - 1 else num_samples_in_full_train
            if start_index >= end_index : continue


            X_train_subset = X_train_full_scaled[start_index:end_index]
            y_train_subset = y_train_full_orig.iloc[start_index:end_index]

            if len(X_train_subset) == 0:
                logging.warning(f"Outer Fold {fold_counter}, Inner Split {i+1}: Training subset is empty. Skipping.")
                continue

            n_neighbors_actual = min(k_value, len(X_train_subset))
            if n_neighbors_actual == 0 :
                logging.warning(f"Outer Fold {fold_counter}, Inner Split {i+1}: n_neighbors_actual became 0. Skipping.")
                continue

            unique_classes_in_subset = np.unique(y_train_subset)
            if len(unique_classes_in_subset) < 2 and len(y_train_subset) > 0 :
                 logging.warning(f"Outer Fold {fold_counter}, Inner Split {i+1}: Training subset has only one class ({unique_classes_in_subset}). KNN might make constant predictions.")
                 if len(y_train_subset) < n_neighbors_actual and len(y_train_subset) > 0:
                     n_neighbors_actual = len(y_train_subset)


            inner_model_start_time = time.time()
            knn = KNeighborsClassifier(n_neighbors=n_neighbors_actual)

            try:
                knn.fit(X_train_subset, y_train_subset)
                if not hasattr(knn, 'classes_') or len(knn.classes_) < 2 :
                    logging.warning(f"Outer Fold {fold_counter}, Inner Split {i+1}: KNN model learned only one class. Predictions might be uniform.")

                predictions = knn.predict(X_test_scaled_fold)
                acc = accuracy_score(y_test_orig_fold, predictions)
                prec = precision_score(y_test_orig_fold, predictions, average='weighted', zero_division=0)
                rec = recall_score(y_test_orig_fold, predictions, average='weighted', zero_division=0)
                f1 = f1_score(y_test_orig_fold, predictions, average='weighted', zero_division=0)
                inner_model_exec_time = time.time() - inner_model_start_time

                all_individual_accuracies.append(acc); all_individual_precisions.append(prec)
                all_individual_recalls.append(rec); all_individual_f1_scores.append(f1)
                all_individual_times.append(inner_model_exec_time)
                current_outer_fold_accuracies.append(acc)

                all_inner_model_details.append({
                    'model_object': knn, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1,
                    'X_test_scaled_fold': X_test_scaled_fold, 'y_test_orig_fold': y_test_orig_fold,
                    'fold_num': fold_counter, 'inner_split_num': i + 1, 'train_subset_size': len(X_train_subset)
                })
            except Exception as e:
                logging.error(f"Outer Fold {fold_counter}, Inner Split {i+1}: Error during model training/prediction: {e}")

        if current_outer_fold_accuracies:
            mean_acc_fold, std_acc_fold = np.mean(current_outer_fold_accuracies), np.std(current_outer_fold_accuracies)
            logging.info(f"Outer Fold {fold_counter} Avg. Acc of Inner Models: {mean_acc_fold:.7f} (Std: {std_acc_fold:.7f})")
        else:
            logging.info(f"Outer Fold {fold_counter}: No valid inner model could be evaluated in this fold.")

    overall_end_time = time.time()
    logging.info(f"\nAll outer folds and inner splits completed. Total time: {overall_end_time - overall_start_time:.2f} seconds")

    model_for_analysis = None
    X_test_for_analysis = None
    y_test_for_analysis = None
    model_description_for_analysis = "N/A"

    if all_inner_model_details:
        if all_individual_accuracies:
            mean_overall_accuracy = np.mean(all_individual_accuracies)
            logging.info(f"Average accuracy of all inner models (for typical model selection): {mean_overall_accuracy:.7f}")

            valid_models_for_typical_selection = [m for m in all_inner_model_details if not np.isnan(m['accuracy'])]
            if valid_models_for_typical_selection:
                typical_inner_model_info = min(valid_models_for_typical_selection, key=lambda item: abs(item['accuracy'] - mean_overall_accuracy))
                model_for_analysis = typical_inner_model_info['model_object']
                X_test_for_analysis = typical_inner_model_info['X_test_scaled_fold']
                y_test_for_analysis = typical_inner_model_info['y_test_orig_fold']
                model_description_for_analysis = "Typical Inner Model"

                logging.info(f"\nIndividual INNER MODEL closest to the average (TYPICAL) selected (for Noise Test and CM):")
                logging.info(f"  Originating Outer Fold: {typical_inner_model_info['fold_num']}")
                logging.info(f"  Originating Inner Split: {typical_inner_model_info['inner_split_num']}")
                logging.info(f"  Training Subset Size: {typical_inner_model_info['train_subset_size']}")
                logging.info(f"  Accuracy of this Model (on its respective outer fold test set): {typical_inner_model_info['accuracy']:.7f}")
                logging.info(f"  This Model's F1 Score: {typical_inner_model_info['f1_score']:.7f}")
            else:
                logging.warning("A typical inner model close to the average could not be selected (no valid models).")
        else:
            logging.warning("Average accuracy could not be calculated, a typical model cannot be selected. The best model will be tried as a fallback.")
            if all_inner_model_details:
                 best_fallback_model_info = max(all_inner_model_details, key=lambda item: item['accuracy'])
                 model_for_analysis = best_fallback_model_info['model_object']
                 X_test_for_analysis = best_fallback_model_info['X_test_scaled_fold']
                 y_test_for_analysis = best_fallback_model_info['y_test_orig_fold']
                 model_description_for_analysis = "Best Inner Model (Fallback)"
                 logging.info(f"\nFALLBACK: Best Inner Model selected.")
                 logging.info(f"  Accuracy: {best_fallback_model_info['accuracy']:.7f}")


    else:
        logging.warning("No inner model details were recorded. A model could not be selected for Noise Test and CM.")

    analyze_results(model_for_cm=model_for_analysis,
                    X_cm=X_test_for_analysis,
                    y_cm=y_test_for_analysis,
                    model_desc=model_description_for_analysis)

    if model_for_analysis and X_test_for_analysis is not None and y_test_for_analysis is not None and len(y_test_for_analysis) > 0 :
        logging.info(f"\nRunning noise robustness test for the selected {model_description_for_analysis}...")
        noise_levels_config = [0.1, 0.2, 0.3, 0.4, 0.5]
        n_bootstrap_config = 100
        evaluate_robustness(
            model_for_analysis, X_test_for_analysis, y_test_for_analysis,
            noise_levels=noise_levels_config, n_bootstrap=n_bootstrap_config,
            model_desc=model_description_for_analysis
        )
    elif not model_for_analysis:
        logging.info(f"A suitable model ({model_description_for_analysis}) for the noise test could not be found.")
    else:
        logging.info(f"For the noise test, {model_description_for_analysis}'s test data is missing/empty.")

    logging.info("Process completed.")

if __name__ == "__main__":
    main()