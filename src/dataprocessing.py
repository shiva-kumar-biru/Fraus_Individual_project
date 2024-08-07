import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import seaborn as sns
import io
from matplotlib.figure import Figure
import panel as pn
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
# Autocorrelation function
def autocorrelation(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]

# Apply Hamming window function
def apply_hamming_window(data):
    window = np.hamming(len(data))
    return data * window

def file_processing(file):
    dfs = []
    print(file)
    df = pd.read_csv(file, header=None, index_col=False)
    df["label"] = 1
    dfs.append(df)
    combined_person_df_test = pd.concat(dfs, ignore_index=True)

    adc_data_selected_columns_for_person = combined_person_df_test.iloc[:, 16:]
    test_features_df = adc_data_selected_columns_for_person.drop(columns='label')
    test_labels_df = adc_data_selected_columns_for_person['label']
    print("going to fft conversiom")
    adc_array = test_features_df.to_numpy()
    sampling_rate = 1953125

    fft_values_list = []
    frequency_list = []

    for row in adc_array:
        autocorr_result = autocorrelation(row)
        windowed_data = apply_hamming_window(autocorr_result)
        fft_result = np.fft.fft(windowed_data)
        freq = np.fft.fftfreq(len(fft_result), d=1/sampling_rate)
        positive_freqs = freq[:len(freq) // 2] / 1000
        positive_fft_values = np.abs(fft_result[:len(freq) // 2])

        if len(frequency_list) == 0:
            frequency_list = positive_freqs
        fft_values_list.append(positive_fft_values)

    fft_values = np.array(fft_values_list)
    fft_df = pd.DataFrame(fft_values, columns=frequency_list)

    range_min, range_max = 30, 50
    filtered_columns = [col for col in fft_df.columns if range_min <= col <= range_max]
    filtered_data = fft_df[filtered_columns]
    normalized_data = (filtered_data - filtered_data.min()) / (filtered_data.max() - filtered_data.min())
    print("done normalization")
    return normalized_data, test_labels_df

def rfc(normalized_data, test_labels_df):
    print("it came to rfc")
    model_rfc = joblib.load('/Users/shivakumarbiru/Desktop/individual_project/rfc/models/rf_classifier')
    y_pred = model_rfc.predict(normalized_data)
    accuracy_direct = accuracy_score(test_labels_df, y_pred)
    precision_direct = precision_score(test_labels_df, y_pred)
    recall_direct = recall_score(test_labels_df, y_pred)
    f1_direct = f1_score(test_labels_df, y_pred)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(test_labels_df, y_pred)
    
    # Create the confusion matrix plot
    fig1, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Person', 'Person'], yticklabels=['Not Person', 'Person'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    pane = pn.pane.Matplotlib(fig1,dpi=144,tight=True,height=500,width=500)

    return accuracy_direct, precision_direct, recall_direct, f1_direct, pane


def svm(normalized_data, test_labels_df):
    print("it came to svm")
    loaded_model = joblib.load('/Users/shivakumarbiru/Desktop/individual_project/rfc/models/svm1_model')

    y_pred_loaded = loaded_model.predict(normalized_data)

    accuracy = accuracy_score(test_labels_df, y_pred_loaded)
    precision = precision_score(test_labels_df, y_pred_loaded)
    recall = recall_score(test_labels_df, y_pred_loaded)
    f1 = f1_score(test_labels_df, y_pred_loaded)
    conf_matrix = confusion_matrix(test_labels_df, y_pred_loaded)
    fig2, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Person', 'Person'], yticklabels=['Not Person', 'Person'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    pane = pn.pane.Matplotlib(fig2,dpi=144,tight=True,height=550,width=550)
    return accuracy, precision, recall, f1, pane


def lr(normalized_data, test_labels_df):
    print("it came to lr")
  # Load the saved model
    loaded_model = joblib.load('logistic_regression_model.pkl')

    # Load the scaler
    scaler = joblib.load('scaler.pkl')

    # Feature scaling
    X_test_scaled = scaler.transform(normalized_data)  # Apply the same scaling as used in training

    # Predict using the loaded model
    y_pred = loaded_model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(test_labels_df, y_pred)
    conf_matrix = confusion_matrix(test_labels_df, y_pred)
    precision= precision_score(test_labels_df, y_pred)
    recall=recall_score(test_labels_df, y_pred)
    f1=f1_score(test_labels_df, y_pred)

    fig3, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Person', 'Person'], yticklabels=['Not Person', 'Person'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    pane = pn.pane.Matplotlib(fig3,dpi=144,tight=True,height=550,width=550)
    return accuracy, precision, recall, f1, pane




def xgboost(normalized_data, test_labels_df):
    print("it came to xgboost")
    # Load the saved model
    model_path = '/Users/shivakumarbiru/Desktop/individual_project/rfc/best_xgb_model_early_stopping.pkl'  # Replace with your actual model file name if different
    xgb_classifier = joblib.load(model_path)

    # Make predictions with the loaded model
    y_pred = xgb_classifier.predict(normalized_data)

    accuracy = accuracy_score(test_labels_df, y_pred)
    conf_matrix = confusion_matrix(test_labels_df, y_pred)
    precision= precision_score(test_labels_df, y_pred)
    recall=recall_score(test_labels_df, y_pred)
    f1=f1_score(test_labels_df, y_pred)

    fig4, ax = plt.subplots(figsize=(2, 2))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Person', 'Person'], yticklabels=['Not Person', 'Person'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    pane = pn.pane.Matplotlib(fig4,dpi=144,tight=True,height=550,width=550)
    return accuracy, precision, recall, f1, pane


def gbm(normalized_data, test_labels_df):
    print("it came to gbm")
    loaded_gbm_model = joblib.load('gbm_model.pkl')
    print('GBM model loaded from gbm_model.pkl')
    y_pred = loaded_gbm_model.predict(normalized_data)
    accuracy = accuracy_score(test_labels_df, y_pred)
    conf_matrix = confusion_matrix(test_labels_df, y_pred)
    precision= precision_score(test_labels_df, y_pred)
    recall=recall_score(test_labels_df, y_pred)
    f1=f1_score(test_labels_df, y_pred)

    fig5, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Person', 'Person'], yticklabels=['Not Person', 'Person'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    pane = pn.pane.Matplotlib(fig5,dpi=144,tight=True,height=550,width=550)
    return accuracy, precision, recall, f1, pane

def cnn(normalized_data, test_labels_df):

    loaded_model = tf.keras.models.load_model('mycnn_model.h5')

    # Predict on the test data using the loaded model
    y_pred_prob_loaded= loaded_model.predict(normalized_data)
    y_pred = np.round(y_pred_prob_loaded)

    # Calculate metrics
    accuracy = accuracy_score(test_labels_df, y_pred)
    conf_matrix = confusion_matrix(test_labels_df, y_pred)
    precision= precision_score(test_labels_df, y_pred)
    recall=recall_score(test_labels_df, y_pred)
    f1=f1_score(test_labels_df, y_pred)

    # Print results
    print('Model Results:')
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    fig6, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Person', 'Person'], yticklabels=['Not Person', 'Person'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    pane = pn.pane.Matplotlib(fig6,dpi=144,tight=True,height=550,width=550)
    return accuracy, precision, recall, f1, pane