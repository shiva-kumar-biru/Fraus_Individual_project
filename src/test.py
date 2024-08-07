
import panel as pn
import io
from dataprocessing import file_processing,rfc,gbm,svm,xgboost,lr,cnn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import pandas as pd

custom_style = {
    'padding': '14px',
    'font-style': 'italic',
    'border-radius': '12px',
    'width': '180px'
}

style_to_display= {
    'padding': '14px',
    'font-style': 'italic',
    'border-radius': '12px',
    'font-size': '13pt',
    'width': 'auto',
    'font-weight': 'bold'
}

logopath=r"/Users/shivakumarbiru/Desktop/individual_project/rfc/Picture 1.png"
# logo = pn.panel(logopath,width=200)
logo = pn.pane.Image(logopath, width=150, height=150)

global_data = {"normalized_data": None, "test_labels_df": None}

def run_classifier(event):
    if global_data["normalized_data"] is not None and global_data["test_labels_df"] is not None:
        selected_classifier = classifier_select.value
        if selected_classifier == "Random Forest":
            normalized_data = global_data["normalized_data"]
            test_labels_df = global_data["test_labels_df"]
            accuracy_direct, precision_direct, recall_direct, f1_direct, CM_rfc = rfc(normalized_data, test_labels_df)
            row1.clear()
            row1.append(CM_rfc)
            metrics_placeholder.object = f"""
            <div style="font-size:16px; font-weight:bold; color:#333;">
            <h3>Metrics for RFC</h3>
            <ul>
                <li>Accuracy: <span style="color:blue;">{accuracy_direct:.4f}</span></li>
                <li>Precision: <span style="color:green;">{precision_direct:.4f}</span></li>
                <li>Recall: <span style="color:orange;">{recall_direct:.4f}</span></li>
                <li>F1-score: <span style="color:red;">{f1_direct:.4f}</span></li>
            </ul>
            </div>
            """
        elif selected_classifier == "SVM":
            normalized_data = global_data["normalized_data"]
            test_labels_df = global_data["test_labels_df"]
            accuracy_direct, precision_direct, recall_direct, f1_direct, cm_svm = svm(normalized_data, test_labels_df)
            row1.clear()
            row1.append(cm_svm)
            metrics_placeholder.object = f"""
            <div style="font-size:16px; font-weight:bold; color:#333;">
            <h3>Metrics for SVM</h3>
            <ul>
                <li>Accuracy: <span style="color:blue;">{accuracy_direct:.4f}</span></li>
                <li>Precision: <span style="color:green;">{precision_direct:.4f}</span></li>
                <li>Recall: <span style="color:orange;">{recall_direct:.4f}</span></li>
                <li>F1-score: <span style="color:red;">{f1_direct:.4f}</span></li>
            </ul>
            </div>
            """
        elif selected_classifier == "Logistic Regression":
            normalized_data = global_data["normalized_data"]
            test_labels_df = global_data["test_labels_df"]
            accuracy_direct, precision_direct, recall_direct, f1_direct, cm_lr = lr(normalized_data, test_labels_df)
            row1.clear()
            row1.append(cm_lr)
            metrics_placeholder.object = f"""
            <div style="font-size:16px; font-weight:bold; color:#333;">
            <h3>Metrics for Logistic Regression</h3>
            <ul>
                <li>Accuracy: <span style="color:blue;">{accuracy_direct:.4f}</span></li>
                <li>Precision: <span style="color:green;">{precision_direct:.4f}</span></li>
                <li>Recall: <span style="color:orange;">{recall_direct:.4f}</span></li>
                <li>F1-score: <span style="color:red;">{f1_direct:.4f}</span></li>
            </ul>
            </div>
            """
        elif selected_classifier == "XG Boost":
            normalized_data = global_data["normalized_data"]
            test_labels_df = global_data["test_labels_df"]
            accuracy_direct, precision_direct, recall_direct, f1_direct, cm_xg = xgboost(normalized_data, test_labels_df)
            row1.clear()
            row1.append(cm_xg)
            metrics_placeholder.object = f"""
            <div style="font-size:16px; font-weight:bold; color:#333;">
            <h3>Metrics for XG Boost</h3>
            <ul>
                <li>Accuracy: <span style="color:blue;">{accuracy_direct:.4f}</span></li>
                <li>Precision: <span style="color:green;">{precision_direct:.4f}</span></li>
                <li>Recall: <span style="color:orange;">{recall_direct:.4f}</span></li>
                <li>F1-score: <span style="color:red;">{f1_direct:.4f}</span></li>
            </ul>
            </div>
            """
        elif selected_classifier == "Gradient Boosting":
            normalized_data = global_data["normalized_data"]
            test_labels_df = global_data["test_labels_df"]
            accuracy_direct, precision_direct, recall_direct, f1_direct, cm_gbm = gbm(normalized_data, test_labels_df)
            row1.clear()
            row1.append(cm_gbm)
            metrics_placeholder.object = f"""
            <div style="font-size:16px; font-weight:bold; color:#333;">
            <h3>Metrics for Gradient Boosting</h3>
            <ul>
                <li>Accuracy: <span style="color:blue;">{accuracy_direct:.4f}</span></li>
                <li>Precision: <span style="color:green;">{precision_direct:.4f}</span></li>
                <li>Recall: <span style="color:orange;">{recall_direct:.4f}</span></li>
                <li>F1-score: <span style="color:red;">{f1_direct:.4f}</span></li>
            </ul>
            </div>
          """
        elif selected_classifier == "CNN":
            normalized_data = global_data["normalized_data"]
            test_labels_df = global_data["test_labels_df"]
            accuracy_direct, precision_direct, recall_direct, f1_direct, cm_cnn = cnn(normalized_data, test_labels_df)
            row1.clear()
            row1.append(cm_cnn)
            metrics_placeholder.object = f"""
            <div style="font-size:16px; font-weight:bold; color:#333;">
            <h3>Metrics for CNN</h3>
            <ul>
                <li>Accuracy: <span style="color:blue;">{accuracy_direct:.4f}</span></li>
                <li>Precision: <span style="color:green;">{precision_direct:.4f}</span></li>
                <li>Recall: <span style="color:orange;">{recall_direct:.4f}</span></li>
                <li>F1-score: <span style="color:red;">{f1_direct:.4f}</span></li>
            </ul>
            </div>
            """           
        else:
            status_message.object = "No classifier selected."
    else:
        status_message.object = "Please upload a file and process it before running the classifier."

file_path_field=pn.widgets.FileInput(name="file", accept=".csv",)

file_upload_button=pn.widgets.Button(name="Upload", button_type='primary',styles=custom_style)

loading_spinner = pn.indicators.LoadingSpinner(value=False, width=50, height=50)

status_message = pn.pane.Markdown("")

def upload_button_click(event):
    if file_path_field.value is not None:
        loading_spinner.value = True
        status_message.object = ""
        try:
            print("Reading file content...")
            out = io.BytesIO()
            file_path_field.save(out)
            out.seek(0)
            content = out.getvalue()
            print("File content preview:")
            print(content[:500])                                                       
            if len(content) == 0:
                raise ValueError("The uploaded file is empty.")
            try:
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                print("CSV file preview:")
                print(df.head())
            except Exception as e:
                raise ValueError(f"Could not parse the file as a CSV: {e}")
            normalized_data, test_labels_df = file_processing(io.BytesIO(content))
            global_data["normalized_data"] = normalized_data
            global_data["test_labels_df"] = test_labels_df
            print("File content read successfully. Processing file...")
            loading_spinner.value = False
            status_message.object = "Data processing is done."
            print("Data processing completed.")
        except Exception as e:
            loading_spinner.value = False
            status_message.object = f"Error: {e}"
            print(f"Error during file processing: {e}")
    else:
        status_message.object = "No file uploaded."
        print("No file uploaded.")

# Attach the callback to the upload button

file_upload_button.on_click(upload_button_click)

classifier_select = pn.widgets.Select(name="Select classifier", options=["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression", "XG Boost","CNN"], width=300,)
run_button = pn.widgets.Button(name="Run",button_type='primary',styles=custom_style)

# Attach the classifier runner to the Run button
run_button.on_click(run_classifier)

conf_matrix_placeholder = pn.pane.Markdown("<div style='font-size:18px;'><b>Confusion Matrix will be displayed here</b></div>", sizing_mode='stretch_width')
metrics_placeholder = pn.pane.Markdown("<div style='font-size:18px;'><b>Metrics will be displayed here</b></div>", sizing_mode='stretch_width')

# Layout setup
# row1 = pn.Row(conf_matrix_placeholder, sizing_mode='stretch_width')
# row2 = pn.Row(metrics_placeholder, sizing_mode='stretch_width')



# main_column = pn.Column(row1, row2, sizing_mode='stretch_width')

# # Template setup
# template = pn.template.BootstrapTemplate(
#     title="GUI of classifiers for person detection in an Office environment",
#     sidebar=pn.Column(
#         pn.pane.Markdown("## Enter file path",styles=style_to_display),
#         file_path_field,
#         file_upload_button,
#         loading_spinner,
#         status_message,
#         pn.pane.Markdown("## Select any classifier",styles=style_to_display),
#         classifier_select,
#         run_button,
#         styles={"width": "100%", "padding": "15px"}
#     ),
#     main=[pn.Row(pn.layout.HSpacer(), logo),main_column],
#     header_background='#6424db',
#     site="Frankfurt University of Applied Sciences",
#     sidebar_width=350,
#     busy_indicator=None
# )

# # Serve the template
# template.servable()



# Layout setup
import panel as pn

# Main content placeholders
row1 = pn.Row(conf_matrix_placeholder, sizing_mode='stretch_width')
row2 = pn.Row(metrics_placeholder, sizing_mode='stretch_width')

# Main column for content
main_column = pn.Column(row1, row2, sizing_mode='stretch_width')

# Logo column with CSS for positioning
logo_column = pn.Column(logo, sizing_mode='fixed', width=100, height=100)

# Template setup
template = pn.template.BootstrapTemplate(
    title="GUI for Optimization of classifiers for person detection",
    sidebar=pn.Column(
        pn.pane.Markdown("## Enter file path", styles=style_to_display),
        file_path_field,
        file_upload_button,
        loading_spinner,
        status_message,
        pn.pane.Markdown("## Select any classifier", styles=style_to_display),
        classifier_select,
        run_button,
        styles={"width": "100%", "padding": "15px"}
    ),
    main=pn.Row(
        pn.Column(main_column, sizing_mode='stretch_width'),
        pn.layout.HSpacer(),  # Adds space between main content and logo
        pn.Column(
            logo_column,
            sizing_mode='fixed',
            width=120,
            height=120,
            styles={'position': 'fixed', 'top': '30px', 'right': '40px'}
        ),
        sizing_mode='stretch_width'
    ),
    header_background='#6424db',
    site="Frankfurt University of Applied Sciences",
    sidebar_width=350,
    busy_indicator=None
)

# Serve the template
template.servable()


# import panel as pn
# import io
# from dataprocessing import file_processing,rfc,gbm,svm,xgboost,lr
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
# import pandas as pd

# custom_style = {
#     'padding': '14px',
#     'font-style': 'italic',
#     'border-radius': '12px',
#     'width': '180px'
# }

# style_to_display= {
#     'padding': '14px',
#     'font-style': 'italic',
#     'border-radius': '12px',
#     'font-size': '13pt',
#     'width': '180px',
#     'font-weight': 'bold'
# }

# logopath=r"C:\Users\tdegner\Desktop\test_project_lpa\logo.PNG"
# logo = pn.panel(logopath,width=200)
# global_data = {"normalized_data": None, "test_labels_df": None}

# def svm():
#     pass
# def Lr():
#     pass
# def XGboost():
#     pass
# def GBM():
#     pass


# def run_classifier(event):
#     if global_data["normalized_data"] is not None and global_data["test_labels_df"] is not None:
#         selected_classifier = classifier_select.value
#         if selected_classifier == "Random Forest":
#             normalized_data = global_data["normalized_data"]
#             test_labels_df = global_data["test_labels_df"]
#             accuracy_direct, precision_direct, recall_direct, f1_direct, pane = rfc(normalized_data, test_labels_df)
#             # conf_matrix_placeholder.object = pane
#             metrics_placeholder.object = f"""
#             ### Metrics for RFC
#             - **Accuracy:** {accuracy_direct:.4f}
#             - **Precision:** {precision_direct:.4f}
#             - **Recall:** {recall_direct:.4f}
#             - **F1-score:** {f1_direct:.4f}
#             """
#         elif selected_classifier == "SVM":
#             normalized_data = global_data["normalized_data"]
#             test_labels_df = global_data["test_labels_df"]
#             accuracy_direct, precision_direct, recall_direct, f1_direct, pane = svm(normalized_data, test_labels_df)
#             # conf_matrix_placeholder.object = pane
#             metrics_placeholder.object = f"""
#             ### Metrics for SVM
#             - **Accuracy:** {accuracy_direct:.4f}
#             - **Precision:** {precision_direct:.4f}
#             - **Recall:** {recall_direct:.4f}
#             - **F1-score:** {f1_direct:.4f}
#             """
#         elif selected_classifier == "Logistic Regression":
#             normalized_data = global_data["normalized_data"]
#             test_labels_df = global_data["test_labels_df"]
#             accuracy_direct, precision_direct, recall_direct, f1_direct, pane = lr(normalized_data, test_labels_df)
#             # conf_matrix_placeholder.object = pane
#             metrics_placeholder.object = f"""
#             ### Metrics for LR
#             - **Accuracy:** {accuracy_direct:.4f}
#             - **Precision:** {precision_direct:.4f}
#             - **Recall:** {recall_direct:.4f}
#             - **F1-score:** {f1_direct:.4f}
#             """
#         elif selected_classifier == "XG Boost":
#             normalized_data = global_data["normalized_data"]
#             test_labels_df = global_data["test_labels_df"]
#             accuracy_direct, precision_direct, recall_direct, f1_direct, pane = xgboost(normalized_data, test_labels_df)
#             # conf_matrix_placeholder.object = pane
#             metrics_placeholder.object = f"""
#             ### Metrics for XG Boosting
#             - **Accuracy:** {accuracy_direct:.4f}
#             - **Precision:** {precision_direct:.4f}
#             - **Recall:** {recall_direct:.4f}
#             - **F1-score:** {f1_direct:.4f}
#             """
#         elif selected_classifier == "Gradient Boosting":
#             normalized_data = global_data["normalized_data"]
#             test_labels_df = global_data["test_labels_df"]
#             accuracy_direct, precision_direct, recall_direct, f1_direct, pane = gbm(normalized_data, test_labels_df)
#             # conf_matrix_placeholder.object = pane
#             metrics_placeholder.object = f"""
#             ### Metrics for Gradient Boosting
#             - **Accuracy:** {accuracy_direct:.4f}
#             - **Precision:** {precision_direct:.4f}
#             - **Recall:** {recall_direct:.4f}
#             - **F1-score:** {f1_direct:.4f}
#             """
#             pass
#         else:
#             status_message.object = "No classifier selected."
#     else:
#         status_message.object = "Please upload a file and process it before running the classifier."

# file_path_field=pn.widgets.FileInput(name="file", accept=".csv")

# file_upload_button=pn.widgets.Button(name="Upload", button_type='primary')

# loading_spinner = pn.indicators.LoadingSpinner(value=False, width=50, height=50)

# status_message = pn.pane.Markdown("")

# def upload_button_click(event):
#     if file_path_field.value is not None:
#         loading_spinner.value = True
#         status_message.object = ""
#         try:
#             print("Reading file content...")
#             out = io.BytesIO()
#             file_path_field.save(out)
#             out.seek(0)
#             content = out.getvalue()
#             print("File content preview:")
#             print(content[:500])                                                       
#             if len(content) == 0:
#                 raise ValueError("The uploaded file is empty.")
#             try:
#                 df = pd.read_csv(io.StringIO(content.decode('utf-8')))
#                 print("CSV file preview:")
#                 print(df.head())
#             except Exception as e:
#                 raise ValueError(f"Could not parse the file as a CSV: {e}")
#             normalized_data, test_labels_df = file_processing(io.BytesIO(content))
#             global_data["normalized_data"] = normalized_data
#             global_data["test_labels_df"] = test_labels_df
#             print("File content read successfully. Processing file...")
#             loading_spinner.value = False
#             status_message.object = "Data processing is done."
#             print("Data processing completed.")
#         except Exception as e:
#             loading_spinner.value = False
#             status_message.object = f"Error: {e}"
#             print(f"Error during file processing: {e}")
#     else:
#         status_message.object = "No file uploaded."
#         print("No file uploaded.")

# # Attach the callback to the upload button

# file_upload_button.on_click(upload_button_click)

# classifier_select = pn.widgets.Select(name="Select classifier", options=["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression", "XG Boost"], width=300)
# run_button = pn.widgets.Button(name="Run", button_type='primary')

# # Attach the classifier runner to the Run button
# run_button.on_click(run_classifier)

# conf_matrix_placeholder = pn.pane.Markdown("**Confusion Matrix will be displayed here**", sizing_mode='stretch_width')
# metrics_placeholder = pn.pane.Markdown("**Metrics will be displayed here**", sizing_mode='stretch_width')
# main_tab = pn.Tabs(
#     ('Confusion Matrix', conf_matrix_placeholder),
#     ('Metrics', metrics_placeholder)
# )
# template = pn.template.BootstrapTemplate(
#     title="GUI of classifiers for person detection in a Office environment",
#     sidebar=pn.Column(
#     pn.pane.Markdown("## Enter file path"),
#     file_path_field,
#     file_upload_button,
#     loading_spinner,
#     status_message,
#     pn.pane.Markdown("## Select any classifer"),
#     classifier_select,
#     run_button,
#     styles={"width": "100%", "padding": "15px"}
# ),
#     header_background='#6424db',
#     site="Frankfurt University of Applied sciences",
#     sidebar_width=400,
#     busy_indicator=None,
#     main=[main_tab]
# )
# template.servable()