"""
Taster Score Classification using the script **train_test_labels.py**

This script trains and evaluates a Ridge classifier to predict categorical labels (e.g. taster identity, wine variety,
cave, age) based on averaged sensory evaluation scores of Champagne wines.
The features correspond to numerical scores given by tasters on different attributes (e.g. acid, balance),
and the labels are drawn from associated metadata.

Data is preprocessed by collapsing replicates per (wine, taster) pair and cleaning non-numeric values.
Stratified K-Fold cross-validation is repeated multiple times to obtain a robust estimate of classification accuracy,
and a normalized confusion matrix is plotted at the end for each classification target to visualize model performance
across classes.

"""

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # --- Parameters ---
    label_columns = ['prod area', 'variety', 'cave', 'age']
    csv_path = "/home/luiscamara/Documents/datasets/Champagnes/test.csv"
    n_splits = 10
    n_repeats = 20
    random_seed = 42

    # --- Load and clean dataset ---
    df = pd.read_csv(csv_path, skiprows=1)
    df = df.iloc[1:]  # Skip redundant header row
    df.columns = [col.strip().lower() for col in df.columns]  # Normalize column names

    # --- Prepare features ---
    df_selected = df.loc[:, 'code vin':'ageing']
    df_selected['taster'] = df['taster']
    df_selected.iloc[:, 1:-1] = df_selected.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce')
    df_selected = df_selected.dropna(axis=1, how='all')
    df_grouped = df_selected.groupby(['code vin', 'taster']).mean().dropna().reset_index()

    # --- Storage for results ---
    conf_matrices = []
    label_encoders = []
    class_counts_per_label = []

    # --- Loop through classification targets ---
    for label_column in label_columns:
        print(f"\nRunning Ridge Classifier for label: {label_column}\n")

        labels_df = df[['code vin', label_column, 'taster']].dropna()
        labels_df = labels_df.drop_duplicates(subset=['code vin', 'taster'])
        labels_df.columns = ['code vin', 'label', 'taster']

        merged_df = df_grouped.merge(labels_df, on=['code vin', 'taster'])
        def balance_classes(df, label_col, max_samples_per_class=10, random_state=42):
            return (
                df.groupby(label_col, group_keys=False)
                .apply(lambda x: x.sample(n=min(len(x), max_samples_per_class), random_state=random_state))
                .reset_index(drop=True)
            )
        # merged_df = balance_classes(merged_df, label_col='label', max_samples_per_class=10)
        wine_labels = merged_df['label'].values
        feature_matrix = merged_df.drop(columns=['code vin', 'taster', 'label']).to_numpy()

        le = LabelEncoder()
        y = le.fit_transform(wine_labels)
        sample_counts = np.bincount(y)
        class_counts = dict(zip(le.classes_, sample_counts))
        class_counts_per_label.append(class_counts)

        # Print sample counts per class
        print("Samples per class:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")


        all_true = []
        all_pred = []

        for repeat in tqdm(range(n_repeats)):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed + repeat)

            for train_idx, test_idx in skf.split(feature_matrix, y):
                X_train, X_test = feature_matrix[train_idx], feature_matrix[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # clf = RidgeClassifier(class_weight='balanced')
                clf = RidgeClassifier(class_weight='balanced')
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)

                all_true.extend(y_test)
                all_pred.extend(y_pred)
            pred_counts = np.bincount(all_pred, minlength=len(le.classes_))
            true_counts = np.bincount(all_true, minlength=len(le.classes_))
            # print("Predicted counts:")
            # for cls, count in zip(le.classes_, pred_counts):
            #     print(f"  {cls}: {count}")
            # print("True counts:")
            # for cls, count in zip(le.classes_, true_counts):
            #     print(f"  {cls}: {count}")

        cm = confusion_matrix(all_true, all_pred, labels=np.arange(len(le.classes_)), normalize='true')
        conf_matrices.append(cm)
        label_encoders.append(le)

    # --- Plot all confusion matrices ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, (cm, le, title) in enumerate(zip(conf_matrices, label_encoders, label_columns)):
        print(f"\nLabel column: {title}")
        print("Label order (classes_):", list(le.classes_))
        print("Class counts:", class_counts_per_label[i])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot(ax=axes[i], cmap="Blues", values_format=".2f", xticks_rotation=45, colorbar=False)
        axes[i].set_title(f"Confusion Matrix – {title}")

    plt.tight_layout()
    plt.show()