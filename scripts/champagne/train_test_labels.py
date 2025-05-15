if __name__ == "__main__":
    """
    Ridge Classifier Evaluation with Cross-Validation
    
    This script loads the Champagne dataset, prepares chemical feature vectors,
    encodes the selected label, and trains a Ridge classifier using Stratified K-Fold
    cross-validation. The whole process is repeated multiple times to get
    a robust estimate of accuracy.
    
    Author: Luis (or your name)
    """

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
    label_column = 'taster'  # Change this to 'prod area', 'variety',  'cave', 'age', 'ageing', etc..
    csv_path = "/home/luiscamara/Documents/datasets/Champagnes/test.csv"
    n_splits = 10               # Inner cross-validation folds
    n_repeats = 50             # How many times to repeat the CV loop
    random_seed = 42

    # --- Load and preprocess dataset ---
    df = pd.read_csv(csv_path, skiprows=1)
    df = df.iloc[1:]
    df.columns = [col.strip().lower() for col in df.columns]

    # Prepare feature matrix and labels
    df_selected = df.loc[:, 'code vin':'acid']
    df_selected['taster'] = df['taster']
    df_selected.iloc[:, 1:-1] = df_selected.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce')
    df_grouped = df_selected.groupby(['code vin', 'taster']).mean().dropna()

    labels_df = df[['code vin', label_column, 'taster']].dropna()
    labels_df = labels_df.drop_duplicates(subset=['code vin', 'taster'])
    labels_df.columns = ['code vin', 'label', 'taster']
    df_grouped = df_grouped.reset_index().merge(labels_df, on=['code vin', 'taster'])
    wine_labels = df_grouped['label'].values
    df_grouped = df_grouped.drop(columns=['code vin', 'taster', 'label'])

    X = StandardScaler().fit_transform(df_grouped)
    le = LabelEncoder()
    y = le.fit_transform(wine_labels)

    # --- CV Loop: accuracy + confusion matrix ---
    all_accuracies = []
    all_true = []
    all_pred = []

    # # --- Print LaTeX tables for sample counts ---
    #
    # # Clean metadata (load again just for counting purposes)
    # metadata = pd.read_csv(csv_path, skiprows=1)
    # metadata = metadata.iloc[1:]
    # metadata.columns = [col.strip().lower() for col in metadata.columns]
    #
    # # List of properties (columns) to process
    # properties = ['taster', 'variety', 'cave', 'prod area', 'age']
    #
    # # Make a cleaned dataframe for counting
    # counting_df = df[['code vin', 'taster', 'variety', 'cave', 'prod area', 'age']].dropna()
    # counting_df = counting_df.drop_duplicates(subset=['code vin', 'taster'])
    #
    # properties = ['taster', 'variety', 'cave', 'prod area', 'age']
    #
    # # After collapsing replicates:
    # counting_df = df[['code vin', 'taster', 'variety', 'cave', 'prod area', 'age']].dropna()
    # counting_df = counting_df.drop_duplicates(subset=['code vin', 'taster'])
    #
    # for prop in properties:
    #     counts = counting_df[prop].value_counts().sort_index()
    #
    #     print(f"\n% --- LaTeX table for {prop} ---\n")
    #     print("\\begin{table}[H]")
    #     print("    \\centering")
    #     print("    \\setlength{\\tabcolsep}{8pt}")
    #     print("    \\renewcommand{\\arraystretch}{1.2}")
    #     print("    \\begin{tabular}{" + "c " * len(counts) + "}")
    #     print("        \\toprule")
    #
    #     # Header: categories (bold)
    #     header_row = " & ".join(f"\\textbf{{{str(label)}}}" for label in counts.index)
    #     print(f"        {header_row} \\\\")
    #
    #     print("        \\midrule")
    #
    #     # Row: corresponding counts
    #     count_row = " & ".join(str(count) for count in counts.values)
    #     print(f"        {count_row} \\\\")
    #
    #     print("        \\bottomrule")
    #     print("    \\end{tabular}")
    #     print(f"    \\caption{{Number of samples per {prop} (after collapsing duplicates)}}")
    #     print(f"    \\label{{tab:{prop.replace(' ', '_')}_counts}}")
    #     print("\\end{table}\n")

    print(f"Running Ridge Classifier with {n_splits}-fold CV repeated {n_repeats} times...\n")

    for repeat in tqdm(range(n_repeats)):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed + repeat)
        repeat_accuracies = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # clf = RidgeClassifier(class_weight='balanced')
            clf = RidgeClassifier(class_weight='balanced')
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            repeat_accuracies.append(acc)

            all_true.extend(y_test)
            all_pred.extend(y_pred)

        all_accuracies.append(np.mean(repeat_accuracies))

    # --- Accuracy Summary ---
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    print(f"\nFinal averaged accuracy over {n_repeats} runs: {mean_acc:.3f} ± {std_acc:.3f}")

    white_to_blue = mcolors.LinearSegmentedColormap.from_list("white_to_blue", ["white", "blue"])

    # --- Confusion Matrix (normalized per true class) ---
    cm = confusion_matrix(all_true, all_pred, normalize='true')

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=white_to_blue, ax=ax, xticks_rotation=45, colorbar=True, values_format=".2f" )
    plt.title(f"Normalized Confusion Matrix – {label_column}")
    plt.tight_layout()
    plt.show()