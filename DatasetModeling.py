import pandas as pd

def balance_dataset(train_df, ratio=2):
    label_freqs = train_df["label"].value_counts()
    min_freq = label_freqs.min()
    max_freq = label_freqs.max()
    print(min_freq, max_freq)
    balanced_train_df = pd.DataFrame(columns=train_df.columns)
    for label in train_df["label"].unique():
        freq = int(max_freq * ratio * (min_freq / max_freq))
        freq = min(freq, label_freqs[label])
        freq = max(freq, min_freq)
        balanced_train_df = balanced_train_df.append(train_df[train_df["label"] == label].sample(freq), ignore_index=True)
    return balanced_train_df

def inductive_dataset(train_df, unseen_labels):
    # Quita cualquier imagen y mascaras que contenga unseen_labels
    inductive_train_df = pd.DataFrame(columns=train_df.columns)
    for image_id, group in train_df.groupby('image'):
        if not group['label'].isin(unseen_labels).any():
            inductive_train_df = inductive_train_df.append(group)
    return inductive_train_df

def transductive_dataset(train_df, seen_labels):
    # Quita cualquier mascara que contenga unseen_labels
    transductive_train_df = pd.DataFrame(columns=train_df.columns)
    for label in seen_labels:
        transductive_train_df = transductive_train_df.append(train_df[train_df['label'] == label])
    return transductive_train_df
