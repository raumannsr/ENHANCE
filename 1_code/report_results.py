import pandas as pd

def report_acc_and_loss(history, filename):
    hist_df = pd.DataFrame(history.history)
    with open(filename, mode='w') as f:
        hist_df.to_csv(f)

def report_auc(df, filename):
    with open(filename, mode='w') as f:
        df.to_csv(f, index=False)