import pandas as pd
from sklearn.metrics import f1_score


NOT_CLASS_VALUE = '-1'


def get_metric(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    pred_df = pred_df.rename({'label': 'predict_label'}, axis=1)
    result = pd.merge(gt_df, pred_df, on='filename', how='left').fillna(NOT_CLASS_VALUE)
    gt = result['label'].values
    pred = result['predict_label'].values
    return f1_score(gt, pred, average='micro')


def get_public_metric(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    gt_df = gt_df.loc[gt_df['test_type'] == 'public'].reset_index(drop=True)
    return get_metric(gt_df, pred_df)
