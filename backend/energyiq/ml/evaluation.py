import calendar
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# percent of the largest test residuals dropped before scoring accuracy
TRIM_PCT = 2

# nesting of the group breakdown; the full key is one threshold group
GROUP_LEVELS = ["utility", "sitename", "meterid", "month", "is_weekend", "hour_bucket"]
PRINT_LEVELS = 4  # print down to month

def trim_mask(y_true, y_pred, trim_pct):
    """
    False for the rows with the largest trim_pct% of absolute residuals, True for the rest.
    The share dropped is fixed, so unlike scoring only unflagged rows, a worse model can't
    remove more of its own error.
    """
    abs_err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    n_drop = int(len(abs_err) * trim_pct / 100)
    keep = np.ones(len(abs_err), dtype=bool)
    if n_drop:  # [-0:] would select every row
        keep[np.argsort(abs_err)[-n_drop:]] = False
    return keep

def accuracy_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ------------------------------------------------------------- group breakdown

def group_stats(meta, y_true, y_pred):
    """
    MAE and RMSE per group at every depth of GROUP_LEVELS, as {depth: frame}.
    Leaf groups are scored on their readings; every level above is the unweighted mean of
    its child groups, so each group counts equally however many readings it has.
    """
    err = np.asarray(y_true) - np.asarray(y_pred)
    frame = meta[GROUP_LEVELS].reset_index(drop=True).assign(abs_err=np.abs(err), sq_err=err ** 2)
    leaves = frame.groupby(GROUP_LEVELS)
    stats = {len(GROUP_LEVELS): pd.DataFrame({
        "MAE": leaves["abs_err"].mean(),
        "RMSE": leaves["sq_err"].mean() ** 0.5,
        "n_groups": 1,
    })}

    for depth in range(len(GROUP_LEVELS) - 1, 0, -1):
        children = stats[depth + 1].groupby(level=list(range(depth)) if depth > 1 else 0)
        stats[depth] = pd.DataFrame({
            "MAE": children["MAE"].mean(),
            "RMSE": children["RMSE"].mean(),
            "n_groups": children["n_groups"].sum(),
        })
    return stats

def group_label(level, value):
    match level:
        case "meterid":
            return f"meter {value}"
        case "month":
            return calendar.month_abbr[value]
        case "is_weekend":
            return "weekend" if value else "weekday"
    return str(value)

def print_group_breakdown(meta, y_true, y_pred, label):
    """Average group MAE/RMSE as a tree: utility > building > meter > month (PRINT_LEVELS deep)."""
    stats = group_stats(meta, y_true, y_pred)

    # (key, name, row) for every printed node; sorting the keys puts each parent right before its children
    lines = []
    for depth in range(1, PRINT_LEVELS + 1):
        for key, row in stats[depth].iterrows():
            key = key if isinstance(key, tuple) else (key,)
            name = "  " * depth + group_label(GROUP_LEVELS[depth - 1], key[-1])
            lines.append((key, name, row))
    lines.sort(key=lambda line: line[0])

    top = stats[1]
    width = max(len(name) for _, name, _ in lines) + 2
    row_fmt = "{:<{w}}MAE {:>10.3f}   RMSE {:>10.3f}   ({} groups)"

    print(f"\nGroup breakdown ({label}; each level is the mean of its child groups)")
    print(row_fmt.format("all", top["MAE"].mean(), top["RMSE"].mean(),
                         int(top["n_groups"].sum()), w=width))
    for _, name, row in lines:
        print(row_fmt.format(name, row["MAE"], row["RMSE"], int(row["n_groups"]), w=width))


# ------------------------------------------------------------- entry point

def evaluate(y_test, y_pred, y_train, train_pred, meta_test, verbose=False):
    """Test accuracy with the largest TRIM_PCT% of residuals trimmed, plus the training MAE."""
    kept = trim_mask(y_test, y_pred, TRIM_PCT)
    y_test, y_pred, meta_test = y_test[kept], y_pred[kept], meta_test[kept]
    if verbose:
        print_group_breakdown(meta_test, y_test, y_pred,
                              label=f"test, largest {TRIM_PCT}% of residuals removed")
    return {
        **accuracy_metrics(y_test, y_pred),
        "train_MAE": float(mean_absolute_error(y_train, train_pred)),
    }