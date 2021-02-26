def remove_low_variance_features(df, threshold=0.0):
    ok_id = []
    for colid, col in enumerate(df.values.T):
        try:
            if np.var(col) > threshold:
                ok_id.append(colid)
        except:
            pass
    
    return df.iloc[:, ok_id]

def remove_high_correlation_features(df, threshold=0.95):
    corrcoef = np.corrcoef(df.T.values.tolist())
    selected_or_not = {}
    for i, array in enumerate(corrcoef):
        if i not in selected_or_not.keys():
            selected_or_not[i] = True
        if selected_or_not[i]:
            for j, ary in enumerate(array): 
                if i < j:
                    if abs(ary) >= threshold:
                        selected_or_not[j] = False

    return df.iloc[:, [i for i, array in enumerate(corrcoef) if selected_or_not[i]]]
