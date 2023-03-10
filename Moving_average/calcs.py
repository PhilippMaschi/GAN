from sklearn.preprocessing import minmax_scale

def scale(group, col):
    group[col] = minmax_scale(group[col])
    return group