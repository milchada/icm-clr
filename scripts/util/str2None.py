def str2None(x):
    if isinstance(x, str) and x == "None":
        return None
    else:
        return float(x)

