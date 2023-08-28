def str2None(x):
    if x == "None":
        return None
    elif isinstance(x, list):
        return list(map(str2None, x))
    else:
        return x

