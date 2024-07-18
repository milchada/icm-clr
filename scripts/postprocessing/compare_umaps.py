#compare_umaps

def comparison(u, df, label1, label2, gridsize=10):
    hist1, xedge, yedge = np.histogram2d(u[:,0], u[:,1], weights=df[label1], bins=gridsize)
    hist2, xedge, yedge = np.histogram2d(u[:,0], u[:,1], weights=df[label2], bins=gridsize)
    mask = (hist1 != 0) * (hist2 != 0)
    return np.corrcoef(hist1[mask], hist2[mask])