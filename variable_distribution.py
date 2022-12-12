# specific columns
def count_elements(seq) -> dict:
    """Tally elements from `seq`."""
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist


# * VARIABLE DISTRIBUTION
# TODO: barplot date and hour

def univariate_analysis(data, fig_name):
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    fig.suptitle('univariate distribution')
    sns.kdeplot(ax=axes[0, 0], data=data['date'])
    sns.kdeplot(ax=axes[0, 1], data=data['hour'])
    sns.kdeplot(ax=axes[0, 2], data=data['consumed energy'])
    sns.kdeplot(ax=axes[1, 0], data=data['exported energy'])
    sns.kdeplot(ax=axes[1, 1], data=data['reactive energy Q1'])
    sns.kdeplot(ax=axes[1, 2], data=data['reactive energy Q2'])
    sns.kdeplot(ax=axes[2, 0], data=data['reactive energy Q3'])
    sns.kdeplot(ax=axes[2, 1], data=data['reactive energy Q4'])
    sns.kdeplot(ax=axes[2, 2], data=data['contacted power P1'])
    plt.savefig(fig_name)
    return plt.show()


# variable correlation
f = plt.figure(figsize=(19, 15))
corrMatrix = data.corr()
ax = sns.heatmap(corrMatrix, annot=True)
plt.show()
