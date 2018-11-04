import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression

def linear_regression(df, group, col='UF TMP'):
    # Select the group from the data frame
    subdf = df[df.group == group]
    # Generate index in a 2D matrix
    idx_ = np.arange(len(subdf.index)).reshape(-1,1)
    # Fit the column selected to the index
    lm = LinearRegression()
    lm.fit(idx_, subdf[col])
    LinearRegression(copy_X=True, fit_intercept=True, normalize=False)

    return lm.coef_, lm.intercept_