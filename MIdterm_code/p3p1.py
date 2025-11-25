import pandas as pd

df = pd.DataFrame({
    'kind': ['cat', 'dog', 'cat', 'dog'],
    'height': [9.1, 6.0, 9.5, 34.0],
    'weight': [7.9, 7.5, 9.9, 198.0]
})

grouped = df.groupby('kind')
print(grouped.max())

# now with cuDF
import cudf
cdf = cudf.DataFrame({
    'kind': ['cat', 'dog', 'cat', 'dog'],
    'height': [9.1, 6.0, 9.5, 34.0],
    'weight': [7.9, 7.5, 9.9, 198.0]
})
grouped_cudf = cdf.groupby('kind')
print(grouped_cudf.max())