import numpy as np
from sklearn.model_selection import train_test_split

n_entries = 1000
n_per_entry = 8

def num_ones(arr):
    count = 0
    for i in arr:
        if i == 1:
            count += 1
    return count

def create_labels(arr):
    labels = []
    for i in arr:
        if num_ones(i) % 2 == 0:
            labels.append(1)
        else:
            labels.append(0)
    return labels

X = np.random.randint(0, 2, (n_entries, n_per_entry), dtype=int)

y = create_labels(X)
y = np.array(y).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=True)
print("-"*10,"^Handling data...^","-"*10)

if __name__ == "__main__":
    print(n_entries)