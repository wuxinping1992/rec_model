import pandas as pd


def load_data(filepath='./data/tiny_train_input.csv'):
    train_data = {}

    raw_data = pd.read_csv(filepath, header=None)
    raw_data.columns = ['c' + str(i) for i in range(raw_data.shape[1])]
    label = raw_data.c0.values
    label = label.reshape(len(label), 1)
    train_data['y_train'] = label


    co_features = pd.DataFrame()
    ca_features = pd.DataFrame()
    co_cols = []
    ca_cols = []
    feat_dict = {}
    cnt = 0
    for i in range(1, raw_data.shape[1]):
        target = raw_data.iloc[:, i]
        col = target.name
        l = len(set(target))
        if l > 10:
            target = (target - target.mean()) / target.std()
            co_features = pd.concat([co_features, target], axis=1)
            feat_dict[col] = cnt
            cnt += 1
            co_cols.append(col)
        else:
            us = target.unique()
            print(us)
            feat_dict[col] = dict(zip(us, range(cnt, len(us) + cnt)))
            ca_features = pd.concat([ca_features, target], axis=1)
            cnt += len(us)
            ca_cols.append(col)
    feat_dim = cnt
    feature_values = pd.concat([co_features, ca_features], axis=1)
    feature_index = feature_values.copy()
    for i in feature_index.columns:
        if i in co_features:
            feature_index[i] = feat_dict[i]
        else:
            feature_index[i] = feature_index[i].map(feat_dict[i])
            feature_values[i] = 1.
    train_data['Xi'] = feature_index.values.tolist()
    train_data['Xv'] = feature_values.values.tolist()
    train_data['feat_dim'] = feat_dim
    return train_data


if __name__ == "__main__":
    train_data = load_data()
    print(train_data['feat_dim'])


