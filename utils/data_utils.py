import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def create_sequences(data, n_past):
    X, Y = [], []
    L = len(data)
    # 預測窗口為 5，包含最後兩天與未來三天
    for i in range(L - (n_past + 3)):
        X.append(data[i:i + n_past])
        # 取第 4 個特徵 (Close)，範圍是從 n_past-2 到 n_past+3
        Y.append(data[i + n_past - 2: i + n_past + 3][:, 3])
    
    return torch.Tensor(np.array(X)), torch.Tensor(np.array(Y))

def preprocess_data(df, train_ratio=0.6, val_ratio=0.2, n_past=20):
    # 移除不需要的欄位
    data = df[[c for c in df.columns if c not in ['Date', 'Adj Close']]].values
    
    # 計算索引
    train_ind = int(len(data) * train_ratio)
    val_ind = int(len(data) * val_ratio)
    
    # 分割資料
    train_raw = data[:train_ind]
    val_raw = data[train_ind : train_ind + val_ind]
    test_raw = data[train_ind + val_ind:]
    
    # 標準化 (僅 fit train 避免 Data Leakage)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)
    
    # 建立序列
    X_train, Y_train = create_sequences(train_scaled, n_past)
    X_val, Y_val = create_sequences(val_scaled, n_past)
    X_test, Y_test = create_sequences(test_scaled, n_past)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler

def stack_average(data):
    batch_size, seq_len = data.shape[0], data.shape[1]
    total_day = batch_size + seq_len - 1
    count_list = []
    a = 0
    for i in range(total_day):
        count = 0
        if i < seq_len - 1:
            path = i
            for k in range(i + 1):
                count += data[k][path]
                path -= 1
            count_list.append(count / (i + 1))
        elif i > total_day - seq_len:
            b = batch_size - (total_day - i)
            path = seq_len - 1
            for l in range(b, total_day - i + b):
                count += data[l][path]
                path -= 1
            count_list.append(count / (total_day - i))
        else:
            path = seq_len - 1
            for j in range(a, seq_len + a):
                count += data[j][path]
                path -= 1
            a += 1
            count_list.append(count / seq_len)
    return np.array(count_list)

def inverse_transform(scaled_data, scaler):
    """
    針對 Close Price (index 3) 進行反標準化
    Formula: x = scaled * std + mean
    """
    mean = scaler.mean_[3]
    std = np.sqrt(scaler.var_[3])
    return (scaled_data * std) + mean