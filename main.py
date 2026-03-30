import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from models.transformer import Transformer
from utils.data_utils import preprocess_data, stack_average, inverse_transform

# 設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_to_save_model = './save_model/'
os.makedirs(path_to_save_model, exist_ok=True)

# 1. 資料載入
df = pd.read_csv("data/2330.TW.csv")
X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler = preprocess_data(df, train_ratio=0.7)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=False)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=32, shuffle=False)

# 2. 初始化
model = Transformer().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 預測函式
def make_prediction(model, src, forecast_window):
    model.eval()
    tgt = src[:, -1, 0].unsqueeze(-1).unsqueeze(-1)
    for _ in range(forecast_window-1):
        prediction = model(src, tgt, device)
        last_val = prediction[:, -1, :].unsqueeze(-1)
        tgt = torch.cat((tgt, last_val.detach()), dim=1)
    return model(src, tgt, device)

# 3. 訓練迴圈
train_losses, val_losses = [], []
min_val_loss = float('inf')
best_model_path = ""

for epoch in range(500):
    model.train()
    temp_train_loss = []
    for data, targets in train_loader:
        data, targets = data.to(device), targets.unsqueeze(-1).to(device)
        optimizer.zero_grad()
        outputs = model(data, targets, device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        temp_train_loss.append(loss.item())
    
    avg_train_loss = sum(temp_train_loss) / len(temp_train_loss)
    train_losses.append(avg_train_loss)

    # 驗證
    model.eval()
    temp_val_loss = []
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.unsqueeze(-1).to(device)
            outputs = make_prediction(model, data, targets.shape[1])
            v_loss = criterion(outputs, targets)
            temp_val_loss.append(v_loss.item())
    
    avg_val_loss = sum(temp_val_loss) / len(temp_val_loss)
    val_losses.append(avg_val_loss)

    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        best_model_path = os.path.join(path_to_save_model, f"best_model.pth")
        torch.save(model.state_dict(), best_model_path)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.6f}, Val Loss {avg_val_loss:.6f}")

# 4. 繪製訓練結果
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.legend(); plt.savefig("train_loss.png")

# 5. 測試與反標準化
model.load_state_dict(torch.load(best_model_path))
all_preds, all_gts = [], []

with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device)
        outputs = make_prediction(model, data, targets.shape[1]).cpu()
        all_preds.append(outputs)
        all_gts.append(targets.unsqueeze(-1))

# 合併、反標準化、平均化
final_preds = inverse_transform(stack_average(torch.cat(all_preds)), scaler)
final_gts = inverse_transform(stack_average(torch.cat(all_gts)), scaler)

# 相關係數與繪圖
corr = np.corrcoef(final_gts, final_preds)[0, 1]
print(f"Final Correlation: {corr:.4f}")

plt.figure(); plt.plot(final_gts, label='Actual'); plt.plot(final_preds, label='Forecast')
plt.legend(); plt.savefig("test_result.png")