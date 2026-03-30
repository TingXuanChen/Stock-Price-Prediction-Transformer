# Stock-Price-Prediction-Transformer
<div align="right">
  <strong>[ 🇹🇼 繁體中文 ](#-繁體中文) | [ 🇺🇸 English ](#-english-version)</strong>
</div>

---

# 🇹🇼 繁體中文

##  2330.TW Stock Prediction: From LSTM to Transformer
本專案實作了一個基於 **Transformer (Encoder-Decoder)** 架構的台積電 (2330.TW) 股價預測模型。有別於傳統的 RNN/LSTM 模型，本專案透過 Self-Attention 機制捕捉時序資料中的長距離依賴關係，並針對滑動視窗預測加入了獨創的資料處理與還原演算法。

###  致謝與參考來源 (Credits)
本專案的基礎數據抓取與時序處理邏輯參考了 [Peaceful0907 的 Medium 文章：Time Series Prediction — LSTM 的各種用法](https://peaceful0907.medium.com/time-series-prediction-lstm%E7%9A%84%E5%90%84%E7%A8%AE%E7%94%A8%E6%B3%95-ed36f0370204)。

** 本專案相對於參考來源的重大改進：**
1. **模型架構升級**：將原本的 LSTM 替換為 Transformer (Encoder-Decoder)。
2. **資料預處理優化**：重新設計資料切割邏輯，引入「重疊預測視窗」。
3. **預測機制**：實作了 **Autoregressive (自回歸)** 預測，確保驗證集與測試集絕對不會接觸到未來資訊。
4. **後處理演算法**：針對滑動視窗產生的重疊預測，開發了 **Stuck Average** 演算法，將碎片化的預測還原為連續時序。

###  資料處理策略 (Data Strategy)
1. **數據分割**：訓練集 (70%)、驗證集 (20%)、測試集 (10%)。滑動視窗以過去 20 天預測未來 5 天。
2. **重疊預測 (Overlapping Target)**：預測的 5 天包含 **「Data 最後 2 天」+「未來 3 天」**。重疊過去兩天是為了讓模型有更好的「錨定基準」，使預測曲線銜接更平滑。

###  模型架構 (Model Architecture)
* **特徵空間擴張 (Dimension Expansion)**：利用 `nn.Linear` 將 5 維特徵擴張至 128 維，解決特徵過少及 Multi-head Attention 的維度要求。
* **自定義 Mask 機制 (Subsequent Mask)**：實作下三角矩陣遮罩，防止模型在訓練時「偷看未來的答案」，迫使模型在預測第 $t$ 天時，只能參考 $1$ 到 $t-1$ 天的資訊。

###  訓練與測試流程
1. **自回歸預測 (Autoregressive Prediction)**：採取「一次生成一天」的策略，將前一天的預測值拼接回 Input，直到完成 5 天的預測。
2. **Stuck Average (預測還原演算法)**：收集時間軸上每個點的所有預測片段，根據重疊次數進行加權平均，將零散的 Sequence 恢復為連續的股價趨勢線。

###  如何執行 (How to run)
1. 將 `2330.TW.csv` 放入 `data/` 資料夾。
2. 執行 `python main.py`，最佳模型將儲存於 `save_model/`。

###  備註 (Note)
本專案的說明文件 (`README.md`) 由 AI 輔助整理與潤飾。若您對程式碼的邏輯、架構設計或內容描述有任何疑問與指教，非常歡迎開啟 [Issue] 或是提交 [Pull Request] 一同交流討論！

<br>
<br>

---

# 🇺🇸 English Version

## 📈 2330.TW Stock Prediction: From LSTM to Transformer
This project implements a stock price prediction model for TSMC (2330.TW) based on the **Transformer (Encoder-Decoder)** architecture. Moving beyond traditional RNN/LSTM models, this project leverages the Self-Attention mechanism to capture long-range dependencies in time-series data and introduces highly customized data processing and sequence reconstruction algorithms for sliding window forecasting.

### 🌟 Credits & References
The foundational data fetching and time-series preprocessing logic were inspired by [Peaceful0907's Medium article: Time Series Prediction — LSTM](https://peaceful0907.medium.com/time-series-prediction-lstm%E7%9A%84%E5%90%84%E7%A8%AE%E7%94%A8%E6%B3%95-ed36f0370204).

** Major Improvements Made in This Project:**
1. **Architecture Upgrade**: Replaced the original LSTM with a full Transformer (Encoder-Decoder) model.
2. **Data Preprocessing Optimization**: Redesigned the data splitting logic by introducing an "Overlapping Prediction Window."
3. **Inference Mechanism**: Implemented **Autoregressive** generation to ensure validation and test sets never access future data points.
4. **Post-processing Algorithm**: Developed a novel **Stuck Average** algorithm to reconstruct continuous time series from overlapping sliding window predictions.

###  Data Strategy
1. **Data Split**: Training (70%), Validation (20%), Testing (10%). The sliding window uses the past 20 days to predict the next 5 days.
2. **Overlapping Target**: The 5-day prediction target consists of **"the last 2 days of input data" + "3 future days."** Overlapping the past 2 days provides the model with a strong "anchoring baseline," ensuring smoother and more accurate trend transitions.

###  Model Architecture
* **Dimension Expansion**: Utilized `nn.Linear` to expand the 5 raw features (Open, High, Low, Close, Volume) to 128 dimensions ($d_{model}$). This enriches feature representation and satisfies the dimensional requirements of Multi-head Attention.
* **Custom Subsequent Mask**: Implemented a lower triangular mask to prevent data leakage during training. This forces the model to strictly rely on information from day $1$ to $t-1$ when predicting day $t$.

###  Training & Evaluation
1. **Autoregressive Prediction**: During evaluation, the model generates one day at a time. The predicted output is concatenated back into the input sequence iteratively until the 5-day forecast window is fulfilled.
2. **Sequence Reconstruction (Stuck Average)**: Since the sliding window approach generates multiple predictions for the same date, this algorithm collects all overlapping prediction fragments for each timestamp and calculates a weighted average, restoring a continuous and coherent stock price trend line.

###  How to Run
1. Place the `2330.TW.csv` file into the `data/` directory.
2. Execute `python main.py`. The best model weights will be saved in the `save_model/` directory.

### 備註 (Note)
 This `README.md` documentation was structured and refined with the assistance of AI. If you have any questions, suggestions, or spot any potential improvements regarding the code logic or descriptions, please feel free to open an [Issue] or submit a [Pull Request]!