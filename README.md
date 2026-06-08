<div align="center">
  <img src="assets/XiaomiMIMO.png" width="60%" alt="Xiaomi-MiMo" />
</div>

<div align="center">
  <h3>
    <b>
      <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span><br/>
      MiMo-V2.5-ASR：支援多語言、方言及複雜聲學場景的強健語音辨識模型<br/>
      <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    </b>
  </h3>
</div>

<br/>

<div align="center" style="line-height: 1;">
  |
  <a href="https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR" target="_blank">🤗 HuggingFace</a>
  &nbsp;|
  <a href="https://huggingface.co/spaces/XiaomiMiMo/MiMo-V2.5-ASR" target="_blank">🚀 線上 Demo</a>
  &nbsp;|
  <a href="https://mimo.xiaomi.com/mimo-v2-5-asr" target="_blank">📰 部落格</a>
  &nbsp;|

  <br/>
</div>

<br/>

## 簡介

**MiMo-V2.5-ASR** 是由小米 MiMo 團隊開發的先進端到端自動語音辨識（ASR）模型。本模型旨在針對中文普通話與英文、多種漢語方言、中英混合（Code-Switching）、歌詞辨識、知識密集型內容、高噪音聲學環境及多發言人會議等場景，提供高度精準且強健的轉錄服務。MiMo-V2.5-ASR 在多項公開基準測試中均取得了領先水準。

## 摘要

現代語音辨識系統需要能夠準確轉錄來自不同語言、方言、口音和領域的語音訊號，並在多種聲學條件下保持穩定。雖然傳統端到端模型在特定領域表現優異，但在面對方言混雜、中英夾雜、專業領域知識、背景噪音干擾以及多發言人交疊等現實挑戰時仍顯不足。

為此，小米 MiMo 團隊推出了端到端語音辨識模型 **MiMo-V2.5-ASR**。透過大規模增量預訓練（Mid-training）、高品質監督微調（SFT）和全新的強化學習（RL）演算法，MiMo-V2.5-ASR 在以下多個維度上實現了系統性的效能突破：

- 🗣️ **漢語方言**：原生支援吳語、粵語、閩南語、四川話等多種方言。
- 🔀 **中英混合（Code-Switch）**：無縫辨識中英文夾雜語音，無需預先標記語言標籤。
- 🎵 **歌詞辨識**：高精度的中英文歌曲轉錄，即使在背景音樂和伴奏混合的情況下也能精準提取歌詞。
- 🔊 **強健抗噪**：在強噪音、遠場收音等不良聲學環境下仍具備極佳的辨識率。
- 👥 **多發言人會議**：精準轉錄多人交疊、交談的會議記錄。
- 🇬🇧 **複雜英文場景**：在 AMI 等挑戰性英文 benchmark 的 Open ASR 排行榜上名列前茅。
- 📚 **知識密集型辨識**：精準辨識古詩詞、專業技術術語、人名、地名等高密度知識內容。
- 📝 **原生標點符號**：模型直接根據語音停頓與 semantic 語意原生生成標點符號，輸出結果即可直接使用，無需額外後處理。

## 效能表現

MiMo-V2.5-ASR 已在標準普通話、英語、漢語方言、歌詞辨識以及多個內部業務場景的廣泛基準上完成評估。下圖展示了 MiMo-V2.5-ASR 在這些場景中的平均表現：

![Results](assets/MiMo_ASR_Results.png)

如需查看每個基準測試的詳細數據與具體質性案例，請參閱我們的[部落格](https://mimo.xiaomi.com/mimo-v2-5-asr)。

## 模型下載

| 模型名稱 | 🤗 Hugging Face 下載連結 |
|-------|-------|
| MiMo-Audio-Tokenizer | [XiaomiMiMo/MiMo-Audio-Tokenizer](https://huggingface.co/XiaomiMiMo/MiMo-Audio-Tokenizer) |
| MiMo-V2.5-ASR | [XiaomiMiMo/MiMo-V2.5-ASR](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR) |

```bash
pip install huggingface-hub

hf download XiaomiMiMo/MiMo-Audio-Tokenizer --local-dir ./models/MiMo-Audio-Tokenizer
hf download XiaomiMiMo/MiMo-V2.5-ASR --local-dir ./models/MiMo-V2.5-ASR
```

## 快速入門

透過內建的 Gradio 應用程式，可在幾分鐘內啟動 MiMo-V2.5-ASR 的互動網頁介面。

### 系統環境要求 (Linux & macOS)

* Python >= 3.10
* PyTorch >= 2.2.0 (支援原生 `scaled_dot_product_attention` SDPA)
* 硬體加速器（支援 CUDA GPU 或 Apple Silicon MPS GPU；若無加速卡則自動退回 CPU 執行）

### 安裝步驟

```bash
git clone https://github.com/jmedzen/MiMo-V2.5-ASR.git
cd MiMo-V2.5-ASR
pip install -r requirements.txt
```

> [!TIP]
> 由於本專案採用 PyTorch 原生的 `scaled_dot_product_attention` (SDPA)，您**不需要**編譯或安裝複雜的 `flash-attn` 套件。模型可以在 macOS (Apple Silicon MPS)、標準 Linux CUDA 容器以及僅有 CPU 的伺服器環境下開箱即用。

### 🍏 macOS 記憶體與長音訊優化 (macOS Memory & Long Audio Optimization)

為了讓 Apple Silicon Mac（特別是統一記憶體架構機型）以及有限記憶體設備能穩定運行長音訊轉錄，我們設計了以下優化機制：
1. **即時記憶體防護 (Memory Guard)**：Web UI 整合了動態記憶體監控面板。在載入模型前，系統會先檢測可用記憶體（最低安全門檻為 22 GB，建議 28 GB 統一記憶體/RAM）。若可用記憶體不足以安全載入，系統會發出警告或拒絕載入，防止觸發系統級的 Out-of-Memory (OOM) 或是過度 Swap 導致系統卡頓。
2. **長音訊切片串流 (Audio Chunking & Streaming)**：針對長音訊檔案（如 1 小時以上的錄音），Gradio Web UI 與轉錄腳本皆已預設採用 300 秒 (5 分鐘) 切片轉錄機制。系統會逐步處理並即時將結果串流（Stream/Yield）至網頁介面。
3. **主動記憶體清理 (Active Cache Emptying)**：在每段音訊轉錄完成後，系統會自動呼叫 `torch.mps.empty_cache()` (macOS) / `torch.cuda.empty_cache()` (Linux) 及進行 Garbage Collection (GC)，將記憶體開銷嚴格限制在極低範圍。

### 啟動 Web UI 介面

執行以下命令以啟動本地的 Gradio 網頁介面。您可以直接上傳音訊檔案，或使用麥克風即時錄音。

```bash
python run_mimo_asr.py
```

![MiMo-V2.5-ASR Demo](assets/MiMo_ASR_Demo.png)

#### 💻 命令行範例 (Command-Line Examples)

您可以透過命令列參數來調整模型的載入路徑與服務配置：

##### 1. 基本啟動 (使用預設配置)
```bash
python run_mimo_asr.py
```

##### 2. 啟動時直接載入指定模型與分詞器路徑
省去在網頁介面手動輸入路徑的步驟，在啟動時自動載入模型：
```bash
python run_mimo_asr.py \
    --model-path ./models/MiMo-V2.5-ASR \
    --tokenizer-path ./models/MiMo-Audio-Tokenizer
```

##### 3. 指定綁定的 IP 位址與連接埠
適用於在遠端伺服器佈署或區域網路共用場景：
```bash
python run_mimo_asr.py --host 127.0.0.1 --port 8080
```

##### 4. 產生臨時的公開分享連結 (Gradio Share Link)
```bash
python run_mimo_asr.py --share
```

##### 5. 啟用除錯模式
```bash
python run_mimo_asr.py --debug
```

##### 6. 多引數組合生產佈署命令
結合自訂路徑、主機與連接埠配置、對外分享和除錯紀錄的完整範例：
```bash
python run_mimo_asr.py \
    --model-path ./models/MiMo-V2.5-ASR \
    --tokenizer-path ./models/MiMo-Audio-Tokenizer \
    --host 0.0.0.0 \
    --port 9000 \
    --share \
    --debug
```

## CLI 命令行轉錄工具 (Command-Line Interface)

本專案提供了一個獨立的 CLI 轉錄工具 `transcribe.py`，適合用來批量處理或在終端機直接轉錄音訊，並自動輸出 **SRT 字幕** 與 **TXT 純文字** 兩種格式。

### 基本用法：
```bash
python transcribe.py <音訊檔案路徑> [選用參數]
```

### 常用參數說明：
- `audio`：必填，音訊檔案路徑（如 `.wav`, `.mp3`, `.m4a`）。
- `--language`：可選，指定辨識語系。支援 `zh` (中文)、`en` (英文) 與 `auto` (自動偵測，預設為 `zh`)。
- `--chunk-sec`：可選，音訊切片長度（預設 `300.0` 秒，已內建自動記憶體清理與垃圾回收）。
- `--output-dir`：可選，指定字幕與文字檔輸出目錄（預設與輸入音訊同目錄）。

### 轉錄範例：
```bash
# 自動語言偵測，並將結果輸出至 ./outputs 目錄
python transcribe.py path/to/audio.mp3 --language auto --output-dir ./outputs
```
執行後將在輸出目錄下產生：
1. `audio.mp3.mimo.srt` (標準 SRT 字幕，可用於影片播放器)
2. `audio.mp3.mimo.txt` (逐段純文字檔)

## Python API 使用範例

使用 `asr_sft` 介面的基本程式碼範例：

```python
from src.mimo_audio.mimo_audio import MimoAudio

# 初始化 ASR 模型
model = MimoAudio(
    model_path="./models/MiMo-V2.5-ASR",
    tokenizer_path="./models/MiMo-Audio-Tokenizer",
)

# 自動語言偵測辨識 (推薦用於中英混合 Code-Switching)
text = model.asr_sft("path/to/audio.wav")
print(text)

# 帶有特定語言標籤的辨識模式
text_zh = model.asr_sft("path/to/audio.wav", audio_tag="<chinese>")
text_en = model.asr_sft("path/to/audio.wav", audio_tag="<english>")
```

## 引用

```bibtex
@misc{coreteam2026mimov25asr,
      title={MiMo-V2.5-ASR: Robust Speech Recognition Across Languages, Dialects, and Complex Acoustic Scenarios},
      author={LLM-Core-Team Xiaomi},
      year={2026},
      url={https://github.com/jmedzen/MiMo-V2.5-ASR},
}
```

## 聯絡方式

如有任何問題，請寄信至 [mimo@xiaomi.com](mailto:mimo@xiaomi.com) 或在 GitHub 提交 Issue。
