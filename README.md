# Eval_LPIPS_2pic

CGH（計算機合成ホログラム）をSLM（空間光変調器）に投影し、カメラで撮像した再生像とターゲット画像の画質をLPIPS・PSNR・SSIMで定量評価するアプリケーションです。

---

## 動作環境

- Python 3.8 以上
- Windows 10/11（SLMのセカンダリモニター表示を使用するため、Windowsを推奨）
- USBカメラ
- SLM（セカンダリモニターとして接続）

---

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/shinker441/Eval_LPIPS_2pic.git
cd Eval_LPIPS_2pic
```

### 2. 仮想環境の作成・有効化

システムのPython環境を汚さないために、仮想環境を使用することを強く推奨します。

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

有効化すると、ターミナルのプロンプトの先頭に `(venv)` と表示されます。

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

> **GPUを使用する場合（任意）:**
> デフォルトではCPU版のPyTorchがインストールされます。GPUを使いたい場合は、お使いのCUDAバージョンに合ったPyTorchを別途インストールしてください。
> 詳細: https://pytorch.org/get-started/locally/

### 4. 設定ファイルの確認・編集

`config.yaml` を開き、環境に合わせて以下を設定してください。

```yaml
camera:
  device_index: 0      # USBカメラのデバイス番号（認識しない場合は 1, 2 に変更）

slm:
  monitor_index: 1     # SLMを表示するモニター番号（セカンダリモニターは通常 1）
```

### 5. アプリの起動

```bash
python main.py
```

---

## 使い方

1. **ターゲット画像フォルダ** と **CGH画像フォルダ** を指定する
2. 画像ペアリング方式を選択する（ソート順 or ファイル名一致）
3. 「計測開始」ボタンを押す
4. SLMにCGHが順番に表示され、カメラで再生像を撮像、LPIPSが自動計算される
5. 計測完了後、結果がテーブルに表示され、CSVファイルとして自動保存される

---

## 仮想環境の終了

作業が終わったら以下のコマンドで仮想環境を無効化できます。

```bash
deactivate
```

---

## 依存パッケージ

| パッケージ | 用途 |
|---|---|
| PyQt5 | GUIフレームワーク |
| opencv-python | カメラ撮像 |
| lpips | LPIPS計算 |
| torch / torchvision | 深層学習バックエンド |
| Pillow / numpy | 画像処理 |
| pandas | 結果データ管理 |
| PyYAML | 設定ファイル読み込み |

---

## トラブルシューティング

**カメラが認識されない**
→ `config.yaml` の `camera.device_index` を `0`, `1`, `2` と順番に試してください。

**SLMに画像が表示されない**
→ `config.yaml` の `slm.monitor_index` を変更してください。モニターが1台しか接続されていない場合は `0` にしてください。

**PyQt5のインストールに失敗する（Linux）**
→ 以下を先に実行してください。
```bash
sudo apt-get install python3-pyqt5
```
