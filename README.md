# wandas-extras

[wandas](https://github.com/kasahart/wandas) にまだない解析機能を、暫定的に置いておくパッケージです。
インターフェースが固まった機能は wandas 本体へ移し、移したあともしばらくは同じ名前で呼べるようにします。

新しい解析は `ChannelFrame` を継承した `ExtrasFrame` のメソッドとして作ります。wandas の独自処理ガイドにある拡張方法です。
読み込みから解析までが wandas と同じチェーンになり、結果は遅延評価のまま lineage を引き継ぎます。
`recipe_registry()` を渡すと、Recipe に保存して別の入力で再実行できます。
本体へ移したあとも、同じ名前と引数で呼べます。

```python
from wandas_extras import ExtrasFrame

tacho = ExtrasFrame.read_wav("tacho.wav")
rpm = tacho.rpm(pulses_per_rev=1)  # 回転数の ChannelFrame（単位 rpm）
```

`ExtrasFrame.read_wav()` などの読み込みが `ExtrasFrame` を返すのは、wandas 0.13.0 の次のリリース（kasahart/wandas#644）からです。
wandas 0.13.0 では `ExtrasFrame.from_numpy()` で作ってください。

まだ `ExtrasFrame` に移していない機能は、フレームを受け取る関数です。

## 機能

| ExtrasFrame のメソッド | 内容 |
|------|------|
| `rpm` | 回転パルスから回転数の時間変化（単位 rpm の波形） |

| 関数 | 入力 | 内容 |
|------|------|------|
| `spectral_peaks` | SpectralFrame | スペクトルの山の周波数・振幅・突出量（dB の放物線補間も可） |
| `integrate` | ChannelFrame | 周波数領域の積分・微分と帯域制限（加速度 → 速度など）。結果は ChannelFrame |
| `order_spectrum`, `order_tracks` | ChannelFrame と回転パルス | 角度リサンプリングによる平均次数スペクトルと次数トラッキング（実効値） |
| `spectral_kurtosis` | ChannelFrame | STFT によるスペクトル尖度（衝撃的な成分が強い帯域を探す） |
| `envelope_spectrum` | ChannelFrame | 帯域通過 → 包絡線 → スペクトル（軸受・ギアの故障周波数） |
| `modal_parameters` | 周波数応答関数（`transfer_function()` の 1 ch） | 固有振動数と減衰比（半値幅法、1 自由度＋剰余項の当てはめ） |
| `signal_stats` | ChannelFrame | RMS、ピーク、波高率、直流分、尖度、dB 値、フルスケール付近のサンプルの割合と振幅分布 |

## 決めごと

- 新しい解析は `ExtrasFrame` のメソッドにする。数値計算は `operations.py` の AudioOperation（`_process`）に置き、メソッドは結果の Frame を組み立てるだけにする。メソッドには `@recipe_operation("wandas_extras.…")` を付けて `recipe_registry()` に登録する。
- 結果は wandas の既存の Frame で返す。新しい Frame は、既存の Frame で軸を表せないときだけ作る。
- 入力は wandas のフレーム。入力のフレームは変更しない。dB の基準値はフレームのチャンネル情報（`ch_units="Pa"` なら 20 µPa）を使う。
- 区間の切り出し、フィルタ、スペクトルなど wandas にある処理は wandas で行い、ここでは作り直さない。
- 既定値は一般的な値にする。特定の課題やデータに合わせた閾値や正解値は置かない。
- 外部ライブラリ（scipy など）に渡す値の境界チェックは足さず、そのライブラリの例外に任せる。
- wandas 本体へ移すときは、テストも一緒に移す。

## 開発

```bash
uv sync --group test
uv run ruff check . && uv run ruff format --check . && uv run ty check src tests && uv run pytest
```
