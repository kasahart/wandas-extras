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

`modal_parameters` は、FRF のフレームを受け取ってモードの表を返す関数です。

## 機能

| ExtrasFrame のメソッド | 内容 |
|------|------|
| `rpm` | 回転パルスから回転数の時間変化（単位 rpm の波形） |
| `full_scale_fraction` | フルスケール付近（既定 99.9 % 以上）のサンプルの割合。`rms` と同じくチャンネルごとの配列 |
| `amplitude_histogram` | [-full_scale, full_scale] の振幅分布（チャンネルごとの度数と区間の端） |
| `angle_resample` | フレーム内の回転パルスのチャンネルを使い、ほかのチャンネルを等角度にリサンプリング。結果は `AngleFrame`（横軸は回転数 rev） |
| `order_tracks` | 指定した次数の片振幅を回転数ごとに並べた表（次数トラッキング）。`find_peaks()` と同じく pandas の表を返す |
| `AngleFrame.order_spectrum` | 角度領域の Welch 平均による次数スペクトル（片振幅）。結果は `OrderSpectralFrame`（横軸は次数） |

```python
frame = ExtrasFrame.from_numpy(np.vstack([vib, tacho]), fs, ch_labels=["vib", "tacho"])
spec = frame.angle_resample("tacho", samples_per_rev=64).order_spectrum(revs_per_block=16)
spec.orders, spec.data  # 次数と片振幅。以前の order_spectrum の実効値は ÷√2（0 次とナイキストを除く）
```

| 関数 | 入力 | 内容 |
|------|------|------|
| `modal_parameters` | 周波数応答関数（`transfer_function()` の 1 ch） | 固有振動数と減衰比（半値幅法、1 自由度＋剰余項の当てはめ） |

### wandas 本体に移ったもの

次の解析は wandas 0.13.0 に入ったので、ここからは外しました。wandas のメソッドを使ってください。

| 以前の関数 | wandas |
|------|------|
| `signal_stats` | `frame.rms`, `frame.peak`, `frame.crest_factor`, `frame.kurtosis`。フルスケールの確認は上の 2 メソッド |
| `spectral_peaks` | `frame.welch().find_peaks(...)` |
| `spectral_kurtosis` | `frame.stft(...).spectral_kurtosis` |
| `integrate` | `frame.integrate().high_pass_filter(cutoff=...)` |
| `envelope_spectrum` | `frame.band_pass_filter(lo, hi).envelope().remove_dc().fft()`（振幅はピーク値。以前の実効値は ÷√2） |

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
