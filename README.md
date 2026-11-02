# wandas-extras

[wandas](https://github.com/kasahart/wandas) にまだない解析機能を、暫定的に置いておくパッケージです。
インターフェースが固まった機能は wandas 本体へ移し、移したあともしばらくは同じ名前で呼べるようにします。

```python
import wandas as wd
from wandas_extras import signal_stats

frame = wd.read_wav("signal.wav")
signal_stats(frame.trim(1.0, 2.0), full_scale=1.0)
```

## 機能

| 関数 | 入力 | 内容 |
|------|------|------|
| `signal_stats` | ChannelFrame | RMS、ピーク、波高率、直流分、尖度、dB 値、フルスケール付近のサンプルの割合と振幅分布 |

## 決めごと

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
