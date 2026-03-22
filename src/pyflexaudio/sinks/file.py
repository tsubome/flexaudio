import logging
import os
import struct
import time

import numpy as np

from pyflexaudio.types import AudioChunk

__all__ = ["FileSink"]

logger = logging.getLogger("pyflexaudio.sinks.file")


class FileSink:
    """WAV ファイルに int16 で書き出す Sink"""

    # WAV 4GB 制限（data チャンクサイズ上限）
    _MAX_DATA_BYTES = 0xFFFFFFFF - 36  # 約 4GB

    def __init__(
        self,
        path: str,
        sample_rate: int | None = None,
        channels: int | None = None,
        *,
        enabled: bool = True,
    ):
        """
        Args:
            path: 出力ファイルパス
            sample_rate: None = ソースのレートをそのまま使用。指定時は内部でリサンプリング
            channels: None = ソースのチャンネル数をそのまま使用
            enabled: False の場合、write() でデータをスキップ
        """
        self.enabled = enabled
        self.pause_exempt = False
        self._base_path = path
        self._target_sample_rate = sample_rate
        self._target_channels = channels

        # ファイル状態
        self._file = None
        self._file_path: str | None = None
        self._file_index = 0  # 分割ファイルの連番
        self._total_data_bytes = 0
        self._last_header_update = 0.0  # monotonic

        # 現在のソースフォーマット（Source 切替検知用）
        self._current_sample_rate: int | None = None
        self._current_channels: int | None = None

        # リサンプラー（sample_rate 指定時のみ）
        self._resampler = None

    def write(self, chunk: AudioChunk) -> None:
        if not self.enabled:
            return

        # Source 切替検知: サンプルレート/チャンネル変更時に新ファイル
        if self._file is not None:
            if (chunk.sample_rate != self._current_sample_rate
                    or chunk.channels != self._current_channels):
                logger.info("Source format changed, splitting file")
                self._close_current_file()

        # ファイルが未オープンなら開く
        if self._file is None:
            self._open_new_file(chunk.sample_rate, chunk.channels)

        # データ変換
        data = chunk.data  # float32, (frames, channels)

        # リサンプリング（target_sample_rate 指定時）
        if self._resampler is not None:
            data = self._resampler.resample_chunk(data)

        # float32 → int16 変換
        int16_data = np.clip(data * 32767, -32768, 32767).astype(np.int16)
        raw_bytes = int16_data.tobytes()

        # 4GB 制限チェック
        if self._total_data_bytes + len(raw_bytes) > self._MAX_DATA_BYTES:
            logger.info("WAV 4GB limit reached, splitting file")
            self._close_current_file()
            self._open_new_file(chunk.sample_rate, chunk.channels)

        # データ書き込み
        self._file.write(raw_bytes)
        self._total_data_bytes += len(raw_bytes)

        # 30秒ごとにヘッダ更新（異常終了対策）
        now = time.monotonic()
        if now - self._last_header_update >= 30.0:
            self._update_wav_header()
            self._last_header_update = now

    def flush(self) -> None:
        if self._file is not None:
            self._update_wav_header()

    def close(self) -> None:
        self._close_current_file()
        self._resampler = None

    # ---- 内部メソッド ----

    def _open_new_file(self, source_sample_rate: int, source_channels: int) -> None:
        """新しい WAV ファイルを開く"""
        self._current_sample_rate = source_sample_rate
        self._current_channels = source_channels

        # 出力フォーマット決定
        out_rate = self._target_sample_rate or source_sample_rate
        out_channels = self._target_channels or source_channels

        # ファイルパス決定（分割時は連番サフィックス）
        if self._file_index == 0:
            self._file_path = self._base_path
        else:
            base, ext = os.path.splitext(self._base_path)
            self._file_path = f"{base}_{self._file_index + 1:03d}{ext}"
        self._file_index += 1

        # リサンプラー初期化（必要時）
        if self._target_sample_rate and self._target_sample_rate != source_sample_rate:
            import soxr
            self._resampler = soxr.ResampleStream(
                source_sample_rate, self._target_sample_rate,
                num_channels=source_channels, dtype=np.float32
            )
        else:
            self._resampler = None

        # ファイルオープン + ヘッダ書き出し
        self._file = open(self._file_path, "wb")
        self._total_data_bytes = 0
        self._write_wav_header(out_rate, out_channels, data_size=0)
        self._last_header_update = time.monotonic()

        logger.info("Opened WAV file: %s (%dHz, %dch)", self._file_path, out_rate, out_channels)

    def _close_current_file(self) -> None:
        """現在のファイルを閉じる"""
        if self._file is not None:
            self._update_wav_header()
            self._file.close()
            self._file = None
            logger.info("Closed WAV file: %s (%d bytes)", self._file_path, self._total_data_bytes)

    def _write_wav_header(self, sample_rate: int, channels: int, data_size: int = 0) -> None:
        """44バイトの RIFF WAV ヘッダを書き出す"""
        f = self._file
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))  # ファイルサイズ - 8
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))              # fmt チャンクサイズ
        f.write(struct.pack('<H', 1))               # PCM
        f.write(struct.pack('<H', channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * channels * 2))  # byte rate
        f.write(struct.pack('<H', channels * 2))    # block align
        f.write(struct.pack('<H', 16))              # bits per sample
        f.write(b'data')
        f.write(struct.pack('<I', data_size))

    def _update_wav_header(self) -> None:
        """ファイル先頭にシークしてヘッダのサイズフィールドを更新"""
        if self._file is None or self._file.closed:
            return
        f = self._file
        f.seek(4)
        f.write(struct.pack('<I', 36 + self._total_data_bytes))
        f.seek(40)
        f.write(struct.pack('<I', self._total_data_bytes))
        f.flush()
        os.fsync(f.fileno())  # ディスクに確実に書き出す
        f.seek(0, 2)  # ファイル末尾に戻る
