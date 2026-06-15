//! flexaudio facade の end-to-end 検証（MockBackend 駆動・ハードウェア不要）。
//!
//! `MockBackend`（合成サイン波）で「backend → RawRing → 加工スレッド →
//! Normalizer → ChunkRing → poll」の全配線が動くことを確認する。

use std::time::{Duration, Instant};

use flexaudio::core::types::{AudioChunk, OutputFormat, StreamConfig};
use flexaudio::{MockBackend, Stream};

/// 指定時間 poll してチャンクを集めるヘルパ。
fn collect_chunks(stream: &mut Stream, dur: Duration) -> Vec<AudioChunk> {
    let mut chunks = Vec::new();
    let start = Instant::now();
    while start.elapsed() < dur {
        while let Some(c) = stream.poll_chunk() {
            chunks.push(c);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    chunks
}

/// mono 44100 入力 → stereo 48000 / 960frame チャンクへ正規化される。
#[test]
fn mock_mono_44100_to_stereo_960_chunks() {
    let backend = Box::new(MockBackend::new(44100, 1, 440.0));
    let mut stream = Stream::open(StreamConfig::default(), backend).expect("open");
    stream.start().expect("start");

    let chunks = collect_chunks(&mut stream, Duration::from_millis(500));
    stream.stop();

    assert!(!chunks.is_empty(), "チャンクが1つも来ていない");
    for c in &chunks {
        assert_eq!(c.frames, 960, "20ms@48k = 960 frame でない");
        assert_eq!(c.data.len(), 960 * 2, "stereo interleaved (960*2) でない");
    }
    // seq は単調増加（DROP_OLDEST が起きても増加は保たれる）
    for w in chunks.windows(2) {
        assert!(w[1].seq > w[0].seq, "seq が単調増加していない: {} -> {}", w[0].seq, w[1].seq);
    }
}

/// 48000 stereo はパススルー経路で 960frame チャンクになる。
#[test]
fn mock_passthrough_48000_stereo() {
    let backend = Box::new(MockBackend::new(48000, 2, 440.0));
    let mut stream = Stream::open(StreamConfig::default(), backend).expect("open");
    stream.start().expect("start");

    let chunks = collect_chunks(&mut stream, Duration::from_millis(400));
    stream.stop();

    assert!(!chunks.is_empty(), "チャンクが来ていない");
    for c in &chunks {
        assert_eq!(c.frames, 960);
        assert_eq!(c.data.len(), 1920);
    }
}

/// 出力 {16000, 1}: 48k/stereo 入力 → 320 frame・320 sample（mono）チャンク。
/// peak/rms が妥当（合成サイン波で 0 でない・1.0 を大きく超えない）。
#[test]
fn mock_output_16k_mono() {
    let backend = Box::new(MockBackend::new(48_000, 2, 440.0));
    let config = StreamConfig {
        output: OutputFormat {
            sample_rate: 16_000,
            channels: 1,
        },
        ..Default::default()
    };
    let mut stream = Stream::open(config, backend).expect("open");
    stream.start().expect("start");

    let chunks = collect_chunks(&mut stream, Duration::from_millis(500));
    stream.stop();

    assert!(!chunks.is_empty(), "16k/mono チャンクが来ていない");
    for c in &chunks {
        assert_eq!(c.frames, 320, "16k 20ms = 320 frame でない");
        assert_eq!(c.data.len(), 320, "mono interleaved (320*1) でない");
        // peak/rms 妥当性（合成サイン波 amplitude 0.5）。
        assert!(c.peak > 0.0 && c.peak <= 1.5, "peak が妥当でない: {}", c.peak);
        assert!(c.rms > 0.0 && c.rms <= 1.0, "rms が妥当でない: {}", c.rms);
        assert!(c.peak >= c.rms, "peak >= rms のはず: peak={} rms={}", c.peak, c.rms);
    }
}

/// 出力 {16000, 2}: → 320 frame・640 sample（stereo）チャンク。
#[test]
fn mock_output_16k_stereo() {
    let backend = Box::new(MockBackend::new(48_000, 2, 440.0));
    let config = StreamConfig {
        output: OutputFormat {
            sample_rate: 16_000,
            channels: 2,
        },
        ..Default::default()
    };
    let mut stream = Stream::open(config, backend).expect("open");
    stream.start().expect("start");

    let chunks = collect_chunks(&mut stream, Duration::from_millis(500));
    stream.stop();

    assert!(!chunks.is_empty(), "16k/stereo チャンクが来ていない");
    for c in &chunks {
        assert_eq!(c.frames, 320, "16k 20ms = 320 frame でない");
        assert_eq!(c.data.len(), 640, "stereo interleaved (320*2) でない");
        assert!(c.peak > 0.0 && c.peak <= 1.5, "peak が妥当でない: {}", c.peak);
        assert!(c.rms > 0.0 && c.rms <= 1.0, "rms が妥当でない: {}", c.rms);
    }
}

/// 既定出力 {48000, 2} の回帰: frames==960 / data.len()==1920 / peak/rms 妥当。
#[test]
fn mock_default_output_regression_with_peak_rms() {
    let backend = Box::new(MockBackend::new(48_000, 2, 440.0));
    let mut stream = Stream::open(StreamConfig::default(), backend).expect("open");
    stream.start().expect("start");

    let chunks = collect_chunks(&mut stream, Duration::from_millis(400));
    stream.stop();

    assert!(!chunks.is_empty(), "チャンクが来ていない");
    for c in &chunks {
        assert_eq!(c.frames, 960);
        assert_eq!(c.data.len(), 1920);
        assert!(c.peak > 0.0 && c.peak <= 1.5, "peak: {}", c.peak);
        assert!(c.rms > 0.0 && c.rms <= 1.0, "rms: {}", c.rms);
    }
}

/// open → start → stop がハング/panic なく完了する（スレッド join の健全性）。
#[test]
fn open_start_stop_is_clean() {
    let backend = Box::new(MockBackend::new(48000, 2, 440.0));
    let mut stream = Stream::open(StreamConfig::default(), backend).expect("open");
    stream.start().expect("start");
    std::thread::sleep(Duration::from_millis(50));
    stream.stop();
}
