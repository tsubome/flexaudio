//! flexaudio facade の end-to-end 検証（MockBackend 駆動・ハードウェア不要）。
//!
//! `MockBackend`（合成サイン波）で「backend → RawRing → 加工スレッド →
//! Normalizer → ChunkRing → poll」の全配線が動くことを確認する。

use std::time::{Duration, Instant};

use flexaudio::core::types::{AudioChunk, StreamConfig};
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

/// open → start → stop がハング/panic なく完了する（スレッド join の健全性）。
#[test]
fn open_start_stop_is_clean() {
    let backend = Box::new(MockBackend::new(48000, 2, 440.0));
    let mut stream = Stream::open(StreamConfig::default(), backend).expect("open");
    stream.start().expect("start");
    std::thread::sleep(Duration::from_millis(50));
    stream.stop();
}
