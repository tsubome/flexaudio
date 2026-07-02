//! mic + system を 1 本に合成する合成バックエンド [`CompositeBackend`]。
//!
//! mic と system の 2 つの子バックエンドを内部に持ち、各子の音声を内部正規形
//! （48kHz/stereo）へ揃えてから側別ゲインで加算合成し、[`Stream`](crate::Stream)
//! からはただの 1 バックエンドに見せる。Stream 本体には手を入れないので、
//! seq/PTS・ウォッチドッグ・pause・グローバル gain・switch_source がそのまま効く。
//!
//! # スレッド構成
//! - 子バックエンドの RT スレッド: それぞれ専用の子 RawRing へ push するだけ（既存
//!   backend のまま・触らない）。
//! - 合成スレッド（1 本・`flexaudio-mix`）: 子リングを pop → 子ごとの [`Normalizer`]
//!   で 48k/stereo 化 → 両側の揃ったフレームを側別ゲインで加算合成（±1.0 クランプ）
//!   → 実 sink へ push。RT ではないのでヒープ確保可（ただしループ内の定常確保は
//!   スクラッチ再利用で避ける）。

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use flexaudio_core::backend::{CaptureBackend, RawSink};
use flexaudio_core::clock::monotonic_now_ns;
use flexaudio_core::normalizer::Normalizer;
use flexaudio_core::raw_ring::{raw_ring, RawConsumer};
use flexaudio_core::types::{Error, OutputFormat, Result, CHANNELS, SAMPLE_RATE};

use crate::stream::RAW_RING_SAMPLES;

/// 片側が供給ゼロのままこの時間を超えたら、不足分を無音（0.0）として合成を続行する。
/// 根拠: 正規化は 20ms チャンク単位でしか出てこないので、2〜3 チャンク分の到着ゆらぎ
/// までは正常とみなし、それを超えた途絶（例: システム側が何も再生していない時間帯）
/// でも録音全体は流れ続けるようにする。
const STARVATION_FILL_THRESHOLD: Duration = Duration::from_millis(60);

/// 片側の正規化済み FIFO の上限（f32 サンプル数）。約 500ms 分
/// （48kHz × 2ch × 0.5s = 48_000）を超えたら古い方から捨てる安全弁。
/// 子クロック間のドリフト補正は次ステージで実装予定なので、v1 は無限成長の防止のみ。
const FIFO_MAX_SAMPLES: usize = 48_000;

/// 両側とも合成する材料が無いときの待ち時間（stream.rs の取り込みスレッドと同じ流儀）。
const IDLE_SLEEP: Duration = Duration::from_millis(2);

/// mic + system の 2 子バックエンドを内部正規形で加算合成する合成バックエンド。
///
/// [`native_format`](CaptureBackend::native_format) は常に内部正規形 `(48000, 2)` を
/// 返すので、Stream 側の第 1 段リサンプラは実質パススルーになる。子はコンストラクタ
/// 注入（テストでは mock を渡せる）。実子の構築は facade の `build_backend` が担う。
pub(crate) struct CompositeBackend {
    mic: Box<dyn CaptureBackend>,
    system: Box<dyn CaptureBackend>,
    mic_gain: f32,
    system_gain: f32,
    /// 合成スレッドへの停止指示。start のたびに新しい Arc に差し替える
    /// （旧スレッドの残骸と混線しない）。
    stopping: Arc<AtomicBool>,
    /// 合成スレッドのハンドル。`Some` なら動作中。
    mixer: Option<JoinHandle<()>>,
}

impl CompositeBackend {
    /// 子 2 つと側別ゲインを注入して作る。ゲインの検証（有限・0 以上）は呼び出し側
    /// （facade の `build_backend`）が済ませていること。
    pub(crate) fn new(
        mic: Box<dyn CaptureBackend>,
        system: Box<dyn CaptureBackend>,
        mic_gain: f32,
        system_gain: f32,
    ) -> Self {
        Self {
            mic,
            system,
            mic_gain,
            system_gain,
            stopping: Arc::new(AtomicBool::new(false)),
            mixer: None,
        }
    }
}

impl CaptureBackend for CompositeBackend {
    fn native_format(&self) -> (u32, u16) {
        // 合成は常に内部正規形で行う。Stream の第 1 段は実質パススルーになる。
        (SAMPLE_RATE, CHANNELS)
    }

    fn start(&mut self, sink: RawSink) -> Result<()> {
        // 動作中の二重 start は no-op（CaptureBackend 契約）。
        if self.mixer.is_some() {
            return Ok(());
        }

        // mic の起動に失敗したら即 Err、system の起動に失敗したら mic を stop してから
        // Err（片肺で起動成功にしない）。
        let mic_lane = start_child(&mut self.mic)?;
        let system_lane = match start_child(&mut self.system) {
            Ok(lane) => lane,
            Err(e) => {
                stop_child(&mut self.mic);
                return Err(e);
            }
        };

        // 合成スレッドを起動する。停止フラグは start ごとに新調する（前回 stop の
        // フラグを引きずらない）。
        self.stopping = Arc::new(AtomicBool::new(false));
        let stopping = self.stopping.clone();
        let mic_gain = self.mic_gain;
        let system_gain = self.system_gain;
        let mixer = thread::Builder::new()
            .name("flexaudio-mix".into())
            .spawn(move || {
                run_mixer(mic_lane, system_lane, mic_gain, system_gain, sink, stopping);
            })
            .map_err(|e| Error::Backend(format!("spawn mix thread: {e}")));
        match mixer {
            Ok(handle) => {
                self.mixer = Some(handle);
                Ok(())
            }
            Err(e) => {
                // スレッドが立たなければ子を止めて失敗を返す（片肺にしない）。
                stop_child(&mut self.mic);
                stop_child(&mut self.system);
                Err(e)
            }
        }
    }

    fn stop(&mut self) {
        // 停止フラグ → 合成スレッド join → 子 2 つを stop。冪等（未起動なら子の stop
        // だけが走るが、子側も冪等契約なので無害）。
        self.stopping.store(true, Ordering::SeqCst);
        if let Some(h) = self.mixer.take() {
            let _ = h.join();
        }
        stop_child(&mut self.mic);
        stop_child(&mut self.system);
    }
}

impl Drop for CompositeBackend {
    fn drop(&mut self) {
        // stop されずに捨てられても合成スレッドと子を残さない。
        self.stop();
    }
}

/// 片側の子の取り込み状態一式（子リング consumer + 正規化器 + 正規化済み FIFO）。
struct ChildLane {
    consumer: RawConsumer,
    /// 子ネイティブ → 内部正規形（48k/stereo）。出力を内部正規形に固定するので
    /// 第 2 段はパススルー。
    normalizer: Normalizer,
    /// 正規化済みサンプル（48k/stereo interleaved）の FIFO。
    fifo: Vec<f32>,
    /// 最後にこの側が正規化済みサンプルを供給した時刻（飢餓判定用）。
    last_supply: Instant,
}

impl ChildLane {
    /// 子リングから pop して正規化し、完成分を FIFO へ積む。
    ///
    /// FIFO が [`FIFO_MAX_SAMPLES`] を超えたら古い方から捨てる（無限成長の安全弁）。
    /// rubato の処理失敗は `Err` で返す（呼び出し側が合成スレッドを終える）。
    fn ingest(&mut self, scratch: &mut [f32]) -> Result<()> {
        let got = self.consumer.pop_slice(scratch);
        if got == 0 {
            return Ok(());
        }
        // pts は sink 側では使われない（配線層が別途取り回す契約）が、正規化器の
        // アンカー用に単調 now を渡しておく。
        self.normalizer.push(&scratch[..got], monotonic_now_ns())?;
        let mut supplied = false;
        while let Some((chunk, _pts)) = self.normalizer.pop_chunk() {
            self.fifo.extend_from_slice(&chunk);
            supplied = true;
        }
        if supplied {
            self.last_supply = Instant::now();
            if self.fifo.len() > FIFO_MAX_SAMPLES {
                let excess = self.fifo.len() - FIFO_MAX_SAMPLES;
                self.fifo.drain(..excess);
            }
        }
        Ok(())
    }

    /// この側が [`STARVATION_FILL_THRESHOLD`] 以上供給ゼロのままか。
    fn is_starved(&self, now: Instant) -> bool {
        now.duration_since(self.last_supply) >= STARVATION_FILL_THRESHOLD
    }
}

/// 子を 1 つ起動する: 専用の子 RawRing（stream.rs と同じ容量）を作り、子ネイティブ
/// フォーマットの [`RawSink`] で `start` する。成功で [`ChildLane`] を返す。
///
/// 子の `start` の panic は catch_unwind で [`Error::Backend`] へ変換する
/// （stream.rs の start_backend_catching と同じ趣旨。合成スレッドや呼び出し側を
/// 連鎖 panic させない）。
fn start_child(child: &mut Box<dyn CaptureBackend>) -> Result<ChildLane> {
    let (rate, channels) = child.native_format();
    if rate == 0 || channels == 0 {
        return Err(Error::InvalidArg(
            "mix child native_format must have non-zero rate and channels".into(),
        ));
    }
    let (producer, consumer) = raw_ring(RAW_RING_SAMPLES);
    let sink = RawSink::new(producer, rate, channels);
    match std::panic::catch_unwind(AssertUnwindSafe(|| child.start(sink))) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(Error::Backend("mix child panicked during start()".into())),
    }
    // 子ネイティブ → 内部正規形（48k/stereo）。この Normalizer の出力を内部正規形に
    // 固定するので、第 2 段は常にパススルー。
    let normalizer = Normalizer::new(
        rate,
        channels,
        OutputFormat {
            sample_rate: SAMPLE_RATE,
            channels: CHANNELS,
        },
    )
    .inspect_err(|_| {
        // 正規化器が作れないなら子を止めてから失敗を返す（起動済みの子を残さない）。
        stop_child(child);
    })?;
    Ok(ChildLane {
        consumer,
        normalizer,
        fifo: Vec::with_capacity(FIFO_MAX_SAMPLES),
        last_supply: Instant::now(),
    })
}

/// 子の `stop` を catch_unwind で包んで呼ぶ（panic を巻き上げない。stream.rs の
/// stop_backend_catching と同じ趣旨）。
fn stop_child(child: &mut Box<dyn CaptureBackend>) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| child.stop()));
}

/// 合成スレッド本体。
///
/// 各子を取り込み（pop → 正規化 → FIFO）、両側の揃ったフレームを側別ゲインで加算
/// 合成して実 sink へ push する。片側が [`STARVATION_FILL_THRESHOLD`] 以上供給ゼロ
/// なら不足分を無音として続行する（システム側が無音の時間帯も録音は流れ続ける）。
/// 両側とも材料が無ければ [`IDLE_SLEEP`] 眠る。
///
/// 正規化の失敗（理論上の rubato 失敗）はループを終える。以降サンプルが流れなく
/// なるので、Stream のウォッチドッグが失速を検知して backend を再オープンする。
fn run_mixer(
    mut mic: ChildLane,
    mut system: ChildLane,
    mic_gain: f32,
    system_gain: f32,
    mut sink: RawSink,
    stopping: Arc<AtomicBool>,
) {
    // pop 用スクラッチ（子リング容量ぶん）と合成出力スクラッチ。ループ内で再利用する。
    let mut scratch = vec![0.0f32; RAW_RING_SAMPLES];
    let mut mixed: Vec<f32> = Vec::with_capacity(FIFO_MAX_SAMPLES);

    loop {
        if stopping.load(Ordering::SeqCst) {
            break;
        }

        if mic.ingest(&mut scratch).is_err() || system.ingest(&mut scratch).is_err() {
            // 正規化が壊れたら合成を終える（ウォッチドッグの再オープンに委ねる）。
            return;
        }

        let pushed = mix_and_push(
            &mut mic,
            &mut system,
            mic_gain,
            system_gain,
            &mut sink,
            &mut mixed,
        );

        if !pushed {
            thread::sleep(IDLE_SLEEP);
        }
    }
}

/// 両側の FIFO から合成できる分を取り出し、側別ゲインで加算合成（±1.0 クランプ）して
/// sink へ push する。何か push したら `true`。
///
/// 取り出し量の決め方:
/// - 両側にデータがある → 揃っているフレーム数 min。
/// - 片側だけデータがあり、もう片側が飢餓（60ms 以上供給ゼロ）→ ある側の全量を
///   無音相手と合成（不足分 0.0 埋め）。
/// - それ以外（両側空・相手がまだ飢餓でない）→ 何もしない（揃うのを待つ）。
fn mix_and_push(
    mic: &mut ChildLane,
    system: &mut ChildLane,
    mic_gain: f32,
    system_gain: f32,
    sink: &mut RawSink,
    mixed: &mut Vec<f32>,
) -> bool {
    let now = Instant::now();
    let both = mic.fifo.len().min(system.fifo.len());
    let (mic_take, system_take) = if both > 0 {
        (both, both)
    } else if !mic.fifo.is_empty() && system.is_starved(now) {
        (mic.fifo.len(), 0)
    } else if !system.fifo.is_empty() && mic.is_starved(now) {
        (0, system.fifo.len())
    } else {
        return false;
    };

    let count = mic_take.max(system_take);
    mixed.clear();
    for i in 0..count {
        let m = if i < mic_take { mic.fifo[i] } else { 0.0 };
        let s = if i < system_take { system.fifo[i] } else { 0.0 };
        mixed.push((m * mic_gain + s * system_gain).clamp(-1.0, 1.0));
    }
    mic.fifo.drain(..mic_take);
    system.fifo.drain(..system_take);

    // pts は stream.rs の取り込みと同様、単調 now でよい（sink 側では別途取り回す契約）。
    sink.push(mixed, monotonic_now_ns());
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use flexaudio_core::raw_ring::raw_ring;
    use std::sync::atomic::AtomicU32;

    /// 定振幅（直流）のサンプルを実時間ペースで push するテスト専用の子バックエンド。
    ///
    /// 正弦波（MockBackend）だと 2 ソース合成時に位相の問題が出るため、合成結果を
    /// 決定論的に検証できる直流信号を使う。`feed_for` を指定すると、その時間だけ給餌
    /// してから push を止める（スレッドは生かしたまま）＝片側飢餓の再現用。
    struct ConstBackend {
        sample_rate: u32,
        channels: u16,
        value: f32,
        feed_for: Option<Duration>,
        running: Arc<AtomicBool>,
        handle: Option<JoinHandle<()>>,
    }

    impl ConstBackend {
        fn new(value: f32, feed_for: Option<Duration>) -> Self {
            Self {
                sample_rate: 48_000,
                channels: 2,
                value,
                feed_for,
                running: Arc::new(AtomicBool::new(false)),
                handle: None,
            }
        }
    }

    impl CaptureBackend for ConstBackend {
        fn native_format(&self) -> (u32, u16) {
            (self.sample_rate, self.channels)
        }

        fn start(&mut self, mut sink: RawSink) -> Result<()> {
            if self.running.load(Ordering::SeqCst) {
                return Ok(());
            }
            self.running.store(true, Ordering::SeqCst);
            let running = self.running.clone();
            let sample_rate = self.sample_rate;
            let channels = self.channels as usize;
            let value = self.value;
            let feed_for = self.feed_for;
            let handle = thread::Builder::new()
                .name("flexaudio-const-gen".into())
                .spawn(move || {
                    let frames_per_block = (sample_rate as usize / 100).max(1); // 10ms
                    let block = vec![value; frames_per_block * channels];
                    let block_dur = Duration::from_millis(10);
                    let start = Instant::now();
                    while running.load(Ordering::SeqCst) {
                        let feeding = feed_for.is_none_or(|d| start.elapsed() < d);
                        if feeding {
                            sink.push(&block, start.elapsed().as_nanos() as i64);
                        }
                        thread::sleep(block_dur);
                    }
                })
                .map_err(|e| Error::Backend(format!("spawn const gen thread: {e}")))?;
            self.handle = Some(handle);
            Ok(())
        }

        fn stop(&mut self) {
            self.running.store(false, Ordering::SeqCst);
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    impl Drop for ConstBackend {
        fn drop(&mut self) {
            self.stop();
        }
    }

    /// `start` が常に Err を返すテスト専用バックエンド。
    struct FailingStartBackend;

    impl CaptureBackend for FailingStartBackend {
        fn native_format(&self) -> (u32, u16) {
            (48_000, 2)
        }
        fn start(&mut self, _sink: RawSink) -> Result<()> {
            Err(Error::Backend("intentional start failure".into()))
        }
        fn stop(&mut self) {}
    }

    /// start / stop の呼び出し回数を共有カウンタへ記録するテスト専用バックエンド。
    struct TrackingBackend {
        starts: Arc<AtomicU32>,
        stops: Arc<AtomicU32>,
    }

    impl CaptureBackend for TrackingBackend {
        fn native_format(&self) -> (u32, u16) {
            (48_000, 2)
        }
        fn start(&mut self, _sink: RawSink) -> Result<()> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn stop(&mut self) {
            self.stops.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// composite を組んで start し、実 sink の consumer を返すヘルパ。
    fn start_composite(
        mic: Box<dyn CaptureBackend>,
        system: Box<dyn CaptureBackend>,
        mic_gain: f32,
        system_gain: f32,
    ) -> (CompositeBackend, RawConsumer) {
        let mut be = CompositeBackend::new(mic, system, mic_gain, system_gain);
        assert_eq!(be.native_format(), (48_000, 2), "内部正規形を名乗るはず");
        let (producer, consumer) = raw_ring(RAW_RING_SAMPLES);
        let sink = RawSink::new(producer, 48_000, 2);
        be.start(sink).expect("composite start");
        (be, consumer)
    }

    /// 期限まで consumer からサンプルを集めるヘルパ。
    fn collect_samples(consumer: &mut RawConsumer, dur: Duration) -> Vec<f32> {
        let mut out = Vec::new();
        let mut scratch = vec![0.0f32; RAW_RING_SAMPLES];
        let start = Instant::now();
        while start.elapsed() < dur {
            let got = consumer.pop_slice(&mut scratch);
            out.extend_from_slice(&scratch[..got]);
            thread::sleep(Duration::from_millis(5));
        }
        out
    }

    /// 既知振幅の直流 2 ソース（0.2 と 0.3）を mic_gain=1.0 / system_gain=2.0 で合成
    /// すると、出力は 0.2×1.0 + 0.3×2.0 = 0.8 になる（48k/stereo 子なので全段
    /// パススルー・値は決定論的）。
    #[test]
    fn mix_sums_two_sources_with_gains() {
        let mic = Box::new(ConstBackend::new(0.2, None));
        let system = Box::new(ConstBackend::new(0.3, None));
        let (mut be, mut consumer) = start_composite(mic, system, 1.0, 2.0);

        let samples = collect_samples(&mut consumer, Duration::from_millis(500));
        be.stop();

        assert!(
            samples.len() > 10_000,
            "500ms で相応のサンプルが出るはず: {}",
            samples.len()
        );
        // 先頭 1 チャンク分（立ち上がり）を捨てて定常部を見る。テスト機の一時的な
        // スケジューリング停滞（>60ms）で片側だけ飢餓埋めされると 0.2 / 0.6 が
        // 瞬間的に混ざり得るので、ごく僅かな外れ値だけ許容する（支配値は 0.8）。
        let steady = &samples[1920..];
        let ok = steady.iter().filter(|&&s| (s - 0.8).abs() < 1e-4).count();
        assert!(
            ok as f64 > steady.len() as f64 * 0.98,
            "合成値は 0.2*1.0 + 0.3*2.0 = 0.8 が支配的なはず: {ok}/{}",
            steady.len()
        );
    }

    /// 合成が ±1.0 を超える組合せ（0.8 + 0.8 = 1.6）はクランプされて 1.0 になる。
    #[test]
    fn mix_clamps_sum() {
        let mic = Box::new(ConstBackend::new(0.8, None));
        let system = Box::new(ConstBackend::new(0.8, None));
        let (mut be, mut consumer) = start_composite(mic, system, 1.0, 1.0);

        let samples = collect_samples(&mut consumer, Duration::from_millis(400));
        be.stop();

        assert!(
            samples.len() > 1920,
            "サンプルが出るはず: {}",
            samples.len()
        );
        // クランプの本質: どのサンプルも 1.0 を超えない。
        for (i, &s) in samples.iter().enumerate() {
            assert!(
                s <= 1.0,
                "クランプ後は 1.0 を超えないはず: samples[{i}] = {s}"
            );
        }
        // 定常部（両側供給）では 1.6 → 1.0 ちょうどが支配的（片側飢餓埋めの瞬間だけ
        // 0.8 になり得るので僅かな外れ値は許容）。
        let steady = &samples[1920..];
        let clamped = steady.iter().filter(|&&s| s == 1.0).count();
        assert!(
            clamped as f64 > steady.len() as f64 * 0.98,
            "クランプ値 1.0 が支配的なはず: {clamped}/{}",
            steady.len()
        );
    }

    /// 片側（system）が途中で供給を止めても出力は止まらず、以降は mic 側の音だけが
    /// 出続ける（飢餓側を無音 0.0 として合成を続行する）。
    #[test]
    fn mix_survives_one_side_starvation() {
        let mic = Box::new(ConstBackend::new(0.2, None));
        // system は 150ms だけ給餌してから止まる（スレッドは生存）。
        let system = Box::new(ConstBackend::new(0.3, Some(Duration::from_millis(150))));
        let (mut be, mut consumer) = start_composite(mic, system, 1.0, 1.0);

        // 前半（両側供給）を捨て、system 停止 + 飢餓閾値経過後の出力を観測する。
        let _ = collect_samples(&mut consumer, Duration::from_millis(400));
        let late = collect_samples(&mut consumer, Duration::from_millis(400));
        be.stop();

        assert!(
            late.len() > 5_000,
            "片側が止まっても出力は流れ続けるはず: {} サンプル",
            late.len()
        );
        // 後半は mic 単独（0.2×1.0 + 0.0×1.0 = 0.2）。境界チャンクに合成残り
        // （0.5）が混ざり得るので、末尾側の大半が 0.2 であることを確認する。
        let tail = &late[late.len() / 2..];
        let mic_only = tail.iter().filter(|&&s| (s - 0.2).abs() < 1e-4).count();
        assert!(
            mic_only as f64 > tail.len() as f64 * 0.9,
            "飢餓後は mic 単独の値 0.2 が支配的なはず: {mic_only}/{}",
            tail.len()
        );
    }

    /// system 子の start が Err を返したら、先に起動した mic 子が stop され、全体も
    /// Err になる（片肺で起動成功にしない）。
    #[test]
    fn mix_start_failure_cleans_up() {
        let starts = Arc::new(AtomicU32::new(0));
        let stops = Arc::new(AtomicU32::new(0));
        let mic = Box::new(TrackingBackend {
            starts: starts.clone(),
            stops: stops.clone(),
        });
        let system = Box::new(FailingStartBackend);

        let mut be = CompositeBackend::new(mic, system, 1.0, 1.0);
        let (producer, _consumer) = raw_ring(RAW_RING_SAMPLES);
        let sink = RawSink::new(producer, 48_000, 2);

        let err = be
            .start(sink)
            .expect_err("system 起動失敗で全体も Err のはず");
        assert!(
            matches!(err, Error::Backend(_)),
            "system 子の Err が伝播するはず: {err:?}"
        );
        assert_eq!(starts.load(Ordering::SeqCst), 1, "mic は一度起動される");
        assert_eq!(
            stops.load(Ordering::SeqCst),
            1,
            "system 失敗時に mic が stop されるはず"
        );
    }

    /// mic 子の start が Err なら system 子には触れず即 Err。stop は冪等で二重に
    /// 呼べる。
    #[test]
    fn mix_mic_start_failure_is_immediate() {
        let starts = Arc::new(AtomicU32::new(0));
        let stops = Arc::new(AtomicU32::new(0));
        let mic = Box::new(FailingStartBackend);
        let system = Box::new(TrackingBackend {
            starts: starts.clone(),
            stops: stops.clone(),
        });

        let mut be = CompositeBackend::new(mic, system, 1.0, 1.0);
        let (producer, _consumer) = raw_ring(RAW_RING_SAMPLES);
        let sink = RawSink::new(producer, 48_000, 2);
        assert!(be.start(sink).is_err(), "mic 起動失敗で即 Err のはず");
        assert_eq!(
            starts.load(Ordering::SeqCst),
            0,
            "mic 失敗なら system は起動されない"
        );

        // stop は未起動でも冪等（子の stop も冪等契約）。
        be.stop();
        be.stop();
    }

    /// 合成バックエンドを実際の [`Stream`](crate::Stream) に載せた end-to-end。
    /// 20ms/960frame のチャンクが流れ、data が合成値（0.2 + 0.3 = 0.5）になる。
    /// Stream 本体無変更で seq・チャンク契約がそのまま効くことの裏取り。
    #[test]
    fn stream_delivers_mixed_chunks_end_to_end() {
        use flexaudio_core::types::StreamConfig;

        let mic = Box::new(ConstBackend::new(0.2, None));
        let system = Box::new(ConstBackend::new(0.3, None));
        let backend = Box::new(CompositeBackend::new(mic, system, 1.0, 1.0));
        let mut stream = crate::Stream::open(StreamConfig::default(), backend).expect("open");
        stream.start().expect("start");

        let mut chunks = Vec::new();
        let deadline = Instant::now() + Duration::from_millis(600);
        while Instant::now() < deadline {
            while let Some(c) = stream.poll_chunk() {
                chunks.push(c);
            }
            thread::sleep(Duration::from_millis(5));
        }
        stream.stop();

        assert!(!chunks.is_empty(), "チャンクが届くはず");
        let mut mixed_samples = 0usize;
        let mut total_samples = 0usize;
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.frames, 960, "20ms@48k = 960 frame");
            assert_eq!(c.data.len(), 960 * 2, "stereo interleaved");
            if i > 0 {
                assert!(c.seq > chunks[i - 1].seq, "seq は単調増加");
            }
            total_samples += c.data.len();
            mixed_samples += c.data.iter().filter(|&&s| (s - 0.5).abs() < 1e-4).count();
        }
        // スケジューリング停滞の飢餓埋め（0.2/0.3 単独）をごく僅かだけ許容しつつ、
        // 合成値 0.5 が支配的であること。
        assert!(
            mixed_samples as f64 > total_samples as f64 * 0.98,
            "合成値 0.5 が支配的なはず: {mixed_samples}/{total_samples}"
        );
    }
}
