//! flexaudio-cli — 実機テスト用キャプチャ CLI。
//!
//! 既定マイク等から N 秒キャプチャし、出力フォーマット（既定 48000 Hz / stereo 2ch /
//! interleaved f32）のチャンクを集めて 16-bit PCM WAV に書き出す。
//! ピーク / RMS(dBFS) / チャンク数 / ドロップ数などのサマリも表示する。
//!
//! 出力フォーマットは `--output-rate <Hz>`（既定 48000）と
//! `--output-channels <1|2>`（既定 2）で指定する。内部正規形（48k/stereo）から
//! 第 2 段リサンプラ（rubato・アンチエイリアス込み）で再変換され、WAV ヘッダ・
//! stdout の rate/ch もこれに追従する。チャンクは時間ベース 20ms 固定なので、
//! レートに応じて 1 チャンクのフレーム数が変わる（48k=960 / 16k=320）。
//!
//! ```text
//! flexaudio-cli --source mic --seconds 5 --out mic.wav
//! flexaudio-cli --source system --output-rate 16000 --output-channels 1 --out 16k.wav --seconds 3
//! ```
//!
//! また `--out -` を指定すると、WAV ではなく **ヘッダ無し raw PCM** を
//! stdout（バイナリ）へチャンク到着次第ストリーミングする。受け手
//! （例: WhisperApp の `spawn('flexaudio-cli', ...)` + stdout 読み）が
//! リアルタイムに音声を受け取れる。`--encoding f32|s16` で標本形式を選ぶ。
//! このモードでは stdout は PCM バイト専用とし、サマリ等のログは stderr へ出す。
//! `--seconds 0` で無限ストリーミング（パイプ切れ / Ctrl-C で綺麗に停止）。
//! raw PCM のレート/ch も出力フォーマットに追従する（受け手の `-r/-c` を合わせること）。
//!
//! ```text
//! flexaudio-cli --source system --out - --encoding s16 --seconds 0 | aplay -f S16_LE -r 48000 -c 2
//! flexaudio-cli --source system --out - --encoding s16 --output-rate 16000 --output-channels 1 --seconds 0 | aplay -f S16_LE -r 16000 -c 1
//! ```
//!
//! 入力デバイスが無い環境（homelab 等）では実キャプチャはできず、
//! 分かりやすいメッセージを表示して非ゼロ終了する（panic しない）。

use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};

use flexaudio::core::{
    AudioChunk, CaptureBackend, Error, OutputFormat, SourceKind, StreamConfig,
};
use flexaudio::Stream;
use flexaudio_mic::CpalMicBackend;
#[cfg(target_os = "linux")]
use flexaudio_os_linux::PwSystemBackend;

/// キャプチャするソース種別（CLI 引数用）。
#[derive(Debug, Clone, Copy, ValueEnum)]
enum SourceArg {
    /// 既定マイク入力。
    Mic,
    // TODO: process（特定プロセスのループバック）は
    // 対応バックエンド（flexaudio-os-*）が配線され次第ここへ追加する。
    /// システム出力ループバック（Linux のみ）。
    System,
    /// プロセス出力ループバック（未対応）。
    Process,
}

/// stdout ストリーミング時の標本形式（`--out -` 専用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum EncodingArg {
    /// interleaved f32 little-endian（固定契約そのまま）。TS の Float32Array 直読み用。
    F32,
    /// interleaved i16 little-endian。`aplay -f S16_LE` 等の外部ツール互換。
    S16,
}

/// flexaudio キャプチャ CLI。
#[derive(Debug, Parser)]
#[command(name = "flexaudio-cli", about = "flexaudio キャプチャ CLI（実機テスト用）")]
struct Cli {
    /// 録音せず、利用可能なオーディオデバイスを一覧表示して終了する
    /// （`devices()` の統合列挙。`--source` 等とは独立に動く）。
    #[arg(long)]
    list_devices: bool,

    /// キャプチャするソース（mic / system[Linux]）。
    #[arg(long, value_enum, default_value_t = SourceArg::Mic)]
    source: SourceArg,

    /// キャプチャ秒数。`0` で無限ストリーミング（`--out -` 想定、Ctrl-C / パイプ切れで停止）。
    #[arg(long, default_value_t = 5)]
    seconds: u64,

    /// 出力先。ファイルパスなら WAV 書き出し、`-` なら stdout へ raw PCM ストリーミング。
    #[arg(long, default_value = "capture.wav")]
    out: PathBuf,

    /// stdout ストリーミング時の標本形式（`--out -` 専用。WAV 出力では無視）。
    #[arg(long, value_enum, default_value_t = EncodingArg::F32)]
    encoding: EncodingArg,

    /// 出力サンプルレート（Hz）。既定 48000。例: `--output-rate 16000` で 16kHz へ
    /// ダウンサンプル。WAV ヘッダ・stdout の標本レートもこれに追従する。
    #[arg(long, default_value_t = 48_000)]
    output_rate: u32,

    /// 出力チャンネル数（1 = mono / 2 = stereo）。既定 2。stereo→mono は L/R 平均。
    #[arg(long, default_value_t = 2)]
    output_channels: u16,
}

impl Cli {
    /// `--out -`（ハイフン 1 文字）かどうか。true なら stdout へ raw PCM ストリーミング。
    fn is_stdout_stream(&self) -> bool {
        self.out == Path::new("-")
    }

    /// CLI 引数から [`OutputFormat`] を組み立てる。
    fn output_format(&self) -> OutputFormat {
        OutputFormat {
            sample_rate: self.output_rate,
            channels: self.output_channels,
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    // stdout ストリーミング時はエラーも stdout を汚さないよう stderr へ。
    // （run 内のログクロージャと同じ方針。main からのエラーは常に stderr で問題ない。）
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("エラー: {msg}");
            ExitCode::FAILURE
        }
    }
}

/// 実処理本体。失敗は人間向けメッセージ（`String`）として返す。
fn run(cli: &Cli) -> std::result::Result<(), String> {
    // --- デバイス一覧モード（録音せず列挙して終了） ---
    // `--source` 等とは独立。最優先で処理する。
    if cli.list_devices {
        return list_devices();
    }

    let stdout_stream = cli.is_stdout_stream();

    // ログ出力先の切り替え:
    //   - stdout ストリーミング時 → 全ログを stderr へ（stdout は PCM 専用）。
    //   - ファイル出力時 → 従来通り stdout サマリでよい。
    // クロージャ 1 つで集約し、以降の println!/eprintln! はこれを通す。
    macro_rules! log {
        ($($arg:tt)*) => {
            if stdout_stream {
                eprintln!($($arg)*);
            } else {
                println!($($arg)*);
            }
        };
    }

    // --- ソース種別から backend / SourceKind / 表示ラベルを解決 ---
    // Stream::open は Box<dyn CaptureBackend> を取る汎用設計なので、
    // 以降のキャプチャ本体は全ソース共通。
    let (backend, kind, source_label): (Box<dyn CaptureBackend>, SourceKind, &str) =
        match cli.source {
            SourceArg::Mic => (
                Box::new(CpalMicBackend::new()),
                SourceKind::Mic,
                "mic（既定入力デバイス）",
            ),
            SourceArg::System => {
                #[cfg(target_os = "linux")]
                {
                    (
                        Box::new(PwSystemBackend::new()),
                        SourceKind::SystemLoopback,
                        "system（既定出力の monitor / PipeWire）",
                    )
                }
                #[cfg(not(target_os = "linux"))]
                {
                    return Err(
                        "--source system（システム出力ループバック）は現在 Linux のみ対応です。"
                            .into(),
                    );
                }
            }
            SourceArg::Process => {
                return Err(
                    "--source process（プロセス出力ループバック）は未対応です。\
                     対応バックエンドが配線され次第サポート予定です。"
                        .into(),
                );
            }
        };

    // --- 出力フォーマット解決・検証 ---
    let output = cli.output_format();
    output.validate().map_err(|e| {
        format!(
            "出力フォーマット {}Hz/{}ch は非対応です: {e}",
            output.sample_rate, output.channels
        )
    })?;
    let out_rate = output.sample_rate;
    let out_ch = output.channels;

    // --- ネイティブフォーマット表示 ---
    let (native_rate, native_ch) = backend.native_format();
    log!("ソース            : {source_label}");
    log!("ネイティブフォーマット: {native_rate} Hz / {native_ch} ch");
    if stdout_stream {
        let enc = match cli.encoding {
            EncodingArg::F32 => "f32 LE",
            EncodingArg::S16 => "s16 LE",
        };
        log!("出力フォーマット   : {out_rate} Hz / {out_ch} ch / {enc} raw PCM（stdout）");
    } else {
        log!("出力フォーマット   : {out_rate} Hz / {out_ch} ch / 16-bit PCM WAV");
    }
    if cli.seconds == 0 {
        log!("キャプチャ秒数     : 無限（Ctrl-C / パイプ切れで停止）");
    } else {
        log!("キャプチャ秒数     : {} 秒", cli.seconds);
    }
    if stdout_stream {
        log!("出力先             : stdout（raw PCM ストリーミング）");
    } else {
        log!("出力パス           : {}", cli.out.display());
    }
    log!("");

    // --- ストリームを開いて開始 ---
    let config = StreamConfig {
        kind,
        output,
        ..Default::default()
    };
    let mut stream = Stream::open(config, backend).map_err(describe_error)?;
    stream.start().map_err(describe_error)?;

    log!("キャプチャ中 ...");

    if stdout_stream {
        run_stdout_stream(cli, &mut stream)
    } else {
        run_wav(cli, &mut stream, output)
    }
}

/// `--list-devices`: 利用可能なオーディオデバイスを `devices()` で取得して表形式で表示する。
///
/// 列: SOURCE（mic/system/process）/ LOOPBACK / DEFAULT / RATE / CH / NAME / ID。
/// id は安定 ID（cpal=デバイス名 / PipeWire=node.name）。デバイスが 1 つも無い環境では
/// その旨を表示する（エラーにはしない）。
fn list_devices() -> std::result::Result<(), String> {
    let devices =
        flexaudio::devices().map_err(|e| format!("デバイス列挙に失敗しました: {e}"))?;

    if devices.is_empty() {
        println!("利用可能なオーディオデバイスが見つかりませんでした。");
        println!(
            "（PipeWire/オーディオデバイスのある実機で実行してください。\
             Linux は system 列挙に PipeWire セッションが必要です。）"
        );
        return Ok(());
    }

    println!("利用可能なオーディオデバイス: {} 件", devices.len());
    println!();
    // ヘッダ。固定幅で揃える（id/name は可変長なので末尾）。
    println!(
        "{:<7} {:<8} {:<7} {:>6} {:>3}  {:<28} ID",
        "SOURCE", "LOOPBACK", "DEFAULT", "RATE", "CH", "NAME"
    );
    println!("{}", "-".repeat(88));
    for d in &devices {
        println!(
            "{:<7} {:<8} {:<7} {:>6} {:>3}  {:<28} {}",
            source_kind_label(d.source_kind),
            if d.is_loopback { "yes" } else { "no" },
            if d.is_default { "*" } else { "" },
            d.sample_rate,
            d.channels,
            truncate(&d.name, 28),
            d.id,
        );
    }
    println!();
    println!("（DEFAULT の * は OS 既定デバイス。ID は --device-id 等で使える安定キー。）");
    Ok(())
}

/// [`SourceKind`] を CLI 表示用の短いラベルへ。
fn source_kind_label(kind: SourceKind) -> &'static str {
    match kind {
        SourceKind::Mic => "mic",
        SourceKind::SystemLoopback => "system",
        SourceKind::ProcessLoopback => "process",
    }
}

/// 表示用に文字列を `max` 文字（char 単位）で切り詰める（超過分は `…`）。
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let keep = max.saturating_sub(1);
        let mut t: String = s.chars().take(keep).collect();
        t.push('…');
        t
    }
}

/// WAV 出力経路（従来挙動）。N 秒（>0）収集して 16-bit WAV を書き出し、stdout にサマリ表示。
/// `output` は出力フォーマット（WAV ヘッダの rate/ch・録れた秒数の算出に使う）。
fn run_wav(
    cli: &Cli,
    stream: &mut Stream,
    output: OutputFormat,
) -> std::result::Result<(), String> {
    // WAV 経路で --seconds 0 は意味を持たない（無限に貯め続けてしまう）。従来通り拒否。
    if cli.seconds == 0 {
        stream.stop();
        return Err(
            "--seconds 0（無限）は raw PCM ストリーミング（--out -）専用です。\
             WAV 出力では 1 以上を指定してください。"
                .into(),
        );
    }

    // --- N 秒間 poll_chunk をループして全チャンクを収集 ---
    let mut chunks: Vec<AudioChunk> = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(cli.seconds);
    while Instant::now() < deadline {
        let mut got_any = false;
        while let Some(chunk) = stream.poll_chunk() {
            chunks.push(chunk);
            got_any = true;
        }
        // poll_event は表示用に消化（イベントがあれば出す）。
        while let Some(ev) = stream.poll_event() {
            println!("  イベント: {ev:?}");
        }
        if !got_any {
            // チャンク 1 つ ≈ 20ms。空転を避けて適度に眠る。
            thread::sleep(Duration::from_millis(10));
        }
    }

    let dropped = stream.dropped_chunks();
    stream.stop();

    // stop 後にもリングへ残ったチャンクを取り切る。
    while let Some(chunk) = stream.poll_chunk() {
        chunks.push(chunk);
    }

    if chunks.is_empty() {
        return Err(
            "チャンクを 1 つも取得できませんでした。\
             デバイスは開けましたがサンプルが流れていません（ミュート/権限等を確認してください）。"
                .into(),
        );
    }

    // --- 解析（ピーク / RMS）と WAV 書き出し ---
    // NOTE: WAV は現状 s16 固定。将来 f32 WAV が必要になれば encoding を WAV にも波及させる余地あり。
    // ヘッダの rate/ch は出力フォーマット（output）に追従する。
    let total_frames: usize = chunks.iter().map(|c| c.frames).sum();
    let stats =
        write_wav(&cli.out, &chunks, output).map_err(|e| format!("WAV 書き出し失敗: {e}"))?;

    let captured_secs = total_frames as f64 / output.sample_rate as f64;
    let rms_dbfs = if stats.rms > 0.0 {
        20.0 * stats.rms.log10()
    } else {
        f64::NEG_INFINITY
    };
    let peak_dbfs = if stats.peak > 0.0 {
        20.0 * (stats.peak as f64).log10()
    } else {
        f64::NEG_INFINITY
    };

    println!();
    println!("=== 結果 ===");
    println!("取得チャンク数     : {}", chunks.len());
    println!("総フレーム数       : {total_frames}");
    println!("録れた秒数         : {captured_secs:.3} 秒");
    println!("ドロップチャンク数 : {dropped}");
    println!(
        "ピーク             : {:.4}（{}）",
        stats.peak,
        fmt_dbfs(peak_dbfs)
    );
    println!(
        "RMS                : {:.6}（{}）",
        stats.rms,
        fmt_dbfs(rms_dbfs)
    );
    println!("WAV 書き出し       : {}", cli.out.display());

    Ok(())
}

/// stdout raw PCM ストリーミング経路。
///
/// チャンク到着次第すぐ stdout へ書き、各回 flush して溜め込まない（低レイテンシ）。
/// `--seconds 0` なら無限（Ctrl-C / パイプ切れ[BrokenPipe] で正常停止）、
/// `--seconds N>0` なら N 秒で停止。どちらも stop 後にリング残チャンクを出し切る。
fn run_stdout_stream(cli: &Cli, stream: &mut Stream) -> std::result::Result<(), String> {
    let infinite = cli.seconds == 0;

    // Ctrl-C(SIGINT) フラグ。無限時に押されたら綺麗に停止する。
    // ctrlc は重複登録すると Err を返すので、無限時のみ登録する。
    let running = Arc::new(AtomicBool::new(true));
    if infinite {
        let r = running.clone();
        ctrlc::set_handler(move || {
            r.store(false, Ordering::SeqCst);
        })
        .map_err(|e| format!("Ctrl-C ハンドラの登録に失敗しました: {e}"))?;
    }

    // stdout をロックして BufWriter で包む。チャンクごとに flush するので溜め込みは無い。
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let deadline = (!infinite).then(|| Instant::now() + Duration::from_secs(cli.seconds));

    let mut wrote_any = false;
    let mut broken_pipe = false;

    // メインループ: ポーリングしてチャンクを stdout へ流す。
    'outer: loop {
        // 停止条件チェック（無限時は Ctrl-C、有限時は deadline）。
        if infinite {
            if !running.load(Ordering::SeqCst) {
                break;
            }
        } else if let Some(dl) = deadline {
            if Instant::now() >= dl {
                break;
            }
        }

        let mut got_any = false;
        while let Some(chunk) = stream.poll_chunk() {
            got_any = true;
            match write_chunk(&mut out, &chunk, cli.encoding) {
                Ok(()) => wrote_any = true,
                Err(e) if e.kind() == io::ErrorKind::BrokenPipe => {
                    // 受け手が閉じた（| head 等）。エラーにせず正常停止へ。
                    broken_pipe = true;
                    break 'outer;
                }
                Err(e) => return Err(format!("stdout への書き込みに失敗しました: {e}")),
            }
        }

        // イベントは stderr へ（stdout は PCM 専用）。
        while let Some(ev) = stream.poll_event() {
            eprintln!("  イベント: {ev:?}");
        }

        if !got_any {
            // チャンク 1 つ ≈ 20ms。空転を避けて適度に眠る。
            thread::sleep(Duration::from_millis(10));
        }
    }

    let dropped = stream.dropped_chunks();
    stream.stop();

    // stop 後にもリングへ残ったチャンクを出し切る（パイプ切れ後はスキップ）。
    if !broken_pipe {
        while let Some(chunk) = stream.poll_chunk() {
            match write_chunk(&mut out, &chunk, cli.encoding) {
                Ok(()) => wrote_any = true,
                Err(e) if e.kind() == io::ErrorKind::BrokenPipe => {
                    broken_pipe = true;
                    break;
                }
                Err(e) => return Err(format!("stdout への書き込みに失敗しました: {e}")),
            }
        }
    }

    // 最終 flush。パイプ切れはここでも正常扱い。
    if !broken_pipe {
        if let Err(e) = out.flush() {
            if e.kind() != io::ErrorKind::BrokenPipe {
                return Err(format!("stdout の flush に失敗しました: {e}"));
            }
            broken_pipe = true;
        }
    }

    // --- サマリ（stderr） ---
    eprintln!();
    eprintln!("=== 結果（stderr） ===");
    if broken_pipe {
        eprintln!("停止理由           : 受け手がパイプを閉じました（正常終了）");
    } else if infinite {
        eprintln!("停止理由           : Ctrl-C（正常終了）");
    } else {
        eprintln!("停止理由           : {} 秒経過", cli.seconds);
    }
    eprintln!("ドロップチャンク数 : {dropped}");

    // パイプ切れ・Ctrl-C は「受け手都合の停止」なので、サンプル 0 でもエラーにしない。
    // 有限秒指定で素直に終わったのに 1 サンプルも出ていない場合だけ警告する。
    if !wrote_any && !broken_pipe && !infinite {
        return Err(
            "チャンクを 1 つも取得できませんでした。\
             デバイスは開けましたがサンプルが流れていません（ミュート/権限等を確認してください）。"
                .into(),
        );
    }

    Ok(())
}

/// 1 チャンクの interleaved f32 を指定 encoding で `out` へ書く（little-endian）。
///
/// 各サンプル単位の小書き込みを避けるため、チャンク分のバイト列を一旦まとめてから
/// 1 回の `write_all` で出す。書き込み後すぐ flush（低レイテンシ・溜め込み無し）。
fn write_chunk<W: Write>(
    out: &mut W,
    chunk: &AudioChunk,
    encoding: EncodingArg,
) -> io::Result<()> {
    match encoding {
        EncodingArg::F32 => {
            // f32 LE: 契約そのまま。1 サンプル 4 byte。
            let mut buf = Vec::with_capacity(chunk.data.len() * 4);
            for &x in &chunk.data {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            out.write_all(&buf)?;
        }
        EncodingArg::S16 => {
            // s16 LE: (clamp(-1,1) * 32767) as i16。1 サンプル 2 byte。
            let mut buf = Vec::with_capacity(chunk.data.len() * 2);
            for &x in &chunk.data {
                let s = (x.clamp(-1.0, 1.0) * 32767.0) as i16;
                buf.extend_from_slice(&s.to_le_bytes());
            }
            out.write_all(&buf)?;
        }
    }
    // 届いたら即出す。BufWriter に溜め込まない。
    out.flush()
}

/// WAV 書き出しと同時に計算する信号統計。
struct Stats {
    /// 全サンプルの絶対値の最大（線形 0.0..=1.0 目安）。
    peak: f32,
    /// 全サンプルの二乗平均平方根（線形）。
    rms: f64,
}

/// チャンク列を `output.sample_rate` / `output.channels` / 16-bit PCM WAV として
/// `path` へ書き出す。
///
/// 各チャンクの interleaved f32 を `(x.clamp(-1,1) * 32767) as i16` で量子化する。
/// 併せてピーク / RMS（線形）を計算して返す。
fn write_wav(
    path: &std::path::Path,
    chunks: &[AudioChunk],
    output: OutputFormat,
) -> hound::Result<Stats> {
    let spec = hound::WavSpec {
        channels: output.channels,
        sample_rate: output.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;

    let mut peak: f32 = 0.0;
    let mut sum_sq: f64 = 0.0;
    let mut count: u64 = 0;

    for chunk in chunks {
        for &x in &chunk.data {
            let a = x.abs();
            if a > peak {
                peak = a;
            }
            sum_sq += (x as f64) * (x as f64);
            count += 1;

            let clamped = x.clamp(-1.0, 1.0);
            let s = (clamped * 32767.0) as i16;
            writer.write_sample(s)?;
        }
    }
    writer.finalize()?;

    let rms = if count > 0 {
        (sum_sq / count as f64).sqrt()
    } else {
        0.0
    };

    Ok(Stats { peak, rms })
}

/// dBFS を読みやすく整形する（無音時は `-inf dBFS`）。
fn fmt_dbfs(db: f64) -> String {
    if db.is_finite() {
        format!("{db:.1} dBFS")
    } else {
        "-inf dBFS（無音）".into()
    }
}

/// `flexaudio` の [`Error`] を人間向けメッセージへ変換する。
///
/// デバイス不在（`DeviceNotFound`）は実機での実行を促す案内に置き換える。
fn describe_error(err: Error) -> String {
    match err {
        Error::DeviceNotFound => {
            "入力デバイスが見つかりません。マイクのある実機で実行してください。".into()
        }
        Error::PermissionDenied => {
            "マイクへのアクセス権限がありません。OS のマイク権限設定を確認してください。".into()
        }
        Error::DeviceLost => {
            "キャプチャ中に入力デバイスが失われました（切断など）。".into()
        }
        other => format!("ストリーム初期化に失敗しました: {other}"),
    }
}
