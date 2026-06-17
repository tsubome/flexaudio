//! Process Tap チェーン共通実装: `CATapDescription` → process tap →
//! private aggregate device → IOProc(block) → start。逆順破棄を
//! [`TapChain`] の `Drop` が担う。
//!
//! システム/プロセス両バックエンドはこの [`build_tap_chain`] を、`TapKind` で
//! INCLUDE/EXCLUDE を切り替えて呼ぶだけ（チェーン本体は共通）。
//!
//! # 破棄順（必須）
//! `AudioDeviceStop` → `AudioDeviceDestroyIOProcID` →
//! `AudioHardwareDestroyAggregateDevice` → `AudioHardwareDestroyProcessTap`。
//! 最後に block（`RcBlock`）と `CATapDescription`（`Retained`）が drop される。
//! [`TapChain`] のフィールド宣言順 + `Drop` 実装でこの順序を保証する。

use std::cell::RefCell;
use std::ptr::NonNull;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::AnyThread;
use objc2_core_audio::{
    kAudioAggregateDeviceIsPrivateKey, kAudioAggregateDeviceIsStackedKey,
    kAudioAggregateDeviceNameKey, kAudioAggregateDeviceTapAutoStartKey,
    kAudioAggregateDeviceTapListKey, kAudioAggregateDeviceUIDKey, kAudioSubTapDriftCompensationKey,
    kAudioSubTapUIDKey, AudioDeviceCreateIOProcIDWithBlock, AudioDeviceDestroyIOProcID,
    AudioDeviceIOProcID, AudioDeviceStart, AudioDeviceStop, AudioHardwareCreateAggregateDevice,
    AudioHardwareCreateProcessTap, AudioHardwareDestroyAggregateDevice,
    AudioHardwareDestroyProcessTap, AudioObjectID, CATapDescription,
};
use objc2_core_audio_types::{AudioBufferList, AudioTimeStamp};
use objc2_core_foundation::CFDictionary;
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSObject};

use flexaudio_core::backend::RawSink;
use flexaudio_core::types::Error;

use crate::common::{map_os_status, now_ns, tap_native_format, FALLBACK_FORMAT, NO_ERR};

/// tap の種別（INCLUDE = 指定プロセス群の mixdown / EXCLUDE = 指定プロセス群を除く全体）。
pub(crate) enum TapKind {
    /// 指定オブジェクト群を含めるステレオ mixdown（プロセスループバック INCLUDE）。
    /// 空 vec は不正（呼び出し側が DeviceNotFound を返す前提）。
    IncludeProcesses(Vec<AudioObjectID>),
    /// 指定オブジェクト群を除く全システム音（`exclude` が空 vec ならシステム全体）。
    ExcludeProcesses(Vec<AudioObjectID>),
}

/// 構築済みの tap チェーン。`Drop` で逆順に破棄する。
///
/// フィールド宣言順は Rust の drop 順（宣言順）と一致させ、`Drop` 実装内で
/// 明示的に Stop→IOProc→aggregate→tap の順で OS リソースを片付けてから
/// `RcBlock` / `Retained<CATapDescription>` を drop させる。
pub(crate) struct TapChain {
    /// IOProc が回っている aggregate device ID。
    aggregate_id: AudioObjectID,
    /// 登録済み IOProc ID（block 駆動）。
    io_proc_id: AudioDeviceIOProcID,
    /// process tap ID。
    tap_id: AudioObjectID,
    /// IOProc に渡した block（`DestroyIOProcID` まで生存必須）。最後に drop。
    _block: RcBlock<
        dyn Fn(
            NonNull<AudioTimeStamp>,
            NonNull<AudioBufferList>,
            NonNull<AudioTimeStamp>,
            NonNull<AudioBufferList>,
            NonNull<AudioTimeStamp>,
        ),
    >,
    /// tap description（aggregate 生存中は保持しておく）。block の後に drop。
    _desc: Retained<CATapDescription>,
}

// SAFETY: TapChain が保持する id 群は u32（Send）。`RcBlock` / `Retained<CATapDescription>`
// は所有スレッド（バックエンドの専用スレッド）の中に閉じ込められ、スレッド境界を跨いで
// 共有されない（TapChain は所有スレッドのローカルで生成・drop される）。バックエンド本体
// （`MacSystemBackend`/`MacProcessBackend`）が `Send` であるために TapChain 自体は
// スレッドを跨がない設計のため、ここでは TapChain に対する Send/Sync は宣言しない。

impl Drop for TapChain {
    fn drop(&mut self) {
        // 破棄順（必須）: Stop → DestroyIOProcID → DestroyAggregateDevice →
        // DestroyProcessTap。失敗は無視（best-effort クリーンアップ）。
        unsafe {
            if self.io_proc_id.is_some() {
                let _ = AudioDeviceStop(self.aggregate_id, self.io_proc_id);
                let _ = AudioDeviceDestroyIOProcID(self.aggregate_id, self.io_proc_id);
            }
            if self.aggregate_id != 0 {
                let _ = AudioHardwareDestroyAggregateDevice(self.aggregate_id);
            }
            if self.tap_id != 0 {
                let _ = AudioHardwareDestroyProcessTap(self.tap_id);
            }
        }
        // ここを抜けると _block → _desc の順で drop（宣言順）。
    }
}

/// `AudioObjectID` 群を `NSArray<NSNumber>`（u32 値）へ。
fn object_ids_to_nsarray(ids: &[AudioObjectID]) -> Retained<NSArray<NSNumber>> {
    let numbers: Vec<Retained<NSNumber>> = ids
        .iter()
        .map(|&id| NSNumber::numberWithUnsignedInt(id))
        .collect();
    NSArray::from_retained_slice(&numbers)
}

/// `&CStr` 鍵（objc2-core-audio が export する `kAudio…Key`）を `NSString` 鍵へ。
fn cstr_key(key: &std::ffi::CStr) -> Retained<NSString> {
    NSString::from_str(key.to_str().unwrap_or(""))
}

/// process tap → aggregate device → IOProc → start までを構築する。
///
/// `sink` は IOProc block へ move され、RT コールバックから [`RawSink::push`] される。
/// 成功時に [`TapChain`] を返す（drop で全リソースを逆順破棄）。失敗時は途中まで作った
/// リソースをその場で破棄してから [`Error`] を返す。
///
/// # Safety
/// CoreAudio 呼び出し。`sink` の所有権を block へ渡す（block は単一 RT スレッドからのみ
/// 呼ばれる前提で `RefCell` 内部可変）。
pub(crate) unsafe fn build_tap_chain(
    kind: TapKind,
    name: &str,
    sink: RawSink,
) -> Result<TapChain, Error> {
    // 1) CATapDescription（INCLUDE = mixdown / EXCLUDE = global-but-exclude）。
    let desc: Retained<CATapDescription> = match &kind {
        TapKind::IncludeProcesses(ids) => {
            let arr = object_ids_to_nsarray(ids);
            CATapDescription::initStereoMixdownOfProcesses(CATapDescription::alloc(), &arr)
        }
        TapKind::ExcludeProcesses(ids) => {
            let arr = object_ids_to_nsarray(ids);
            CATapDescription::initStereoGlobalTapButExcludeProcesses(CATapDescription::alloc(), &arr)
        }
    };
    desc.setName(&NSString::from_str(name));
    desc.setPrivate(true);
    // aggregate の sub-tap UID として使う tap UUID 文字列。
    let uuid_str: Retained<NSString> = desc.UUID().UUIDString();

    // 2) process tap を作る。
    let mut tap_id: AudioObjectID = 0;
    let status = AudioHardwareCreateProcessTap(Some(&desc), &mut tap_id as *mut AudioObjectID);
    if status != NO_ERR {
        return Err(map_os_status("AudioHardwareCreateProcessTap", status));
    }
    if tap_id == 0 {
        return Err(Error::Backend(
            "AudioHardwareCreateProcessTap returned null tap id".into(),
        ));
    }

    // tap の ASBD を読み、実機の native フォーマット（rate/channels・interleaved/planar）を
    // デバッグ出力する（実音スモークで実測を確認するため。Stream の native_format は
    // 構築時に backend のフォールバック値を採るので、ここでの読みは情報目的）。
    if std::env::var_os("FLEXAUDIO_DEBUG").is_some() {
        match tap_native_format(tap_id) {
            Some((rate, ch)) => eprintln!(
                "[flexaudio-os-macos] tap ASBD: rate={rate} channels={ch} (fallback would be {FALLBACK_FORMAT:?})"
            ),
            None => eprintln!(
                "[flexaudio-os-macos] tap ASBD unavailable; using fallback {FALLBACK_FORMAT:?}"
            ),
        }
    }

    // 3) private aggregate device を作る。失敗時は tap を破棄してから返す。
    let aggregate_id = match create_aggregate_device(name, &uuid_str) {
        Ok(id) => id,
        Err(e) => {
            let _ = AudioHardwareDestroyProcessTap(tap_id);
            return Err(e);
        }
    };

    // 4) IOProc block を作る。sink を block へ move（RefCell で内部可変）。
    //    block は単一 RT スレッドからのみ呼ばれる前提。
    let sink_cell = RefCell::new(sink);
    let block = RcBlock::new(
        move |_in_now: NonNull<AudioTimeStamp>,
              in_input: NonNull<AudioBufferList>,
              _in_input_time: NonNull<AudioTimeStamp>,
              _out: NonNull<AudioBufferList>,
              _out_time: NonNull<AudioTimeStamp>| {
            // RT コールバック。借用に失敗（再入）したら何もしない。
            if let Ok(mut sink) = sink_cell.try_borrow_mut() {
                // SAFETY: in_input は有効な AudioBufferList（CoreAudio が供給）。
                unsafe { push_buffer_list(&mut sink, in_input.as_ptr()) };
            }
        },
    );

    // 5) IOProc を登録（queue=None で device 既定の RT スレッド）。
    let mut io_proc_id: AudioDeviceIOProcID = None;
    let status = AudioDeviceCreateIOProcIDWithBlock(
        NonNull::from(&mut io_proc_id),
        aggregate_id,
        None,
        // AudioDeviceIOBlock = *mut DynBlock<...>。RcBlock を生 DynBlock ポインタへ。
        RcBlock::as_ptr(&block),
    );
    if status != NO_ERR || io_proc_id.is_none() {
        let _ = AudioHardwareDestroyAggregateDevice(aggregate_id);
        let _ = AudioHardwareDestroyProcessTap(tap_id);
        return Err(map_os_status("AudioDeviceCreateIOProcIDWithBlock", status));
    }

    // 6) start。
    let status = AudioDeviceStart(aggregate_id, io_proc_id);
    if status != NO_ERR {
        let _ = AudioDeviceDestroyIOProcID(aggregate_id, io_proc_id);
        let _ = AudioHardwareDestroyAggregateDevice(aggregate_id);
        let _ = AudioHardwareDestroyProcessTap(tap_id);
        return Err(map_os_status("AudioDeviceStart", status));
    }

    Ok(TapChain {
        aggregate_id,
        io_proc_id,
        tap_id,
        _block: block,
        _desc: desc,
    })
}

/// private aggregate device を作り、その `AudioObjectID` を返す。
///
/// 辞書（AudioCap 写経）:
/// `{ Name, UID(生成UUID), IsPrivate:true, IsStacked:false, TapAutoStart:true,
///    TapList:[{SubTapUID: tap UUID, SubTapDriftCompensation:true}] }`。
/// NSDictionary で組み、toll-free bridge で `&CFDictionary` として渡す。
fn create_aggregate_device(
    name: &str,
    sub_tap_uid: &NSString,
) -> Result<AudioObjectID, Error> {
    // sub-tap 辞書: { uid: <tap uuid>, drift: true }。
    let drift_true = NSNumber::numberWithBool(true);
    let sub_tap: Retained<NSDictionary<NSString, NSObject>> = NSDictionary::from_slices::<NSString>(
        &[&cstr_key(kAudioSubTapUIDKey), &cstr_key(kAudioSubTapDriftCompensationKey)],
        &[sub_tap_uid.as_ref(), drift_true.as_ref()],
    );
    let tap_list: Retained<NSArray<NSObject>> =
        NSArray::from_retained_slice(&[Retained::into_super(sub_tap)]);

    // aggregate 自身の UID（一意な UUID 文字列）。
    let agg_uid = NSString::from_str(&new_uuid_string());
    let agg_name = NSString::from_str(name);
    let is_private = NSNumber::numberWithBool(true);
    let is_stacked = NSNumber::numberWithBool(false);
    let tap_auto_start = NSNumber::numberWithBool(true);

    let keys: [&NSString; 6] = [
        &cstr_key(kAudioAggregateDeviceNameKey),
        &cstr_key(kAudioAggregateDeviceUIDKey),
        &cstr_key(kAudioAggregateDeviceIsPrivateKey),
        &cstr_key(kAudioAggregateDeviceIsStackedKey),
        &cstr_key(kAudioAggregateDeviceTapAutoStartKey),
        &cstr_key(kAudioAggregateDeviceTapListKey),
    ];
    let values: [&NSObject; 6] = [
        agg_name.as_ref(),
        agg_uid.as_ref(),
        is_private.as_ref(),
        is_stacked.as_ref(),
        tap_auto_start.as_ref(),
        tap_list.as_ref(),
    ];
    let dict: Retained<NSDictionary<NSString, NSObject>> =
        NSDictionary::from_slices::<NSString>(&keys, &values);

    // NSDictionary は CFDictionary と toll-free bridge。ポインタを &CFDictionary に。
    // SAFETY: NSDictionary と CFDictionary は toll-free bridged（同一 ObjC オブジェクト）。
    // dict は本関数末尾まで生存し、ポインタはその間有効。
    let cf: &CFDictionary = unsafe {
        &*(Retained::as_ptr(&dict) as *const CFDictionary)
    };

    let mut device_id: AudioObjectID = 0;
    // SAFETY: cf は有効な CFDictionary、device_id は有効なローカル。
    let status = unsafe {
        AudioHardwareCreateAggregateDevice(cf, NonNull::from(&mut device_id))
    };
    if status != NO_ERR {
        return Err(map_os_status("AudioHardwareCreateAggregateDevice", status));
    }
    if device_id == 0 {
        return Err(Error::Backend(
            "AudioHardwareCreateAggregateDevice returned null device id".into(),
        ));
    }
    Ok(device_id)
}

/// 一意な UUID 文字列を生成する（aggregate の UID 用）。
fn new_uuid_string() -> String {
    use objc2_foundation::NSUUID;
    NSUUID::new().UUIDString().to_string()
}

/// IOProc に渡る `AudioBufferList` を interleaved f32 として [`RawSink::push`] へ流す。
///
/// - interleaved（`mNumberBuffers == 1`）: そのまま push。
/// - planar（`mNumberBuffers >= 2`）: フレーム毎に L,R,L,R… へインターリーブして push
///   （スレッドローカル scratch Vec を再利用してアロケート回避）。
/// - size0 / null は無音とみなし push しない。
///
/// # Safety
/// `list` は有効な `AudioBufferList` を指すこと（CoreAudio が IOProc に供給）。
unsafe fn push_buffer_list(sink: &mut RawSink, list: *const AudioBufferList) {
    if list.is_null() {
        return;
    }
    let num_buffers = (*list).mNumberBuffers as usize;
    if num_buffers == 0 {
        return;
    }
    // 実機での interleaved/planar 判定を一度だけデバッグ出力（FLEXAUDIO_DEBUG 時）。
    log_buffer_shape_once(num_buffers);
    // mBuffers は可変長配列の先頭。num_buffers 本ぶんをスライスとして読む。
    let buffers = std::slice::from_raw_parts((*list).mBuffers.as_ptr(), num_buffers);

    if num_buffers == 1 {
        // interleaved: そのまま f32 として push。
        let buf = &buffers[0];
        let n = buf.mDataByteSize as usize / core::mem::size_of::<f32>();
        if n == 0 || buf.mData.is_null() {
            return;
        }
        let slice = std::slice::from_raw_parts(buf.mData as *const f32, n);
        sink.push(slice, now_ns());
        return;
    }

    // planar: 各バッファ = 1ch ぶん。フレーム数は最小バッファに合わせる。
    let channels = num_buffers;
    let mut min_frames = usize::MAX;
    for b in buffers.iter() {
        if b.mData.is_null() {
            return;
        }
        let frames = b.mDataByteSize as usize / core::mem::size_of::<f32>();
        min_frames = min_frames.min(frames);
    }
    if min_frames == 0 || min_frames == usize::MAX {
        return;
    }

    // スレッドローカル scratch を再利用してインターリーブ（RT 経路のアロケート回避）。
    INTERLEAVE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let total = min_frames * channels;
        scratch.resize(total, 0.0);
        for ch in 0..channels {
            let src = std::slice::from_raw_parts(buffers[ch].mData as *const f32, min_frames);
            let mut idx = ch;
            for &s in src.iter() {
                scratch[idx] = s;
                idx += channels;
            }
        }
        sink.push(&scratch[..total], now_ns());
    });
}

thread_local! {
    /// planar→interleaved 用スレッドローカル scratch（RT スレッドでのアロケート回避）。
    static INTERLEAVE_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// バッファ構成のデバッグ出力を 1 度だけ行うためのフラグ（RT スレッドローカル）。
    static LOGGED_SHAPE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// 最初の IOProc コールバックで一度だけ、interleaved（mNumberBuffers==1）/ planar
/// （>=2）の別を `FLEXAUDIO_DEBUG` 時に stderr へ出す（実機実測の確認用）。
fn log_buffer_shape_once(num_buffers: usize) {
    if std::env::var_os("FLEXAUDIO_DEBUG").is_none() {
        return;
    }
    LOGGED_SHAPE.with(|c| {
        if !c.get() {
            c.set(true);
            let kind = if num_buffers == 1 { "interleaved" } else { "planar" };
            eprintln!("[flexaudio-os-macos] IOProc buffer shape: mNumberBuffers={num_buffers} ({kind})");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `new_uuid_string` は 36 文字の UUID 形式（8-4-4-4-12）を返す。
    #[test]
    fn uuid_string_has_expected_shape() {
        let s = new_uuid_string();
        assert_eq!(s.len(), 36);
        assert_eq!(s.matches('-').count(), 4);
    }

    /// object_ids_to_nsarray が要素数を保つ。
    #[test]
    fn object_ids_array_preserves_count() {
        let arr = object_ids_to_nsarray(&[1, 2, 3]);
        assert_eq!(arr.count(), 3);
    }
}
