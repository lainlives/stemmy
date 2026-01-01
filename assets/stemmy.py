import importlib
import json
import logging
import os
import sys

import gradio as gr
from audio_separator.separator import Separator

from assets.model_tools import download_file, download_files_from_txt

now_dir = os.getcwd()
sys.path.append

# Configurables
theme_name = "blurple"
out_dir = os.path.join(now_dir, "outputs")
asset_dir = os.path.join(now_dir, "assets")
models_dir = os.path.join(asset_dir, "models")

os.makedirs(models_dir, exist_ok=True)


#  Important Constants
HF_TOKEN = os.getenv("HF_TOKEN")
extensions = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
output_format = ["mp3", "wav", "flac", "ogg", "m4a"]
threads = os.cpu_count()
user = os.environ.get("GRADIOUSER")
userpw = os.environ.get("USERPW")
config_file = os.path.join(asset_dir, "config.json")
models_file = "https://huggingface.co/lainlives/audio-separator-models/resolve/main/assets/luvr5-ui/models.txt"
confirm_file = os.path.join(asset_dir, "files_downloaded")
default_settings_file = os.path.join(asset_dir, "default_settings.json")
custom_settings_file = os.path.join(asset_dir, "custom_settings.json")
components = {"Roformer": {}, "MDX23C": {}, "MDX-NET": {}, "VR Arch": {}, "Demucs": {}}


#  Dynamically import theme .py
module_path = f"assets.themes.{theme_name}"
theme_module = importlib.import_module(module_path)
theme = getattr(theme_module, theme_name)


# =========================#
#     Roformer Models     #
# =========================#

roformer_models = {
    "MDX23C | Main": "MDX23C_D1581.ckpt",
    "MDX23C | InstVoc HQ": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
    "MDX23C | InstVoc HQ v2": "MDX23C-8KFFT-InstVoc_HQ_2.ckpt",
    "MDX23C | De-Reverb": "MDX23C-De-Reverb-aufr33-jarredou.ckpt",
    "MDX23C | Drum Seperator": "MDX23C-DrumSep-aufr33-jarredou.ckpt",
    "BS-Roformer-Viperx-1297": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "BS-Roformer-Viperx-1296": "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    "BS-Roformer-Viperx-1053": "model_bs_roformer_ep_937_sdr_10.5309.ckpt",
    "Mel-Roformer-Viperx-1143": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
    "BS-Roformer-De-Reverb": "deverb_bs_roformer_8_384dim_10depth.ckpt",
    "Mel-Roformer-Crowd-Aufr33-Viperx": "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt",
    "Mel-Roformer-Denoise-Aufr33": "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    "Mel-Roformer-Denoise-Aufr33-Aggr": "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
    "MelBand Roformer | Denoise-Debleed by Gabox": "mel_band_roformer_denoise_debleed_gabox.ckpt",
    "Mel-Roformer-Karaoke-Aufr33-Viperx": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "MelBand Roformer | Karaoke by Gabox": "mel_band_roformer_karaoke_gabox.ckpt",
    "MelBand Roformer | Karaoke by becruily": "mel_band_roformer_karaoke_becruily.ckpt",
    "MelBand Roformer | Vocals by Kimberley Jensen": "vocals_mel_band_roformer.ckpt",
    "MelBand Roformer Kim | FT by unwa": "mel_band_roformer_kim_ft_unwa.ckpt",
    "MelBand Roformer Kim | FT 2 by unwa": "mel_band_roformer_kim_ft2_unwa.ckpt",
    "MelBand Roformer Kim | FT 2 Bleedless by unwa": "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt",
    "MelBand Roformer Kim | FT 3 by unwa": "mel_band_roformer_kim_ft3_unwa.ckpt",
    "MelBand Roformer Kim | Inst V1 by Unwa": "melband_roformer_inst_v1.ckpt",
    "MelBand Roformer Kim | Inst V1 Plus by Unwa": "melband_roformer_inst_v1_plus.ckpt",
    "MelBand Roformer Kim | Inst V1 (E) by Unwa": "melband_roformer_inst_v1e.ckpt",
    "MelBand Roformer Kim | Inst V1 (E) Plus by Unwa": "melband_roformer_inst_v1e_plus.ckpt",
    "MelBand Roformer Kim | Inst V2 by Unwa": "melband_roformer_inst_v2.ckpt",
    "MelBand Roformer Kim | InstVoc Duality V1 by Unwa": "melband_roformer_instvoc_duality_v1.ckpt",
    "MelBand Roformer Kim | InstVoc Duality V2 by Unwa": "melband_roformer_instvox_duality_v2.ckpt",
    "MelBand Roformer | Vocals by becruily": "mel_band_roformer_vocals_becruily.ckpt",
    "MelBand Roformer | Instrumental by becruily": "mel_band_roformer_instrumental_becruily.ckpt",
    "MelBand Roformer | Vocals Fullness by Aname": "mel_band_roformer_vocal_fullness_aname.ckpt",
    "BS Roformer | Vocals by Gabox": "bs_roformer_vocals_gabox.ckpt",
    "MelBand Roformer | Vocals by Gabox": "mel_band_roformer_vocals_gabox.ckpt",
    "MelBand Roformer | Vocals FV1 by Gabox": "mel_band_roformer_vocals_fv1_gabox.ckpt",
    "MelBand Roformer | Vocals FV2 by Gabox": "mel_band_roformer_vocals_fv2_gabox.ckpt",
    "MelBand Roformer | Vocals FV3 by Gabox": "mel_band_roformer_vocals_fv3_gabox.ckpt",
    "MelBand Roformer | Vocals FV4 by Gabox": "mel_band_roformer_vocals_fv4_gabox.ckpt",
    "MelBand Roformer | Instrumental by Gabox": "mel_band_roformer_instrumental_gabox.ckpt",
    "MelBand Roformer | Instrumental 2 by Gabox": "mel_band_roformer_instrumental_2_gabox.ckpt",
    "MelBand Roformer | Instrumental 3 by Gabox": "mel_band_roformer_instrumental_3_gabox.ckpt",
    "MelBand Roformer | Instrumental Bleedless V1 by Gabox": "mel_band_roformer_instrumental_bleedless_v1_gabox.ckpt",
    "MelBand Roformer | Instrumental Bleedless V2 by Gabox": "mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt",
    "MelBand Roformer | Instrumental Bleedless V3 by Gabox": "mel_band_roformer_instrumental_bleedless_v3_gabox.ckpt",
    "MelBand Roformer | Instrumental Fullness V1 by Gabox": "mel_band_roformer_instrumental_fullness_v1_gabox.ckpt",
    "MelBand Roformer | Instrumental Fullness V2 by Gabox": "mel_band_roformer_instrumental_fullness_v2_gabox.ckpt",
    "MelBand Roformer | Instrumental Fullness V3 by Gabox": "mel_band_roformer_instrumental_fullness_v3_gabox.ckpt",
    "MelBand Roformer | Instrumental Fullness Noisy V4 by Gabox": "mel_band_roformer_instrumental_fullness_noise_v4_gabox.ckpt",
    "MelBand Roformer | INSTV5 by Gabox": "mel_band_roformer_instrumental_instv5_gabox.ckpt",
    "MelBand Roformer | INSTV5N by Gabox": "mel_band_roformer_instrumental_instv5n_gabox.ckpt",
    "MelBand Roformer | INSTV6 by Gabox": "mel_band_roformer_instrumental_instv6_gabox.ckpt",
    "MelBand Roformer | INSTV6N by Gabox": "mel_band_roformer_instrumental_instv6n_gabox.ckpt",
    "MelBand Roformer | INSTV7 by Gabox": "mel_band_roformer_instrumental_instv7_gabox.ckpt",
    "MelBand Roformer | INSTV7N by Gabox": "mel_band_roformer_instrumental_instv7n_gabox.ckpt",
    "MelBand Roformer | INSTV8 by Gabox": "mel_band_roformer_instrumental_instv8_gabox.ckpt",
    "MelBand Roformer | INSTV8N by Gabox": "mel_band_roformer_instrumental_instv8n_gabox.ckpt",
    "MelBand Roformer | FVX by Gabox": "mel_band_roformer_instrumental_fvx_gabox.ckpt",
    "MelBand Roformer | De-Reverb by anvuew": "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    "MelBand Roformer | De-Reverb Less Aggressive by anvuew": "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
    "MelBand Roformer | De-Reverb Mono by anvuew": "dereverb_mel_band_roformer_mono_anvuew.ckpt",
    "MelBand Roformer | De-Reverb Big by Sucial": "dereverb_big_mbr_ep_362.ckpt",
    "MelBand Roformer | De-Reverb Super Big by Sucial": "dereverb_super_big_mbr_ep_346.ckpt",
    "MelBand Roformer | De-Reverb-Echo by Sucial": "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt",
    "MelBand Roformer | De-Reverb-Echo V2 by Sucial": "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
    "MelBand Roformer | De-Reverb-Echo Fused by Sucial": "dereverb_echo_mbr_fused.ckpt",
    "MelBand Roformer Kim | SYHFT by SYH99999": "MelBandRoformerSYHFT.ckpt",
    "MelBand Roformer Kim | SYHFT V2 by SYH99999": "MelBandRoformerSYHFTV2.ckpt",
    "MelBand Roformer Kim | SYHFT V2.5 by SYH99999": "MelBandRoformerSYHFTV2.5.ckpt",
    "MelBand Roformer Kim | SYHFT V3 by SYH99999": "MelBandRoformerSYHFTV3Epsilon.ckpt",
    "MelBand Roformer Kim | Big SYHFT V1 by SYH99999": "MelBandRoformerBigSYHFTV1.ckpt",
    "MelBand Roformer Kim | Big Beta 4 FT by unwa": "melband_roformer_big_beta4.ckpt",
    "MelBand Roformer Kim | Big Beta 5e FT by unwa": "melband_roformer_big_beta5e.ckpt",
    "MelBand Roformer | Big Beta 6 by unwa": "melband_roformer_big_beta6.ckpt",
    "MelBand Roformer | Big Beta 6X by unwa": "melband_roformer_big_beta6x.ckpt",
    "BS Roformer | Chorus Male-Female by Sucial": "model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt",
    "BS Roformer | Male-Female by aufr33": "bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt",
    "MelBand Roformer | Aspiration by Sucial": "aspiration_mel_band_roformer_sdr_18.9845.ckpt",
    "MelBand Roformer | Aspiration Less Aggressive by Sucial": "aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt",
    "MelBand Roformer | Bleed Suppressor V1 by unwa-97chris": "mel_band_roformer_bleed_suppressor_v1.ckpt",
}

# =========================#
#     MDXN-NET Models     #
# =========================#
mdxnet_models = [
    "UVR-MDX-NET-Inst_full_292.onnx",
    "UVR-MDX-NET_Inst_187_beta.onnx",
    "UVR-MDX-NET_Inst_82_beta.onnx",
    "UVR-MDX-NET_Inst_90_beta.onnx",
    "UVR-MDX-NET_Main_340.onnx",
    "UVR-MDX-NET_Main_390.onnx",
    "UVR-MDX-NET_Main_406.onnx",
    "UVR-MDX-NET_Main_427.onnx",
    "UVR-MDX-NET_Main_438.onnx",
    "UVR-MDX-NET-Inst_HQ_1.onnx",
    "UVR-MDX-NET-Inst_HQ_2.onnx",
    "UVR-MDX-NET-Inst_HQ_3.onnx",
    "UVR-MDX-NET-Inst_HQ_4.onnx",
    "UVR-MDX-NET-Inst_HQ_5.onnx",
    "UVR_MDXNET_Main.onnx",
    "UVR-MDX-NET-Inst_Main.onnx",
    "UVR_MDXNET_1_9703.onnx",
    "UVR_MDXNET_2_9682.onnx",
    "UVR_MDXNET_3_9662.onnx",
    "UVR-MDX-NET-Inst_1.onnx",
    "UVR-MDX-NET-Inst_2.onnx",
    "UVR-MDX-NET-Inst_3.onnx",
    "UVR_MDXNET_KARA.onnx",
    "UVR_MDXNET_KARA_2.onnx",
    "UVR_MDXNET_9482.onnx",
    "UVR-MDX-NET-Voc_FT.onnx",
    "Kim_Vocal_1.onnx",
    "Kim_Vocal_2.onnx",
    "Kim_Inst.onnx",
    "Reverb_HQ_By_FoxJoy.onnx",
    "UVR-MDX-NET_Crowd_HQ_1.onnx",
    "kuielab_a_vocals.onnx",
    "kuielab_a_other.onnx",
    "kuielab_a_bass.onnx",
    "kuielab_a_drums.onnx",
    "kuielab_b_vocals.onnx",
    "kuielab_b_other.onnx",
    "kuielab_b_bass.onnx",
    "kuielab_b_drums.onnx",
]

# ========================#
#     VR-ARCH Models     #
# ========================#
vrarch_models = [
    "1_HP-UVR.pth",
    "2_HP-UVR.pth",
    "3_HP-Vocal-UVR.pth",
    "4_HP-Vocal-UVR.pth",
    "5_HP-Karaoke-UVR.pth",
    "6_HP-Karaoke-UVR.pth",
    "7_HP2-UVR.pth",
    "8_HP2-UVR.pth",
    "9_HP2-UVR.pth",
    "10_SP-UVR-2B-32000-1.pth",
    "11_SP-UVR-2B-32000-2.pth",
    "12_SP-UVR-3B-44100.pth",
    "13_SP-UVR-4B-44100-1.pth",
    "14_SP-UVR-4B-44100-2.pth",
    "15_SP-UVR-MID-44100-1.pth",
    "16_SP-UVR-MID-44100-2.pth",
    "17_HP-Wind_Inst-UVR.pth",
    "UVR-De-Echo-Aggressive.pth",
    "UVR-De-Echo-Normal.pth",
    "UVR-DeEcho-DeReverb.pth",
    "UVR-De-Reverb-aufr33-jarredou.pth",
    "UVR-DeNoise-Lite.pth",
    "UVR-DeNoise.pth",
    "UVR-BVE-4B_SN-44100-1.pth",
    "MGM_HIGHEND_v4.pth",
    "MGM_LOWEND_A_v4.pth",
    "MGM_LOWEND_B_v4.pth",
    "MGM_MAIN_v4.pth",
]

# =======================#
#     DEMUCS Models     #
# =======================#
demucs_models = [
    "htdemucs_ft.yaml",
    "htdemucs_6s.yaml",
    "htdemucs.yaml",
    "hdemucs_mmi.yaml",
]


def get_all_models():
    if os.path.isfile(confirm_file):
        return
    else:
        print(f"Downloading models from {models_file}")
        download_file(models_file, "/tmp")
        download_files_from_txt("/tmp/models.txt", models_dir)
        with open(confirm_file, "a"):
            pass


found_files = []
logs = []


def read_main_config():
    try:
        with open(config_file, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading main config file '{config_file}': {e}")
        gr.Warning("Error reading main config file")


def load_settings_from_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading settings file '{filepath}': {e}")
        gr.Warning("Error reading settings file")
        return None


initial_settings = load_settings_from_file(default_settings_file)


def roformer_separator(
    audio,
    model_key,
    out_format,
    segment_size,
    override_seg_size,
    overlap,
    batch_size,
    norm_thresh,
    amp_thresh,
    progress=gr.Progress(track_tqdm=True),
):
    roformer_model = roformer_models[model_key]
    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=models_dir,
        output_dir=out_dir,
        output_format=out_format,
        # use_autocast=use_autocast,
        normalization_threshold=norm_thresh,
        amplification_threshold=amp_thresh,
        mdxc_params={
            "segment_size": segment_size,
            "override_model_segment_size": override_seg_size,
            "batch_size": batch_size,
            "overlap": overlap,
        },
    )

    progress(0.2, desc="Loading model...")
    separator.load_model(model_filename=roformer_model)

    progress(0.7, desc="Separating audio...")
    separation = separator.separate(audio)

    stems = [os.path.join(out_dir, file_name) for file_name in separation]

    return stems[0], stems[1]


def mdxc_separator(
    audio,
    model,
    out_format,
    segment_size,
    override_seg_size,
    overlap,
    batch_size,
    norm_thresh,
    amp_thresh,
    progress=gr.Progress(track_tqdm=True),
):
    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=models_dir,
        output_dir=out_dir,
        output_format=out_format,
        # use_autocast=use_autocast,
        normalization_threshold=norm_thresh,
        amplification_threshold=amp_thresh,
        mdxc_params={
            "segment_size": segment_size,
            "override_model_segment_size": override_seg_size,
            "batch_size": batch_size,
            "overlap": overlap,
        },
    )

    progress(0.2, desc="Loading model...")
    separator.load_model(model_filename=model)

    progress(0.7, desc="Separating audio...")
    separation = separator.separate(audio)

    stems = [os.path.join(out_dir, file_name) for file_name in separation]

    return stems[0], stems[1]


def mdxnet_separator(
    audio,
    model,
    out_format,
    hop_length,
    segment_size,
    denoise,
    overlap,
    batch_size,
    norm_thresh,
    amp_thresh,
    progress=gr.Progress(track_tqdm=True),
):

    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=models_dir,
        output_dir=out_dir,
        output_format=out_format,
        # use_autocast=use_autocast,
        normalization_threshold=norm_thresh,
        amplification_threshold=amp_thresh,
        mdx_params={
            "hop_length": hop_length,
            "segment_size": segment_size,
            "overlap": overlap,
            "batch_size": batch_size,
            "enable_denoise": denoise,
        },
    )

    progress(0.2, desc="Loading model...")
    separator.load_model(model_filename=model)

    progress(0.7, desc="Separating audio...")
    separation = separator.separate(audio)

    stems = [os.path.join(out_dir, file_name) for file_name in separation]

    return stems[0], stems[1]


def vrarch_separator(
    audio,
    model,
    out_format,
    window_size,
    aggression,
    tta,
    post_process,
    post_process_threshold,
    high_end_process,
    batch_size,
    norm_thresh,
    amp_thresh,
    progress=gr.Progress(track_tqdm=True),
):
    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=models_dir,
        output_dir=out_dir,
        output_format=out_format,
        # use_autocast=use_autocast,
        normalization_threshold=norm_thresh,
        amplification_threshold=amp_thresh,
        vr_params={
            "batch_size": batch_size,
            "window_size": window_size,
            "aggression": aggression,
            "enable_tta": tta,
            "enable_post_process": post_process,
            "post_process_threshold": post_process_threshold,
            "high_end_process": high_end_process,
        },
    )

    progress(0.2, desc="Loading model...")
    separator.load_model(model_filename=model)

    progress(0.7, desc="Separating audio...")
    separation = separator.separate(audio)

    stems = [os.path.join(out_dir, file_name) for file_name in separation]

    return stems[0], stems[1]


def demucs_separator(
    audio,
    model,
    out_format,
    shifts,
    segment_size,
    segments_enabled,
    overlap,
    batch_size,
    norm_thresh,
    amp_thresh,
    progress=gr.Progress(track_tqdm=True),
):
    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=models_dir,
        output_dir=out_dir,
        output_format=out_format,
        # use_autocast=use_autocast,
        normalization_threshold=norm_thresh,
        amplification_threshold=amp_thresh,
        demucs_params={
            "batch_size": batch_size,
            "segment_size": segment_size,
            "shifts": shifts,
            "overlap": overlap,
            "segments_enabled": segments_enabled,
        },
    )

    progress(0.2, desc="Loading model...")
    separator.load_model(model_filename=model)

    progress(0.7, desc="Separating audio...")
    separation = separator.separate(audio)

    stems = [os.path.join(out_dir, file_name) for file_name in separation]

    if model == "htdemucs_6s.yaml":
        return stems[0], stems[1], stems[2], stems[3], stems[4], stems[5]
    else:
        return stems[0], stems[1], stems[2], stems[3], None, None


def update_stems(model):
    if model == "htdemucs_6s.yaml":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def roformer_batch(
    path_input,
    path_output,
    model_key,
    out_format,
    segment_size,
    override_seg_size,
    overlap,
    batch_size,
    norm_thresh,
    amp_thresh,
    single_stem,
    progress=gr.Progress(),
):
    found_files.clear()
    logs.clear()
    roformer_model = roformer_models[model_key]
    model_path = os.path.join(models_dir, roformer_model)

    if not os.path.exists(model_path):
        gr.Info(
            f"This is the first time the {model_key} model is being used. The separation will take a little longer because the model needs to be downloaded."
        )

    for audio_files in os.listdir(path_input):
        if audio_files.endswith(extensions):
            found_files.append(audio_files)
    total_files = len(found_files)

    if total_files == 0:
        logs.append("No valid audio files.")
        return "\n".join(logs)
    else:
        logs.append(f"{total_files} audio files found")
        found_files.sort()
        progress(0, desc="Starting processing...")

        for i, audio_files in enumerate(found_files):
            progress((i / total_files), desc=f"Processing file {i + 1}/{total_files}")
            file_path = os.path.join(path_input, audio_files)
            try:
                separator = Separator(
                    log_level=logging.WARNING,
                    model_file_dir=models_dir,
                    output_dir=path_output,
                    output_format=out_format,
                    # use_autocast=use_autocast,
                    normalization_threshold=norm_thresh,
                    amplification_threshold=amp_thresh,
                    mdxc_params={
                        "segment_size": segment_size,
                        "override_model_segment_size": override_seg_size,
                        "batch_size": batch_size,
                        "overlap": overlap,
                    },
                )

                logs.append("Loading model...")
                separator.load_model(model_filename=roformer_model)

                logs.append(f"Separating file: {audio_files}")
                separator.separate(file_path)
                logs.append(f"File: {audio_files} separated!")
            except Exception as e:
                raise RuntimeError(
                    f"BS/Mel Roformer batch separation failed: {e}"
                ) from e

        progress(1.0, desc="Processing complete")
        return "\n".join(logs)


def mdx23c_batch(
    path_input,
    path_output,
    model,
    out_format,
    segment_size,
    override_seg_size,
    overlap,
    batch_size,
    norm_thresh,
    amp_thresh,
    single_stem,
    progress=gr.Progress(),
):
    found_files.clear()
    logs.clear()
    model_path = os.path.join(models_dir, model)

    if not os.path.exists(model_path):
        gr.Info(
            f"This is the first time the {model} model is being used. The separation will take a little longer because the model needs to be downloaded."
        )

    for audio_files in os.listdir(path_input):
        if audio_files.endswith(extensions):
            found_files.append(audio_files)
    total_files = len(found_files)

    if total_files == 0:
        logs.append("No valid audio files.")
        return "\n".join(logs)
    else:
        logs.append(f"{total_files} audio files found")
        found_files.sort()
        progress(0, desc="Starting processing...")

        for i, audio_files in enumerate(found_files):
            progress((i / total_files), desc=f"Processing file {i + 1}/{total_files}")
            file_path = os.path.join(path_input, audio_files)
            try:
                separator = Separator(
                    log_level=logging.WARNING,
                    model_file_dir=models_dir,
                    output_dir=path_output,
                    output_format=out_format,
                    # use_autocast=use_autocast,
                    normalization_threshold=norm_thresh,
                    amplification_threshold=amp_thresh,
                    mdxc_params={
                        "segment_size": segment_size,
                        "override_model_segment_size": override_seg_size,
                        "batch_size": batch_size,
                        "overlap": overlap,
                    },
                )

                logs.append("Loading model...")
                separator.load_model(model_filename=model)

                logs.append(f"Separating file: {audio_files}")
                separator.separate(file_path)
                logs.append(f"File: {audio_files} separated!")
            except Exception as e:
                raise RuntimeError(f"MDXC batch separation failed: {e}") from e

        progress(1.0, desc="Processing complete")
        return "\n".join(logs)


def mdxnet_batch(
    path_input,
    path_output,
    model,
    out_format,
    hop_length,
    segment_size,
    denoise,
    overlap,
    batch_size,
    norm_thresh,
    amp_thresh,
    single_stem,
    progress=gr.Progress(),
):
    found_files.clear()
    logs.clear()
    model_path = os.path.join(models_dir, model)

    if not os.path.exists(model_path):
        gr.Info(
            f"This is the first time the {model} model is being used. The separation will take a little longer because the model needs to be downloaded."
        )

    for audio_files in os.listdir(path_input):
        if audio_files.endswith(extensions):
            found_files.append(audio_files)
    total_files = len(found_files)

    if total_files == 0:
        logs.append("No valid audio files.")
        return "\n".join(logs)
    else:
        logs.append(f"{total_files} audio files found")
        found_files.sort()
        progress(0, desc="Starting processing...")

        for i, audio_files in enumerate(found_files):
            progress((i / total_files), desc=f"Processing file {i + 1}/{total_files}")
            file_path = os.path.join(path_input, audio_files)
            try:
                separator = Separator(
                    log_level=logging.WARNING,
                    model_file_dir=models_dir,
                    output_dir=path_output,
                    output_format=out_format,
                    # use_autocast=use_autocast,
                    normalization_threshold=norm_thresh,
                    amplification_threshold=amp_thresh,
                    mdx_params={
                        "hop_length": hop_length,
                        "segment_size": segment_size,
                        "overlap": overlap,
                        "batch_size": batch_size,
                        "enable_denoise": denoise,
                    },
                )

                logs.append("Loading model...")
                separator.load_model(model_filename=model)

                logs.append(f"Separating file: {audio_files}")
                separator.separate(file_path)
                logs.append(f"File: {audio_files} separated!")
            except Exception as e:
                raise RuntimeError(f"MDX-NET batch separation failed: {e}") from e

        progress(1.0, desc="Processing complete")
        return "\n".join(logs)


def vrarch_batch(
    path_input,
    path_output,
    model,
    out_format,
    window_size,
    aggression,
    tta,
    post_process,
    post_process_threshold,
    high_end_process,
    batch_size,
    norm_thresh,
    amp_thresh,
    single_stem,
    progress=gr.Progress(),
):
    found_files.clear()
    logs.clear()
    model_path = os.path.join(models_dir, model)

    if not os.path.exists(model_path):
        gr.Info(
            f"This is the first time the {model} model is being used. The separation will take a little longer because the model needs to be downloaded."
        )

    for audio_files in os.listdir(path_input):
        if audio_files.endswith(extensions):
            found_files.append(audio_files)
    total_files = len(found_files)

    if total_files == 0:
        logs.append("No valid audio files.")
        return "\n".join(logs)
    else:
        logs.append(f"{total_files} audio files found")
        found_files.sort()
        progress(0, desc="Starting processing...")

        for i, audio_files in enumerate(found_files):
            progress((i / total_files), desc=f"Processing file {i + 1}/{total_files}")
            file_path = os.path.join(path_input, audio_files)
            try:
                separator = Separator(
                    log_level=logging.WARNING,
                    model_file_dir=models_dir,
                    output_dir=path_output,
                    output_format=out_format,
                    # use_autocast=use_autocast,
                    normalization_threshold=norm_thresh,
                    amplification_threshold=amp_thresh,
                    vr_params={
                        "batch_size": batch_size,
                        "window_size": window_size,
                        "aggression": aggression,
                        "enable_tta": tta,
                        "enable_post_process": post_process,
                        "post_process_threshold": post_process_threshold,
                        "high_end_process": high_end_process,
                    },
                )

                logs.append("Loading model...")
                separator.load_model(model_filename=model)

                logs.append(f"Separating file: {audio_files}")
                separator.separate(file_path)
                logs.append(f"File: {audio_files} separated!")
            except Exception as e:
                raise RuntimeError(f"VR Arch batch separation failed: {e}") from e

        progress(1.0, desc="Processing complete")
        return "\n".join(logs)


def demucs_batch(
    path_input,
    path_output,
    model,
    out_format,
    shifts,
    segment_size,
    segments_enabled,
    overlap,
    batch_size,
    norm_thresh,
    amp_thresh,
    progress=gr.Progress(),
):
    found_files.clear()
    logs.clear()
    model_path = os.path.join(models_dir, model)

    if not os.path.exists(model_path):
        gr.Info(
            f"This is the first time the {model} model is being used. The separation will take a little longer because the model needs to be downloaded."
        )

    for audio_files in os.listdir(path_input):
        if audio_files.endswith(extensions):
            found_files.append(audio_files)
    total_files = len(found_files)

    if total_files == 0:
        logs.append("No valid audio files.")
        return "\n".join(logs)
    else:
        logs.append(f"{total_files} audio files found")
        found_files.sort()
        progress(0, desc="Starting processing...")

        for i, audio_files in enumerate(found_files):
            progress((i / total_files), desc=f"Processing file {i + 1}/{total_files}")
            file_path = os.path.join(path_input, audio_files)
            try:
                separator = Separator(
                    log_level=logging.WARNING,
                    model_file_dir=models_dir,
                    output_dir=path_output,
                    output_format=out_format,
                    # use_autocast=use_autocast,
                    normalization_threshold=norm_thresh,
                    amplification_threshold=amp_thresh,
                    demucs_params={
                        "batch_size": batch_size,
                        "segment_size": segment_size,
                        "shifts": shifts,
                        "overlap": overlap,
                        "segments_enabled": segments_enabled,
                    },
                )

                logs.append("Loading model...")
                separator.load_model(model_filename=model)

                logs.append(f"Separating file: {audio_files}")
                separator.separate(file_path)
                logs.append(f"File: {audio_files} separated!")
            except Exception as e:
                raise RuntimeError(f"Demucs batch separation failed: {e}") from e

        progress(1.0, desc="Processing complete")
        return "\n".join(logs)


get_all_models()
