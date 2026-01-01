import copy
import json
import logging
import os
import sys
import gradio as gr
import torch
from audio_separator.separator import Separator
from huggingface_hub import snapshot_download
import importlib

now_dir = os.getcwd()
sys.path.append

# Configurables
zgpuduration = 30  # Zerogpu allocation time.
model_repo = "audio-separator-models"
repo_owner = "lainlives"
theme_name = "blurple"
models_dir = "/tmp/models"
asset_dir = os.path.join(now_dir, "assets")
out_dir = os.path.join(now_dir, "outputs")

os.makedirs(models_dir, exist_ok=True)

if torch.cuda.is_available():
    gpuconcurrency = torch.cuda.device_count()
    device = "cuda"
    use_autocast = device == "cuda"
else:
    gpuconcurrency = 2
    device = "cpu"

#  Important Constants
HF_TOKEN = os.getenv("HF_TOKEN")
extensions = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
output_format = ["mp3", "wav", "flac", "ogg", "m4a"]
threads = os.cpu_count()
user = os.environ.get("GRADIOUSER")
userpw = os.environ.get("USERPW")
repo_id = repo_owner + "/" + model_repo
config_file = os.path.join(asset_dir, "config.json")
models_file = os.path.join(asset_dir, "models.json")
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
        snapshot_download(repo_id=repo_id, local_dir=models_dir)  # predownload models
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


def get_initial_settings():
    main_config = read_main_config()
    load_custom = main_config.get("load_custom_settings", False)

    settings_to_load = {}
    default_settings = load_settings_from_file(default_settings_file)

    if load_custom:
        print("Attempting to load custom settings...")
        custom_settings = load_settings_from_file(custom_settings_file)
        if custom_settings:
            settings_to_load = copy.deepcopy(default_settings)
            for section, params in custom_settings.items():
                if section in settings_to_load:
                    for key, value in params.items():
                        settings_to_load[section][key] = value
                else:
                    settings_to_load[section] = params
            print("Custom settings loaded successfully.")
        else:
            print(
                "Custom settings file not found or invalid. Falling back to default settings."
            )
            settings_to_load = default_settings
    else:
        print("Loading default settings...")
        settings_to_load = default_settings

    return settings_to_load


initial_settings = get_initial_settings()


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
        use_autocast=use_autocast,
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
        use_autocast=use_autocast,
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
        use_autocast=use_autocast,
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
        use_autocast=use_autocast,
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
        use_autocast=use_autocast,
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
                    use_autocast=use_autocast,
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


# if os.environ.get("SPACES_ZERO_GPU") is not None:
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
                    use_autocast=use_autocast,
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
                    use_autocast=use_autocast,
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
                    use_autocast=use_autocast,
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
                    use_autocast=use_autocast,
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
with gr.Blocks(
    title="Audio Separator",
) as app:
    all_configurable_inputs = []
    with gr.Tabs():
        with gr.TabItem("Roformers"):
            with gr.Row():
                roformer_model = gr.Dropdown(
                    label="Select the model",
                    choices=list(roformer_models.keys()),
                    value=initial_settings["Roformer"]["model"],
                    interactive=True,
                )
                roformer_output_format = gr.Dropdown(
                    label="Select the output format",
                    choices=output_format,
                    value=initial_settings["Roformer"]["output_format"],
                    interactive=True,
                )

            with gr.Group("Configuration"):
                with gr.Group():
                    with gr.Row():
                        roformer_segment_size = gr.Slider(
                            label="Segment size",
                            info="Larger consumes more resources, but may give better results",
                            minimum=32,
                            maximum=4096,
                            step=32,
                            value=initial_settings["Roformer"]["segment_size"],
                            interactive=True,
                        )
                        roformer_override_segment_size = gr.Checkbox(
                            label="Override segment size",
                            info="Override model default segment size instead of using the model default value",
                            value=initial_settings["Roformer"]["override_segment_size"],
                            interactive=True,
                        )
                    with gr.Row():
                        roformer_overlap = gr.Slider(
                            label="Overlap",
                            info="Amount of overlap between prediction windows",
                            minimum=2,
                            maximum=10,
                            step=1,
                            value=initial_settings["Roformer"]["overlap"],
                            interactive=True,
                        )
                        roformer_batch_size = gr.Slider(
                            label="Batch size",
                            info="Larger consumes more RAM but may process slightly faster",
                            minimum=1,
                            maximum=128,
                            step=1,
                            value=initial_settings["Roformer"]["batch_size"],
                            interactive=True,
                        )
                    with gr.Row():
                        roformer_normalization_threshold = gr.Slider(
                            label="Normalization threshold",
                            info="The threshold for audio normalization",
                            minimum=0.1,
                            maximum=1.0,
                            step=0.1,
                            value=initial_settings["Roformer"][
                                "normalization_threshold"
                            ],
                            interactive=True,
                        )
                        roformer_amplification_threshold = gr.Slider(
                            label="Amplification threshold",
                            info="The threshold for audio amplification",
                            minimum=0.1,
                            maximum=1.0,
                            step=0.1,
                            value=initial_settings["Roformer"][
                                "amplification_threshold"
                            ],
                            interactive=True,
                        )

                components["Roformer"] = {
                    "model": roformer_model,
                    "output_format": roformer_output_format,
                    "segment_size": roformer_segment_size,
                    "override_segment_size": roformer_override_segment_size,
                    "overlap": roformer_overlap,
                    "batch_size": roformer_batch_size,
                    "normalization_threshold": roformer_normalization_threshold,
                    "amplification_threshold": roformer_amplification_threshold,
                }
                all_configurable_inputs.extend(components["Roformer"].values())

            with gr.Row():
                roformer_audio = gr.Audio(
                    label="Input audio", type="filepath", interactive=True
                )

            with gr.Row():
                roformer_button = gr.Button("Separate!", variant="primary")
            with gr.Row():
                roformer_stem1 = gr.Audio(
                    label="Stem 1", type="filepath", interactive=False
                )
                roformer_stem2 = gr.Audio(
                    label="Stem 2", type="filepath", interactive=False
                )

            roformer_button.click(
                roformer_separator,
                [
                    roformer_audio,
                    roformer_model,
                    roformer_output_format,
                    roformer_segment_size,
                    roformer_override_segment_size,
                    roformer_overlap,
                    roformer_batch_size,
                    roformer_normalization_threshold,
                    roformer_amplification_threshold,
                ],
                [roformer_stem1, roformer_stem2],
                concurrency_limit=gpuconcurrency,
                concurrency_id="gpu_queue",
            )

        with gr.TabItem("MDX-NET"):
            with gr.Row():
                mdxnet_model = gr.Dropdown(
                    label="Select the model",
                    choices=mdxnet_models,
                    value=initial_settings.get("MDX-NET", {}).get("model", None),
                    interactive=True,
                )
                mdxnet_output_format = gr.Dropdown(
                    label="Select the output format",
                    choices=output_format,
                    value=initial_settings.get("MDX-NET", {}).get(
                        "output_format", None
                    ),
                    interactive=True,
                )
            with gr.Group("Configuration"):
                with gr.Group():
                    with gr.Row():
                        mdxnet_hop_length = gr.Slider(
                            label="Stride length",
                            info="Audio frames per step. Lower should yield better results at a cost of more processing time.",
                            minimum=32,
                            maximum=2048,
                            step=32,
                            value=initial_settings.get("MDX-NET", {}).get(
                                "hop_length", 128
                            ),
                            interactive=True,
                        )
                        mdxnet_segment_size = gr.Slider(
                            minimum=32,
                            maximum=4096,
                            step=32,
                            label="Segment size",
                            info="Larger consumes more resources, but may give better results",
                            value=initial_settings.get("MDX-NET", {}).get(
                                "segment_size", 256
                            ),
                            interactive=True,
                        )
                        mdxnet_denoise = gr.Checkbox(
                            label="Denoise",
                            info="Enable denoising during separation",
                            value=initial_settings.get("MDX-NET", {}).get(
                                "denoise", True
                            ),
                            interactive=True,
                        )
                    with gr.Row():
                        mdxnet_overlap = gr.Slider(
                            label="Overlap",
                            info="Amount of overlap between prediction windows",
                            minimum=0.001,
                            maximum=0.999,
                            step=0.001,
                            value=initial_settings.get("MDX-NET", {}).get(
                                "overlap", 0.25
                            ),
                            interactive=True,
                        )
                        mdxnet_batch_size = gr.Slider(
                            label="Batch size",
                            info="Larger consumes more RAM but may process slightly faster",
                            minimum=1,
                            maximum=128,
                            step=1,
                            value=initial_settings.get("MDX-NET", {}).get(
                                "batch_size", 128
                            ),
                            interactive=True,
                        )
                    with gr.Row():
                        mdxnet_normalization_threshold = gr.Slider(
                            label="Normalization threshold",
                            info="The threshold for audio normalization",
                            minimum=0.1,
                            maximum=1,
                            step=0.1,
                            value=initial_settings.get("MDX-NET", {}).get(
                                "normalization_threshold", 0.9
                            ),
                            interactive=True,
                        )
                        mdxnet_amplification_threshold = gr.Slider(
                            label="Amplification threshold",
                            info="The threshold for audio amplification",
                            minimum=0.1,
                            maximum=1,
                            step=0.1,
                            value=initial_settings.get("MDX-NET", {}).get(
                                "amplification_threshold", 0.7
                            ),
                            interactive=True,
                        )
                    components["MDX-NET"] = {
                        "model": mdxnet_model,
                        "output_format": mdxnet_output_format,
                        "hop_length": mdxnet_hop_length,
                        "segment_size": mdxnet_segment_size,
                        "denoise": mdxnet_denoise,
                        "overlap": mdxnet_overlap,
                        "batch_size": mdxnet_batch_size,
                        "normalization_threshold": mdxnet_normalization_threshold,
                        "amplification_threshold": mdxnet_amplification_threshold,
                    }
                    all_configurable_inputs.extend(components["MDX-NET"].values())

            with gr.Row():
                mdxnet_audio = gr.Audio(
                    label="Input audio", type="filepath", interactive=True
                )

            with gr.Row():
                mdxnet_button = gr.Button("Separate!", variant="primary")
            with gr.Row():
                mdxnet_stem1 = gr.Audio(
                    label="Stem 1", type="filepath", interactive=False
                )
                mdxnet_stem2 = gr.Audio(
                    label="Stem 2", type="filepath", interactive=False
                )

            mdxnet_button.click(
                mdxnet_separator,
                [
                    mdxnet_audio,
                    mdxnet_model,
                    mdxnet_output_format,
                    mdxnet_hop_length,
                    mdxnet_segment_size,
                    mdxnet_denoise,
                    mdxnet_overlap,
                    mdxnet_batch_size,
                    mdxnet_normalization_threshold,
                    mdxnet_amplification_threshold,
                ],
                [mdxnet_stem1, mdxnet_stem2],
                concurrency_limit=gpuconcurrency,
                concurrency_id="gpu_queue",
            )
        with gr.TabItem("VR ARCH"):
            with gr.Row():
                vrarch_model = gr.Dropdown(
                    label="Select the model",
                    choices=vrarch_models,
                    value=initial_settings.get("VR Arch", {}).get("model", None),
                    interactive=True,
                )
                vrarch_output_format = gr.Dropdown(
                    label="Select the output format",
                    choices=output_format,
                    value=initial_settings.get("VR Arch", {}).get(
                        "output_format", None
                    ),
                    interactive=True,
                )
            with gr.Group("Configuration"):
                with gr.Group():
                    with gr.Row():
                        vrarch_window_size = gr.Slider(
                            label="Window size",
                            info="Balance quality and speed. 1024 = fast but lower, 320 = slower but better quality",
                            minimum=320,
                            maximum=10240,
                            step=32,
                            value=initial_settings.get("VR Arch", {}).get(
                                "window_size", 2048
                            ),
                            interactive=True,
                        )
                        vrarch_agression = gr.Slider(
                            minimum=1,
                            maximum=100,
                            step=1,
                            label="Agression",
                            info="Intensity of primary stem extraction",
                            value=initial_settings.get("VR Arch", {}).get(
                                "aggression", 5
                            ),
                            interactive=True,
                        )
                        vrarch_tta = gr.Checkbox(
                            label="TTA",
                            info="Enable Test-Time-Augmentation; slow but improves quality",
                            value=initial_settings.get("VR Arch", {}).get("tta", True),
                            visible=True,
                            interactive=True,
                        )
                    with gr.Row():
                        vrarch_post_process = gr.Checkbox(
                            label="Post process",
                            info="Identify leftover artifacts within vocal output; may improve separation for some songs",
                            value=initial_settings.get("VR Arch", {}).get(
                                "post_process", False
                            ),
                            visible=True,
                            interactive=True,
                        )
                        vrarch_post_process_threshold = gr.Slider(
                            label="Post process threshold",
                            info="Threshold for post-processing",
                            minimum=0.1,
                            maximum=0.3,
                            step=0.1,
                            value=initial_settings.get("VR Arch", {}).get(
                                "post_process_threshold", 0.2
                            ),
                            interactive=True,
                        )
                    with gr.Row():
                        vrarch_high_end_process = gr.Checkbox(
                            label="High end process",
                            info="Mirror the missing frequency range of the output",
                            value=initial_settings.get("VR Arch", {}).get(
                                "high_end_process", False
                            ),
                            visible=True,
                            interactive=True,
                        )
                        vrarch_batch_size = gr.Slider(
                            label="Batch size",
                            info="Larger consumes more RAM but may process slightly faster",
                            minimum=1,
                            maximum=128,
                            step=1,
                            value=initial_settings.get("VR Arch", {}).get(
                                "batch_size", 128
                            ),
                            interactive=True,
                        )
                    with gr.Row():
                        vrarch_normalization_threshold = gr.Slider(
                            label="Normalization threshold",
                            info="The threshold for audio normalization",
                            minimum=0.1,
                            maximum=1,
                            step=0.1,
                            value=initial_settings.get("VR Arch", {}).get(
                                "normalization_threshold", 0.9
                            ),
                            interactive=True,
                        )
                        vrarch_amplification_threshold = gr.Slider(
                            label="Amplification threshold",
                            info="The threshold for audio amplification",
                            minimum=0.1,
                            maximum=1,
                            step=0.1,
                            value=initial_settings.get("VR Arch", {}).get(
                                "amplification_threshold", 0.7
                            ),
                            interactive=True,
                        )
                    components["VR Arch"] = {
                        "model": vrarch_model,
                        "output_format": vrarch_output_format,
                        "window_size": vrarch_window_size,
                        "aggression": vrarch_agression,
                        "tta": vrarch_tta,
                        "post_process": vrarch_post_process,
                        "post_process_threshold": vrarch_post_process_threshold,
                        "high_end_process": vrarch_high_end_process,
                        "batch_size": vrarch_batch_size,
                        "normalization_threshold": vrarch_normalization_threshold,
                        "amplification_threshold": vrarch_amplification_threshold,
                    }
                    all_configurable_inputs.extend(components["VR Arch"].values())

            with gr.Row():
                vrarch_audio = gr.Audio(
                    label="Input audio", type="filepath", interactive=True
                )

            with gr.Row():
                vrarch_button = gr.Button("Separate!", variant="primary")
            with gr.Row():
                vrarch_stem1 = gr.Audio(
                    interactive=False,
                    type="filepath",
                    label="Stem 1",
                )
                vrarch_stem2 = gr.Audio(
                    interactive=False,
                    type="filepath",
                    label="Stem 2",
                )

            vrarch_button.click(
                vrarch_separator,
                [
                    vrarch_audio,
                    vrarch_model,
                    vrarch_output_format,
                    vrarch_window_size,
                    vrarch_agression,
                    vrarch_tta,
                    vrarch_post_process,
                    vrarch_post_process_threshold,
                    vrarch_high_end_process,
                    vrarch_batch_size,
                    vrarch_normalization_threshold,
                    vrarch_amplification_threshold,
                ],
                [vrarch_stem1, vrarch_stem2],
                concurrency_limit=gpuconcurrency,
                concurrency_id="gpu_queue",
            )

        with gr.TabItem("Demucs"):
            with gr.Row():
                demucs_model = gr.Dropdown(
                    label="Select the model",
                    choices=demucs_models,
                    value=initial_settings.get("Demucs", {}).get("model", None),
                    interactive=True,
                )
                demucs_output_format = gr.Dropdown(
                    label="Select the output format",
                    choices=output_format,
                    value=initial_settings.get("Demucs", {}).get("output_format", None),
                    interactive=True,
                )
            with gr.Group("Configuration"):
                with gr.Group():
                    with gr.Row():
                        demucs_shifts = gr.Slider(
                            label="Shifts",
                            info="Number of predictions with random shifts, higher = slower but better quality",
                            minimum=1,
                            maximum=40,
                            step=1,
                            value=initial_settings.get("Demucs", {}).get("shifts", 20),
                            interactive=True,
                        )
                        demucs_segment_size = gr.Slider(
                            label="Segment size",
                            info="Size of segments into which the audio is split. Higher = slower but better quality",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=initial_settings.get("Demucs", {}).get(
                                "segment_size", 100
                            ),
                            interactive=True,
                        )
                        demucs_segments_enabled = gr.Checkbox(
                            label="Segment-wise processing",
                            info="Enable segment-wise processing",
                            value=initial_settings.get("Demucs", {}).get(
                                "segments_enabled", True
                            ),
                            interactive=True,
                        )
                    with gr.Row():
                        demucs_overlap = gr.Slider(
                            label="Overlap",
                            info="Overlap between prediction windows. Higher = slower but better quality",
                            minimum=0.001,
                            maximum=0.999,
                            step=0.001,
                            value=initial_settings.get("Demucs", {}).get(
                                "overlap", 0.9
                            ),
                            interactive=True,
                        )
                        demucs_batch_size = gr.Slider(
                            label="Batch size",
                            info="Larger consumes more RAM but may process slightly faster",
                            minimum=1,
                            maximum=128,
                            step=1,
                            value=initial_settings.get("Demucs", {}).get(
                                "batch_size", 512
                            ),
                            interactive=True,
                        )
                    with gr.Row():
                        demucs_normalization_threshold = gr.Slider(
                            label="Normalization threshold",
                            info="The threshold for audio normalization",
                            minimum=0.1,
                            maximum=1,
                            step=0.1,
                            value=initial_settings.get("Demucs", {}).get(
                                "normalization_threshold", 0.9
                            ),
                            interactive=True,
                        )
                        demucs_amplification_threshold = gr.Slider(
                            label="Amplification threshold",
                            info="The threshold for audio amplification",
                            minimum=0.1,
                            maximum=1,
                            step=0.1,
                            value=initial_settings.get("Demucs", {}).get(
                                "amplification_threshold", 0.7
                            ),
                            interactive=True,
                        )
                    components["Demucs"] = {
                        "model": demucs_model,
                        "output_format": demucs_output_format,
                        "shifts": demucs_shifts,
                        "segment_size": demucs_segment_size,
                        "segments_enabled": demucs_segments_enabled,
                        "overlap": demucs_overlap,
                        "batch_size": demucs_batch_size,
                        "normalization_threshold": demucs_normalization_threshold,
                        "amplification_threshold": demucs_amplification_threshold,
                    }
                    all_configurable_inputs.extend(components["Demucs"].values())

            with gr.Row():
                demucs_audio = gr.Audio(
                    label="Input audio", type="filepath", interactive=True
                )
            with gr.Row():
                demucs_button = gr.Button("Separate!", variant="primary")
            with gr.Row():
                demucs_stem1 = gr.Audio(
                    interactive=False,
                    type="filepath",
                    label="Stem 1",
                )
                demucs_stem2 = gr.Audio(
                    interactive=False,
                    type="filepath",
                    label="Stem 2",
                )
            with gr.Row():
                demucs_stem3 = gr.Audio(
                    interactive=False,
                    type="filepath",
                    label="Stem 3",
                )
                demucs_stem4 = gr.Audio(
                    interactive=False,
                    type="filepath",
                    label="Stem 4",
                )
            with gr.Row(visible=False) as stem6:
                demucs_stem5 = gr.Audio(
                    interactive=False,
                    type="filepath",
                    label="Stem 5",
                )
                demucs_stem6 = gr.Audio(
                    interactive=False,
                    type="filepath",
                    label="Stem 6",
                )

            demucs_model.change(update_stems, inputs=[demucs_model], outputs=stem6)

            demucs_button.click(
                demucs_separator,
                [
                    demucs_audio,
                    demucs_model,
                    demucs_output_format,
                    demucs_shifts,
                    demucs_segment_size,
                    demucs_segments_enabled,
                    demucs_overlap,
                    demucs_batch_size,
                    demucs_normalization_threshold,
                    demucs_amplification_threshold,
                ],
                [
                    demucs_stem1,
                    demucs_stem2,
                    demucs_stem3,
                    demucs_stem4,
                    demucs_stem5,
                    demucs_stem6,
                ],
                concurrency_limit=gpuconcurrency,
                concurrency_id="gpu_queue",
            )

app.queue()
app.launch(
    ssr_mode=True,
    show_error=True,
    pwa=True,
    footer_links=[""],
    max_threads=256,
    theme=theme,
)
