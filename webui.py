import gradio as gr
import torch

from assets.stemmy import (
    components,
    demucs_models,
    demucs_separator,
    initial_settings,
    mdxnet_models,
    mdxnet_separator,
    output_format,
    roformer_models,
    roformer_separator,
    theme,
    update_stems,
    vrarch_models,
    vrarch_separator,
)

if torch.cuda.is_available():
    gpuconcurrency = torch.cuda.device_count()
    device = "cuda"
    use_autocast = device == "cuda"
else:
    gpuconcurrency = 1
    device = "cpu"

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
