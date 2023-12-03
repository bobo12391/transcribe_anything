# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 12:57:02 2023

@author: Administrator
"""

import gradio as gr
from PIL import Image

def run(path, style: str="audio"):
    print(path)
    if type(path) == dict:
        return "视频", "视频", "视频"
    return "音频", "音频", "音频"

with gr.Blocks() as demo:
    gr.Markdown("## EchoTranscribe: 智能语音转写")
    gr.Markdown("![logo](circular_echo_transcribe_logo.png)")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="点击选择音频", type="filepath")
            video_input = gr.Video(label="点击选择视频")
        with gr.Column():
            audio_transcribe_button = gr.Button("音频转录")
            video_transcribe_button = gr.Button("视频转录")
            
            with gr.Tabs():
                with gr.TabItem("详细"):
                    transcription_output1 = gr.Textbox(label = "详细转录结果", placeholder="转录结果", interactive=True, lines=40, max_lines=100)
                with gr.TabItem("简略"):
                    transcription_output2 = gr.Textbox(label = "简略转录结果", placeholder="转录结果", interactive=True, lines=40, max_lines=100)
                with gr.TabItem("文本"):
                    transcription_output3 = gr.Textbox(label = "纯文本", placeholder="转录结果", interactive=True, lines=40, max_lines=100)

    # Interactions
    audio_transcribe_button.click(run, inputs=audio_input, outputs=[transcription_output1, transcription_output2, transcription_output3])
    video_transcribe_button.click(run, inputs=video_input, outputs=[transcription_output1, transcription_output2, transcription_output3])

demo.launch(server_name='127.0.0.1', server_port=7777)