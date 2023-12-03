import gradio as gr
import whisperx
import argparse

from Simplify_Chinese.convert_simple import convert_t2s

class WisperAPI:   
    def __init__(self, model_path: str, device: str="cuda", language: str='zh'):
        self.device = device
        self.model_path = model_path
        self.language = language
        self.model = whisperx.load_model(self.model_path, 
                                         device=self.device, 
                                         compute_type="float32", 
                                         language=self.language
                                         )
        self.model_a, self.metadata = whisperx.load_align_model(language_code=self.language, 
                                                                device=self.device
                                                                )
        self.diarize_model = whisperx.DiarizationPipeline(use_auth_token="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",
                                                     device=self.device)
        
        self.transcribe_list = []
        
        with gr.Blocks() as demo:
            gr.Markdown("## EchoTranscribe: 智能语音转写")
            gr.Image("circular_echo_transcribe_logo.png")
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
            audio_transcribe_button.click(self.run, inputs=audio_input, outputs=[transcription_output1, transcription_output2, transcription_output3])
            video_transcribe_button.click(self.run, inputs=[video_input, "video"], outputs=[transcription_output1, transcription_output2, transcription_output3])

        demo.launch(server_name='127.0.0.1', server_port=7777)

    def convert_time(self, time_str):
        # 将字符串转换为浮点数
        time_in_seconds = float(time_str)
        # 计算分钟和秒数
        minutes = int(time_in_seconds // 60)
        seconds = time_in_seconds % 60
        # 格式化输出，保留1位小数
        return f"{minutes}分{seconds:.1f}秒"

    def is_sentence_ending(self, word, sentence_endings):
        return word in sentence_endings

    def needs_preprocessing(self, word, previous_word, sentence_endings, digits):
        return word in digits or (previous_word in sentence_endings and word not in sentence_endings)

    def insert_timestamps(self, words, idx):
        for i in range(idx, -1, -1):
            if 'end' in words[i]:
                words[idx]['start'] = words[i]['end']
                break
        for j in range(idx + 1, len(words)):
            if 'start' in words[j]:
                words[idx]['end'] = words[j]['start']
                break
        return words

    def split_sentences(self, segments):
        words = segments['words']
        split_segments = []
        current_word = None
        start_time = None
        end_time = None
        sentence_endings = ["?", "!", "。", "？", "！", "."]
        digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '《']
    
        # 预处理步骤
        for idx, word_info in enumerate(words):
            if idx == 0 or 'start' in word_info:
                continue
            previous_word = words[idx - 1]['word']
            if self.needs_preprocessing(word_info['word'], previous_word, sentence_endings, digit):
                words = self.insert_timestamps(words, idx)
            elif not self.is_sentence_ending(word_info['word'], sentence_endings) and not self.is_sentence_ending(previous_word, sentence_endings):
                words[idx]['start'] = words[idx - 1]['end']
                words[idx]['end'] = words[idx - 1]['end']
    
        # 分割句子
        for idx, word_info in enumerate(words):
            word = word_info['word']
            if start_time is None:
                if current_word in sentence_endings and word in sentence_endings:
                    split_segments[-1]['text'] += word
                    split_segments[-1]['words'].append(word_info)
                    continue
                start_time = word_info.get('start')
            if word in sentence_endings or word_info is words[-1]:
                end_time = words[idx - 1].get('end', segments['end'])
                new_segment_text = ''.join([w['word'] for w in words if w.get('start', start_time - 1) >= start_time and w.get('end', end_time + 1) <= end_time]) + word
                new_segment_words = [w for w in words if w.get('start', start_time - 1) >= start_time and w.get('end', end_time + 1) <= end_time]
                new_segment_words.append(word_info)
                new_segment = {'start': start_time, 'end': end_time, 'text': new_segment_text, 'words': new_segment_words}
                split_segments.append(new_segment)
                start_time = None
    
            current_word = word
    
        return split_segments
    
    def run(self, audio_path: str,
            style: str="audio",
            batch_size: int=16,
            language: str='zh') -> str:
        if audio_path == "video":
            audio_path = audio_path['name']
        if audio_path is None:
            return "请先上传音频文件。", "请先上传音频文件。", "请先上传音频文件。"
        audio = whisperx.load_audio(audio_path)
        segment_text = ""
        segment_text2 = ""
        segment_text3 = ""
        transcription = self.model.transcribe(audio,
                                          batch_size=16, 
                                          language='zh'
                                          )
        transcription = whisperx.align(transcription["segments"], 
                                       self.model_a, 
                                       self.metadata, 
                                       audio, 
                                       self.device, 
                                       return_char_alignments=False)
        seg_split = transcription["segments"]
        seg_list = []
        for seg in seg_split:
            seg_list.extend(self.split_sentences(seg))
        transcription["segments"] = seg_list
        diarize_segments = self.diarize_model(audio)
        transcription = whisperx.assign_word_speakers(diarize_segments, transcription)
        transcription = list(transcription["segments"])
        self.transcribe_list = []
        for seg in transcription:
            seg_info = {
                'speaker': seg['speaker'],
                'text': seg['text'],
                'start': seg['start'],
                'end': seg['end'],
                }
            self.transcribe_list.append(seg_info)
            
        for idx in range(len(transcription)):
            if idx == 0:
                continue
            while idx < len(transcription) and transcription[idx]['speaker'] == transcription[idx-1]['speaker']:
                transcription[idx-1]['text'] += transcription[idx]['text']
                transcription[idx-1]['end'] = transcription[idx]['end']
                transcription.pop(idx)
              
        for aud in transcription:
            start, end = self.convert_time(aud['start']), self.convert_time(aud['end']) 
            segment_text += f"【{start} =========> {end}】\n{aud['speaker']}: {aud['text']}\n"
            segment_text2 += f"{aud['speaker']}: {aud['text']}\n"
            segment_text3 += aud['text'] + '\n'
        transcriptions = convert_t2s(segment_text)
        transcriptions2 = convert_t2s(segment_text2)
        transcriptions3 = convert_t2s(segment_text3)
        return transcriptions, transcriptions2, transcriptions3
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="whisper")
    parser.add_argument('--language', dest='language', type=str, default='zh')
    parser.add_argument('--device', dest='device', type=str, default='cuda')
    parser.add_argument('--model_path', dest='model_path', type=str, default='E:/code/faster-whisper/faster-whisper-large-v2')
    args = parser.parse_args()
    
    # load whisper model
    whisper_api = WisperAPI(args.model_path)
    
    
    
    

