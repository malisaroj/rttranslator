import torch
import numpy as np
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Wav2Vec2Processor, Wav2Vec2ForCTC, MarianMTModel, MarianTokenizer, BertTokenizer, BertModel, Trainer, TrainingArguments
import soundfile as sf
from datasets import load_dataset
import librosa
from googletrans import Translator
from gtts import gTTS
import matplotlib.pyplot as plt
import random
import pandas as pd
import os

# Load transformer models for context embedding
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


# Cloud-based models
asr_processor_cloud = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
asr_model_cloud = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# Local-based models
asr_processor_local = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
asr_model_local = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
translator_model_local = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
translator_tokenizer_local = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

# Bandwidth and complexity thresholds
B_THRESH = 1000
COMPLEXITY_THRESH = 1.5

# Load your own audio file
def load_audio_file(file_path):
    audio_array, original_sampling_rate = sf.read(file_path)
    target_sampling_rate = 16000
    if original_sampling_rate != target_sampling_rate:
        audio_array = librosa.resample(audio_array, orig_sr=original_sampling_rate, target_sr=target_sampling_rate)
    return audio_array, target_sampling_rate

# Speech complexity calculation
def calculate_complexity(audio_array, sampling_rate=16000):
    if not isinstance(audio_array, np.ndarray):
        raise ValueError("audio_array should be a numpy array")
    tempo, _ = librosa.beat.beat_track(y=audio_array, sr=sampling_rate)
    return tempo

# Get context embedding using BERT
def get_context_embedding(context):
    inputs = bert_tokenizer(context, return_tensors="pt")
    with torch.no_grad():
        context_embeddings = bert_model(**inputs).last_hidden_state
    return context_embeddings

# Adaptive ASR model switch based on bandwidth and complexity
def switch_asr_model(bandwidth, complexity):
    cloud_score = bandwidth / (B_THRESH + 1e-6)
    complexity_factor = complexity / (COMPLEXITY_THRESH + 1e-6)
    threshold = 0.5 + 0.1 * (complexity_factor - 1)

    if cloud_score > threshold:
        return asr_processor_cloud, asr_model_cloud
    else:
        return asr_processor_local, asr_model_local

# ASR Processing
def speech_to_text(audio_array, sampling_rate, bandwidth, complexity):
    if sampling_rate != 16000:
        raise ValueError("Audio sampling rate should be 16,000 Hz.")

    if isinstance(audio_array, list):
        audio_array = np.array(audio_array)

    asr_processor, asr_model = switch_asr_model(bandwidth, complexity)
    inputs = asr_processor(audio_array, return_tensors="pt", sampling_rate=sampling_rate, padding=True)

    with torch.no_grad():
        logits = asr_model(inputs['input_values']).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_processor.batch_decode(predicted_ids)

    return transcription[0]

# Translation Processing

def translate_text_google_cloud(text, target_language='en'):
    translator = Translator()

    try:
        # Translate text
        translation = translator.translate(text, dest=target_language)
        translated_text = translation.text
    except Exception as e:
        print(f"Error during translation: {e}")
        translated_text = ""

    print(f"Translation output: {translated_text}")
    return translated_text

def get_context_embedding(context):
    inputs = bert_tokenizer(context, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        context_embeddings = bert_model(**inputs).last_hidden_state
    return context_embeddings

def translate_text(text, bandwidth, context="", target_language="en"):
    """
    Translate the ASR-translated text to a target language with cloud-based translation if bandwidth is sufficient.
    """
    context_embedding = get_context_embedding(context)
    text_with_context = f"{context} {text}"

    if bandwidth >= B_THRESH:
        # Use Google Cloud Translation if bandwidth is high
        print("Using Google Cloud Translation for translation")
        translated = translate_text_google_cloud(text_with_context, target_language=target_language)
    else:
        # Local translation fallback (using MarianMT or another fallback mechanism)
        translated = translator_tokenizer_local([text_with_context], return_tensors="pt", padding=True, truncation=True, max_length=128)
        translated_tokens = translator_model_local.generate(**translated)
        translated = translator_tokenizer_local.decode(translated_tokens[0], skip_special_tokens=True)

    return translated

# Reward function
def get_reward(prediction_accuracy, bandwidth, processing_time, user_preference=None):
    if user_preference:
        accuracy_weight, efficiency_weight, latency_weight = user_preference
    else:
        accuracy_weight = 1.0
        efficiency_weight = 0.5
        latency_weight = 0.5

    max_processing_time = 10.0
    normalized_processing_time = min(processing_time / max_processing_time, 1.0)

    accuracy_reward = accuracy_weight * prediction_accuracy
    efficiency_reward = efficiency_weight * (1 - normalized_processing_time)
    bandwidth_reward = latency_weight * min(bandwidth / B_THRESH, 1.0)

    total_reward = accuracy_reward + efficiency_reward + bandwidth_reward
    return total_reward

# RL agent for bandwidth and complexity management
class RLAgent:
    def __init__(self, initial_bandwidth, complexity_levels, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.bandwidth = initial_bandwidth
        self.complexity_levels = complexity_levels
        self.current_level = 0
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_state(self):
        return (self.bandwidth, self.current_level)

    def choose_action(self, state):
        self.epsilon = max(0.01, self.epsilon * 0.995)
        state = tuple(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(['increase', 'decrease', 'maintain'])
        else:
            actions = ['increase', 'decrease', 'maintain']
            q_values = [self.q_table.get((state, action), 0) for action in actions]
            max_q_value = max(q_values)
            max_actions = [actions[i] for i, q in enumerate(q_values) if q == max_q_value]
            return np.random.choice(max_actions)

    def update(self, state, action, reward, new_state):
        state = tuple(state)
        new_state = tuple(new_state)

        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0

        future_q_values = [self.q_table.get((new_state, a), 0) for a in ['increase', 'decrease', 'maintain']]
        max_future_q_value = max(future_q_values)

        old_q_value = self.q_table[(state, action)]
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * max_future_q_value - old_q_value)
        self.q_table[(state, action)] = new_q_value

    def adjust_complexity(self):
        state = self.get_state()
        action = self.choose_action(state)

        if action == 'increase' and self.current_level < len(self.complexity_levels) - 1:
            self.current_level += 1
        elif action == 'decrease' and self.current_level > 0:
            self.current_level -= 1

        return self.complexity_levels[self.current_level], action

def fine_tune_model(model, tokenizer, translated_texts, correct_texts):
    encodings = tokenizer(translated_texts, truncation=True, padding=True, return_tensors="pt")
    decodings = tokenizer(correct_texts, truncation=True, padding=True, return_tensors="pt")

    class TranslationDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, decodings):
            self.encodings = encodings
            self.decodings = decodings

        def __len__(self):
            return len(self.encodings.input_ids)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.decodings.input_ids[idx])
            return item

    dataset = TranslationDataset(encodings, decodings)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir="./logs",
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    model.save_pretrained("./fine-tuned-model")
    tokenizer.save_pretrained("./fine-tuned-model")

def user_feedback_loop(translated_text, correct_text, translation_model, tokenizer):
    feedback_score = sum(1 for a, b in zip(translated_text, correct_text) if a == b) / len(correct_text)

    print(f"Feedback Score: {feedback_score:.2f}")

    if feedback_score < 0.8:
        print("Feedback score is below threshold. Fine-tuning model...")
        fine_tune_model(translation_model, tokenizer, [translated_text], [correct_text])

    return feedback_score

def text_to_speech(text, lang='en', output_file='output.mp3'):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_file)
    print(f"Audio saved as {output_file}")

# Full translation pipeline with debug prints
def translation_pipeline(audio_file, bandwidth):
    rl_agent = RLAgent(bandwidth, ['low', 'medium', 'high'])

    # Load and preprocess audio
    audio_array, sampling_rate = load_audio_file(audio_file)
    print(f"Loaded audio file. Sample rate: {sampling_rate}, Audio length: {len(audio_array)}")

    # Calculate speech complexity
    complexity = calculate_complexity(audio_array)
    print(f"Calculated complexity: {complexity}")

    # Initial state
    state = rl_agent.get_state()
    print(f"Initial state: {state}")

    # Adjust model complexity based on RL agent
    new_complexity, action = rl_agent.adjust_complexity()
    print(f"Adjusted complexity: {new_complexity}, Action taken: {action}")

    # ASR processing and latency measurement
    start_time = time.time()
    text = speech_to_text(audio_array, sampling_rate, bandwidth, complexity)
    asr_latency = time.time() - start_time
    print(f"ASR output: {text}")
    print(f"ASR latency: {asr_latency:.2f} seconds")

    # Translation processing and latency measurement
    start_time = time.time()
    translation = translate_text(text, bandwidth)
    translation_latency = time.time() - start_time
    print(f"Translation output: {translation}")
    print(f"Translation latency: {translation_latency:.2f} seconds")

    # Convert translated text to speech
    text_to_speech(translation, lang='en', output_file='translated_output.mp3')

    # Total processing time
    total_processing_time = asr_latency + translation_latency

    # Test the function with ASR output and reference text
    reference_text = "周末你有什么打算？我早就想好了，请你吃饭，看电影，喝咖啡。请我？是啊，我已经找好饭馆了，电影票也买好了。我还没想好要不要跟你去呢。"
    asr_output = text
    prediction_accuracy = compute_translation_accuracy(asr_output, reference_text)
    print(f"Translation Accuracy: {prediction_accuracy:.2f}%")

    # Calculate reward
    reward = get_reward(prediction_accuracy, bandwidth, total_processing_time)
    print(f"Reward: {reward}")

    # New state after processing
    new_state = rl_agent.get_state()
    print(f"New state: {new_state}")

    # Update RL agent
    rl_agent.update(state, action, reward, new_state)

    '''
    # Process user feedback if provided
    if user_correct_translation:
        feedback_score = user_feedback_loop(translation, user_correct_translation, translator_model_new, translator_tokenizer_new)
        print(f"User Feedback Score: {feedback_score:.2f}")
    '''

    # Simulate model type switch (for tracking cloud/local usage)
    _, action = rl_agent.adjust_complexity()
    model_type = "cloud" if bandwidth >= B_THRESH else "local"

    record_metrics(bandwidth, complexity, asr_latency, translation_latency, prediction_accuracy, reward, model_type)

    return translation

# Initialize metrics dictionaries for both models
metrics_cloud = {
    "bandwidth": [],
    "complexity": [],
    "asr_latency": [],
    "translation_latency": [],
    "translation_accuracy": [],
    "reward": []
}

metrics_cloud_edge = {
    "bandwidth": [],
    "complexity": [],
    "asr_latency": [],
    "translation_latency": [],
    "translation_accuracy": [],
    "reward": [],
    "model_switches": {"cloud": 0, "local": 0}
}

def record_metrics_cloud(bandwidth, complexity, asr_latency, translation_latency, translation_accuracy, reward, model_type):
    metrics_cloud['bandwidth'].append(bandwidth)
    metrics_cloud['complexity'].append(complexity)
    metrics_cloud['asr_latency'].append(asr_latency)
    metrics_cloud['translation_latency'].append(translation_latency)
    metrics_cloud['translation_accuracy'].append(translation_accuracy)
    metrics_cloud['reward'].append(reward)

def record_metrics(bandwidth, complexity, asr_latency, translation_latency, translation_accuracy, reward, model_type):
    metrics_cloud_edge['bandwidth'].append(bandwidth)
    metrics_cloud_edge['complexity'].append(complexity)
    metrics_cloud_edge['asr_latency'].append(asr_latency)
    metrics_cloud_edge['translation_latency'].append(translation_latency)
    metrics_cloud_edge['translation_accuracy'].append(translation_accuracy)
    metrics_cloud_edge['reward'].append(reward)
    metrics_cloud_edge['model_switches'][model_type] += 1



# Function to simulate Cloud only pipeline
def cloud_only_pipeline(audio_file, bandwidth):
    # Load and preprocess audio
    audio_array, sampling_rate = load_audio_file(audio_file)
    print(f"Loaded audio file. Sample rate: {sampling_rate}, Audio length: {len(audio_array)}")

    # Calculate speech complexity (still needed for consistent comparison)
    complexity = calculate_complexity(audio_array)
    print(f"Calculated complexity: {complexity}")

    # ASR processing
    start_time = time.time()
    text = speech_to_text(audio_array, sampling_rate, bandwidth, complexity)
    asr_latency = time.time() - start_time
    print(f"ASR output (Cloud-Only): {text}")
    print(f"ASR latency: {asr_latency:.2f} seconds")

    # Translation processing using Google Cloud Translation API
    start_time = time.time()
    translated = translate_text_google_cloud(text, target_language='en')  # Translate to English (or another target language)
    translation_latency = time.time() - start_time
    print(f"Translation output (Cloud-Only): {translated}")
    print(f"Translation latency: {translation_latency:.2f} seconds")

    # Calculate total processing time and reward
    total_latency = asr_latency + translation_latency

    # Test the function with ASR output and reference text
    reference_text = "周末你有什么打算？我早就想好了，请你吃饭，看电影，喝咖啡。请我？是啊，我已经找好饭馆了，电影票也买好了。我还没想好要不要跟你去呢。"
    asr_output = text

    prediction_accuracy = compute_translation_accuracy(asr_output, reference_text)
    print(f"Translation Accuracy: {prediction_accuracy:.2f}%")
    reward = get_reward(prediction_accuracy, bandwidth, total_latency)

    # Metrics collection for cloud-only model
    record_metrics_cloud(bandwidth, complexity, asr_latency, translation_latency, prediction_accuracy, reward, model_type="cloud")

    return translated

import jieba
import re
from difflib import SequenceMatcher

def remove_punctuation(text):
    """Remove punctuation from a given text."""
    return re.sub(r'[^\w\s]', '', text)

def compute_translation_accuracy(asr_output, reference_text):
    """
    Compute the word-level accuracy between ASR-translated output and the reference text.

    Args:
        asr_output (str): The ASR-translated text.
        reference_text (str): The ground truth reference text.

    Returns:
        float: Accuracy as a percentage.
    """
    # Remove punctuation from both texts
    asr_output = remove_punctuation(asr_output)
    reference_text = remove_punctuation(reference_text)
    
    # Tokenize using jieba
    asr_tokens = list(jieba.cut(asr_output))
    reference_tokens = list(jieba.cut(reference_text))
    
    # Compute sequence similarity
    matcher = SequenceMatcher(None, asr_tokens, reference_tokens)
    matches = sum(triple.size for triple in matcher.get_matching_blocks())
    
    # Calculate accuracy
    accuracy = (matches / len(reference_tokens)) * 100 if reference_tokens else 0
    return accuracy

def plot_latency(metrics):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['bandwidth'], metrics['asr_latency'], marker='o', label='ASR Latency')
    plt.plot(metrics['bandwidth'], metrics['translation_latency'], marker='x', label='Translation Latency')
    plt.xlabel('Bandwidth (Kbps)')
    plt.ylabel('Latency (seconds)')
    plt.title('ASR and Translation Latency vs Bandwidth')
    plt.legend()
    plt.savefig('latency_plot.png')
    plt.show()

def plot_accuracy(metrics):
    plt.figure(figsize=(6, 4))
    plt.plot(metrics['bandwidth'], metrics['translation_accuracy'], marker='s', color='green', label='Translation Accuracy')
    plt.xlabel('Bandwidth (Kbps)')
    plt.ylabel('Translation Accuracy (%)')
    plt.title('Translation Accuracy vs Bandwidth')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()

def plot_reward(metrics):
    plt.figure(figsize=(6, 4))
    plt.plot(metrics['bandwidth'], metrics['reward'], marker='d', color='purple', label='Reward')
    plt.xlabel('Bandwidth (Kbps)')
    plt.ylabel('Reward')
    plt.title('Reward vs Bandwidth')
    plt.legend()
    plt.savefig('reward_plot.png')
    plt.show()

def plot_model_switches(metrics):
    plt.figure(figsize=(6, 4))
    labels = ['Cloud', 'Local']
    switches = [metrics['model_switches']['cloud'], metrics['model_switches']['local']]
    plt.bar(labels, switches, color=['blue', 'orange'])
    plt.title('Model Switching Frequency')
    plt.ylabel('Number of Switches')
    plt.savefig('switch_plot.png')
    plt.show()

def plot_latency_cloudvedge(metrics_cloud_edge, metrics_cloud):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_cloud['bandwidth'], metrics_cloud['asr_latency'], label='Cloud-Only ASR Latency', marker='o')
    plt.plot(metrics_cloud_edge['bandwidth'], metrics_cloud_edge['asr_latency'], label='Cloud-Edge ASR Latency', marker='x')
    plt.plot(metrics_cloud['bandwidth'], metrics_cloud['translation_latency'], label='Cloud-Only Translation Latency', marker='s')
    plt.plot(metrics_cloud_edge['bandwidth'], metrics_cloud_edge['translation_latency'], label='Cloud-Edge Translation Latency', marker='^')
    plt.xlabel('Bandwidth (Kbps)')
    plt.ylabel('Latency (seconds)')
    plt.title('Latency Comparison: Cloud-Only vs Cloud-Edge')
    plt.legend()
    plt.show()

def plot_accuracy_cloudvedge(metrics_cloud_edge, metrics_cloud):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_cloud['bandwidth'], metrics_cloud['translation_accuracy'], label='Cloud-Only Accuracy', marker='o')
    plt.plot(metrics_cloud_edge['bandwidth'], metrics_cloud_edge['translation_accuracy'], label='Cloud-Edge Accuracy', marker='x')
    plt.xlabel('Bandwidth (Kbps)')
    plt.ylabel('Translation Accuracy (%)')
    plt.title('Translation Accuracy Comparison')
    plt.legend()
    plt.show()

def plot_reward_cloudvedge(metrics_cloud_edge, metrics_cloud):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_cloud['bandwidth'], metrics_cloud['reward'], label='Cloud-Only Reward', marker='o')
    plt.plot(metrics_cloud_edge['bandwidth'], metrics_cloud_edge['reward'], label='Cloud-Edge Reward', marker='x')
    plt.xlabel('Bandwidth (Kbps)')
    plt.ylabel('Reward')
    plt.title('Reward Comparison')
    plt.legend()
    plt.show()


bandwidths = [500, 800, 1000, 1200, 1500]
audio_file = "C:\\Users\\rog\\Downloads\\a.wav"

print(f"File exists: {os.path.exists(audio_file)}")
rl_agent = RLAgent(initial_bandwidth=800, complexity_levels=['low', 'medium', 'high'])

# Run Cloud-Edge Model
for bandwidth in bandwidths:
    translation_pipeline(audio_file, bandwidth)  # Use cloud-edge metrics collection

# Run Cloud-Only Model
for bandwidth in bandwidths:
    cloud_only_pipeline(audio_file, bandwidth)

# Plot results and show data table
plot_latency(metrics_cloud_edge)
plot_reward(metrics_cloud_edge)
plot_model_switches(metrics_cloud_edge)
plot_accuracy(metrics_cloud_edge)
plot_latency_cloudvedge(metrics_cloud_edge, metrics_cloud)
plot_reward_cloudvedge(metrics_cloud_edge, metrics_cloud)
plot_accuracy_cloudvedge(metrics_cloud_edge, metrics_cloud)

def main():
    bandwidths = [500, 800, 1000, 1200, 1500]
    #audio_file = "/content/1.wav"
    audio_file = "C:\\Users\\rog\\Downloads\\a.wav"

    print(f"File exists: {os.path.exists(audio_file)}")
    rl_agent = RLAgent(initial_bandwidth=800, complexity_levels=['low', 'medium', 'high'])

    # Run Cloud-Only Model
    for bandwidth in bandwidths:
        cloud_only_pipeline(audio_file, bandwidth)

    # Run Cloud-Edge Model
    for bandwidth in bandwidths:
        translation_pipeline(audio_file, bandwidth)  # Use cloud-edge metrics collection

    # Plot results and show data table
    plot_latency(metrics_cloud_edge)
    plot_reward(metrics_cloud_edge)
    plot_model_switches(metrics_cloud_edge)
    plot_accuracy(metrics_cloud_edge)

    # Show the data table
    df = pd.DataFrame({
        "Bandwidth (Kbps)": metrics_cloud_edge['bandwidth'],
        "Complexity": metrics_cloud_edge['complexity'],
        "ASR Latency (s)": metrics_cloud_edge['asr_latency'],
        "Translation Latency (s)": metrics_cloud_edge['translation_latency'],
        "Translation Accuracy (%)": [acc * 100 for acc in metrics_cloud_edge['translation_accuracy']],
        "Reward": metrics_cloud_edge['reward']
    })

    print(df)
    df.to_csv("translation_metrics.csv", index=False)


    # Create DataFrame for the table
    df_cloud = pd.DataFrame({
        "Bandwidth (Kbps)": metrics_cloud['bandwidth'],
        "Complexity": metrics_cloud['complexity'],
        "ASR Latency (s)": metrics_cloud['asr_latency'],
        "Translation Latency (s)": metrics_cloud['translation_latency'],
        "Translation Accuracy (%)": [acc * 100 for acc in metrics_cloud['translation_accuracy']],
        "Reward": metrics_cloud['reward']
    })

    # Print DataFrame as a table
    print(df_cloud)

    # Save to CSV if needed
    df_cloud.to_csv("translation_metrics_cloud.csv", index=False)

if __name__ == "__main__":
    main()

