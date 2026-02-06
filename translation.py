import random
import time
import numpy as np
import torch
import evaluate
import re
import gc
import io
import jieba
from contextlib import contextmanager

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    MarianMTModel, MarianTokenizer
)
import soundfile as sf
from googletrans import Translator

# Import sacrebleu
try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    print("sacrebleu not available. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "sacrebleu"])
    import sacrebleu
    SACREBLEU_AVAILABLE = True

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000
MAX_SAMPLES = 200
BANDWIDTH_LEVELS = [500, 800, 1200]
BTHRESH = 1000
COMPLEXITY_LEVELS = ["low", "medium", "high"]

# Visualization settings
SAVE_PLOTS = True
PLOT_FORMAT = 'png'
SHOW_INTERACTIVE = False

# Initialize metrics - using sacrebleu instead of evaluate for BLEU
chrf_metric = evaluate.load("chrf")

# Initialize googletrans translator
google_translator = Translator()

# Initialize jieba for nepali tokenization
jieba.initialize()

class SystemVisualizer:
    """Visualization class for the speech translation system"""

    # ... [Keep all the visualization methods exactly the same as before] ...
    def __init__(self):
        self.history = {
            'rewards': [],
            'latencies': [],
            'bleu_scores': [],
            'chrf_scores': [],
            'cloud_usage': [],
            'bandwidth': [],
            'complexity': [],
            'actions': [],
            'states': []
        }
        self.colors = {
            'cloud': '#FF6B6B',
            'edge': '#4ECDC4',
            'ne': '#45B7D1',
            'en': '#96CEB4',
            'increase': '#FFEAA7',
            'decrease': '#DDA0DD',
            'maintain': '#98D8C8'
        }

    def add_data_point(self, **kwargs):
        """Add data point to history"""
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)

    def plot_training_progress(self, save_path=None):
        """Plot training progress over time"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('RL Training Progress', fontsize=16, fontweight='bold')

        # Plot 1: Reward over time
        axes[0, 0].plot(self.history['rewards'], 'o-', linewidth=2, markersize=4)
        axes[0, 0].set_title('Reward Evolution')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Latency over time
        axes[0, 1].plot(self.history['latencies'], 's-', color='orange', linewidth=2, markersize=4)
        axes[0, 1].set_title('Latency Evolution')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Latency (s)')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: BLEU scores
        if self.history['bleu_scores']:
            bleu_scores = np.array(self.history['bleu_scores'])
            if bleu_scores.ndim == 2 and bleu_scores.shape[1] == 2:
                axes[1, 0].plot(bleu_scores[:, 0], 'o-', label='nepali BLEU', alpha=0.7)
                axes[1, 0].plot(bleu_scores[:, 1], 's-', label='English BLEU', alpha=0.7)
                axes[1, 0].legend()
            axes[1, 0].set_title('BLEU Scores')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('BLEU Score')
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Cloud usage pattern
        if self.history['cloud_usage']:
            cloud_usage = np.array(self.history['cloud_usage'])
            cumulative_cloud = np.cumsum(cloud_usage) / np.arange(1, len(cloud_usage) + 1)
            axes[1, 1].plot(cumulative_cloud, '^-', color='purple', linewidth=2)
            axes[1, 1].set_title('Cloud Usage Rate (Cumulative)')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Cloud Usage Rate')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)

        # Plot 5: Action distribution
        if self.history['actions']:
            actions = self.history['actions']
            action_counts = pd.Series(actions).value_counts()
            axes[2, 0].bar(action_counts.index, action_counts.values,
                          color=[self.colors.get(a, '#95A5A6') for a in action_counts.index])
            axes[2, 0].set_title('Action Distribution')
            axes[2, 0].set_xlabel('Action')
            axes[2, 0].set_ylabel('Count')
            for i, v in enumerate(action_counts.values):
                axes[2, 0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

        # Plot 6: Complexity levels over time
        if self.history['complexity']:
            complexity_map = {'low': 0, 'medium': 1, 'high': 2}
            complexity_numeric = [complexity_map.get(c, 1) for c in self.history['complexity']]
            axes[2, 1].plot(complexity_numeric, 'D-', color='red', linewidth=2, markersize=6)
            axes[2, 1].set_yticks([0, 1, 2])
            axes[2, 1].set_yticklabels(['Low', 'Medium', 'High'])
            axes[2, 1].set_title('Complexity Level Evolution')
            axes[2, 1].set_xlabel('Training Step')
            axes[2, 1].set_ylabel('Complexity')
            axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}_training_progress.{PLOT_FORMAT}', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_decision_surface(self, agent, save_path=None):
        """Plot decision surface for RL agent"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create grid of states
        bandwidth_grid = np.array(BANDWIDTH_LEVELS)
        complexity_grid = np.arange(len(COMPLEXITY_LEVELS))

        # Create meshgrid
        X, Y = np.meshgrid(bandwidth_grid, complexity_grid)
        Z = np.zeros_like(X, dtype=float)

        # Fill with Q-values for "maintain" action
        for i, complexity in enumerate(complexity_grid):
            for j, bandwidth in enumerate(bandwidth_grid):
                state_key = f"{bandwidth}_{COMPLEXITY_LEVELS[complexity]}"
                q_value = agent.q_table.get((state_key, "maintain"), 0.0)
                Z[i, j] = q_value

        # Create heatmap
        im = ax.imshow(Z, cmap='RdYlGn', aspect='auto', origin='lower')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(bandwidth_grid)))
        ax.set_xticklabels([f'{b} kbps' for b in bandwidth_grid])
        ax.set_yticks(np.arange(len(complexity_grid)))
        ax.set_yticklabels(COMPLEXITY_LEVELS)

        # Add text annotations
        for i in range(len(complexity_grid)):
            for j in range(len(bandwidth_grid)):
                text = ax.text(j, i, f'{Z[i, j]:.2f}',
                              ha="center", va="center",
                              color="white" if Z[i, j] < np.max(Z)/2 else "black",
                              fontweight='bold')

        ax.set_title('RL Agent Q-Value Surface (Maintain Action)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Bandwidth', fontsize=12)
        ax.set_ylabel('Complexity', fontsize=12)
        plt.colorbar(im, ax=ax, label='Q-Value')

        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}_decision_surface.{PLOT_FORMAT}', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_performance_comparison(self, save_path=None):
        """Compare performance across different conditions"""
        if not self.history['states']:
            return

        # Create DataFrame for analysis
        df = pd.DataFrame({
            'reward': self.history['rewards'],
            'latency': self.history['latencies'],
            'cloud_used': self.history['cloud_usage'],
            'bandwidth': self.history['bandwidth'],
            'complexity': self.history['complexity'],
            'action': self.history['actions']
        })

        if len(df) == 0:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Performance Comparison Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Reward by bandwidth
        bandwidth_groups = df.groupby('bandwidth')['reward'].mean()
        axes[0, 0].bar(bandwidth_groups.index.astype(str), bandwidth_groups.values,
                      color=['#FF9999', '#66B2FF', '#99FF99'])
        axes[0, 0].set_title('Average Reward by Bandwidth')
        axes[0, 0].set_xlabel('Bandwidth (kbps)')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # Plot 2: Latency by complexity
        complexity_groups = df.groupby('complexity')['latency'].mean()
        complexity_order = ['low', 'medium', 'high']
        axes[0, 1].bar([c for c in complexity_order if c in complexity_groups.index],
                      [complexity_groups.get(c, 0) for c in complexity_order],
                      color=['#FFD700', '#FFA500', '#FF4500'])
        axes[0, 1].set_title('Average Latency by Complexity')
        axes[0, 1].set_xlabel('Complexity')
        axes[0, 1].set_ylabel('Average Latency (s)')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Plot 3: Cloud usage by complexity and bandwidth
        if 'cloud_used' in df.columns:
            cloud_rate = df.groupby(['complexity', 'bandwidth'])['cloud_used'].mean().unstack()
            if not cloud_rate.empty:
                cloud_rate.plot(kind='bar', ax=axes[0, 2], rot=0)
                axes[0, 2].set_title('Cloud Usage Rate by Complexity & Bandwidth')
                axes[0, 2].set_xlabel('Complexity')
                axes[0, 2].set_ylabel('Cloud Usage Rate')
                axes[0, 2].legend(title='Bandwidth')
                axes[0, 2].grid(True, alpha=0.3, axis='y')

        # Plot 4: Action effectiveness
        action_groups = df.groupby('action')['reward'].agg(['mean', 'std'])
        if not action_groups.empty:
            x_pos = np.arange(len(action_groups))
            axes[1, 0].bar(x_pos, action_groups['mean'],
                          yerr=action_groups['std'],
                          capsize=5,
                          color=[self.colors.get(a, '#95A5A6') for a in action_groups.index])
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(action_groups.index)
            axes[1, 0].set_title('Reward by Action Type')
            axes[1, 0].set_xlabel('Action')
            axes[1, 0].set_ylabel('Average Reward ± STD')
            axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Plot 5: Scatter plot of reward vs latency
        scatter = axes[1, 1].scatter(df['latency'], df['reward'],
                                     c=df['bandwidth'], cmap='viridis',
                                     alpha=0.6, s=50)
        axes[1, 1].set_title('Reward vs Latency (colored by bandwidth)')
        axes[1, 1].set_xlabel('Latency (s)')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='Bandwidth')

        # Plot 6: Cumulative performance
        axes[1, 2].plot(np.cumsum(df['reward']), 'b-', linewidth=2, label='Cumulative Reward')
        axes[1, 2].plot(np.cumsum(df['latency']), 'r-', linewidth=2, label='Cumulative Latency')
        axes[1, 2].set_title('Cumulative Performance')
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Cumulative Value')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}_performance_comparison.{PLOT_FORMAT}', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_interactive_dashboard(self):
        """Create interactive dashboard with Plotly"""
        if len(self.history['rewards']) < 2:
            return

        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Reward Evolution', 'Latency Trend', 'BLEU Scores',
                           'Cloud Usage Pattern', 'Complexity Levels', 'Action Distribution',
                           'Reward vs Latency', 'Cumulative Performance', 'Q-Value Heatmap'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'heatmap'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # 1. Reward evolution
        fig.add_trace(
            go.Scatter(y=self.history['rewards'], mode='lines+markers',
                      name='Reward', line=dict(color='blue', width=2)),
            row=1, col=1
        )

        # 2. Latency trend
        fig.add_trace(
            go.Scatter(y=self.history['latencies'], mode='lines+markers',
                      name='Latency', line=dict(color='red', width=2)),
            row=1, col=2
        )

        # 3. BLEU scores
        if self.history['bleu_scores']:
            bleu_scores = np.array(self.history['bleu_scores'])
            if bleu_scores.ndim == 2 and bleu_scores.shape[1] == 2:
                fig.add_trace(
                    go.Scatter(y=bleu_scores[:, 0], mode='lines',
                              name='nepali BLEU', line=dict(color='green', width=2)),
                    row=1, col=3
                )
                fig.add_trace(
                    go.Scatter(y=bleu_scores[:, 1], mode='lines',
                              name='English BLEU', line=dict(color='orange', width=2)),
                    row=1, col=3
                )

        # 4. Cloud usage pattern
        if self.history['cloud_usage']:
            cloud_usage = np.array(self.history['cloud_usage'])
            cumulative_cloud = np.cumsum(cloud_usage) / np.arange(1, len(cloud_usage) + 1)
            fig.add_trace(
                go.Scatter(y=cumulative_cloud, mode='lines',
                          name='Cloud Usage Rate', line=dict(color='purple', width=2)),
                row=2, col=1
            )

        # 5. Complexity levels
        if self.history['complexity']:
            complexity_map = {'low': 0, 'medium': 1, 'high': 2}
            complexity_numeric = [complexity_map.get(c, 1) for c in self.history['complexity']]
            fig.add_trace(
                go.Scatter(y=complexity_numeric, mode='lines+markers',
                          name='Complexity', line=dict(color='black', width=2)),
                row=2, col=2
            )

        # 6. Action distribution
        if self.history['actions']:
            actions = self.history['actions']
            action_counts = pd.Series(actions).value_counts()
            fig.add_trace(
                go.Bar(x=action_counts.index, y=action_counts.values,
                      marker_color=[self.colors.get(a, '#95A5A6') for a in action_counts.index]),
                row=2, col=3
            )

        # 7. Reward vs Latency scatter
        fig.add_trace(
            go.Scatter(x=self.history['latencies'], y=self.history['rewards'],
                      mode='markers',
                      marker=dict(size=10, color=self.history['bandwidth'],
                                 colorscale='Viridis', showscale=True,
                                 colorbar=dict(title="Bandwidth")),
                      name='Reward vs Latency'),
            row=3, col=1
        )

        # 8. Cumulative performance
        cumulative_reward = np.cumsum(self.history['rewards'])
        cumulative_latency = np.cumsum(self.history['latencies'])
        fig.add_trace(
            go.Scatter(y=cumulative_reward, mode='lines',
                      name='Cumulative Reward', line=dict(color='blue', width=2)),
            row=3, col=2
        )
        fig.add_trace(
            go.Scatter(y=cumulative_latency, mode='lines',
                      name='Cumulative Latency', line=dict(color='red', width=2)),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Speech Translation System - Interactive Dashboard",
            title_font_size=20
        )

        # Update axes labels
        fig.update_xaxes(title_text="Step", row=1, col=1)
        fig.update_xaxes(title_text="Step", row=1, col=2)
        fig.update_xaxes(title_text="Step", row=1, col=3)
        fig.update_xaxes(title_text="Step", row=2, col=1)
        fig.update_xaxes(title_text="Step", row=2, col=2)
        fig.update_xaxes(title_text="Action", row=2, col=3)
        fig.update_xaxes(title_text="Latency (s)", row=3, col=1)
        fig.update_xaxes(title_text="Step", row=3, col=2)

        fig.update_yaxes(title_text="Reward", row=1, col=1)
        fig.update_yaxes(title_text="Latency (s)", row=1, col=2)
        fig.update_yaxes(title_text="BLEU Score", row=1, col=3)
        fig.update_yaxes(title_text="Usage Rate", row=2, col=1)
        fig.update_yaxes(title_text="Complexity Level", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=3)
        fig.update_yaxes(title_text="Reward", row=3, col=1)
        fig.update_yaxes(title_text="Cumulative Value", row=3, col=2)

        if SHOW_INTERACTIVE:
            fig.show()

        # Save as HTML
        if SAVE_PLOTS:
            fig.write_html("interactive_dashboard.html")

    def plot_system_architecture(self, save_path=None):
        """Visualize the system architecture"""
        fig, ax = plt.subplots(figsize=(14, 8))

        # Define positions
        positions = {
            'Audio Input': (0.1, 0.5),
            'Bandwidth Sensor': (0.3, 0.7),
            'Complexity Estimator': (0.3, 0.3),
            'RL Agent': (0.5, 0.5),
            'Decision Engine': (0.7, 0.5),
            'Cloud Models': (0.9, 0.7),
            'Edge Models': (0.9, 0.3),
            'Output': (1.1, 0.5)
        }

        # Draw boxes
        box_props = dict(boxstyle="round,pad=0.3", facecolor="lightblue", edgecolor="black", alpha=0.8)

        for name, (x, y) in positions.items():
            ax.text(x, y, name, transform=ax.transAxes,
                   bbox=box_props, ha='center', va='center',
                   fontsize=10, fontweight='bold')

        # Draw arrows
        arrows = [
            ('Audio Input', 'Bandwidth Sensor'),
            ('Audio Input', 'Complexity Estimator'),
            ('Bandwidth Sensor', 'RL Agent'),
            ('Complexity Estimator', 'RL Agent'),
            ('RL Agent', 'Decision Engine'),
            ('Decision Engine', 'Cloud Models'),
            ('Decision Engine', 'Edge Models'),
            ('Cloud Models', 'Output'),
            ('Edge Models', 'Output')
        ]

        for start, end in arrows:
            x1, y1 = positions[start]
            x2, y2 = positions[end]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7),
                       transform=ax.transAxes)

        # Add feedback loop
        ax.annotate('Feedback', xy=(0.85, 0.55), xytext=(1.05, 0.6),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2, linestyle='--', alpha=0.7),
                   transform=ax.transAxes,
                   ha='center', va='center', fontsize=9, color='green')

        ax.set_xlim(0, 1.2)
        ax.set_ylim(0, 1)
        ax.set_title('Adaptive Speech Translation System Architecture', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}_architecture.{PLOT_FORMAT}', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_summary_statistics(self, agent, cloud_ratio, save_path=None):
        """Plot summary statistics at the end"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('System Summary Statistics', fontsize=16, fontweight='bold')

        # 1. Overall performance pie chart
        if len(self.history['cloud_usage']) > 0:
            cloud_count = sum(self.history['cloud_usage'])
            edge_count = len(self.history['cloud_usage']) - cloud_count
            sizes = [cloud_count, edge_count]
            labels = ['Cloud', 'Edge']
            colors = [self.colors['cloud'], self.colors['edge']]

            axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                          startangle=90, shadow=True)
            axes[0, 0].set_title('Model Deployment Distribution')

        # 2. Reward distribution histogram
        if self.history['rewards']:
            axes[0, 1].hist(self.history['rewards'], bins=20, edgecolor='black', alpha=0.7)
            axes[0, 1].axvline(np.mean(self.history['rewards']), color='red', linestyle='--',
                              label=f'Mean: {np.mean(self.history["rewards"]):.3f}')
            axes[0, 1].set_title('Reward Distribution')
            axes[0, 1].set_xlabel('Reward')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Learning curve (moving average)
        if self.history['rewards']:
            window_size = min(10, len(self.history['rewards']) // 5)
            if window_size > 1:
                rewards_ma = np.convolve(self.history['rewards'],
                                        np.ones(window_size)/window_size,
                                        mode='valid')
                axes[1, 0].plot(range(len(rewards_ma)), rewards_ma,
                               'g-', linewidth=2, label=f'{window_size}-step MA')
                axes[1, 0].set_title('Learning Curve (Moving Average)')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Reward (MA)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

        # 4. Q-table statistics
        if agent and agent.q_table:
            q_values = list(agent.q_table.values())
            axes[1, 1].hist(q_values, bins=15, edgecolor='black', alpha=0.7, color='purple')
            axes[1, 1].axvline(np.mean(q_values), color='red', linestyle='--',
                              label=f'Mean Q: {np.mean(q_values):.3f}')
            axes[1, 1].set_title('Q-Value Distribution')
            axes[1, 1].set_xlabel('Q-Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}_summary.{PLOT_FORMAT}', dpi=300, bbox_inches='tight')
        plt.show()

# Initialize visualizer
visualizer = SystemVisualizer()

@contextmanager
def torch_no_grad():
    """Context manager for inference with memory cleanup"""
    with torch.no_grad():
        try:
            yield
        finally:
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

print(f"Loading models on {DEVICE}...")

try:
    # Load CLOUD models (high accuracy, larger)
    print("Loading CLOUD models (high accuracy)...")

    # Cloud models for both languages (larger Whisper models)
    asr_cloud_en_proc = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
    asr_cloud_en = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-medium.en"
    ).to(DEVICE).eval()

    asr_cloud_ne_proc = WhisperProcessor.from_pretrained("kpriyanshu256/whisper-medium-ne-NP-20-16-1e-05-pretrain-hi")
    asr_cloud_ne = WhisperForConditionalGeneration.from_pretrained(
        "kpriyanshu256/whisper-medium-ne-NP-20-16-1e-05-pretrain-hi"
    ).to(DEVICE).eval()

    # Load EDGE models (lightweight, efficient)
    print("Loading EDGE models (lightweight)...")

    # Best choice: Whisper Tiny models
    asr_edge_ne_proc = WhisperProcessor.from_pretrained("kpriyanshu256/whisper-medium-ne-NP-20-16-1e-05-pretrain-hi")
    asr_edge_ne = WhisperForConditionalGeneration.from_pretrained(
        "kpriyanshu256/whisper-medium-ne-NP-20-16-1e-05-pretrain-hi"
    ).to(DEVICE).eval()

    # asr_edge_ne_proc = WhisperProcessor.from_pretrained("carlot/whisper-tiny-ne")
    # asr_edge_ne = WhisperForConditionalGeneration.from_pretrained(
    #     "carlot/whisper-tiny-ne"
    # ).to(DEVICE).eval()

    # Even better for English: English-only Whisper Tiny
    asr_edge_en_proc = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    asr_edge_en = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny.en"
    ).to(DEVICE).eval()

    print("Loading translation models...")
    ne_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ne-en")
    ne_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ne-en").to(DEVICE).eval()

    en_ne_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ne")
    en_ne_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ne").to(DEVICE).eval()

    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    # Fallback to CPU if CUDA fails
    DEVICE = "cpu"
    print(f"Falling back to {DEVICE}")

def decode_audio_bytes(audio_bytes, target_sr=16000):
    """Decode audio bytes to numpy array"""
    try:
        if isinstance(audio_bytes, dict):
            # Already decoded
            if 'array' in audio_bytes and 'sampling_rate' in audio_bytes:
                array = audio_bytes['array']
                sr = audio_bytes['sampling_rate']
                if sr != target_sr:
                    # Simple resampling by truncation/duplication (for demo)
                    ratio = target_sr / sr
                    new_length = int(len(array) * ratio)
                    indices = np.linspace(0, len(array) - 1, new_length).astype(int)
                    array = array[indices]
                return array
        elif isinstance(audio_bytes, bytes):
            # Decode bytes
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            if sr != target_sr:
                ratio = target_sr / sr
                new_length = int(len(audio_data) * ratio)
                indices = np.linspace(0, len(audio_data) - 1, new_length).astype(int)
                audio_data = audio_data[indices]
            return audio_data
        elif isinstance(audio_bytes, np.ndarray):
            return audio_bytes
    except Exception as e:
        print(f"Audio decoding error: {e}")
    return None

def preprocess_nepali_text(text):
    """Preprocess nepali text for BLEU calculation"""
    if not text:
        return ""

    # Remove extra spaces and normalize
    text = re.sub(r'\s+', ' ', text.strip())

    # Tokenize using jieba
    tokens = list(jieba.cut(text, cut_all=False))

    # Join with spaces for BLEU
    return ' '.join(tokens)

def preprocess_english_text(text):
    """Preprocess English text for BLEU calculation"""
    if not text:
        return ""

    # Lowercase and remove extra spaces
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)

    # Remove punctuation for better matching (optional)
    text = re.sub(r'[^\w\s]', ' ', text)

    return text

def calculate_sacrebleu_score(prediction, reference, language="en"):
    """Calculate BLEU score using sacrebleu"""
    if not prediction or not reference:
        return 0.0

    try:
        # sacrebleu expects references as a list of strings
        # For single reference, wrap it in a list
        references = [reference]

        # Calculate BLEU score
        bleu_score = sacrebleu.sentence_bleu(
            prediction,
            references,
            smooth_method='exp',  # Exponential smoothing
            smooth_value=0.0,
            lowercase=False,  # Don't lowercase - sacrebleu handles tokenization
            use_effective_order=True
        )

        # Convert to 0-1 scale (sacrebleu returns percentage)
        return bleu_score.score / 100.0

    except Exception as e:
        print(f"  SacreBLEU calculation error ({language}): {e}")
        return 0.0

def calculate_bleu_score(prediction, reference, language="en"):
    """Calculate BLEU score - wrapper for sacrebleu"""
    return calculate_sacrebleu_score(prediction, reference, language)

# def calculate_chrf_score(prediction, reference):
#     """Calculate ChrF score as a more robust alternative"""
#     if not prediction or not reference:
#         return 0.0

#     try:
#         # Using sacrebleu for ChrF as well (more reliable)
#         chrf_score = sacrebleu.sentence_chrf(
#             prediction,
#             [reference],  # References as list
#             beta=2.0,  # chrF2
#             order=6,   # Default order
#             remove_whitespace=True
#         )

#         # Convert to 0-1 scale
#         return chrf_score.score / 100.0

#     except Exception as e:
#         print(f"  ChrF calculation error: {e}")
#         try:
#             # Fallback to evaluate library if sacrebleu fails
#             result = chrf_metric.compute(
#                 predictions=[prediction],
#                 references=[[reference]],
#                 word_order=1  # chrF2
#             )
#             return result["score"] / 100.0  # Convert from percentage to 0-1 scale
#         except:
#             return 0.0

def calculate_chrf_score(prediction, reference):
    """Calculate ChrF score using sacrebleu"""
    if not prediction or not reference:
        return 0.0

    try:
        # Using sacrebleu for ChrF with correct parameters
        chrf_score = sacrebleu.sentence_chrf(
            hypothesis=prediction,
            references=[reference],  # References as list
            beta=2.0,  # chrF2
            # order parameter is not supported in sentence_chrf
            # order=6,   # Remove this line
            # Instead, use char_order and word_order if needed
            char_order=6,  # Character n-gram order
            word_order=0,  # Word n-gram order (0 means chrF, not chrF++)
        )

        # Convert to 0-1 scale (sacrebleu returns percentage)
        return chrf_score.score / 100.0

    except Exception as e:
        print(f"  ChrF calculation error: {e}")
        try:
            # Alternative: Use corpus_chrf if sentence_chrf fails
            chrf_score = sacrebleu.corpus_chrf(
                hypotheses=[prediction],
                references=[[reference]],
                beta=2.0,
            )
            return chrf_score.score / 100.0
        except Exception as e2:
            print(f"  Fallback ChrF also failed: {e2}")
            try:
                # Final fallback to evaluate library
                result = chrf_metric.compute(
                    predictions=[prediction],
                    references=[[reference]],
                    word_order=2  # chrF2
                )
                return result["score"] / 100.0  # Convert from percentage to 0-1 scale
            except:
                return 0.0

FILLERS = ["um", "ah", "like", "uh", "hmm", "erm"]
FILLER_PATTERN = re.compile(r"\b(" + "|".join(FILLERS) + r")\b", re.IGNORECASE)

def remove_disfluencies(text):
    """Remove filler words from text"""
    if not text:
        return ""
    text_clean = FILLER_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text_clean).strip()

def tts_latency(model):
    """Simulate TTS latency based on model location"""
    return 0.1 if model == "edge" else 0.4

def calculate_reward(bleu, latency, disfluency_count=0, feedback_bonus=0.0):
    """Calculate reward with multiple factors"""
    # Normalize BLEU to 0-1 scale, penalize latency
    return bleu - 0.2 * latency - (disfluency_count * 0.01) + feedback_bonus

def simulate_feedback(prediction, reference, source_lang, target_lang):
    """Simulate user feedback based on translation quality"""
    if not reference.strip() or not prediction.strip():
        return 0.0

    try:
        # Use ChrF for translation evaluation (more robust for different languages)
        chrf = calculate_chrf_score(prediction, reference)

        # Provide bonus for good translations, penalty for bad ones
        if chrf > 0.7:
            return 0.1  # Good translation bonus
        elif chrf < 0.3:
            return -0.05  # Poor translation penalty
        else:
            return 0.0  # Neutral for moderate translations
    except:
        return 0.0

class RLAgent:
    """Reinforcement Learning agent for bandwidth/complexity adaptation"""

    def __init__(self, lr=0.1, gamma=0.9, epsilon=0.2):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.action_space = ["increase", "decrease", "maintain"]

    def get_state_key(self, bandwidth, complexity):
        """Create a hashable state key"""
        return f"{bandwidth}_{complexity}"

    def act(self, state_key):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(self.action_space)

        # Get Q-values for all actions in current state
        q_values = [self.q_table.get((state_key, action), 0.0)
                   for action in self.action_space]

        # If all Q-values are 0, return random action
        if all(q == 0.0 for q in q_values):
            return random.choice(self.action_space)

        return self.action_space[np.argmax(q_values)]

    def update(self, state, action, reward, next_state):
        """Update Q-table using Q-learning"""
        current_q = self.q_table.get((state, action), 0.0)

        # Max future Q-value
        future_q = max([self.q_table.get((next_state, a), 0.0)
                       for a in self.action_space])

        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * future_q - current_q)
        self.q_table[(state, action)] = new_q

def load_fleurs_stream(lang_code, split="train", max_samples=MAX_SAMPLES):
    """Stream FLEURS dataset samples with proper audio decoding"""
    try:
        print(f"Loading FLEURS {lang_code} dataset...")

        # Use a smaller subset for testing
        ds_stream = load_dataset(
            "google/fleurs",
            lang_code,
            split=f"{split}[:{max_samples}]",  # Get first N samples
            trust_remote_code=True
        )

        print(f"Loaded {len(ds_stream)} samples")

        for i, sample in enumerate(ds_stream):
            # Debug info
            if i == 0:
                print(f"Sample 0 keys: {list(sample.keys())}")
                print(f"Sample 0 audio type: {type(sample.get('audio', 'No audio'))}")

            # Decode audio
            audio_data = None
            if "audio" in sample and sample["audio"] is not None:
                audio_data = decode_audio_bytes(sample["audio"], TARGET_SR)

            # Get transcription, prefer raw_transcription if available
            transcription = sample.get("raw_transcription", sample.get("transcription", ""))

            # Prepare sample dict
            processed_sample = {
                "audio_array": audio_data,
                "transcription": transcription,
                "raw_transcription": transcription,
                "language": lang_code,
                "id": sample.get("id", f"sample_{i}")
            }

            yield processed_sample

    except Exception as e:
        print(f"Error loading FLEURS stream for {lang_code}: {e}")
        import traceback
        traceback.print_exc()
        # Return empty generator
        return

def google_translate(text, src="ne", dest="en"):
    """Translate using googletrans with retry logic"""
    if not text.strip():
        return ""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = google_translator.translate(text, src=src, dest=dest)
            return result.text
        except Exception as e:
            print(f"  Google Translate attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5)  # Wait before retry
            else:
                print("  Google Translate failed, using local fallback")
                return ""  # Will trigger local fallback

    return ""

def run_system(audio_array, language, bandwidth, complexity):
    """
    Run speech recognition pipeline

    Args:
        audio_array: Audio data as numpy array
        language: "ne" for nepali, "en" for English
        bandwidth: Available bandwidth in kbps
        complexity: Complexity level ("low", "medium", "high")

    Returns:
        tuple: (transcription, model_used, total_latency)
    """
    # Decision logic
    use_cloud = bandwidth >= BTHRESH and complexity != "low"

    # Ensure audio is valid
    if audio_array is None or len(audio_array) == 0:
        print("Warning: Empty audio input")
        return "", "edge", 0.0

    start_time = time.time()

    try:
        if language == "ne":
            if use_cloud:
                # Use Whisper for nepali (cloud model)
                print(f"  Using Whisper (cloud) for nepali, complexity={complexity}, bandwidth={bandwidth}")

                inputs = asr_cloud_ne_proc(
                    audio_array,
                    sampling_rate=TARGET_SR,
                    return_tensors="pt"
                ).to(DEVICE)

                with torch_no_grad():
                    predicted_ids = asr_cloud_ne.generate(**inputs)
                    transcription = asr_cloud_ne_proc.batch_decode(
                        predicted_ids,
                        skip_special_tokens=True
                    )[0]

                model_used = "cloud"
            else:
                # Use Whisper tiny for nepali (Edge model)
                print(f"  Using Whisper (tiny) for nepali, complexity={complexity}, bandwidth={bandwidth}")

                inputs = asr_edge_ne_proc(
                    audio_array,
                    sampling_rate=TARGET_SR,
                    return_tensors="pt"
                ).to(DEVICE)

                with torch_no_grad():
                    predicted_ids = asr_edge_ne.generate(**inputs)
                    transcription = asr_edge_ne_proc.batch_decode(
                        predicted_ids,
                        skip_special_tokens=True
                    )[0]

                model_used = "edge"

        elif language == "en":
            if use_cloud:
                # Use Whisper for English (cloud model)
                print(f"  Using Whisper (cloud) for English, complexity={complexity}, bandwidth={bandwidth}")

                inputs = asr_cloud_en_proc(
                    audio_array,
                    sampling_rate=TARGET_SR,
                    return_tensors="pt"
                ).to(DEVICE)

                with torch_no_grad():
                    predicted_ids = asr_cloud_en.generate(**inputs)
                    transcription = asr_cloud_en_proc.batch_decode(
                        predicted_ids,
                        skip_special_tokens=True
                    )[0]

                model_used = "cloud"
            else:
                # Use Whisper for English (edge model)
                print(f"  Using Whisper (edge) for English, complexity={complexity}, bandwidth={bandwidth}")

                inputs = asr_edge_en_proc(
                    audio_array,
                    sampling_rate=TARGET_SR,
                    return_tensors="pt"
                ).to(DEVICE)

                with torch_no_grad():
                    predicted_ids = asr_edge_en.generate(**inputs)
                    transcription = asr_edge_en_proc.batch_decode(
                        predicted_ids,
                        skip_special_tokens=True
                    )[0]

                model_used = "edge"
        else:
            print(f"  Unsupported language: {language}")
            return "", "edge", 0.0

        asr_latency = time.time() - start_time
        print(f"  ASR transcription: {transcription[:80]}...")

        # Clean transcription
        transcription_clean = remove_disfluencies(transcription)

        # Total latency
        total_latency = asr_latency + tts_latency(model_used)

        print(f"  Latency: ASR={asr_latency:.2f}s, Total={total_latency:.2f}s")

        return transcription_clean, model_used, total_latency

    except Exception as e:
        print(f"Error in run_system: {e}")
        import traceback
        traceback.print_exc()
        return "", "edge", 0.0

def translate_text(text, source_lang, target_lang):
    """Translate text between languages"""
    if not text.strip():
        return ""

    start_time = time.time()

    try:
        if source_lang == "ne" and target_lang == "en":
            inputs = ne_en_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128
            ).to(DEVICE)

            with torch_no_grad():
                outputs = ne_en_model.generate(**inputs)
                translation = ne_en_tokenizer.decode(outputs[0], skip_special_tokens=True)

        elif source_lang == "en" and target_lang == "ne":
            inputs = en_ne_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128
            ).to(DEVICE)

            with torch_no_grad():
                outputs = en_ne_model.generate(**inputs)
                translation = en_ne_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            print(f"  Unsupported translation direction: {source_lang}->{target_lang}")
            translation = ""

        latency = time.time() - start_time
        print(f"  Translation: {translation[:80]}... (took {latency:.2f}s)")

        # Check if translation is correct
        if translation.strip():
            print(f"  ✓ Translation generated successfully")
        else:
            print(f"  ✗ Translation failed or empty")

        return translation

    except Exception as e:
        print(f"Translation error: {e}")
        return ""

def rl_training(ne_stream, en_stream, rounds=1):
    """Train RL agent for adaptive speech translation"""
    agent = RLAgent(lr=0.1, gamma=0.9, epsilon=0.3)
    cloud_usage = []
    metrics = []

    for round_num in range(rounds):
        print(f"\n--- Training Round {round_num + 1}/{rounds} ---")

        # Convert streams to lists for easier iteration
        ne_samples = list(ne_stream)
        en_samples = list(en_stream)

        min_samples = min(len(ne_samples), len(en_samples))
        print(f"Processing {min_samples} sample pairs")

        complexity_idx = 1  # Start with "medium" complexity

        for idx in range(min_samples):
            ne_sample = ne_samples[idx]
            en_sample = en_samples[idx]

            # Skip if no audio
            if ne_sample["audio_array"] is None or en_sample["audio_array"] is None:
                print(f"Skipping sample {idx}: No audio data")
                continue

            # Random bandwidth for this sample
            bandwidth = random.choice(BANDWIDTH_LEVELS)
            current_complexity = COMPLEXITY_LEVELS[complexity_idx]
            state_key = agent.get_state_key(bandwidth, current_complexity)

            # Get action from agent
            action = agent.act(state_key)

            # Update complexity based on action
            old_idx = complexity_idx
            if action == "increase" and complexity_idx < len(COMPLEXITY_LEVELS) - 1:
                complexity_idx += 1
            elif action == "decrease" and complexity_idx > 0:
                complexity_idx -= 1
            # For "maintain", complexity_idx stays the same

            new_complexity = COMPLEXITY_LEVELS[complexity_idx]
            next_state_key = agent.get_state_key(bandwidth, new_complexity)

            # Get references
            ne_reference = ne_sample.get("transcription", "") or ne_sample.get("raw_transcription", "")
            en_reference = en_sample.get("transcription", "") or en_sample.get("raw_transcription", "")

            print(f"\nSample {idx + 1}/{min_samples}:")
            print(f"  Bandwidth: {bandwidth}, Complexity: {old_idx}({current_complexity}) -> {complexity_idx}({new_complexity})")
            print(f"  Action: {action}")
            print(f"  nepali ref: {ne_reference[:80]}...")
            print(f"  English ref: {en_reference[:80]}...")

            # Run system for nepali audio (match with nepali reference)
            print("\n  Processing nepali audio:")
            ne_transcription, ne_model_used, ne_latency = run_system(
                ne_sample["audio_array"], "ne", bandwidth, new_complexity
            )

            # Run system for English audio (match with English reference)
            print("\n  Processing English audio:")
            en_transcription, en_model_used, en_latency = run_system(
                en_sample["audio_array"], "en", bandwidth, new_complexity
            )

            # Translate nepali transcription to English for cross-language evaluation
            print("\n  Translating nepali to English:")
            ne_to_en_translation = translate_text(ne_transcription, "ne", "en")

            # Translate English transcription to nepali for cross-language evaluation
            print("\n  Translating English to nepali:")
            en_to_ne_translation = translate_text(en_transcription, "en", "ne")

            # Calculate BLEU scores using sacrebleu
            print("\n  Calculating metrics:")
            ne_bleu = calculate_bleu_score(ne_transcription, ne_reference, language="ne")
            en_bleu = calculate_bleu_score(en_transcription, en_reference, language="en")

            # Also calculate ChrF scores for comparison (using sacrebleu if available)
            ne_chrf = calculate_chrf_score(ne_transcription, ne_reference)
            en_chrf = calculate_chrf_score(en_transcription, en_reference)

            # Count disfluencies
            ne_disfluency_count = len(FILLER_PATTERN.findall(ne_transcription)) if ne_transcription else 0
            en_disfluency_count = len(FILLER_PATTERN.findall(en_transcription)) if en_transcription else 0
            total_disfluency_count = ne_disfluency_count + en_disfluency_count

            # Get feedback bonuses for translations
            ne_to_en_feedback = simulate_feedback(ne_to_en_translation, en_reference, "ne", "en") if ne_to_en_translation else 0.0
            en_to_ne_feedback = simulate_feedback(en_to_ne_translation, ne_reference, "en", "ne") if en_to_ne_translation else 0.0
            total_feedback = ne_to_en_feedback + en_to_ne_feedback

            # Calculate combined scores (use the average of BLEU and ChrF for robustness)
            ne_score = ne_chrf
            en_score = en_chrf
            avg_score = (ne_score + en_score) / 2
            avg_latency = (ne_latency + en_latency) / 2

            # Calculate reward
            reward = calculate_reward(
                avg_score,
                avg_latency,
                total_disfluency_count,
                total_feedback
            )

            # Update agent
            agent.update(state_key, action, reward, next_state_key)

            # Add data to visualizer
            visualizer.add_data_point(
                rewards=reward,
                latencies=avg_latency,
                bleu_scores=[ne_bleu, en_bleu],
                chrf_scores=[ne_chrf, en_chrf],
                cloud_usage=ne_model_used == "cloud" or en_model_used == "cloud",
                bandwidth=bandwidth,
                complexity=new_complexity,
                actions=action,
                states=state_key
            )

            # Track metrics
            cloud_used = ne_model_used == "cloud" or en_model_used == "cloud"
            cloud_usage.append(1 if cloud_used else 0)
            metrics.append({
                'ne_bleu': ne_bleu,
                'en_bleu': en_bleu,
                'ne_chrf': ne_chrf,
                'en_chrf': en_chrf,
                'avg_score': avg_score,
                'ne_latency': ne_latency,
                'en_latency': en_latency,
                'avg_latency': avg_latency,
                'reward': reward,
                'cloud_used': cloud_used
            })

            print(f"  nepali BLEU: {ne_bleu:.4f}, ChrF: {ne_chrf:.4f}")
            print(f"  English BLEU: {en_bleu:.4f}, ChrF: {en_chrf:.4f}")
            print(f"  Average Score: {avg_score:.4f}, Average Latency: {avg_latency:.2f}s")
            print(f"  Reward: {reward:.4f}")
            print(f"  Cloud used: {cloud_used}")

            # Display tokenization debug info
            if ne_bleu < 0.1:
                print(f"  ⚠ nepali text debug:")
                print(f"    Reference: {ne_reference[:100]}...")
                print(f"    Prediction: {ne_transcription[:100]}...")

            if en_bleu < 0.1:
                print(f"  ⚠ English text debug:")
                print(f"    Reference: {en_reference[:100]}...")
                print(f"    Prediction: {en_transcription[:100]}...")

    # Print summary statistics
    if metrics:
        avg_score = np.mean([m['avg_score'] for m in metrics])
        avg_latency = np.mean([m['avg_latency'] for m in metrics])
        avg_reward = np.mean([m['reward'] for m in metrics])
        cloud_ratio = np.mean(cloud_usage) if cloud_usage else 0.0

        print(f"\nSummary Statistics:")
        print(f"  Average Score: {avg_score:.4f}")
        print(f"  Average Latency: {avg_latency:.2f}s")
        print(f"  Average Reward: {avg_reward:.4f}")
        print(f"  Cloud Usage Ratio: {cloud_ratio:.2%}")

        # Language-specific stats
        avg_ne_bleu = np.mean([m['ne_bleu'] for m in metrics])
        avg_en_bleu = np.mean([m['en_bleu'] for m in metrics])
        avg_ne_chrf = np.mean([m['ne_chrf'] for m in metrics])
        avg_en_chrf = np.mean([m['en_chrf'] for m in metrics])

        print(f"  nepali BLEU: {avg_ne_bleu:.4f}, ChrF: {avg_ne_chrf:.4f}")
        print(f"  English BLEU: {avg_en_bleu:.4f}, ChrF: {avg_en_chrf:.4f}")

        # Print sacrebleu information
        print(f"\nSacreBLEU Info:")
        print(f"  Version: {sacrebleu.__version__}")
        print(f"  Tokenizer: {sacrebleu.DEFAULT_TOKENIZER}")

    return agent, np.mean(cloud_usage) if cloud_usage else 0.0

def main():
    """Main execution function"""
    print("=" * 60)
    print("Adaptive Speech Translation System with RL")
    print("Using SacreBLEU for evaluation")
    print(f"Device: {DEVICE}")
    print(f"Max samples: {MAX_SAMPLES}")
    print("=" * 60)

    try:
        # Show system architecture
        print("\nVisualizing system architecture...")
        visualizer.plot_system_architecture("system_architecture")

        # Load datasets
        print("\nLoading datasets...")
        ne_stream = load_fleurs_stream("ne_np", "train")
        en_stream = load_fleurs_stream("en_us", "train")

        # Convert to lists for debugging
        ne_samples = list(ne_stream)
        en_samples = list(en_stream)

        print(f"Loaded {len(ne_samples)} Nepali samples")
        print(f"Loaded {len(en_samples)} English samples")

        # Check first samples
        if ne_samples and ne_samples[0]['audio_array'] is not None:
            print(f"\nFirst nepali sample:")
            print(f"  Audio shape: {ne_samples[0]['audio_array'].shape}")
            print(f"  Transcription: {ne_samples[0]['transcription'][:100] if ne_samples[0]['transcription'] else 'No transcription'}")

        if en_samples and en_samples[0]['audio_array'] is not None:
            print(f"\nFirst English sample:")
            print(f"  Audio shape: {en_samples[0]['audio_array'].shape}")
            print(f"  Transcription: {en_samples[0]['transcription'][:100] if en_samples[0]['transcription'] else 'No transcription'}")

        # Train RL agent
        print("\nStarting RL training...")
        agent, cloud_ratio = rl_training(ne_samples, en_samples, rounds=1)

        # Generate visualizations
        print("\n" + "="*60)
        print("Generating Visualizations...")
        print("="*60)

        # 1. Training progress
        print("\n1. Plotting training progress...")
        visualizer.plot_training_progress("training_progress")

        # 2. Decision surface
        print("\n2. Plotting decision surface...")
        visualizer.plot_decision_surface(agent, "decision_surface")

        # 3. Performance comparison
        print("\n3. Plotting performance comparison...")
        visualizer.plot_performance_comparison("performance_comparison")

        # 4. Interactive dashboard (if plotly is available)
        try:
            print("\n4. Creating interactive dashboard...")
            visualizer.plot_interactive_dashboard()
        except ImportError:
            print("  Plotly not available, skipping interactive dashboard")

        # 5. Summary statistics
        print("\n5. Plotting summary statistics...")
        visualizer.plot_summary_statistics(agent, cloud_ratio, "summary_statistics")

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Cloud Usage Ratio: {cloud_ratio:.2%}")
        print(f"Q-table size: {len(agent.q_table)}")
        print("=" * 60)

        # Display learned policies
        if agent.q_table:
            print("\nLearned Policies (top 5):")
            sorted_policies = sorted(agent.q_table.items(), key=lambda x: x[1], reverse=True)[:5]
            for (state_action), q_value in sorted_policies:
                state, action = state_action
                print(f"  State {state}, Action {action}: Q = {q_value:.4f}")
        else:
            print("\nNo policies learned (Q-table is empty)")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError in main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("\nSystem shutdown complete.")

if __name__ == "__main__":
    # Install required packages if not present
    required_packages = ["soundfile", "jieba", "evaluate", "matplotlib", "seaborn", "pandas", "sacrebleu"]
    import subprocess
    import importlib

    for package in required_packages:
        try:
            importlib.import_module(package if package != "evaluate" else "evaluate")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call(["pip", "install", package])

    # Try to install plotly for interactive visualizations
    try:
        import plotly
    except ImportError:
        try:
            print("Installing plotly for interactive visualizations...")
            subprocess.check_call(["pip", "install", "plotly"])
            SHOW_INTERACTIVE = True
        except:
            print("Plotly installation failed, using static plots only")
            SHOW_INTERACTIVE = False

    # Now import soundfile
    import soundfile as sf

    main()
