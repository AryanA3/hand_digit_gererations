import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(
    page_title="MNIST Digit Generator - VAE",
    page_icon="🔢",
    layout="wide"
)

# Your exact VAE architecture from training
class HighQualityVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(HighQualityVAE, self).__init__()
        
        # Convolutional Encoder with residual connections
        self.encoder_conv = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Second conv block with residual
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 7x7 -> 7x7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 7x7 -> 4x4 (actually 3x3)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Calculate the flattened size after convolutions
        self.conv_output_size = 512 * 3 * 3  # 512 channels * 3 * 3
        
        # Dense layers for encoding
        self.encoder_dense = nn.Sequential(
            nn.Linear(self.conv_output_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Dense decoder
        self.decoder_dense = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, self.conv_output_size),
            nn.ReLU(),
        )
        
        # Convolutional Decoder with upsampling
        self.decoder_conv = nn.Sequential(
            # First deconv block
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 3x3 -> 6x6
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Second deconv block  
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),  # 6x6 -> 6x6
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Third deconv block
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 6x6 -> 12x12
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Fourth deconv block
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 12x12 -> 24x24
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Final layer to get exact size
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1, padding=0),    # 24x24 -> 28x28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Output layer
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),             # 28x28 -> 28x28
            nn.Sigmoid()
        )
        
        # Attention mechanism for better reconstruction
        self.attention = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        # Convolutional encoding
        conv_out = self.encoder_conv(x)
        conv_flat = conv_out.view(conv_out.size(0), -1)
        
        # Dense encoding
        dense_out = self.encoder_dense(conv_flat)
        
        # Latent parameters
        mu = self.fc_mu(dense_out)
        logvar = self.fc_logvar(dense_out)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        # Dense decoding
        dense_out = self.decoder_dense(z)
        
        # Reshape for convolution
        conv_input = dense_out.view(-1, 512, 3, 3)
        
        # Convolutional decoding
        conv_out = self.decoder_conv(conv_input)
        
        # Apply attention for refinement
        attention_weights = self.attention(conv_out)
        refined_output = conv_out * attention_weights + conv_out * (1 - attention_weights)
        
        return refined_output
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

from huggingface_hub import hf_hub_download
import os

@st.cache_resource
def load_model():
    """Download and load the trained VAE model from Hugging Face Hub"""
    model_path = "best_high_quality_vae.pth"
    
    try:
        # Download model from Hugging Face
        hf_model_repo = "your-username/mnist-highquality-vae"  # ✅ Replace with your repo
        hf_filename = "best_high_quality_vae.pth"
        
        # Download (cached automatically by HF)
        cached_model_path = hf_hub_download(
            repo_id=hf_model_repo,
            filename=hf_filename,
            repo_type="model"
        )
        
        # Load model
        model = HighQualityVAE(latent_dim=128)
        checkpoint = torch.load(cached_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def generate_digit_samples(model, target_digit=None, num_samples=5, diversity_factor=1.0):
    """Generate diverse samples using advanced sampling techniques"""
    if model is None:
        return None
    
    model.eval()
    with torch.no_grad():
        samples = []
        
        # Method 1: Standard sampling with controlled variance
        z_std = torch.randn(num_samples//2 + 1, 128) * diversity_factor
        std_samples = model.decode(z_std)
        
        # Method 2: Spherical sampling for better coverage
        z_sphere = torch.randn(num_samples//2 + 1, 128)
        z_sphere = F.normalize(z_sphere, p=2, dim=1) * np.sqrt(128) * 0.8 * diversity_factor
        sphere_samples = model.decode(z_sphere)
        
        # Combine and select best samples
        all_samples = torch.cat([std_samples, sphere_samples], dim=0)
        
        # Convert to numpy
        samples = all_samples[:num_samples].cpu().numpy()
        
        # Post-process for better visibility
        samples = np.clip(samples, 0, 1)
        
        return samples

def create_sample_grid(samples, grid_cols=5):
    """Create a visual grid of samples"""
    if samples is None:
        return None
    
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
    
    if n_samples == 1:
        axes = [axes]
    
    for i, sample in enumerate(samples):
        axes[i].imshow(sample[0], cmap='gray', interpolation='bilinear')
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_interpolation(model, num_steps=5):
    """Create interpolation between two random points in latent space"""
    if model is None:
        return None
    
    model.eval()
    with torch.no_grad():
        # Two random points in latent space
        z1 = torch.randn(1, 128)
        z2 = torch.randn(1, 128)
        
        # Create interpolation
        interpolations = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2
            sample = model.decode(z_interp)
            interpolations.append(sample[0, 0].cpu().numpy())
        
        return interpolations

def main():
    st.title("🔢 MNIST Digit Generator (VAE)")
    st.markdown("Generate high-quality handwritten digits using a Variational Autoencoder!")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Model could not be loaded. Please check if 'best_high_quality_vae.pth' is available.")
        st.markdown("""
        **To use this app:**
        1. Upload your trained model file `best_high_quality_vae.pth`
        2. Refresh the page
        3. Start generating digits!
        """)
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Generation Controls")
    
    diversity = st.sidebar.slider(
        "Diversity Factor",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Higher values create more diverse but potentially less realistic samples"
    )
    
    num_samples = st.sidebar.selectbox(
        "Number of samples",
        options=[5, 8, 10],
        index=0
    )
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Generate Samples")
        
        if st.button("🎲 Generate Random Digits", type="primary", use_container_width=True):
            st.session_state.trigger_generation = True
        
        st.markdown("---")
        
        st.subheader("🔄 Interpolation")
        if st.button("Create Interpolation", use_container_width=True):
            st.session_state.trigger_interpolation = True
    
    with col2:
        st.subheader("📸 Generated Results")
        
        # Generate samples
        if st.session_state.get('trigger_generation', False):
            with st.spinner("🎨 Generating high-quality samples..."):
                samples = generate_digit_samples(
                    model, 
                    num_samples=num_samples, 
                    diversity_factor=diversity
                )
                
                if samples is not None:
                    # Display samples in a grid
                    cols = st.columns(min(num_samples, 5))
                    for i, sample in enumerate(samples):
                        col_idx = i % 5
                        with cols[col_idx]:
                            # Convert to PIL Image for better display
                            img_array = (sample[0] * 255).astype(np.uint8)
                            img = Image.fromarray(img_array, mode='L')
                            st.image(img, caption=f"Sample {i+1}", use_column_width=True)
                    
                    st.success(f"✨ Generated {num_samples} unique digit samples!")
                    
                    # Store in session state
                    st.session_state.last_samples = samples
                else:
                    st.error("Failed to generate samples")
            
            st.session_state.trigger_generation = False
        
        # Generate interpolation
        if st.session_state.get('trigger_interpolation', False):
            with st.spinner("🌈 Creating latent space interpolation..."):
                interpolations = create_interpolation(model, num_steps=5)
                
                if interpolations is not None:
                    st.subheader("🔄 Latent Space Interpolation")
                    cols = st.columns(5)
                    for i, interp in enumerate(interpolations):
                        with cols[i]:
                            img_array = (interp * 255).astype(np.uint8)
                            img = Image.fromarray(img_array, mode='L')
                            st.image(img, caption=f"Step {i+1}", use_column_width=True)
                    
                    st.info("This shows smooth transitions in the learned latent space")
                else:
                    st.error("Failed to create interpolation")
            
            st.session_state.trigger_interpolation = False
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ About this Model", expanded=False):
        st.markdown("""
        ### 🧠 Model Architecture
        This app uses a **High-Quality Variational Autoencoder (VAE)** with:
        
        - **Complex Convolutional Architecture**: Multi-layer encoder/decoder with attention mechanism
        - **Large Latent Space**: 128-dimensional latent representation for detailed generation
        - **Advanced Training**: Perceptual loss, KL annealing, and data augmentation
        - **Quality Focus**: Designed for high-fidelity MNIST digit generation
        
        ### 🎯 Features
        - **Diverse Generation**: Multiple sampling strategies for variety
        - **Latent Interpolation**: Smooth transitions between generated samples
        - **Adjustable Diversity**: Control the creativity vs realism trade-off
        
        ### 🔬 Technical Details
        - Trained on Google Colab with T4 GPU
        - 20 epochs with cosine annealing scheduler
        - Advanced loss function with perceptual components
        - Attention mechanism for improved reconstruction quality
        """)
    
    # Display last generated samples if available
    if 'last_samples' in st.session_state:
        with st.expander("📋 Previous Generation", expanded=False):
            samples = st.session_state.last_samples
            cols = st.columns(min(len(samples), 5))
            for i, sample in enumerate(samples):
                col_idx = i % 5
                with cols[col_idx]:
                    img_array = (sample[0] * 255).astype(np.uint8)
                    img = Image.fromarray(img_array, mode='L')
                    st.image(img, caption=f"Sample {i+1}", use_column_width=True)

if __name__ == "__main__":
    # Initialize session state
    if 'trigger_generation' not in st.session_state:
        st.session_state.trigger_generation = False
    if 'trigger_interpolation' not in st.session_state:
        st.session_state.trigger_interpolation = False
    
    main()
