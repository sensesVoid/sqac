import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import string
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

CHARS = string.printable
VOCAB_SIZE = len(CHARS)
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}

@dataclass
class Stage1Result:
    codes: np.ndarray
    reconstructed: np.ndarray
    reconstruction_error: float
    codebook_utilization: float
    final_loss: float

class CharEmbeddingEncoder(nn.Module):
    def __init__(self, chunk_size: int, latent_dim: int, embed_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(chunk_size * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.argmax(x, dim=-1)
        emb = self.embedding(idx)
        x = emb.view(emb.size(0), -1)
        return self.net(x)

class CharEmbeddingDecoder(nn.Module):
    def __init__(self, chunk_size: int, latent_dim: int, embed_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, chunk_size * embed_dim)
        )
        self.out = nn.Linear(embed_dim, VOCAB_SIZE)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        x = x.view(-1, self.chunk_size, self.embed_dim)
        logits = self.out(x)
        return logits.view(-1, VOCAB_SIZE)

class PQVectorQuantizer(nn.Module):
    """Product Quantization style sub-codebook quantizer with dead code restart."""
    
    def __init__(self, latent_dim: int, num_subspaces: int = 4, subspace_entries: int = 64, beta: float = 0.25):
        super().__init__()
        assert latent_dim % num_subspaces == 0
        self.latent_dim = latent_dim
        self.num_subspaces = num_subspaces
        self.subspace_dim = latent_dim // num_subspaces
        self.subspace_entries = subspace_entries
        self.beta = beta
        
        self.codebooks = nn.ModuleList([
            nn.Embedding(subspace_entries, self.subspace_dim)
            for _ in range(num_subspaces)
        ])
        for emb in self.codebooks:
            nn.init.uniform_(emb.weight, -1.0 / subspace_entries, 1.0 / subspace_entries)
        
        self.register_buffer('usage_counts', torch.zeros(num_subspaces, subspace_entries))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = z.size(0)
        z_reshaped = z.view(B, self.num_subspaces, self.subspace_dim)
        
        all_indices = []
        all_quantized = []
        total_embed_loss = 0.0
        total_commit_loss = 0.0
        
        for i, codebook in enumerate(self.codebooks):
            z_sub = z_reshaped[:, i, :]
            distances = (
                torch.sum(z_sub ** 2, dim=1, keepdim=True)
                + torch.sum(codebook.weight ** 2, dim=1)
                - 2 * torch.matmul(z_sub, codebook.weight.t())
            )
            indices = torch.argmin(distances, dim=1)
            quantized_sub = codebook(indices)
            
            all_indices.append(indices)
            all_quantized.append(quantized_sub)
            
            self.usage_counts[i] += torch.bincount(indices, minlength=self.subspace_entries)
            
            commit_loss = self.beta * torch.mean((quantized_sub.detach() - z_sub) ** 2)
            embed_loss = torch.mean((quantized_sub - z_sub.detach()) ** 2)
            total_commit_loss += commit_loss
            total_embed_loss += embed_loss
        
        quantized = torch.cat(all_quantized, dim=1)
        indices = torch.stack(all_indices, dim=1)
        quantized_st = z + (quantized - z).detach()
        
        return quantized_st, indices, total_embed_loss / self.num_subspaces, total_commit_loss / self.num_subspaces
    
    @torch.no_grad()
    def restart_dead_codes(self, z: torch.Tensor):
        """Reset dead codebook entries to random encoder outputs."""
        B = z.size(0)
        z_reshaped = z.view(B, self.num_subspaces, self.subspace_dim)
        
        for i, codebook in enumerate(self.codebooks):
            z_sub = z_reshaped[:, i, :]
            distances = (
                torch.sum(z_sub ** 2, dim=1, keepdim=True)
                + torch.sum(codebook.weight ** 2, dim=1)
                - 2 * torch.matmul(z_sub, codebook.weight.t())
            )
            indices = torch.argmin(distances, dim=1)
            usage = torch.bincount(indices, minlength=self.subspace_entries)
            
            dead_mask = usage == 0
            if dead_mask.sum() > 0:
                num_dead = dead_mask.sum().item()
                random_indices = torch.randint(0, B, (num_dead,))
                new_vectors = z_sub[random_indices].detach()
                codebook.weight.data[dead_mask] = new_vectors
                print(f"  [Restart] Subspace {i}: reset {num_dead} dead codes")

class VQVAE(nn.Module):
    def __init__(self, chunk_size: int = 8, latent_dim: int = 128, codebook_size: int = 256, 
                 num_subspaces: int = 4, beta: float = 0.25):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.num_subspaces = num_subspaces
        self.encoder = CharEmbeddingEncoder(chunk_size, latent_dim)
        self.quantizer = PQVectorQuantizer(latent_dim, num_subspaces, codebook_size // num_subspaces, beta)
        self.decoder = CharEmbeddingDecoder(chunk_size, latent_dim)

    def forward(self, x: torch.Tensor, restart_dead: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        quantized, indices, embedding_loss, commitment_loss = self.quantizer(z)
        if restart_dead and self.training:
            with torch.no_grad():
                self.quantizer.restart_dead_codes(z.detach())
        recon_logits = self.decoder(quantized)
        return recon_logits, indices, embedding_loss, commitment_loss

    def encode(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            _, indices, _, _ = self.forward(x, restart_dead=False)
        return indices.cpu().numpy()

    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            indices = torch.tensor(codes, dtype=torch.long)
            if indices.ndim == 1:
                # Single subspace: repeat to fill full latent_dim
                quantized_sub = self.quantizer.codebooks[0](indices)
                num_subspaces = self.quantizer.num_subspaces
                quantized = quantized_sub.repeat(1, self.quantizer.num_subspaces)
            else:
                quantized_parts = []
                for i, codebook in enumerate(self.quantizer.codebooks):
                    q = codebook(indices[:, i])
                    quantized_parts.append(q)
                quantized = torch.cat(quantized_parts, dim=1)
            recon_logits = self.decoder(quantized)
            recon = torch.argmax(recon_logits, dim=-1)
        return recon.cpu().numpy()

def text_to_chunks(text: str, chunk_size: int) -> List[np.ndarray]:
    chunks = []
    text = text[:len(text) // chunk_size * chunk_size]
    for i in range(0, len(text), chunk_size):
        chunk = np.zeros((chunk_size, VOCAB_SIZE), dtype=np.float32)
        for j, c in enumerate(text[i:i+chunk_size]):
            if c in CHAR_TO_IDX:
                chunk[j, CHAR_TO_IDX[c]] = 1.0
        chunks.append(chunk)
    return chunks

def chunks_to_text(chunks: np.ndarray, chunk_size: int = None) -> str:
    text = []
    for chunk in chunks:
        if np.isscalar(chunk):
            text.append(IDX_TO_CHAR[int(chunk)])
        elif hasattr(chunk, 'ndim') and chunk.ndim == 1:
            for idx in chunk:
                text.append(IDX_TO_CHAR[int(idx)])
        else:
            for row in chunk:
                idx = np.argmax(row)
                text.append(IDX_TO_CHAR[idx])
    return ''.join(text)

def reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    original_idx = np.argmax(original, axis=-1)
    correct = np.sum(original_idx == reconstructed)
    total = original_idx.size
    return float(correct) / total

def codebook_utilization(indices: np.ndarray, codebook_size: int, num_subspaces: int = 1) -> float:
    used_per_subspace = []
    for i in range(num_subspaces):
        subspace_indices = indices[:, i] if indices.ndim > 1 else indices
        used = len(np.unique(subspace_indices))
        used_per_subspace.append(used / (codebook_size // num_subspaces))
    return np.mean(used_per_subspace)

def train_vqvae(text: str, chunk_size: int = 8, latent_dim: int = 128,
                codebook_size: int = 256, num_subspaces: int = 4, epochs: int = 400,
                batch_size: int = 512, lr: float = 5e-4, device: str = "cpu",
                entropy_weight: float = 0.0, restart_interval: int = 50) -> Tuple[VQVAE, Stage1Result]:
    chunks = text_to_chunks(text, chunk_size)
    if len(chunks) < 1:
        raise ValueError("Not enough text for given chunk_size")
    dataset = torch.tensor(np.stack(chunks), dtype=torch.float32)

    model = VQVAE(chunk_size, latent_dim, codebook_size, num_subspaces).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    recon_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(dataset))
        epoch_recon = 0.0
        epoch_embed = 0.0
        epoch_commit = 0.0
        epoch_entropy = 0.0
        for i in range(0, len(dataset), batch_size):
            batch_idx = perm[i:i+batch_size]
            x = dataset[batch_idx].to(device)
            recon_logits, indices, embedding_loss, commitment_loss = model(x, restart_dead=False)
            recon_loss = recon_loss_fn(recon_logits.view(-1, VOCAB_SIZE), x.argmax(-1).view(-1))
            loss = recon_loss + embedding_loss + commitment_loss
            
            if entropy_weight > 0:
                entropy = 0.0
                for j in range(model.quantizer.num_subspaces):
                    counts = torch.bincount(indices[:, j], minlength=model.quantizer.subspace_entries)
                    probs = counts.float() / counts.sum().clamp(min=1)
                    entropy += -torch.sum(probs * torch.log(probs + 1e-10))
                entropy = entropy / model.quantizer.num_subspaces
                loss = loss - entropy_weight * entropy
                epoch_entropy += entropy.item() * len(batch_idx)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_recon += recon_loss.item() * len(batch_idx)
            epoch_embed += embedding_loss.item() * len(batch_idx)
            epoch_commit += commitment_loss.item() * len(batch_idx)
        
        if (epoch + 1) % restart_interval == 0:
            with torch.no_grad():
                model.eval()
                all_z = []
                for i in range(0, len(dataset), batch_size):
                    x = dataset[i:i+batch_size].to(device)
                    z = model.encoder(x)
                    all_z.append(z)
                all_z = torch.cat(all_z, dim=0)
                model.quantizer.restart_dead_codes(all_z)
                model.train()
        
        scheduler.step()
        epoch_recon /= len(dataset)
        epoch_embed /= len(dataset)
        epoch_commit /= len(dataset)
        epoch_entropy /= len(dataset)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d} | Recon: {epoch_recon:.4f} | Embed: {epoch_embed:.4f} | Commit: {epoch_commit:.4f} | Entropy: {epoch_entropy:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            torch.save(model.state_dict(), '/teamspace/studios/this_studio/Project_ SynthQuant/src/stage1_improved_trained.pt')
            print(f"  [Checkpoint] Saved model at epoch {epoch+1}")

    model.eval()
    with torch.no_grad():
        all_indices = []
        all_recon = []
        for i in range(0, len(dataset), batch_size):
            x = dataset[i:i+batch_size].to(device)
            recon_logits, indices, _, _ = model(x)
            all_indices.append(indices.cpu().numpy())
            recon_chunks = torch.argmax(recon_logits, dim=-1).cpu().numpy()
            recon_chunks = recon_chunks.reshape(-1, chunk_size)
            all_recon.append(recon_chunks)
        all_indices = np.concatenate(all_indices)
        all_recon = np.concatenate(all_recon)
        original_np = dataset.numpy()
        rec_error = reconstruction_error(original_np, all_recon)
        util = codebook_utilization(all_indices, codebook_size, num_subspaces)

    result = Stage1Result(
        codes=all_indices,
        reconstructed=all_recon,
        reconstruction_error=rec_error,
        codebook_utilization=util,
        final_loss=epoch_recon
    )
    
    torch.save(model.state_dict(), '/teamspace/studios/this_studio/Project_ SynthQuant/src/stage1_improved_trained.pt')
    print(f"Saved improved Stage 1 model")
    return model, result

if __name__ == "__main__":
    import glob
    doc_texts = []
    for path in glob.glob("/teamspace/studios/this_studio/Project_ SynthQuant/docs/*.md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            doc_texts.append(f.read())
    text = "\n".join(doc_texts)
    print(f"Loaded {len(text)} characters from docs")
    
    print("\n=== Improved Stage 1: dead code restart + entropy ===")
    model, result = train_vqvae(text, chunk_size=8, latent_dim=128, codebook_size=256, 
                                num_subspaces=4, epochs=300, batch_size=512, lr=5e-4, 
                                entropy_weight=0.02, restart_interval=50)
    print(f"Reconstruction error (char acc): {result.reconstruction_error:.4f}")
    print(f"Codebook utilization: {result.codebook_utilization:.2%}")
    print(f"Final loss: {result.final_loss:.4f}")
    recon_text = chunks_to_text(result.reconstructed[:200])
    print("Sample reconstructed text:")
    print(recon_text[:500])
