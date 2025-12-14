import torch
import torch.nn as nn
import math
from conv_net import ConvNet
from torch.distributions import Categorical


import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A basic residual block for MLP with LayerNorm and ReLU.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # Add residual connection
        return x + self.layer(x)

class MLPActionPredictor(nn.Module):
    """
    MLP-based action predictor with residual connections.

    Args:
        input_dim (int): Flattened input dimension (e.g., from positional encoding).
        hidden_dim (int): Hidden layer dimension.
        output_dim (int): Final output dimension (e.g., num_actions * action_dims).
        num_layers (int): Number of residual blocks (default: 6).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6):
        super().__init__()
        
        # First linear projection to hidden space
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim) for _ in range(num_layers)
        ])

        # Final output layer
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)      # Project input
        x = self.res_blocks(x)      # Pass through residual MLP stack
        return self.output_proj(x)  # Project to output


class FactorNetPPO(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=256, num_actions=161, 
                 order_dim=4, scaler_dim=2, mu_dim = 1, use_conv=False, 
                 input_channels=4, conv_out_channels=8):
        super(FactorNetPPO, self).__init__()
        
        self.num_actions = num_actions
        self.order_dim = order_dim
        self.scaler_dim = scaler_dim
        self.mu_dim = mu_dim
        self.action_dims = self.order_dim + self.scaler_dim + self.mu_dim - 1
        self.use_conv = use_conv
        
        # MLP input dim: 2 for the two normalized integers
        mlp_input_dim = 2
        
        # print(self.action_dims)
        # assert 0
        
        if self.use_conv:
            mlp_input_dim += (order_dim - 1)

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions * self.action_dims)
        )
                
        # nn.init.zeros_(self.mlp[-1].bias)
        # nn.init.zeros_(self.mlp[-1].weight)

        print(f"init mlp with input dimension {mlp_input_dim}")

        first_order_values = torch.linspace(0, 1, num_actions)
        second_order_values = torch.linspace(-2, 0, num_actions)
        order_values = torch.linspace(-1, 1, num_actions)
        scaler_values = torch.linspace(-0.05, 0.05, num_actions)
        mu_values = torch.cat((torch.tensor([0.0]), torch.linspace(0.5, 0.99, num_actions - 1)))
        
        action_values = []
        for i in range(self.action_dims):
            if i == 0:
                action_values.append(first_order_values)
            elif i == 1 and i < self.order_dim - 1:
                action_values.append(second_order_values)
            elif i < self.order_dim - 1:
                action_values.append(order_values)
            elif i < self.order_dim + self.scaler_dim - 1:
                action_values.append(scaler_values)
            else:
                action_values.append(mu_values)
        self.register_buffer('action_values', torch.stack(action_values))

    def normalize_input(self, x):
        # Normalize two integers (assumed 0-999) to [0, 1]
        return x.float() 
        # x = x.float()
        # y = x
        # y[:,-1] = y[:,-1] / 10
        # return y

    def compute_cosine_similarity(self, epsilon):
        if epsilon is None:
            return torch.zeros(epsilon.shape[0], self.order_dim - 1, device=epsilon.device)
        
        # epsilon shape: [B, order_dim, ...]
        B = epsilon.shape[0]
        # Flatten each epsilon in the sequence
        epsilon_flat = epsilon.view(B, self.order_dim, -1)  # [B, order_dim, features]
        
        # Get first epsilon as reference
        first_epsilon = epsilon_flat[:, 0:1, :]  # [B, 1, features]
        
        # Compute cosine similarity for each subsequent epsilon
        cos_sims = []
        for i in range(1, self.order_dim):
            subsequent_epsilon = epsilon_flat[:, i, :]  # [B, features]
            first_epsilon_squeezed = first_epsilon.squeeze(1)  # [B, features]
            cos_sim = nn.functional.cosine_similarity(
                subsequent_epsilon, first_epsilon_squeezed, dim=-1
            )  # [B]
            cos_sims.append(cos_sim.unsqueeze(-1))  # [B, 1]
        
        return torch.cat(cos_sims, dim=-1)  # [B, order_dim - 1]

    def forward(self, x_dict, actions=None):
        if actions is None:
            return self.sample_action(x_dict)
        return self.get_action_probs(x_dict, actions)
    
    def forward_(self, x_dict):
        # Extract components from input dictionary
        x = x_dict['x']  # Shape: [B, 2]
        epsilon = x_dict.get('epsilon', None)  # epsilon is optional
        
        # Normalize the input
        normalized_x = self.normalize_input(x)  # Shape: [B, 2]
        
        # Prepare MLP input
        if self.use_conv:
            # Compute cosine similarity for epsilon
            cos_sim = self.compute_cosine_similarity(epsilon)  # [B, order_dim - 1]
            flattened = torch.cat([normalized_x, cos_sim], dim=-1)  # [B, 2 + (order_dim - 1)]
        else:
            flattened = normalized_x  # [B, 2]
        
        logits = self.mlp(flattened)
        logits = logits.view(-1, self.action_dims, self.num_actions)
        
        probs = torch.softmax(logits/0.01, dim=-1) # xxx
        return probs

    def sample_action(self, x_dict):
        probs = self.forward_(x_dict)
        action_idx = torch.multinomial(probs.view(-1, self.num_actions), num_samples=1)
        action_idx = action_idx.view(-1, self.action_dims)
                
        batch_size = x_dict['x'].shape[0]
        action_values_expanded = self.action_values.unsqueeze(0).expand(batch_size, -1, -1)
        sampled_actions = torch.gather(action_values_expanded, 2, action_idx.unsqueeze(-1)).squeeze(-1)
        action_probs = probs.gather(dim=2, index=action_idx.unsqueeze(-1)).squeeze(-1)
        return sampled_actions, action_probs

    def get_action_probs(self, x_dict, actions):
        probs = self.forward_(x_dict)
        actions = actions.to(probs.device)
                
        action_indices = torch.zeros_like(actions, dtype=torch.long)
        for dim in range(self.action_dims):
            values = self.action_values[dim]
            diffs = torch.abs(actions[:, dim].unsqueeze(-1) - values)
            action_indices[:, dim] = diffs.argmin(dim=-1)

        dist = Categorical(probs=probs)
        entropy = dist.entropy()/torch.log(torch.as_tensor(probs.shape[2], dtype=probs.dtype, device=probs.device))

        selected_probs = probs.gather(dim=2, index=action_indices.unsqueeze(-1)).squeeze(-1)
        return selected_probs, entropy



class MuNetPPO(nn.Module):
    def __init__(self, input_dim=1, num_actions=21):
        super().__init__()
        
        self.num_actions = num_actions
        self.action_dim = 1
        
        # Single-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, num_actions)
        )
        
        # Define action values for mu (0 to 0.99)
        self.register_buffer('action_values', torch.linspace(1.0, 2.0, num_actions))

    def normalize_input(self, x):
        # Normalize input to [0, 1]
        return x.float()

    def forward(self, x, actions=None):
        if actions is None:
            return self.sample_action(x)
        return self.get_action_probs(x, actions)

    def forward_(self, x):
        # Normalize input
        normalized_x = self.normalize_input(x)  # Shape: [B, 1]
        
        # Get logits from MLP
        logits = self.mlp(normalized_x)  # Shape: [B, num_actions]
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)  # Shape: [B, num_actions]
        return probs

    def sample_action(self, x):
        probs = self.forward_(x)  # Shape: [B, num_actions]
        
        # Sample action indices
        action_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)  # Shape: [B]
        
        # Get corresponding action values
        batch_size = x.shape[0]
        action_values_expanded = self.action_values.unsqueeze(0).expand(batch_size, -1)  # Shape: [B, num_actions]
        sampled_actions = torch.gather(action_values_expanded, 1, action_idx.unsqueeze(-1)).squeeze(-1)  # Shape: [B]
        
        # Get probabilities for sampled actions
        action_probs = probs.gather(dim=1, index=action_idx.unsqueeze(-1)).squeeze(-1)  # Shape: [B]
        
        return sampled_actions, action_probs

    def get_action_probs(self, x, actions):
        probs = self.forward_(x)  # Shape: [B, num_actions]
        actions = actions.to(probs.device)
        
        # Convert actions to indices
        diffs = torch.abs(actions.unsqueeze(-1) - self.action_values)  # Shape: [B, num_actions]
        action_indices = diffs.argmin(dim=-1)  # Shape: [B]
        
        # Get distribution and entropy
        dist = Categorical(probs=probs)
        entropy = dist.entropy() / torch.log(torch.as_tensor(self.num_actions, dtype=probs.dtype, device=probs.device))
        
        # Get probabilities for specified actions
        selected_probs = probs.gather(dim=1, index=action_indices.unsqueeze(-1)).squeeze(-1)  # Shape: [B]
        
        return selected_probs, entropy

    
if __name__ == "__main__":
    # Test with conv
    net_with_conv = FactorNetPPO(use_conv=True, input_channels=1)
    batch_size = 4
    x_dict = {
        'x': torch.randn(batch_size, 2) * 1000,
        'epsilon': torch.randn(batch_size, 4, 1, 32, 32)  # [B, order_dim, C, H, W]
    }
    
    probs = net_with_conv(x_dict)
    actions, action_probs = net_with_conv.sample_action(x_dict)
    
    print("With conv:")
    print("Probability distribution shape:", probs.shape)
    print("Sampled actions shape:", actions.shape)
    print("Sampled actions:", actions)
    print("Action probabilities shape:", action_probs.shape)
    
    # Test without conv
    net_without_conv = FactorNetPPO(use_conv=False)
    x_dict_no_eps = {
        'x': torch.randn(batch_size, 2) * 1000
    }
    probs = net_without_conv(x_dict_no_eps)
    actions, action_probs = net_without_conv.sample_action(x_dict_no_eps)
    
    print("\nWithout conv:")
    print("Probability distribution shape:", probs.shape)
    print("Sampled actions shape:", actions.shape)