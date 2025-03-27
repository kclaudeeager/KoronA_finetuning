import torch
import torch.nn as nn
import torch.nn.functional as F


class KronALinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        a1: int,
        a2: int,
        scale: float = 1.0,
        dropout: float = 0.0
    ):
        """
        Kronecker Adapter for linear layers, inspired by LoRA.
        
        Args:
            in_dim: Input dimension of the original linear layer.
            out_dim: Output dimension of the original linear layer.
            a1: First dimension of the Kronecker factor A_k.
            a2: Second dimension of the Kronecker factor A_k.
            scale: Scaling factor for the Kronecker product.
            dropout: Dropout probability for the input.
        """
        super().__init__()
        assert a1 * a2 == in_dim, f"a1 * a2 must equal in_dim: {a1} * {a2} != {in_dim}"
        b1 = a2
        b2 = a1
        assert b1 * b2 == out_dim, f"b1 * b2 must equal out_dim: {b1} * {b2} != {out_dim}"
        
        # Original frozen linear layer
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.linear.weight.requires_grad = False  # Freeze original weights
        
        # Kronecker factors
        self.A_k = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(a1, a2), a=5**0.5))
        self.B_k = nn.Parameter(torch.zeros(b1, b2))
        
        # Scaling factor
        self.scale = scale
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for KronALinear.
        
        Args:
            x: Input tensor of shape [batch_size, in_dim].
        
        Returns:
            Output tensor of shape [batch_size, out_dim].
        """
        # Original frozen output
        frozen_out = self.linear(x)
        
        # Apply dropout to the input
        x = self.dropout(x)
        
        # Reshape input for Kronecker computation
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, -1)  # Flatten inputs
        
        # Compute Kronecker product output
        intermediate = x_reshaped.view(batch_size, self.B_k.size(0), self.B_k.size(1))
        intermediate = torch.bmm(intermediate, self.B_k.unsqueeze(0).expand(batch_size, -1, -1))
        krona_out = torch.bmm(intermediate, self.A_k.t().unsqueeze(0).expand(batch_size, -1, -1))
        krona_out = krona_out.view(x.size())  # Reshape back to original dimensions
        
        # Scale the Kronecker output
        krona_out = self.scale * krona_out
        
        # Add the Kronecker output to the frozen output
        return frozen_out + krona_out

class KronAB(nn.Module):
    """
    Kronecker Adapter applied in parallel to FFN blocks (similar to LoRA).
    """
    def __init__(self, hidden_dim, a1=48, a2=32, scale=16.0, dropout=0.0, num_biases=1):
        super().__init__()
        # Ensure dimensions match
        assert a1 * a2 == hidden_dim, f"a1 * a2 must equal hidden_dim: {a1}*{a2}!={hidden_dim}"
        
        # Original frozen linear layer
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear.weight.requires_grad = False  # Freeze original weights
        
        # Initialize Kronecker factors
        self.A_k = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(a1, a2), a=5**0.5))
        self.B_k = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(a2, a1), a=5**0.5))
        
        # Store dimensions for debugging
        self.a1 = a1
        self.a2 = a2
        
        # Scaling factor
        self.scale = scale
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        
        # Optional biases
        self.num_biases = num_biases
        if num_biases > 0:
            self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
            if num_biases > 1:
                self.bias2 = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        """
        Args:
            x: input tensor [batch_size, seq_len, hidden_dim]
        Returns:
            Output tensor with Kronecker adaptation applied.
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        # Original frozen output
        frozen_out = self.linear(x.view(-1, hidden_dim)).view(batch_size, seq_len, hidden_dim)
        
        # Apply dropout to the input
        x = self.dropout(x)
        
        # Flatten input
        x_flat = x.view(-1, hidden_dim)
        
        # Manual Kronecker product-like computation
        try:
            # Reshape input to [batch_size * seq_len, a2, a1]
            x_reshaped = x_flat.view(-1, self.a2, self.a1)
            
            # Perform matrix multiplications manually
            # First, multiply by B_k along the second dimension
            intermediate = torch.matmul(x_reshaped, self.B_k.t())
            
            # Then, multiply by A_k along the second dimension
            krona_out = torch.matmul(intermediate, self.A_k.t())
            
            # Reshape back to original dimensions
            krona_out = krona_out.view(batch_size, seq_len, hidden_dim)
            
            # Add biases if specified
            if self.num_biases > 0:
                krona_out = krona_out + self.bias1
                if self.num_biases > 1:
                    krona_out = krona_out + self.bias2
            
            # Scale the Kronecker output
            krona_out = self.scale * krona_out
            
            # Add the Kronecker output to the frozen output
            return frozen_out + krona_out
        
        except Exception as e:
            print(f"Error in Kronecker computation: {e}")
            print(f"Input shapes - x_flat: {x_flat.shape}, B_k: {self.B_k.shape}, A_k: {self.A_k.shape}")
            print(f"Reshaped input: {x_reshaped.shape}")
            raise

class KronABres(nn.Module):
    """
    KronAB with learnable residual connection.
    """
    def __init__(self, hidden_dim, a1=32, a2=24, scale=16.0, dropout=0.0, num_biases=1):
        super().__init__()
        # Base KronAB module
        self.kronab = KronAB(hidden_dim, a1, a2, scale, dropout, num_biases)
        
        # Learnable residual scaling factor
        self.res_scale = nn.Parameter(torch.ones(1))
        self.a1=a1
        self.a2=a2

    def forward(self, x):
        """
        Args:
            x: input tensor [batch_size, seq_len, hidden_dim]
        Returns:
            Output tensor with Kronecker adaptation and residual connection.
        """
        # Apply KronAB
        kronab_output = self.kronab(x)
        
        # Add scaled residual connection
        return kronab_output + self.res_scale * x