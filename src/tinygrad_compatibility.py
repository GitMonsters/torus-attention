#!/usr/bin/env python3
"""
TinyGrad Compatibility Layer for Enhanced Multi-PINNACLE System
==============================================================

Implements missing PyTorch components for TinyGrad compatibility.
"""

from tinygrad.tensor import Tensor
from tinygrad import nn
import math
from typing import List, Optional, Callable, Any

class Sequential:
    """TinyGrad-compatible Sequential layer implementation"""
    
    def __init__(self, *layers):
        self.layers = list(layers)
    
    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        """Get all parameters from all layers"""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params


class MultiheadAttention:
    """TinyGrad-compatible Multi-head Attention implementation"""
    
    def __init__(self, embed_dim: int, num_heads: int, batch_first: bool = True):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Create linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def __call__(self, query: Tensor, key: Optional[Tensor] = None, value: Optional[Tensor] = None) -> Tensor:
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size = query.shape[0] if self.batch_first else query.shape[1]
        seq_len = query.shape[1] if self.batch_first else query.shape[0]
        
        # Project to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)  
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        if self.batch_first:
            Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            Q = Q.reshape(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 2)
            K = K.reshape(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 2)
            V = V.reshape(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (Q @ K.transpose(-2, -1)) * scale
        attn_weights = scores.softmax(axis=-1)
        attn_output = attn_weights @ V
        
        # Reshape back
        if self.batch_first:
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        else:
            attn_output = attn_output.transpose(0, 2).reshape(seq_len, batch_size, self.embed_dim)
        
        return self.out_proj(attn_output)
    
    def parameters(self):
        """Get all parameters"""
        params = []
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            params.extend(layer.parameters())
        return params


class LSTM:
    """TinyGrad-compatible LSTM implementation using LSTMCell"""
    
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool = True, num_layers: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        
        # Create LSTM cells for each layer
        self.cells = []
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(nn.LSTMCell(cell_input_size, hidden_size))
    
    def __call__(self, x: Tensor, hidden: Optional[tuple] = None) -> tuple:
        if not self.batch_first:
            x = x.transpose(0, 1)  # Convert to batch_first
        
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden states if not provided
        if hidden is None:
            h = [Tensor.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
            c = [Tensor.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h, c = hidden
        
        outputs = []
        
        # Process each timestep
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Process through each layer
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](x_t, (h[layer], c[layer]))
                x_t = h[layer]  # Input to next layer
            
            outputs.append(h[-1])  # Output from last layer
        
        # Manually stack outputs for TinyGrad
        if len(outputs) > 0:
            # Reshape each output to (batch, 1, hidden_size)
            reshaped_outputs = [o.reshape(batch_size, 1, self.hidden_size) for o in outputs]
            # Use manual concatenation
            output = reshaped_outputs[0]
            for i in range(1, len(reshaped_outputs)):
                output = output.cat(reshaped_outputs[i], dim=1)
        else:
            output = Tensor.zeros(batch_size, 0, self.hidden_size)
        
        if not self.batch_first:
            output = output.transpose(0, 1)  # Convert back if needed
        
        return output, (h, c)
    
    def parameters(self):
        """Get all parameters"""
        params = []
        for cell in self.cells:
            params.extend(cell.parameters())
        return params


class GRU:
    """TinyGrad-compatible GRU implementation (simplified)"""
    
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        
        # GRU gates
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.new_gate = nn.Linear(input_size + hidden_size, hidden_size)
    
    def __call__(self, x: Tensor, hidden: Optional[Tensor] = None) -> tuple:
        if not self.batch_first:
            x = x.transpose(0, 1)  # Convert to batch_first
        
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state if not provided
        if hidden is None:
            h = Tensor.zeros(batch_size, self.hidden_size)
        else:
            h = hidden
        
        outputs = []
        
        # Process each timestep
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Concatenate input and hidden state
            combined = x_t.cat(h, dim=1)
            
            # Compute gates
            r = self.reset_gate(combined).sigmoid()
            z = self.update_gate(combined).sigmoid()
            
            # Reset gate applied to hidden state
            reset_h = r * h
            combined_reset = x_t.cat(reset_h, dim=1)
            n = self.new_gate(combined_reset).tanh()
            
            # Update hidden state
            h = (1 - z) * n + z * h
            outputs.append(h)
        
        # Manually stack outputs for TinyGrad
        if len(outputs) > 0:
            # Reshape each output to (batch, 1, hidden_size)
            reshaped_outputs = [o.reshape(batch_size, 1, self.hidden_size) for o in outputs]
            # Use manual concatenation
            output = reshaped_outputs[0]
            for i in range(1, len(reshaped_outputs)):
                output = output.cat(reshaped_outputs[i], dim=1)
        else:
            output = Tensor.zeros(batch_size, 0, self.hidden_size)
        
        if not self.batch_first:
            output = output.transpose(0, 1)  # Convert back if needed
        
        return output, h
    
    def parameters(self):
        """Get all parameters"""
        params = []
        for layer in [self.reset_gate, self.update_gate, self.new_gate]:
            params.extend(layer.parameters())
        return params


class Sigmoid:
    """Sigmoid activation function"""
    
    def __call__(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh:
    """Tanh activation function"""
    
    def __call__(self, x: Tensor) -> Tensor:
        return x.tanh()


class ReLU:
    """ReLU activation function"""
    
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()


class Dropout:
    """Dropout layer (simplified - always returns input during inference)"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, x: Tensor) -> Tensor:
        # For inference, just return the input
        # In training, would apply dropout
        return x


class GELU:
    """GELU activation function"""
    
    def __call__(self, x: Tensor) -> Tensor:
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return x * 0.5 * (1.0 + (x * (1.0 + 0.044715 * x * x) * (2.0/3.14159265359)**0.5).tanh())


class BaseModel:
    """Base model class to provide register_buffer functionality for TinyGrad"""
    
    def __init__(self):
        self._buffers = {}
    
    def register_buffer(self, name: str, tensor: Tensor):
        """Register a buffer (similar to PyTorch's register_buffer)"""
        self._buffers[name] = tensor
        setattr(self, name, tensor)
    
    def named_buffers(self):
        """Return named buffers"""
        return self._buffers.items()
    
    def parameters(self):
        """Override in child classes"""
        return []


# Monkey-patch missing components into nn
nn.Sequential = Sequential
nn.MultiheadAttention = MultiheadAttention
nn.LSTM = LSTM
nn.GRU = GRU
nn.Sigmoid = Sigmoid
nn.Tanh = Tanh
nn.ReLU = ReLU
nn.Dropout = Dropout
nn.GELU = GELU

print("TinyGrad compatibility layer loaded successfully!")