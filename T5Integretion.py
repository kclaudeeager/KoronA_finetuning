from transformers import T5ForConditionalGeneration, T5Config
from KornA import KronALinear, KronAB, KronABres
import torch.nn as nn

class KronAForT5(nn.Module):
    """
    Apply KronA to T5 model's query and value matrices, as recommended in the paper
    """
    def __init__(self, model_name="t5-base", scale=1.0):
        super().__init__()
        # Load the pretrained T5 model
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Add KronA adapters to query and value matrices in all attention layers
        self.kronas = nn.ModuleDict()
        
        for encoder_idx in range(len(self.model.encoder.block)):
            # Encoder self-attention query
            q_dim = self.model.encoder.block[encoder_idx].layer[0].SelfAttention.q.weight.shape
            self.kronas[f"encoder.{encoder_idx}.q"] = KronALinear(q_dim[1], q_dim[0],32,24, scale=scale)
            
            # Encoder self-attention value
            v_dim = self.model.encoder.block[encoder_idx].layer[0].SelfAttention.v.weight.shape
            self.kronas[f"encoder.{encoder_idx}.v"] = KronALinear(v_dim[1], v_dim[0],32,24, scale=scale)
        
        for decoder_idx in range(len(self.model.decoder.block)):
            # Decoder self-attention query
            q_dim = self.model.decoder.block[decoder_idx].layer[0].SelfAttention.q.weight.shape
            self.kronas[f"decoder.{decoder_idx}.self.q"] = KronALinear(q_dim[1], q_dim[0], 32,24,scale=scale)
            
            # Decoder self-attention value
            v_dim = self.model.decoder.block[decoder_idx].layer[0].SelfAttention.v.weight.shape
            self.kronas[f"decoder.{decoder_idx}.self.v"] = KronALinear(v_dim[1], v_dim[0],32,24, scale=scale)
            
            # Decoder cross-attention query
            q_dim = self.model.decoder.block[decoder_idx].layer[1].EncDecAttention.q.weight.shape
            self.kronas[f"decoder.{decoder_idx}.cross.q"] = KronALinear(q_dim[1], q_dim[0], 32,24,scale=scale)
            
            # Decoder cross-attention value
            v_dim = self.model.decoder.block[decoder_idx].layer[1].EncDecAttention.v.weight.shape
            self.kronas[f"decoder.{decoder_idx}.cross.v"] = KronALinear(v_dim[1], v_dim[0], 32,24,scale=scale)
        
    def forward(self, **kwargs):
        # This hook is called during forward pass to modify the weights
        def apply_krona_weights_hook(module, input):
            # Apply KronA weights by extracting module name from the hook
            if hasattr(module, "_krona_name") and module._krona_name in self.kronas:
                krona = self.kronas[module._krona_name]
                weight = module.weight
                # Create modified input with KronA contribution
                modified_input = input[0] + krona(input[0], weight)
                return (modified_input,) + input[1:]
            return input
        
        # Register forward pre hooks for all relevant modules
        hooks = []
        for name, krona in self.kronas.items():
            parts = name.split('.')
            if parts[0] == "encoder":
                module = self.model.encoder.block[int(parts[1])].layer[0].SelfAttention
                if parts[2] == "q":
                    module = module.q
                elif parts[2] == "v":
                    module = module.v
            else:  # decoder
                module = self.model.decoder.block[int(parts[1])].layer
                if parts[2] == "self":
                    module = module[0].SelfAttention
                else:  # cross
                    module = module[1].EncDecAttention
                
                if parts[3] == "q":
                    module = module.q
                elif parts[3] == "v":
                    module = module.v
            
            module._krona_name = name
            hooks.append(module.register_forward_pre_hook(apply_krona_weights_hook))
        
        # Forward pass through model
        outputs = self.model(**kwargs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return outputs
    
    def merge_weights(self):
        """
        Merge KronA weights with original model weights for efficient inference
        """
        for name, krona in self.kronas.items():
            parts = name.split('.')
            if parts[0] == "encoder":
                module = self.model.encoder.block[int(parts[1])].layer[0].SelfAttention
                if parts[2] == "q":
                    module = module.q
                elif parts[2] == "v":
                    module = module.v
            else:  # decoder
                module = self.model.decoder.block[int(parts[1])].layer
                if parts[2] == "self":
                    module = module[0].SelfAttention
                else:  # cross
                    module = module[1].EncDecAttention
                
                if parts[3] == "q":
                    module = module.q
                elif parts[3] == "v":
                    module = module.v
            
            # Merge weights
            delta = krona.get_delta_weight()
            with torch.no_grad():
                module.weight.add_(delta)


# For KronAB and KronABres, let's integrate with T5's FFN blocks
class KronABT5(nn.Module):
    """
    Apply KronAB to T5 model's FFN blocks
    """
    def __init__(self, model_name="t5-base", scale=16.0, use_residual=False, num_biases=1):
        super().__init__()
        # Load the pretrained T5 model
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Add KronAB adapters to FFN modules
        self.kronas = nn.ModuleDict()
        
        # Get hidden dimension from the model
        hidden_dim = self.model.config.d_model
        
        # Create adapters for each FFN block
        for encoder_idx in range(len(self.model.encoder.block)):
            if use_residual:
                self.kronas[f"encoder.{encoder_idx}.ffn"] = KronABres(hidden_dim, scale=scale, num_biases=num_biases)
            else:
                self.kronas[f"encoder.{encoder_idx}.ffn"] = KronAB(hidden_dim, scale=scale, num_biases=num_biases)
        
        for decoder_idx in range(len(self.model.decoder.block)):
            if use_residual:
                self.kronas[f"decoder.{decoder_idx}.ffn"] = KronABres(hidden_dim, scale=scale, num_biases=num_biases)
            else:
                self.kronas[f"decoder.{decoder_idx}.ffn"] = KronAB(hidden_dim, scale=scale, num_biases=num_biases)
    
    def forward(self, **kwargs):
        # Define a custom forward for FFN blocks with KronAB adapters
        def custom_ffn_forward(ffn_module, krona_module, x):
            # Original FFN forward
            ffn_output = ffn_module(x)
            # Add KronAB output
            return ffn_output + krona_module(x)
        
        # Save original forward methods
        encoder_ffn_forwards = []
        decoder_ffn_forwards = []
        
        # Replace FFN forward methods with our custom ones
        for encoder_idx in range(len(self.model.encoder.block)):
            ffn = self.model.encoder.block[encoder_idx].layer[1].DenseReluDense
            krona = self.kronas[f"encoder.{encoder_idx}.ffn"]
            
            # Save original
            encoder_ffn_forwards.append(ffn.forward)
            
            # Replace with custom forward that uses KronAB
            ffn.forward = lambda x, ffn=ffn, krona=krona: custom_ffn_forward(ffn, krona, x)
        
        for decoder_idx in range(len(self.model.decoder.block)):
            ffn = self.model.decoder.block[decoder_idx].layer[2].DenseReluDense
            krona = self.kronas[f"decoder.{decoder_idx}.ffn"]
            
            # Save original
            decoder_ffn_forwards.append(ffn.forward)
            
            # Replace with custom forward that uses KronAB
            ffn.forward = lambda x, ffn=ffn, krona=krona: custom_ffn_forward(ffn, krona, x)
        
        # Forward pass through model
        outputs = self.model(**kwargs)
        
        # Restore original forward methods
        for encoder_idx, original_forward in enumerate(encoder_ffn_forwards):
            self.model.encoder.block[encoder_idx].layer[1].DenseReluDense.forward = original_forward
        
        for decoder_idx, original_forward in enumerate(decoder_ffn_forwards):
            self.model.decoder.block[decoder_idx].layer[2].DenseReluDense.forward = original_forward
            
        return outputs