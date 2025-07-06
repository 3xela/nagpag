import os
from tasks import Text2ImgInput, Img2ImgInput, ImgOutput
from fal.toolkit import Image
from typing import Optional


model_name = "black-forest-labs/FLUX.1-dev"


class NagPagText2Img:
    def __init__(self, *args, **kwargs):
        from diffusers import FluxPipeline
        import torch
        import torch.nn.functional as F

        self.pipeline = FluxPipeline.from_pretrained(*args, **kwargs)

        # Speed optimizations without quality loss
        self.pipeline.enable_attention_slicing()  # Process attention in slices
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

        # Initialize default NAG-PAG parameters
        self.nag_scale = 0.5
        self.alpha = 0.5

        # Cache for prompt embeddings (speed optimization)
        self._prompt_cache = {}

        self._replace_attention_processors()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    # figure out how to use @torch.no_grad() here, importing torch is weird
    def __call__(
        self, prompt, negative_prompt=None, nag_scale=0.5, alpha=0.5, *args, **kwargs
    ):
        """Generate images with optional negative prompt using NAG-PAG attention.

        Args:
            prompt: Positive text prompt
            negative_prompt: Optional negative text prompt for guidance
            nag_scale: Scale factor for negative guidance (0.0-1.0)
            alpha: Blending factor for NAG-PAG output (0.0-1.0)
        """
        self.pipeline._nag_scale = nag_scale
        self.pipeline._alpha = alpha

        import torch

        with torch.no_grad():
            if negative_prompt is None or negative_prompt == "":
                # Clear any stored negative embeddings
                self.pipeline._negative_embeddings = None
                return self.pipeline(prompt, *args, **kwargs)

            # Encode and store negative embeddings for attention processors to access
            neg_embeds, _ = self._encode_prompt(negative_prompt)
            self.pipeline._negative_embeddings = neg_embeds

            # Run with positive prompt - attention processors will access stored negative embeddings
            return self.pipeline(prompt, *args, **kwargs)

    def __getattr__(self, name):
        if name == "__call__":
            # Don't delegate __call__ - we have our own custom implementation
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self.pipeline, name)

    def to(self, device):
        self.pipeline = self.pipeline.to(device)
        return self

    def _replace_attention_processors(self):
        for name, attn in self.pipeline.transformer.named_modules():
            from diffusers.models.attention import Attention

            if isinstance(attn, Attention):
                processor = NPFluxAttnProcessor2_0()
                processor._pipeline_ref = self.pipeline
                attn.set_processor(processor)

    def _encode_prompt(self, prompt):
        """Encode a prompt using FLUX's dual text encoders (CLIP + T5).

        Returns concatenated embeddings with shape [batch, seq_len, 4864].
        """
        import torch

        # Check cache first for speed optimization
        if prompt in self._prompt_cache:
            return self._prompt_cache[prompt]

        with torch.no_grad():
            # Tokenize prompt for CLIP encoder
            text_inputs = self.pipeline.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            # Tokenize prompt for T5 encoder
            text_inputs_2 = self.pipeline.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=self.pipeline.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

        # Encode with both text encoders
        prompt_embeds = self.pipeline.text_encoder(
            text_inputs.input_ids.to(self.pipeline.device),
            output_hidden_states=True,
        )
        pooled_prompt_embeds = prompt_embeds.pooler_output  # CLIP pooled embeddings
        prompt_embeds = prompt_embeds.hidden_states[-2]

        prompt_embeds_2 = self.pipeline.text_encoder_2(
            text_inputs_2.input_ids.to(self.pipeline.device),
            output_hidden_states=True,
        )
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]  # T5 hidden states only

        # Concatenate the hidden states from both encoders
        if prompt_embeds.shape[1] < prompt_embeds_2.shape[1]:
            # Pad CLIP embeddings to match T5 length
            padding = torch.zeros(
                prompt_embeds.shape[0],
                prompt_embeds_2.shape[1] - prompt_embeds.shape[1],
                prompt_embeds.shape[2],
                device=prompt_embeds.device,
                dtype=prompt_embeds.dtype,
            )
            prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)
        elif prompt_embeds_2.shape[1] < prompt_embeds.shape[1]:
            # Pad T5 embeddings to match CLIP length
            padding = torch.zeros(
                prompt_embeds_2.shape[0],
                prompt_embeds.shape[1] - prompt_embeds_2.shape[1],
                prompt_embeds_2.shape[2],
                device=prompt_embeds_2.device,
                dtype=prompt_embeds_2.dtype,
            )
            prompt_embeds_2 = torch.cat([prompt_embeds_2, padding], dim=1)

        # Now concatenate along the feature dimension
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

        # Cache the result for future use (speed optimization)
        result = (prompt_embeds, pooled_prompt_embeds)
        self._prompt_cache[prompt] = result

        return result

    def _clear_negative_embeddings(self):
        """Clear stored negative embeddings"""
        self.pipeline._negative_embeddings = None


class NagPagImg2Img:
    def __init__(self, *args, **kwargs):
        from diffusers import FluxImg2ImgPipeline
        import torch
        import torch.nn.functional as F

        # Handle both from_pretrained and component-based initialization
        if args and isinstance(args[0], str):
            # from_pretrained case
            self.pipeline = FluxImg2ImgPipeline.from_pretrained(*args, **kwargs)
        else:
            # Component-based initialization
            self.pipeline = FluxImg2ImgPipeline(*args, **kwargs)

        # Speed optimizations without quality loss
        self.pipeline.enable_attention_slicing()  # Process attention in slices
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

        # Initialize default NAG-PAG parameters
        self.nag_scale = 0.5
        self.alpha = 0.5

        # Cache for prompt embeddings (speed optimization)
        self._prompt_cache = {}

        self._replace_attention_processors()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    # figure out how to use @torch.no_grad() here, importing torch is weird
    def __call__(
        self, prompt, negative_prompt=None, nag_scale=0.5, alpha=0.5, *args, **kwargs
    ):
        """Generate images with optional negative prompt using NAG-PAG attention.

        Args:
            prompt: Positive text prompt
            negative_prompt: Optional negative text prompt for guidance
            nag_scale: Scale factor for negative guidance (0.0-1.0)
            alpha: Blending factor for NAG-PAG output (0.0-1.0)
        """
        self._nag_scale = nag_scale
        self._alpha = alpha

        import torch

        with torch.no_grad():
            if negative_prompt is None or negative_prompt == "":
                # Clear any stored negative embeddings
                self.pipeline._negative_embeddings = None
                return self.pipeline(prompt, *args, **kwargs)

            # Encode and store negative embeddings for attention processors to access
            neg_embeds, _ = self._encode_prompt(negative_prompt)
            self.pipeline._negative_embeddings = neg_embeds

            # Run with positive prompt - attention processors will access stored negative embeddings
            return self.pipeline(prompt, *args, **kwargs)

    def __getattr__(self, name):
        if name == "__call__":
            # Don't delegate __call__ - we have our own custom implementation
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self.pipeline, name)

    def to(self, device):
        self.pipeline = self.pipeline.to(device)
        return self

    def _replace_attention_processors(self):
        for name, attn in self.pipeline.transformer.named_modules():
            from diffusers.models.attention import Attention

            if isinstance(attn, Attention):
                processor = NPFluxAttnProcessor2_0()
                processor._pipeline_ref = self.pipeline
                attn.set_processor(processor)

    def _encode_prompt(self, prompt):
        """Encode a prompt using FLUX's dual text encoders (CLIP + T5).

        Returns concatenated embeddings with shape [batch, seq_len, 4864].
        """
        import torch

        # Check cache first for speed optimization
        if prompt in self._prompt_cache:
            return self._prompt_cache[prompt]

        with torch.no_grad():
            # Tokenize prompt for CLIP encoder
            text_inputs = self.pipeline.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            # Tokenize prompt for T5 encoder
            text_inputs_2 = self.pipeline.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=self.pipeline.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

        # Encode with both text encoders
        prompt_embeds = self.pipeline.text_encoder(
            text_inputs.input_ids.to(self.pipeline.device),
            output_hidden_states=True,
        )
        pooled_prompt_embeds = prompt_embeds.pooler_output  # CLIP pooled embeddings
        prompt_embeds = prompt_embeds.hidden_states[-2]

        prompt_embeds_2 = self.pipeline.text_encoder_2(
            text_inputs_2.input_ids.to(self.pipeline.device),
            output_hidden_states=True,
        )
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]  # T5 hidden states only

        # Concatenate the hidden states from both encoders
        if prompt_embeds.shape[1] < prompt_embeds_2.shape[1]:
            # Pad CLIP embeddings to match T5 length
            padding = torch.zeros(
                prompt_embeds.shape[0],
                prompt_embeds_2.shape[1] - prompt_embeds.shape[1],
                prompt_embeds.shape[2],
                device=prompt_embeds.device,
                dtype=prompt_embeds.dtype,
            )
            prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)
        elif prompt_embeds_2.shape[1] < prompt_embeds.shape[1]:
            # Pad T5 embeddings to match CLIP length
            padding = torch.zeros(
                prompt_embeds_2.shape[0],
                prompt_embeds.shape[1] - prompt_embeds_2.shape[1],
                prompt_embeds_2.shape[2],
                device=prompt_embeds_2.device,
                dtype=prompt_embeds_2.dtype,
            )
            prompt_embeds_2 = torch.cat([prompt_embeds_2, padding], dim=1)

        # Now concatenate along the feature dimension
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

        # Cache the result for future use (speed optimization)
        result = (prompt_embeds, pooled_prompt_embeds)
        self._prompt_cache[prompt] = result

        return result

    def _clear_negative_embeddings(self):
        """Clear stored negative embeddings"""
        self.pipeline._negative_embeddings = None


class FluxPredictor:
    def __init__(self):
        import torch
        from diffusers import FluxImg2ImgPipeline

        self.torch = torch
        self.FluxPipeline = NagPagText2Img
        self.FluxImg2ImgPipeline = FluxImg2ImgPipeline
        self._model_path = model_name
        self._setup_env()
        self.load_model()

    def _setup_env(self):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            from huggingface_hub import login

            login(token=hf_token)

    def _load_txt_2_img_model(self):
        self._pipe_txt2img = self.FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=self.torch.bfloat16,
        ).to("cuda")

    def _load_img_2_img_model(self):
        self._pipe_img2img = self.FluxImg2ImgPipeline(
            vae=self._pipe_txt2img.vae,
            text_encoder=self._pipe_txt2img.text_encoder,
            tokenizer=self._pipe_txt2img.tokenizer,
            text_encoder_2=self._pipe_txt2img.text_encoder_2,
            tokenizer_2=self._pipe_txt2img.tokenizer_2,
            transformer=self._pipe_txt2img.transformer,
            scheduler=self._pipe_txt2img.scheduler,
        ).to("cuda")

    def load_model(self):
        self._load_txt_2_img_model()
        self._load_img_2_img_model()

    def do_text_2_img(self, request: Text2ImgInput) -> ImgOutput:
        raw_output = self._pipe_txt2img(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            nag_scale=request.nag_scale,
            alpha=request.alpha if request.negative_prompt != "" else 0,
            height=request.height,
            width=request.width,
            guidance_scale=request.guidance,
            num_inference_steps=request.steps,
            generator=self.torch.Generator().manual_seed(request.seed),
        )
        image = Image.from_pil(raw_output.images[0])
        return ImgOutput(image=image)

    def do_img_2_img(self, request: Img2ImgInput) -> ImgOutput:
        image = request.image.to_pil().resize((request.width, request.height))
        raw_output = self._pipe_img2img(
            image=image,
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            strength=request.strength,
            guidance_scale=request.guidance,
            num_inference_steps=request.steps,
            generator=self.torch.Generator().manual_seed(request.seed),
        )
        image = Image.from_pil(raw_output.images[0])
        return ImgOutput(image=image)


class NPFluxAttnProcessor2_0:
    """NAG-PAG attention processor for FLUX transformer blocks.

    Implements the NAG-PAG (Negative-Aware Guidance with Perturbed Attention Guidance)
    attention mechanism by replacing normal attention computation with identity matrix
    attention for negative prompts.
    """

    def __init__(self):
        import torch
        import torch.nn.functional as F

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "NPFluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: "Attention",
        hidden_states: "torch.FloatTensor",
        encoder_hidden_states: "torch.FloatTensor" = None,
        attention_mask: Optional["torch.FloatTensor"] = None,
        image_rotary_emb: Optional["torch.Tensor"] = None,
    ) -> "torch.FloatTensor":
        """Process attention with optional NAG-PAG negative guidance.

        Args:
            attn: FLUX attention layer
            hidden_states: Input hidden states
            encoder_hidden_states: Text encoder states
            attention_mask: Optional attention mask
            image_rotary_emb: Optional rotary embeddings

        Returns:
            Processed hidden states with NAG-PAG guidance applied if negative embeddings present
        """
        import torch
        import torch.nn.functional as F

        # Get current parameters from pipeline
        nag_scale = getattr(self._pipeline_ref, "_nag_scale", 0.3)
        alpha = getattr(self._pipeline_ref, "_alpha", 0.99)

        negative_embeddings = None
        if hasattr(self, "_pipeline_ref") and hasattr(
            self._pipeline_ref, "_negative_embeddings"
        ):
            negative_embeddings = self._pipeline_ref._negative_embeddings

        if negative_embeddings is not None:
            apply_nagpag = True
        else:
            apply_nagpag = False

        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Compute attention for the full batch (positive + negative) with optimized backend
        attention_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Apply NAG-PAG guidance if negative embeddings are available
        if apply_nagpag:
            # Create projection layer to convert concatenated embeddings (4864) to FLUX format (3072)
            if not hasattr(self, "_text_proj"):
                import torch.nn as nn

                device = negative_embeddings.device
                dtype = negative_embeddings.dtype
                self._text_proj = nn.Linear(4864, 3072, device=device, dtype=dtype)

            # Project negative embeddings to match encoder_hidden_states format
            neg_encoder_hidden_states = self._text_proj(
                negative_embeddings
            )  # [batch, 512, 3072]

            # Handle both cross-attention and self-attention layers in FLUX
            if hasattr(attn, "add_v_proj") and attn.add_v_proj is not None:
                # Cross-attention layer: use add_v_proj for encoder states
                neg_value_proj = attn.add_v_proj(neg_encoder_hidden_states)
            else:
                # Self-attention layer: use standard to_v projection
                neg_value_proj = attn.to_v(neg_encoder_hidden_states)

            # Reshape to match FLUX attention format
            neg_value_proj = neg_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            # Apply identity matrix attention for NAG-PAG negative guidance

            # Create negative attention output with identity matrix behavior
            neg_attention_output = torch.zeros_like(attention_output)

            # Apply negative embeddings to encoder portion (first 512 tokens)
            neg_seq_len = neg_value_proj.shape[2]
            neg_attention_output[:, :, :neg_seq_len, :] = neg_value_proj

            # Preserve hidden state portion from positive attention
            neg_attention_output[:, :, neg_seq_len:, :] = attention_output[
                :, :, neg_seq_len:, :
            ]

            # Apply NAG-PAG guidance equations
            attention_output = self.apply_nagpag_equations(
                attention_output, neg_attention_output, nag_scale, alpha
            )

        # Reshape attention output back to original format
        hidden_states = attention_output.transpose(1, 2).reshape(
            attention_output.shape[0], -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split encoder and hidden state portions
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # Apply output projections
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

    def apply_nagpag_equations(self, z_positive, z_negative, nag_scale, alpha):
        """Apply NAG-PAG equations 7-10 from arxiv:2505.21179.

        Args:
            z_positive: Positive attention output
            z_negative: Negative attention output (computed with identity matrix times v_negative)

        Returns:
            Combined attention output using NAG-PAG guidance
        """
        import torch

        # equation 7: z_tilde = z_pos + nag_scale * (z_pos - z_neg)
        z_tilde = z_positive + nag_scale * (z_positive - z_negative)

        # equation 8: compute norms and ratio
        norm_positive = torch.norm(z_positive, p=1, dim=-1, keepdim=True)
        norm_tilde = torch.norm(z_tilde, p=1, dim=-1, keepdim=True)
        ratio = norm_tilde / norm_positive

        # equation 9: apply threshold
        tau = 2.5  # Default tau value
        z_hat = torch.where(ratio > tau, tau, ratio) / ratio * z_tilde
        # equation 10: final combination
        z_nagpag = alpha * z_hat + (1 - alpha) * z_positive

        return z_nagpag
