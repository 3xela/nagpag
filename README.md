# NAG-PAG with FLUX.1-dev

Implementation of NAG-PAG (Negative Attention Guidance with Perturbed Attention Guidance) for FLUX.1-dev diffusion model.

## Features

- **Custom attention processors** for all 57 FLUX transformer layers
- **Identity matrix attention** for negative prompt guidance
- **Dual text encoding** using CLIP + T5 encoders
- **Dynamic parameter control** via `nag_scale`
- **Optimized performance** with attention slicing and caching

## Setup

```bash
./setup.sh
```

## Usage

```bash
fal run main.py::MyApp
```

## API

- **Text2Img**: `POST /` with `prompt`, `negative_prompt`, `nag_scale`
- **Img2Img**: `POST /img2img` with `image`, `prompt`, `negative_prompt`

## Parameters

- `nag_scale`: Controls negative prompt strength (default: 0.3)
- `alpha`: Fixed at 0.5 for balanced blending
- `tau`: Fixed at 2.5 for thresholding (best choice from paper for flux)