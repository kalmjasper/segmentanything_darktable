# Darktable Mask Generator

A Python tool that helps create Darktable masks using AI-powered image segmentation. This tool allows you to automatically generate path masks for Darktable by selecting points on an image. (still very experimental)

## Features

- Interactive image segmentation using SAM2 (Segment Anything Model 2)
- Converts segmentation masks into Darktable-compatible path masks

## How to install

```bash
# Clone the repository and submodules
git clone --recursive https://github.com/kalmjasper/darktable-mask-generator
cd darktable-mask-generator

# If you already cloned without submodules, initialize them with:
git submodule update --init --recursive

# Download the SAM2 checkpoint file and place it in `./checkpoints/sam2.1_hiera_large.pt`
cd checkpoints && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt && cd ..

# Install dependencies
poetry install
```

## How to run

Run the script with an image:
```bash
poetry run python3 src/main.py --file example_images/truck.jpg
```

- Use the following shortcuts to interact with the image:
  - Left click: Add positive points
  - Right click: Add negative points
  - 'c': Clear all points
  - 'q': Finish and generate mask
  
When you press 'q', the script will generate the outline and print the XMP string to the console. This xmp string is compatible with the "darktable:mask_points" field in the XMP metadata of a darktable image. (Not completely working yet)

```bash
poetry run python3 src/main.py
```



