# roma-v-hml
Quantitative evaluation of RoMa against HomoMatcherLite for Image Stitching and 3D Reconstruction

## Project Structure
```
roma-v-hml/
├── app/
├── analytics/ 
├── data/               
│   ├── hpatches-sequences-release/           
│   └── eth3d/              
├── output/               
├── Dockerfile            
├── entrypoint.sh         
├── requirements.txt      
└── README.md            
```

## Requirements
- Python 3.8+
- Docker (optional, for containerized execution)

## Dependencies
- numpy >= 1.21
- opencv-python-headless
- torch >= 1.10
- matplotlib >= 3.4
- scikit-image >= 0.18
- scipy >= 1.7
- tqdm >= 4.62
- romatch (from RoMA repository)
- PoseLib

## Installation

### Python-based Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/roma-v-hml.git
cd roma-v-hml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker-based Installation
1. Build the Docker image:
```bash
docker build -t evaluator .
```

## Usage

### Python-based Usage
1. Place your input data in the `data` directory:
   - For image stitching evaluation: Place image pairs in `data/hpatches-sequences-release/`
   - For 3D reconstruction evaluation: Place datasets in `data/eth3d/`

2. Run the evaluation:
```bash
# For image stitching
python app/evaluation_image.py

# For 3D reconstruction
python app/evaluation_3d.py
```

### Docker-based Usage

After adding the data into the folder:

```bash
# For image stitching
docker run -v "$(pwd)/data:/app/data" -v "$(pwd)/output:/app/output" evaluator image

# For 3D reconstruction
docker run -v "$(pwd)/data:/app/data" -v "$(pwd)/output:/app/output" evaluator 3d
```

## Analytics

After running the evaluations and having the CSV files generated:

### Script for plotting
```bash
# For image stitching
python analytics/analyze_hpatches_detailed.py

# For 3D reconstruction
python analytics/analyze_3d.py

```

## Output
Results will be saved in the `output` directory, organized by dataset name and evaluation type. There are also plots from the analytics of performance.