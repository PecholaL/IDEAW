# IDEAW
Robust Neural Audio Watermark with Invertible Dual-Embedding

## Training IDEAW
"(*)" means **necessary** steps
### Data Processing
1. (*)Modify ../data/dataConfig.yaml based on the specific location of your dataset.
2. (*)Build Dataset and dump into pickle:
   `python3 ../data/process.py`
3. Test your built Dataset, DataLoader, STFT&ISTFT:
   `python3 [test]data.py`

### Training

### Embedding and Extraction

### Customizing Attack

### Thanks
