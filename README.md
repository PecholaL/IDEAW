# IDEAW
Robust Neural Audio Watermark with Invertible Dual-Embedding

## Training IDEAW
"**(*)**" means **necessary** steps
### Data Processing
1. **(*)**Modify `./data/config.yaml` based on the specific location of your dataset.
2. **(*)**Build Dataset and dump into pickle:
   `python3 ./data/process.py`
3. Test your built Dataset, DataLoader:
   `python3 ./data/test_data.py`
   DataLoader provides data batch shaped [batch_size, sr * audio_limit_len] each time.

### Training

### Embedding and Extraction

### Customizing Attack

### Thanks
