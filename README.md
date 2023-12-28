# IDEAW
Robust Neural Audio Watermarking with Invertible Dual-Embedding

**☕️☕️☕️**

## Training IDEAW
"**(\*)**" means **necessary** steps
### Data Processing
1. **(\*)** Modify *./data/config.yaml* based on the specific location of your dataset.
2. **(\*)** Build Dataset and dump into pickle:
   `python3 ./data/process.py`
3. Test your built Dataset, DataLoader:
   `python3 ./data/test_data.py`
   DataLoader provides data batch shaped [batch_size, sr * audio_limit_len] each time.

### Building Model
**IDEAW** is composed of MIHNETs, a discriminator, a balance block and an attack layer.
These networks are defined in different .py files:
1. *mihnet.py* defines the core multi-INN of **IDEAW**.
2. *componentNet.py* defines the discriminator and the balance block.
3. *attackLayer.py* defines the attack simulate layer.

The components of MIHNET, i.e. INN block is defined in *innBlock.py*.
Then, in *ideaw.py*, these components are assembled. The IDEAW in *ideaw.py* also provides STFT/ISTFT/Embed/Extract operations.
Finally, solver.py provides the optimizer of IDEAW.
Note that all the configurations of the model are located in *./models/config.yaml*.

### Training
1. **(\*)** Modify the training configuration *./config.yaml* and the paths of the related configurations and dataset in *./train.sh*.
2. **(\*)** Train the model:
   `./train.sh`
By default, the two stages of training each account for half of the iterations.

## Embedding and Extraction
*embed_extract.py* provides the embedding and extracting processes of audio watermark via the trained **IDEAW**.
Note that if you have used the DataParallel version of *solver.py* to train **IDEAW**, please use the DataParallel version of *embed_extract.py* to test the trained model. *embed_extract.py* showcases how to use **IDEAW** for watermark embedding and extraction. For simplicity, hard coding is used, and the trained **IDEAW** can be encapsulated based on the provided approach.

## Customizing Attack
The attack types in attackLayer can be customized in *./models/attackLayer.py* to enhance the specific robustness of **IDEAW**. The factors of attacks are in *./models/config.yaml*["AttackLayer"]
The code provides 8 types of common attacks on audio, including Gaussian additive noise, lower-pass filter, mpeg3 compression, etc.
Remember to modify the configuration if more attacks are included: 
*./models/config.yaml*["AttackLayer"]["att_num"].

## Thanks
