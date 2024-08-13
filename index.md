# IDEAW

## Abstract
<p align="justify">
Audio watermarking embeds messages into audio and accurately extracts the watermark even after the watermarked audio has been damaged. Compared to traditional digital watermarking algorithms, neural watermarking using neural networks has better robustness for various attacks are considered during training. However, neural watermarking methods suffer to low capacity and undesirable imperceptibility. In addition, in practical scenarios, watermarks are redundantly embedded in audio according to the demand, and the audio is also subjected to cropping and splicing, which makes the efficient locating of watermarks a problem worth exploring. In this paper, we design an invertible neural network to realize a dual-embedding watermarking model for efficient locating, at the same time, we consider the impact of the attack layer on the invertible neural network in robustness training and improve the model so as to enhance the reasonableness and stability. Experiment shows that the proposed model, **IDEAW**, can withstand various attacks and has higher capacity with more efficient locating ability compared to the state-of-the-art methods.
</p>

## Overview
<p align="justify">

</p>

<div style="text-align: center;">
<img src="assets/IDEAW.png" width = 1000 />
</div>
<p align="center">The Architecture of IDEAW.</p>
<p>&nbsp;</p> 

## Watermarked Audio&Waveform Samples
Audio samples are taken from the VCTK corpus and FMA corpus. The capacity of **IDEAW** achieves 46 bits per second (maintaining SNR at above 30 dB).
The red and dark yellow waveform in the figures stand for watermarked audio and host audio respectively (500-point details of watermarked audio segments).

Audio samples are randomly selected from FMA dataset (musica) and VCTK dataset (speech).

### What can 46-bit watermark message accomplish?

<script>
function pauseOthers(ele) {
    $("audio").not(ele).each(function (index, audio) {audio.pause();});
}
</script>
<p>&nbsp;</p> 

#### · Embedding 7*6 binary figures into audios.
##### I
<img src="assets/wm_msg/I.png" width = 200 />
<div style="text-align: center;">
<img src="assets/wmd_waveforms/070_1.png" width = 500 />
</div>
 <audio controls id="player" onplay="pauseOthers(this);"><source src="assets/wmd_audios/070.mp3" type="audio/mpeg"></audio> embedded 5 times, SNR=35.14, ACC=1(280/280)
<p>&nbsp;</p> 

##### D
<img src="assets/wm_msg/D.png" width = 200 />
<div style="text-align: center;">
<img src="assets/wmd_waveforms/554.png" width = 500 />
</div>
<audio controls id="player" onplay="pauseOthers(this);"><source src="assets/wmd_audios/554.mp3" type="audio/mpeg"></audio> embedded 5 times, SNR=36.29, ACC=0.9928(278/280)
<p>&nbsp;</p> 

##### E
<img src="assets/wm_msg/E.png" width = 200 />
<div style="text-align: center;">
<img src="assets/wmd_waveforms/012.png" width = 500 />
</div>
<audio controls id="player" onplay="pauseOthers(this);"><source src="assets/wmd_audios/012.mp3" type="audio/mpeg"></audio> embedded 5 times, SNR=35.87, ACC=0.9964(279/280)
<p>&nbsp;</p> 

##### A
<img src="assets/wm_msg/A.png" width = 200 />
<div style="text-align: center;">
<img src="assets/wmd_waveforms/022.png" width = 500 />
</div>
<audio controls id="player" onplay="pauseOthers(this);"><source src="assets/wmd_audios/022.mp3" type="audio/mpeg"></audio> embedded 5 times, SNR=33.12, ACC=1(280/280)
<p>&nbsp;</p> 

##### W
<img src="assets/wm_msg/W.png" width = 200 />
<div style="text-align: center;">
<img src="assets/wmd_waveforms/024.png" width = 500 />
</div>
<audio controls id="player" onplay="pauseOthers(this);"><source src="assets/wmd_audios/024.mp3" type="audio/mpeg"></audio> embedded 5 times, SNR=32.43, ACC=0.9928(278/280)
<p>&nbsp;</p> 

<p>&nbsp;</p> 

#### · Embedding characters which are encoded to 5-bit codes into audios. (46 bits watermark -> 9 characters)
Embedding "IDEAWOKAY" (01000-00100-00101-00001-10111-01111-01011-00001-11001-0) into audio.

##### "IDEAWOKAY"
<div style="text-align: center;">
<img src="assets/wmd_waveforms/331.png" width = 500 />
</div>
<audio controls id="player" onplay="pauseOthers(this);"><source src="assets/wmd_audios/331.mp3" type="audio/mpeg"></audio> embedded 5 times, SNR=34.54, ACC=0.9964(279/280)
<p>&nbsp;</p>

<p>&nbsp;</p> 


# IDEAW Code
We will release our code as soon as the paper is accepted. 🍹
