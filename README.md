<h1 align="center"> ESFP: Event-based Shape from Polarization (CVPR 2023) </h1>

<br>

This repository contains the code and download links to our dataset for our work on "**Event-based Shape-from-Polarization**",   [CVPR 2023](https://cvpr2023.thecvf.com/) by [Manasi Muglikar](https://manasi94.github.io/),  [Leonard Bauersfeld](https://lbfd.github.io/), [Diederik Moeys](https://scholar.google.ch/citations?user=RjfwsaIAAAAJ&hl=en), and [Davide Scaramuzza](https://rpg.ifi.uzh.ch/people_scaramuzza.html).

<h2 align="center"> 
  
[Project Page](https://rpg.ifi.uzh.ch/esfp.html) | [Paper](https://rpg.ifi.uzh.ch/docs/CVPR23_Muglikar.pdf) | [Video](https://youtu.be/sF3Ue2Zkpec) | [Dataset](https://rpg.ifi.uzh.ch/esfp.html)
</h2>

[![Event-based Shape from Polarization](images/CVPR23_Muglikar_yt.png)](https://youtu.be/sF3Ue2Zkpec)

## Citation
If you use this code in an academic context, please cite the following work:

```
@InProceedings{Muglikar23CVPR,
  author = {Manasi Muglikar and Leonard Bauersfeld and Diederik Moeys and Davide Scaramuzza},
  title = {Event-based Shape from Polarization},
  booktitle = {IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  month = {Jun},
  year = {2023}
}
```

## Installation

Install metavision from [here](https://docs.prophesee.ai/3.1.2/installation/index.html)
```bash
conda create -y -n esfp
conda activate esfp
conda install -y -c anaconda numpy scipy
conda install -y -c conda-forge opencv tqdm matplotlib pybind11 h5py blosc-hdf5-plugin
pip install --no-cache-dir -r training_code/requirements.txt

```
## Dataset
We present the *first large scale dataset* consisting of several objects with different textures and shapes, and featuring multiple illumination and scene depths, for a total of 100 synthetic and 90 real scenes. 
  
Download the dataset from [here](rpg.ifi.uzh.ch/esfp.html)

To download the mistuba dataset and real dataset use the following links respectively:

```
wget https://download.ifi.uzh.ch/rpg/ESfP/mitsuba_dataset.zip
wget https://download.ifi.uzh.ch/rpg/ESfP/realworld_dataset.zip
```

## Train
To train the network to predict surface normals, use the following training scripts:
`bash training_code/scripts/train_events_esfp_syn.sh`

