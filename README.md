<h2 align="left">PerfGAT</h1>

###

Official implementation of paper: Spatiotemporal Graph Neural Network Modelling Perfusion MRI (*MICCAI 2024*)
[SpringerLink](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_39)
[Arxiv](https://arxiv.org/abs/2406.06434)

###

<h3 align="left">Workflow overview</h2>

###

<div align="center">
  <img height="450" src="https://media.springernature.com/full/springer-static/image/chp%3A10.1007%2F978-3-031-72069-7_39/MediaObjects/631670_1_En_39_Fig1_HTML.png?as=webp"  />
</div>

###

<p align="left">The proposed PerfGAT firstly constructs a spatiotemporal graph using DSC-MRI, brain atlas and tumor masks to incorporate focal tumors. Subsequently, spatial and temporal graph features and local tumor features are generated using distinct encoders. To reduce noisy temporal connections, we employ a graph structure learning approach based on edge attentions. The dual-attention feature fusion mechanism strategically integrates features across local tumor and spatial and temporal graphs, yielding comprehensive representations from pMRI. Finally, to tackle the class-imbalance issue while avoiding data distortion, we employ a recombining augmentation mechanism tailored to spatiotemporal graph.</p>

###

<h3 align="left">Abstract</h2>

###

<p align="left">Perfusion MRI (pMRI) offers valuable insights into tumor vascularity and promises to predict tumor genotypes, thus benefiting prognosis for glioma patients, yet effective models tailored to 4D pMRI are still lacking. This study presents the first attempt to model 4D pMRI using a GNN-based spatiotemporal model (PerfGAT), integrating spatial information and temporal kinetics to predict Isocitrate DeHydrogenase (IDH) mutation status in glioma patients. Specifically, we propose a graph structure learning approach based on edge attention and negative graphs to optimize temporal correlations modeling. In addition, we design a dual-attention feature fusion module to integrate spatiotemporal features while addressing tumor-related brain regions. Further, we develop a class-balanced augmentation methods tailored to spatiotemporal data, which could mitigate the common label imbalance issue in clinical datasets. Our experimental results demonstrate that the proposed method outperforms other state-of-the-art approaches, promising to model pMRI effectively for patient characterization.</p>

###
### Contact

For any question, please feel free to email: ry309@cam.ac.uk
