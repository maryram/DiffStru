# DiffStru

This is an implementation of paper titled with "Joint Inference of Diffusion and Structure in Partially Observed Social Networks Using Coupled Matrix Factorization" authored by Maryam Ramezani, Aryan Ahadinia, Amirmohammad Ziaei Bideh, and Hamid R. Rabiee which is published by ACM Transaction on Knowledge Discovery from Data (TKDD).

We have two versions of code: (1) a version which utilize side information and (2) a version which ignore side information in way to do the computations faster.

## How to Run

To train the model, run command below:

```bash
python main.py --dataset_path <path to dataset: POSIX path> --dim <dimension of latent space: int> --train
```

And to test the model, run 

```bash
python main.py --dataset_path <path to dataset: POSIX path> --dim <dimension of latent space: int> --burn <Gibbs sampling burn-in: int> --thinning <Gibbs sampling thinning: int> --e_threshold <delta_G: float> --test
```

## Parameters

- `--dataset_path`: Path of the dataset, must contain files named with `Observed_C.txt`, `Observed_G.txt`, `Groundtruth_C.txt`, and `Groundtruth_G.txt` Where $C$ is the cascades matrix in which each element indicates time of participating user in cascade and $G$ is a directed matrix containing the users interactions.
- `--burn`: Number of burn-in for Gibbs sampling.
- `--thinning`: Number of `thinning` for Gibbs sampling.
- `--dim`: Dimension of latent space of embeddings.
- `--train`, `--test`: Training or Testing mode.
- `--cascade`: Print cascades according to rank of nodes.
- `--e_threshold`: $\delta_G$
