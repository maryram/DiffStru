# DiffStru

This is an implementation of paper titled with "Joint Inference of Diffusion and Structure in Partially Observed Social Networks Using Coupled Matrix Factorization" authored by Maryam Ramezani, Aryan Ahadinia, Amirmohammad Ziaei Bideh, and Hamid R. Rabiee which is published by ACM Transaction on Knowledge Discovery from Data (TKDD) [1].

We have two versions of code: (1) a version which utilize side information and (2) a version which ignore side information in way to do the computations faster.

## Cite Us

```
@article{10.1145/3599237,
  author = {Ramezani, Maryam and Ahadinia, Aryan and Ziaei Bideh, Amirmohammad and Rabiee, Hamid R.},
  title = {Joint Inference of Diffusion and Structure in Partially Observed Social Networks Using Coupled Matrix Factorization},
  year = {2023},
  issue_date = {November 2023},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {17},
  number = {9},
  issn = {1556-4681},
  url = {https://doi.org/10.1145/3599237},
  doi = {10.1145/3599237},
  abstract = {Access to complete data in large-scale networks is often infeasible. Therefore, the problem of missing data is a crucial and unavoidable issue in the analysis and modeling of real-world social networks. However, most of the research on different aspects of social networks does not consider this limitation. One effective way to solve this problem is to recover the missing data as a pre-processing step. In this paper, a model is learned from partially observed data to infer unobserved diffusion and structure networks. To jointly discover omitted diffusion activities and hidden network structures, we develop a probabilistic generative model called “DiffStru.” The interrelations among links of nodes and cascade processes are utilized in the proposed method via learning coupled with low-dimensional latent factors. Besides inferring unseen data, latent factors such as community detection may also aid in network classification problems. We tested different missing data scenarios on simulated independent cascades over LFR networks and real datasets, including Twitter and Memetracker. Experiments on these synthetic and real-world datasets show that the proposed method successfully detects invisible social behaviors, predicts links, and identifies latent features.},
  journal = {ACM Trans. Knowl. Discov. Data},
  month = {jul},
  articleno = {132},
  numpages = {28},
  keywords = {matrix factorization, network structure, social network, partially observed data, cascade completion, Information diffusion, link prediction}
}
```

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

## References

[1] Maryam Ramezani, Aryan Ahadinia, Amirmohammad Ziaei Bideh, and Hamid R. Rabiee. 2023. Joint Inference of Diffusion and Structure in Partially Observed Social Networks Using Coupled Matrix Factorization. ACM Trans. Knowl. Discov. Data 17, 9, Article 132 (November 2023), 28 pages. https://doi.org/10.1145/3599237
