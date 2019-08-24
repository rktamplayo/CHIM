# CHIM
[EMNLP2019] Rethinking Attribute Representation and Injection for Sentiment Classification

This PyTorch code was used in the experiments of the research paper

Reinald Kim Amplayo.
[**Rethinking Attribute Representation and Injection for Sentiment Classification**. _EMNLP_, 2019.]()

### Data

IMDB, Yelp2013, and Yelp2014 datasets are originally from [here](https://drive.google.com/open?id=1PxAkmPLFMnfom46FMMXkHeqIxDbA16oy). I did some changes, which I cannot unfortunately recall everything, with the format (not the content) of the file (e.g., changed the ordering of the input and output, etc.).

I am therefore sharing my version of the above datasets, as well as the Amazon datasets in the paper, in [this link]().

If you are using any of the above three datasets, please cite the original paper. The BibTeX is shown at the end.

Decompress all files, and save all the dataset directories in a directory named `data`.

### Train and Evaluate CHIM

To train the model, simply run using the format:

`python src/train.py <dataset> weight.chunk.imp <inject_location> 300 <chunk_ratio> <gpu_device>`

where:
- `dataset` is the name of the dataset directory dataset folder (e.g., `yelp2013`).
- `inject_location` is the location to inject the attributes. Choose from the following: `embed`, `encode`, `pool`, `classify`. Multiple locations can also be used by separating them with a comma (e.g., `embed,encode`)
- `chunk_ratio` is the chunk size factors discussed in the paper. The one used in the paper is `15`.
- `gpu_device` is the GPU device number.

Evaluation is done similarly, but with another file:

`python src/evaluate.py <dataset> weight.chunk.imp <inject_location> 300 <chunk_ratio> <gpu_device>`

### Cite the Necessary Papers

To cite the paper/code/data splits, please use this BibTeX:

```
@inproceedings{amplayo2019rethinking,
	Author = {Reinald Kim Amplayo},
	Booktitle = {EMNLP},
	Location = {Hong Kong, China},
	Year = {2019},
	Title = {Rethinking Attribute Representation and Injection for Sentiment Classification},
}
```

If using either of the IMDB/Yelp2013/Yelp2014 datasets, please also cite the original authors of the datasets:

```
@inproceedings{tang2015learning,
	Author = {Duyu Tang and Bing Qin and Ting Liu},
	Booktitle = {ACL},
	Location = {Beijing, China},
	Year = {2015},
	Title = {Learning Semantic Representations of Users and Products for Document Level Sentiment Classification},
}
```

If there are any questions, please send me an email: reinald.kim at ed dot ac dot uk