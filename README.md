# SBKGEX
<h2>Building Knowledge Subgraphs in Question Answering over Knowledge Graphs</h2>
<h4>Abstract</h4>
<p>Question answering over knowledge graphs targets to leverage facts in knowledge graphs to answer natural language questions. The presence of large number of facts, particularly in huge and well-known knowledge graphs such as DBpedia, makes it difficult to access the knowledge graph for each given question. This paper describes a generic solution based on Personal Page Rank for extracting a small subset from the knowledge graph as a knowledge subgraph which is likely to capture the answer of the question. Given a natural language question, relevant facts are determined by a bi-directed propagation process based on Personal Page Rank. Experiments are conducted over FreeBase, DBPedia and WikiMovie to demonstrate the effectiveness of the approach in terms of recall and size of the extracted knowledge subgraphs.</p>

<h4>Keywords: </h4>
<p>Knowledge graphs, Question answering systems, Knowledge subgraph, Personal Page Rank</p>

<h4>Acknowledgement</h4>
This work is part of "Building Knowledge Subgraphs in Question Answering over Knowledge Graphs". Therefore, if you use any code from this repository, please cite our work.

```
@inproceedings{10.1007/978-3-031-09917-5_16,
author = {Aghaei, Sareh and Angele, Kevin and Fensel, Anna},
title =  {Building Knowledge Subgraphs in Question Answering over Knowledge Graphs},
year = {2022},
isbn = {978-3-031-09916-8},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-09917-5_16},
doi = {10.1007/978-3-031-09917-5_16},DOI = {10.3390/s22072763},
booktitle = {Web Engineering: 22nd International Conference, ICWE 2022, Bari, Italy, July 5–8, 2022, Proceedings},
pages = {237–251},
numpages = {15},
keywords = {Knowledge subgraph, Knowledge graphs, Personal Page Rank, Question answering systems},
location = {Bari, Italy}
}
```
We follow [GraftNet](https://github.com/haitian-sun/GraftNet) to conduct this work.

To set-up a dataset(e.g., DBPedia) end point on a local server, please find the instructions from [here](https://github.com/IBCNServices/pyRDF2Vec/wiki/Fast-generation-of-RDF2Vec-embeddings-with-a-SPARQL-endpoint)

