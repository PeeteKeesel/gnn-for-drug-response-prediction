- [1. AGMI: Attention-Guided Multi-omics Integration for Drug Response Prediction with Graph Neural Networks](#1-agmi-attention-guided-multi-omics-integration-for-drug-response-prediction-with-graph-neural-networks)
  - [Papers Contributions](#papers-contributions)
  - [Experiment](#experiment)
  - [Questions](#questions)
- [2. My Knowledge Summary](#2-my-knowledge-summary)
  - [2.1. MPNN - Message Passing Neural Network](#21-mpnn---message-passing-neural-network)
  - [3. GRU - Gated Recurrent Unit](#3-gru---gated-recurrent-unit)
  - [4. LSTM - Long Short-Term Memory](#4-lstm---long-short-term-memory)

---
# 1. AGMI: Attention-Guided Multi-omics Integration for Drug Response Prediction with Graph Neural Networks
- [paper](https://arxiv.org/pdf/2112.08366.pdf)
- [code](https://github.com/yivan-WYYGDSG/AGMI)

> presents Attention-Guided Multi-omics Integration (AGMI) approach for DRP

AGMI: 
1. Constructs a Multi-edge Graph (MeG) for each cell line
2. Aggregates multi-omics features to predict drug response using a Graph edge-aware Network (GeNet)

- approach explores __gene constraint-based multi-omics integration__ for DRP with __the whole-genome using GNNs__.
- _Drug Response Prediction_: aims to predict the response of a patient to a given drug
- DRP methods before: ta ke single type of omics data as input to predict drug response e.g. 
  - gene expression profiles or 
  - aberrations
- more omics data can improve the model predictive power
- However, crucial issues in multi-omics integration for DRP
- --> Essential: develop effective multi-omics integration for more accurate DRP

AGMI: 
1. models each cell line as a _Multi-edge Graph_ (MeG)
2. conducts attention-guided feature aggregation for multi-omics features with a _Graph edge-aware Network_ (GeNet) structure

MeG: 
- considers explicit priors of genes 
- encodes 3 basic types of genome features as node features
  1. gene expression
  2. mutation
  3. CNV
  - and several other 
    - omics data and
    - biological priors on gene relations ...
  - ... as multi-edges e.g. 
    - Protein-Protein Interaction (PPI)
    - gene pathways
    - Pearson correlation coefficient (PCC) of gene expression

GeNet:
- developed based on _Message Passing Neural Networks_ (MPNNs)
- by introducing 2 _Gated Recurrent Units_ ([GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit)s)
  1. _nGRU_: is at the node-level inside the Basic Layer guiding feature extraction of multiple edges and the other 
  2. _gGRU_: is at the graph-level converging gene features of the whole MeG and fusing multi-scale features
- besides, it adopts a _Graph Isomorphism Network_ (GIN)
  - to generate a drug features vector and 
  - concatenate it with a cell line feature vector for final prediction

## Papers Contributions
- model cell lines as a graph with __multiple types of edges__ (MeG)
- use both node-level __and__ graph-level GRU to capture complex features from MeG

## Experiment
They obtained
1. genomic profiles from the CCLE dataset 
  - including gene expression, gene mutation, CNV
2. response scores of cell-drug pairs (IC50 values) from GDSC 
Then they align the genomic profiles (1.) with the $ln$ of the IC50 values (2.) for each cell-drug pair.

## Questions

> Each node $v_i \in \mathbb{V}, i \in {1, ..., N_v}$, represents a gene, with its amount of expression, gene mutation state, and amount of CNV as node features.
- Why is there no mathematical notation of the node features, same as for the edge features

> Edges built based on these three relations are respectively denoted by $e^p, e^s$, and $e^c$.
- Does that mean that every edge can have either one of the features or all of them? Koennen also manche edges keine Info zu $e^s$ haben?
- Sind das feature, oder einfach nur die Bezeichnung der Edges? 




---
# 2. My Knowledge Summary
## 2.1. MPNN - Message Passing Neural Network
- [paper](https://arxiv.org/pdf/1704.01212v2.pdf)
- [code](https://github.com/brain-research/mpnn)
- [papers with code](https://paperswithcode.com/method/mpnn)
- [Keras tutorial](https://keras.io/examples/graph/mpnn-molecular-graphs/)
- [towardsdatascience Introduction](https://towardsdatascience.com/introduction-to-message-passing-neural-networks-e670dc103a87)

Forward pass has 2 phases
1. _Message passing phase_- [1. AGMI: Attention-Guided Multi-omics Integration for Drug Response Prediction with Graph Neural Networks](#1-agmi-attention-guided-multi-omics-integration-for-drug-response-prediction-with-graph-neural-networks)
- [My Knowledge Summary](#my-knowledge-summary)
  - [MPNN - Message Passing Neural Network](#mpnn---message-passing-neural-network)
2. _Readout phase_

<ins>Message passing phase:</ins> 
- runs for $T$ time steps 
- defined in terms of 
  - message functions $M_t$ and 
  - vertex update functions $U_t$
> during this phase hidden states $h_v^t$ at each node n the graph are updated based on messages $m_v^{t+1}$ according to
$$
m_v^{t+1} = \sum_{w \in N(v)} M_t(h_v^t, h_w^t, e_{vw}) \\
h_v^{t+1} = U_t(h_v^t, m_v^{t+1})
$$

- $h_v^t$: hidden state of node $v$ at time $t$ 
- $h_w^t$: hidden state of $v$'s neighbor node $w$ at time $t$ 
- $e_{vw}$: edge feature of the edge from node $v$ to $w$
- $N(v)$: neighbors of $v$ in graph $G$ 

<ins>Readout phase:</ins>
- computes feature vector for whole graph $G$ 
- using some readout function $R$ according to 
$$
\hat{y} = R(\{ h_v^T | v \in G \})
$$

The message functions $M_t$, vertex update functions $U_t$, and readout function $R$ are all learned differentiable functions

- $R$: operates on the set of node states and must be invariant to permutations of the node states in order for the MPNN to be invariant to graph isomorphism

## 3. GRU - Gated Recurrent Unit
- [wikipedia](https://en.wikipedia.org/wiki/Gated_recurrent_unit)
  
> are a gated mechanism in a RNN

> is like a LSTM with forget gate,  but has fewer parameters than LSTM, as it lacks an output gate

## 4. LSTM - Long Short-Term Memory