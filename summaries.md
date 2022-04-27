
- [1. How to get started with Graph Machine Learning](#1-how-to-get-started-with-graph-machine-learning)
  - [1.1. What is Graph ML?](#11-what-is-graph-ml)
  - [1.2. Types of Graphs](#12-types-of-graphs)
    - [1.2.1. Undirected Graphs](#121-undirected-graphs)
    - [1.2.2. Directed Graphs](#122-directed-graphs)
    - [1.2.3. Multigraphs](#123-multigraphs)
    - [1.2.4. Hypergraphs](#124-hypergraphs)
    - [1.2.5. Graphs with self-edges](#125-graphs-with-self-edges)
    - [1.2.6. Graphs w/o self-edges](#126-graphs-wo-self-edges)
  - [1.3. Example](#13-example)
  - [1.4. Properties of Graphs](#14-properties-of-graphs)
    - [1.4.1. Homopholic](#141-homopholic)

# 1. [How to get started with Graph Machine Learning](https://gordicaleksa.medium.com/how-to-get-started-with-graph-machine-learning-afa53f6f963a)

## 1.1. What is Graph ML?
Graphs consists of
- __Nodes__: may have feature vectors associated with them
- __Edges__: may have feature vectors associated with them

What can be done with graphs? 
- Graph classification/regression
- Node/Edge classification/regression

Subfields
- GNNs
- Graph embedding methods
- ...

## 1.2. Types of Graphs
### 1.2.1. Undirected Graphs
### 1.2.2. Directed Graphs
### 1.2.3. Multigraphs
### 1.2.4. Hypergraphs
### 1.2.5. Graphs with self-edges
### 1.2.6. Graphs w/o self-edges

## 1.3. Example
Node regression task:
- Nodes in the graphs are __tweets__
- we want to regress the _probability_ that a certain tweet (node) is __spreading fake news__
- model would associate a number $\in [0,1]$ to that noode

We can use GNNs to help beat cancer! By taking the protein/gene and drug interaction data and modeling the problem as a graph and throwing some GNN weaponry at it we can __find a list of molecules that help beat/prevent cancer__.

Once we have the most effective CBMs (cancer-beating molecules) we can create a map of foods that are richest in these CBMs, also called hyperfoods.

- You can treat transformers as Graph Attention Networks operating on fully-connected graphs and you can treat images/videos as regular graphs (aka grids).

## 1.4. Properties of Graphs 
### 1.4.1. Homopholic 
= connected nodes share a similar label

- “homos”-same
- “philie”- love/friendship