# Introduction

Attempts to break down text from various data sources into chunks and then ingests the data into Elastic Search. The chunking algorithm favors natural paragraphs if it can fit into the max sequence length of the model. If it cannot, it attempts to break the text down into sentences and combines up to 3 sentences together at a time. 

Data sources are stored in a ./data repository. This is just a collection of data repos used by the chatbot. 