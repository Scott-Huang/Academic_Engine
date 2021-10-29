# Academic Engine
Draft work for establishing an academic search engine for users with no prior knowledge about the searching concept to better locate papers they are looking for and provide a general elaboration of the problem they are dealing with.
***

### Environment
- `python==3.7`
### Used Packages
- Whoosh
- SpaCy
- Gensim
- Flask
***

## Data
The dataset is a mirror of the original ArXiv data, which can be downloaded in [Kaggle](https://www.kaggle.com/Cornell-University/arxiv).

The computer science keywords set can be downloaded [here](https://drive.google.com/file/d/1edIDjsOCUi3htZbAk2ePL-6lCBn-M4s3/view?usp=sharing).
***

## Usage
The `concept_search.ipynb` notebook generates an engine storing the keywords appeared in each abstract, and provides a ranking function to display the search results.

The `context_search.ipynb` notebook is a draft work trying to provide an explanation of the relation between a query and searched concepts.

The `word2vec.ipynb` notebook generates a word and phrase embedding model.

After finishing all precomputations to set up the draft web engine, run `app.py`, inside the folder `web_engine`, to set up the testing server of the search engine.
***

### A short demo
[Here](https://youtu.be/Oa8guSThoUk) is a short demo for the interface of the engine.
