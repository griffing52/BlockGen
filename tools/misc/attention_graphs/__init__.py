"""Can an LLM's self-attention be read as an adjacency matrix over the parts of a build?

See README.md. Modules: ``scenes`` (structures + ground-truth graphs), ``serialize``
(one part per line, with token spans), ``extract`` (attention + node pooling),
``adjacency`` (graphs, baselines, metrics), ``probe`` (trained edge probes),
``viz`` (figures), ``experiment`` (the whole sweep), ``__main__`` (CLI).
"""
