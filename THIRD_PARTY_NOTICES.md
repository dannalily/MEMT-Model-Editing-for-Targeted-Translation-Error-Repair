# Third-party notices

The MEMT release contains or relies on the following third-party components and
datasets. Please follow their original terms in addition to the MEMT license.

## Data sources

- **IdiomKB**: idiom inventory and meanings used during benchmark construction.
  Repository: https://github.com/lishuang-w/IdiomKB
- **FLORES-200**: the release contains the fixed shuffled 1,000-example subset
  used for locality evaluation. Source: https://huggingface.co/datasets/facebook/flores
- **MMMLU**: the release contains the fixed shuffled 1,000-example subset used
  for reasoning/locality evaluation. Source: https://huggingface.co/datasets/openai/MMMLU

## Software and algorithms

The editing implementations are based on publicly described methods including
FT, ROME, MEMIT, AlphaEdit, UNKE, WISE, and GRACE. Consult the corresponding
papers and upstream implementations when redistributing or modifying those
components.

MetricX is an external evaluation package and should be installed from its
upstream repository as described in the documentation.

## Generated content

The MEMT benchmark examples, translations, and judgments were generated or
filtered using Claude. The dataset is released for non-commercial research and
evaluation use only under CC BY-NC 4.0.
