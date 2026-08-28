# Data layout

Download the MEMT dataset from the Hugging Face dataset repository listed in the root README and make it available as `MEMT/` in the experiment working directory.

The editing entry points expect paths such as:

```text
MEMT/Editing_Samples/Qwen2-5-3B-Instruct/en2x.json
MEMT/ParaIdiomSent_generality/en2x.json
MEMT/ParaIdiomSent_locality/en2x.json
MEMT/idiom_span_en2x.json
```

The release includes the fixed shuffled subsets used in the paper:

```text
MEMT/locality_flores_shuffle1000.json
MEMT/locality_MMMLU_shuffle1000.json
```

Use these files as provided. Do not reshuffle them when reproducing the
reported results. They are derived from the original FLORES and MMMLU datasets;
see `THIRD_PARTY_NOTICES.md` for attribution and source links.
