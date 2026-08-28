# MEMT data card

MEMT is a multilingual benchmark for targeted model editing in machine translation. The data were generated and filtered from English and Chinese idioms, with translations into English, Chinese, Japanese, Arabic, French, and German.

## Included subsets

- `ParaIdiomSent`: primary idiom-containing sentences and reference translations
- `ParaIdiomSent_generality`: three in-scope generalization examples per idiom
- `ParaIdiomSent_locality`: three out-of-scope locality examples per idiom
- `Editing_Samples`: model-specific translation errors used as editing descriptors
- `Keep_Samples`: retained correct samples
- `idiom_span_en2x.json`, `idiom_span_zh2x.json`: idiom span annotations
- `locality_flores_shuffle1000.json`: fixed shuffled FLORES locality subset
- `locality_MMMLU_shuffle1000.json`: fixed shuffled MMMLU locality subset

## Important provenance

The primary data were constructed using IdiomKB and generated with Claude Sonnet 4. The editing samples contain outputs from the evaluated translation models. The repository does not contain model weights.

The release includes only the fixed shuffled FLORES and MMMLU subsets needed to reproduce the paper's locality and reasoning evaluations. They remain third-party-derived data; users must comply with the original terms and attribution requirements.

## Known limitations

Some translations and quality labels were generated or judged by language models. The benchmark should therefore be treated as a research evaluation resource, not as human-certified translation data.

## License

The dataset is released for non-commercial research and evaluation under
[CC BY-NC 4.0](LICENSE). Users must provide attribution and comply with the
terms of any third-party source data. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
