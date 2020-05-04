This is a README file for the paper "Adaptive Complex Word Identificationthrough False Friend Detection", submitted to UMAP 2020.

The [fastText wrapper library](https://github.com/Babylonpartners/fastText_multilingual) (included in this folder) is developed by [babylon health](https://github.com/Babylonpartners).

In order to replicate our experiments, you need to download the vectors from here: https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md (download the vectors in text format)
If the link is not valid any more, you can probably find the new link at [the fastText project page](https://github.com/facebookresearch/fastText).

To align vectors, run `align_new_languages.py`.

```
usage: align_new_languages.py [-h] lang1 lang2 out1 out2 dict

positional arguments:
  lang1       Language 1 (modified)
  lang2       Language 2 (not modified)
  out1        Output vectors for lang1
  out2        Output vectors for lang2
  dict        Dictionary used to align the languages (format: lang1 [tab]
              lang2)

optional arguments:
  -h, --help  show this help message and exit
```

For `lang2` parameter, the pivot language should be selected (Italian, in our case).
For `dict` parameter, you can use the `Dict_*` files provided in the `data` folder.

Example:
```
python3 align_new_languages.py cc.fr.300.vec cc.it.300.vec fr.vec it.vec data/Dict_it_fr_filtered.txt
```

After the new vectors have been created, you should run the second script `cosine_calc.py`.

```
usage: cosine_calc.py [-h] lang skip_lang lang_p_vec lang_vec synonyms test output

positional arguments:
  lang        Language 1 (modified)
  skip_lang   Language to skip
  lang_p_vec  Input vectors for pivot language
  lang_vec    Input vectors for lang
  synonyms    Synonyms TSV file
  test        Test file
  output      Output folder

optional arguments:
  -h, --help  show this help message and exit
```

In this case, `lang` is the language (fr, en, it). Words in `skip_lang` test file are removed from the training vectors.

Example:
```
python3 cosine_calc.py fr en data/it.vec data/fr.vec data/synonyms.txt data/test-en.txt output
```

In the output folder, you'll find the training file for libSVM.

Examples:
```
svm-train -t [type] -c [cost] -e [epsilon] output-folder/skipfr_fr_nosyn_vec.txt models-folder/skipfr_fr_nosyn_vec.model
svm-predict output-folder/skipfr_en_nosyn_vec.txt models-folder/skipfr_fr_nosyn_vec.model output-folder/skipfr_fr_nosyn_vec.out | tee log-folder/skipfr_fr_nosyn_vec.log
```

### Data

The data folder contains some resources required to run our tool:

- Dict_it_en_filtered.txt is a TSV two-column file containing word pairs in Italian and English. It is used to align the multilingual embeddings.
- Dict_it_fr_filtered.txt, same as above, with pairs in Italian and French.
- synonyms.txt is a TSV file containing Italian synonyms. The first column contains the source word, the second column contains a comma-separated list of synonyms for it. This was used to extract some of the features for the classifier (not mandatory).
- test-en.txt is a TSV file that contains manually annotated data in English. In the first column one can find the annotation ("co" for cognates, "ff" for false friends) that describes the relation between the following two columns. This file was used for training the classifier with ten-fold cross-validation, and as training/test set for the cross-language experiments.
- test-fr.txt, same as above, in French.
