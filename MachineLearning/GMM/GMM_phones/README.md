## Example of GMMs for phonemes
Source: https://raw.githubusercontent.com/sknadig/ASR_2018_T01/master/README.md
To use download TIMIT dataset (example is in TIMIT folder)

## Data pre processing
Run ```python import_timit.py --timit=./TIMIT --preprocessed=False```
to compute the features and store them in a folder.
This script also converts the [NIST "SPHERE" file format](https://www.isip.piconepress.com/projects/speech/software/tutorials/production/fundamentals/v1.0/section_02/s02_01_p04.html) to [WAVE PCM format](http://soundfile.sapp.org/doc/WaveFormat/).
If you have already converted the files, set ```--preprocessed=True``` to skip the conversion process.

## References:
- Mel Frequency Cepstral Coefficient (MFCC) tutorial :
    - [http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
- TIMIT related documents: 
    - [https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir4930.pdf](https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir4930.pdf) 
    - [https://github.com/philipperemy/timit](https://github.com/philipperemy/timit)
- Implementation references:
    - [http://scikit-learn.org/stable/index.html](http://scikit-learn.org/stable/index.html)
    - [http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
    - [http://www.pitt.edu/~naraehan/python2/pickling.html](http://www.pitt.edu/~naraehan/python2/pickling.html)
    - [https://github.com/belambert/asr-evaluation](https://github.com/belambert/asr-evaluation)
## This repo contains
- [x] Code to read files and compute MFCC features
- [x] Computing MFCC for time slices given in .PHN files
- [x] Dumping computed features to a folder
- [x] Dumping phone-wise features to a folder
- [x] GMM training
- [x] GMM model dumping
- [x] GMM evaluation
- [x] PER computation

