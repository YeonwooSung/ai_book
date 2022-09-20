from extractor import PacSumExtractor
from datasets.processing_funcs import PIPELINE
from datasets.fbp_dataset import load_texts, FBPSummaryDataset

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["tune", "test"], help="tune or test"
    )
    parser.add_argument(
        "--extract_num", type=int, default=3, help="number of extracted sentences"
    )
    parser.add_argument(
        "--bert_config_file",
        type=str,
        default="/disk/scratch1/s1797858/bert_model_path/uncased_L-12_H-768_A-12/bert_config.json",
        help="bert configuration file",
    )
    parser.add_argument("--bert_model_file", type=str, help="bert model file")
    parser.add_argument(
        "--bert_vocab_file",
        type=str,
        default="/disk/scratch1/s1797858/bert_model_path/uncased_L-12_H-768_A-12/vocab.txt",
        help="bert vocabulary file",
    )

    parser.add_argument("--beta", type=float, default=0.0, help="beta")
    parser.add_argument("--lambda1", type=float, default=0.0, help="lambda1")
    parser.add_argument("--lambda2", type=float, default=1.0, help="lambda2")

    parser.add_argument(
        "--tune_data_path", type=str, help="data for tunining hyperparameters"
    )
    parser.add_argument("--test_data_path", type=str, help="data for testing")

    args = parser.parse_args()
    print(args)

    extractor = PacSumExtractor(
        model_name_or_path=args.bert_model_file,
        beta=args.beta,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
    )
    # tune
    if args.mode == "tune":
        texts, _ = load_texts(args.tune_data_path, PIPELINE, 0.01)
        tune_dataset = FBPSummaryDataset(texts)
        extractor.tune_hparams(tune_dataset)

    # test
    texts, _ = load_texts(args.test_data_path, PIPELINE, 0.01)
    test_dataset = FBPSummaryDataset(texts)
    extractor.extract_summary(test_dataset)
