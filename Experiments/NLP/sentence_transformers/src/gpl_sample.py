import gpl
import jsonlines


data = [
    'TSDAE (Tranformer and Sequential Denoising AutoEncoder) is one of most popular unsupervised training method. The main idea is reconstruct the original sentence from a corrupted sentence. The TSDAE model consists of two parts: an encoder and a decoder.',
    ' TSDAE encodes corrupted sentences into fixed-sized vectors and requires the decoder to reconstruct the original sentences from this sentence embedding.',
]


def prepare_data_for_gpl(data):
    gpl_data = []
    counter = 1
    for i in data:
        gpl_data.append({
            "_id": str(counter),
            "title": "",
            "text": i,
            "metadata": {}
        })
        counter+=1
    return gpl_data


gpl_data = prepare_data_for_gpl(data)
with jsonlines.open('/content/jd-data/corpus.jsonl', 'w') as writer:
    writer.write_all(gpl_data)
dataset = 'jd-data'


gpl.train(
    path_to_generated_data=f"generated/{dataset}",
    base_ckpt="thenlper/gte-base",  
    # The starting checkpoint of the experiments in the paper
    gpl_score_function="dot",# Note that GPL uses MarginMSE loss, which works with dot-product
    batch_size_gpl=4,
    gpl_steps=3,
    new_size=-1, # Resize the corpus, -1 means take all data
    queries_per_passage=3,# Number of Queries Per Passage (QPP)
    output_dir=f"output/{dataset}",
    evaluation_data=f"./{dataset}",
    evaluation_output=f"evaluation/{dataset}",
    generator="BeIR/query-gen-msmarco-t5-base-v1",
    retrievers=["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
    retriever_score_functions=["cos_sim", "cos_sim"],
    # Note that these two retriever model work with cosine-similarity
    cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
    qgen_prefix="qgen",
    # This prefix will appear as part of the (folder/file) names for query-generation results: For example, we will have "qgen-qrels/" and "qgen-queries.jsonl" by default.
    do_evaluation=False,
    # use_amp=True   # One can use this flag for enabling the efficient float16 precision
)
