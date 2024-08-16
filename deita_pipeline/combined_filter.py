import logging
import argparse
from deita.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--other_data_path", type=str, default=None)
parser.add_argument("--threshold", type=float, default=0.9)
parser.add_argument("--data_size", type=int, default=10)
parser.add_argument("--chunk_size", type=int, default=100000)
parser.add_argument("--sort_key", type=str, default="ifd_ppl")
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--distance_metric", type=str, default="cosine")
parser.add_argument("--embedding_field", type=str, default="embedding")
parser.add_argument("--is_compression", type=bool, default=False)

args = parser.parse_args()

filter_pipeline = Pipeline("filter_pipeline",
                          data_path = args.data_path,  # json file with sharegpt format
                          other_data_path = args.other_data_path,  # embedding file path (pickle format)
                          threshold = args.threshold,  # filter threshold default: 0.9
                          data_size = args.data_size,  # size of selected data
                          chunk_size = args.chunk_size,  # used for more efficient GPU computing  default: 100000
                          sort_key = args.sort_key,  # default: "complexity_scores,quality_scores"
                          output_path = args.output_path,  # json format output path
                          distance_metric = args.distance_metric,  # default: cosine
                          embedding_field = args.embedding_field,  # default: embedding
                          is_compression = args.is_compression)  # default: False)

filter_pipeline.run()