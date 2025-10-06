# main.py
import argparse
from config import SOURCE_FOLDER
from processing import RAGPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-folder", "-s", default=SOURCE_FOLDER)
    args = parser.parse_args()

    rag = RAGPipeline(args.source_folder)
    rag.run = lambda : rag.build_index(rag.tag_chunks(rag.process_files()))  # Inline run method
    rag.run()
