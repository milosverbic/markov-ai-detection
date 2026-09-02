from pathlib import Path
import pandas as pd


class IndexedCorpus:

    def __init__(self, eof_marker="EOF"):
        self.eof_marker = eof_marker
        self.tokens = []

    def load_from_csvs(self, csv_paths, text_column="pure_text"):
        """Loads texts from CSV files, joins them with EOF markers, and tokenizes."""
        all_texts = []

        for p in csv_paths:
            path = Path(p)
            if not path.exists():
                print(f"Skipping missing file: {path}")
                continue

            df = pd.read_csv(path)
            if text_column in df.columns:
                # Drop empty or missing text entries
                valid_texts = df[text_column].dropna().astype(str).tolist()
                all_texts.extend(valid_texts)

        # Join texts with the EOF marker surrounded by spaces
        separator = f" {self.eof_marker} "
        full_raw_text = separator.join(all_texts)

        # Split strictly on spaces/newlines into a single token array
        self.tokens = full_raw_text.split()
        print(
            f"Corpus loaded: {len(all_texts)} texts, {len(self.tokens):,} total words."
        )

    def read_from_word(self, start_word_idx, num_words=100):
        """Returns a string starting at word index `start_word_idx` for `num_words`."""
        if start_word_idx < 0 or start_word_idx >= len(self.tokens):
            return ""

        end_idx = min(start_word_idx + num_words, len(self.tokens))
        return " ".join(self.tokens[start_word_idx:end_idx])

    def save_raw_corpus(self, output_file_path):
        """Saves the combined single-string text to a file."""
        out_path = Path(output_file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(" ".join(self.tokens), encoding="utf-8")
        print(f"Saved combined raw text to {out_path}")


# Example usage:
csv_list = ["leto/enron_emails/sally.csv"]  # Replace with your list of CSV paths
corpus = IndexedCorpus(eof_marker="EOF")

corpus.load_from_csvs(csv_list, text_column="pure_text")

corpus.save_raw_corpus("leto/enron_emails/sally.txt")

csv_list2 = ["leto/enron_emails/susan.csv"]  # Replace with your list of CSV paths
corpus2 = IndexedCorpus(eof_marker="EOF")

corpus2.load_from_csvs(csv_list2, text_column="pure_text")

corpus2.save_raw_corpus("leto/enron_emails/susan.txt")



# Read 50 words starting from exactly word 5000
# snippet = corpus.read_from_word(start_word_idx=5000, num_words=50)
# print("\n--- Snippet starting at Word 5000 ---")
# print(snippet)