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
csv_list = ["leto/enron_emails/kay_mann.csv"]  # Replace with your list of CSV paths
corpus = IndexedCorpus(eof_marker="EOF")
corpus.load_from_csvs(csv_list, text_column="pure_text")
corpus.save_raw_corpus("leto/enron_emails/kay_mann.txt")

# csv_list2 = ["leto/enron_emails/debra_perlingiere.csv"]  # Replace with your list of CSV paths
# corpus2 = IndexedCorpus(eof_marker="EOF")
# corpus2.load_from_csvs(csv_list2, text_column="pure_text")
# corpus2.save_raw_corpus("leto/enron_emails/debra_perlingiere.txt")

# csv_list3 = ["leto/enron_emails/sara_shackleton.csv"]  # Replace with your list of CSV paths
# corpus3 = IndexedCorpus(eof_marker="EOF")
# corpus3.load_from_csvs(csv_list3, text_column="pure_text")
# corpus3.save_raw_corpus("leto/enron_emails/sara_shackleton.txt")

# csv_list4 = ["leto/enron_emails/chris_germany.csv"]  # Replace with your list of CSV paths
# corpus4 = IndexedCorpus(eof_marker="EOF")
# corpus4.load_from_csvs(csv_list4, text_column="pure_text")
# corpus4.save_raw_corpus("leto/enron_emails/chris_germany.txt")

# csv_list5 = ["leto/enron_emails/gerald_nemec.csv"]  # Replace with your list of CSV paths
# corpus5 = IndexedCorpus(eof_marker="EOF")
# corpus5.load_from_csvs(csv_list5, text_column="pure_text")
# corpus5.save_raw_corpus("leto/enron_emails/gerald_nemec.txt")

# csv_list6 = ["leto/enron_emails/mark_taylor.csv"]  # Replace with your list of CSV paths
# corpus6 = IndexedCorpus(eof_marker="EOF")
# corpus6.load_from_csvs(csv_list6, text_column="pure_text")
# corpus6.save_raw_corpus("leto/enron_emails/mark_taylor.txt")

# csv_list7 = ["leto/enron_emails/vince_kaminski.csv"]  # Replace with your list of CSV paths
# corpus7 = IndexedCorpus(eof_marker="EOF")
# corpus7.load_from_csvs(csv_list7, text_column="pure_text")
# corpus7.save_raw_corpus("leto/enron_emails/vince_kaminski.txt")

# csv_list8 = ["leto/enron_emails/tana_jones.csv"]  # Replace with your list of CSV paths
# corpus8 = IndexedCorpus(eof_marker="EOF")
# corpus8.load_from_csvs(csv_list8, text_column="pure_text")
# corpus8.save_raw_corpus("leto/enron_emails/tana_jones.txt")


# Read 50 words starting from exactly word 5000
# snippet = corpus.read_from_word(start_word_idx=5000, num_words=50)
# print("\n--- Snippet starting at Word 5000 ---")
# print(snippet)