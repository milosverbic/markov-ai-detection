import email
import re
import time
from collections import Counter
import pandas as pd

t0 = time.time()


def extract_pure_body(raw_email_str):
    """Parses raw RFC 822 email text, extracts sender, and strips all headers,

    forwarded text, reply blocks, non-breaking space artifacts, and quote lines.
    """
    if not isinstance(raw_email_str, str):
        return None, ""

    msg = email.message_from_string(raw_email_str)
    sender = msg.get("From", "")
    if sender:
        sender = sender.strip().lower()

    body = msg.get_payload()
    if not isinstance(body, str):
        return sender, ""

    # Clean non-breaking spaces (\xa0) common in Enron date/time headers
    body = body.replace("\xa0", " ")

    # Truncate at forwarded or reply header markers
    split_patterns = [
        r"(?:^|\n)\s*-{5,}\s*Forwarded by.*",
        r"(?:^|\n)\s*-{5,}\s*Original Message\s*-{5,}",
        r"(?:^|\n)\s*From:\s*",
        r"(?:^|\n)\s*Sent:\s*",
        r"(?:^|\n)\s*To:\s*",
        r"(?:^|\n)\s*>+",
    ]

    for pattern in split_patterns:
        body = re.split(pattern, body, flags=re.IGNORECASE)[0]

    # Strip inline quoted lines starting with '>' or '|'
    lines = body.splitlines()
    clean_lines = [
        line for line in lines if not line.strip().startswith((">", "|"))
    ]
    clean_body = "\n".join(clean_lines).strip()

    return sender, clean_body


def is_authored_by_owner(file_path, sender):
    """Checks if the email path is in a sent folder and matches the sender's identity."""
    file_path = str(file_path).lower()
    sender = str(sender).lower()

    if not any(
        sent_dir in file_path
        for sent_dir in ["/sent/", "/sent_items/", "/_sent_mail/"]
    ):
        return False

    mailbox_owner = file_path.split("/")[0]
    owner_parts = [part for part in mailbox_owner.split("-") if len(part) > 1]
    return any(part in sender for part in owner_parts)


def get_top_n_words_whitespace_split(csv_path, top_n=50, lower_case=True):
    """Counts top words by splitting purely on whitespace (spaces and newlines),

    stripping outer punctuation while ignoring numeric tokens.
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    print("Extracting headers and pure email text...")
    parsed_data = df["message"].apply(extract_pure_body)
    df["sender"] = [item[0] for item in parsed_data]
    df["pure_body"] = [item[1] for item in parsed_data]

    # 1. Filter missing senders and empty bodies
    df = df.dropna(subset=["sender"])
    df = df[df["pure_body"].str.strip() != ""]

    # 2. Strict authorship verification
    print("Applying authorship verification filter...")
    authorship_mask = df.apply(
        lambda row: is_authored_by_owner(row["file"], row["sender"]), axis=1
    )
    df = df[authorship_mask].copy()

    # 3. Deduplication
    print("Deduplicating duplicate bodies...")
    df = df.drop_duplicates(subset=["sender", "pure_body"], keep="first")

    print(
        f"Processing word frequencies across {len(df)} unique authored emails..."
    )
    word_counter = Counter()

    for text in df["pure_body"]:
        # Standardize curly apostrophes
        text = text.replace("’", "'")

        if lower_case:
            text = text.lower()

        # Split strictly on spaces and newlines
        tokens = text.split()

        clean_tokens = []
        for token in tokens:
            # Strip surrounding punctuation (quotes, commas, periods, brackets, colons, etc.)
            clean_token = token.strip('.,!?:;()"[]{}<>*#~`-')

            # Ignore empty tokens, pure digits/numbers, or currency values
            if (
                clean_token
                and not clean_token.replace(".", "", 1).isdigit()
                and not clean_token.isdigit()
            ):
                clean_tokens.append(clean_token)

        word_counter.update(clean_tokens)

    # Convert to DataFrame
    top_words_df = pd.DataFrame(
        word_counter.most_common(top_n), columns=["Word", "Frequency"]
    )

    return top_words_df


# Example usage:
top_words = get_top_n_words_whitespace_split(
    "leto/emails.csv", top_n=200, lower_case=True
)
print("\nTop Most Common Words (Whitespace Split):")
print(top_words.to_string(index=False))

print(f"\nCompleted in: {time.time() - t0:.2f} seconds")