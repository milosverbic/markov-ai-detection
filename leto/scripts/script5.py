import email
import re
import time
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


def process_enron_deduplicated_stats(csv_path):
    df = pd.read_csv(csv_path)

    # 1. Extract sender and cleaned body
    parsed_data = df["message"].apply(extract_pure_body)
    df["sender"] = [item[0] for item in parsed_data]
    df["pure_body"] = [item[1] for item in parsed_data]

    # 2. Filter missing senders and empty bodies
    df = df.dropna(subset=["sender"])
    df = df[df["pure_body"].str.strip() != ""]

    # 3. Apply strict authorship attribution verification
    authorship_mask = df.apply(
        lambda row: is_authored_by_owner(row["file"], row["sender"]), axis=1
    )
    df = df[authorship_mask].copy()

    # 4. DEDUPLICATION: Remove duplicate pure bodies sent by the same author
    df = df.drop_duplicates(subset=["sender", "pure_body"], keep="first")

    # 5. Calculate word counts on unique emails
    df["word_count"] = df["pure_body"].apply(lambda text: len(text.split()))

    # 6. Group by sender and aggregate unique email count and total words
    stats = (
        df.groupby("sender")
        .agg(
            unique_authored_emails=("pure_body", "count"),
            total_words=("word_count", "sum"),
        )
        .reset_index()
    )

    # Sort by total unique authored emails descending
    stats = stats.sort_values(by="unique_authored_emails", ascending=False)
    return stats


# Example usage:
stats_df = process_enron_deduplicated_stats("leto/emails.csv")
print(stats_df.head(25))

print(f"\nExecution time: {time.time() - t0:.2f} seconds")