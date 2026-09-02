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


def export_multiple_authors_deduplicated(
    csv_file_path, author_emails, output_prefix="author2"
):
    """Parses dataset once, strictly verifies authorship, deduplicates pure texts,

    and exports clean CSV/JSON files per author.
    """
    target_authors = {email.strip().lower() for email in author_emails}

    print(f"Loading dataset from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    print(f"Processing email bodies for {len(target_authors)} authors...")
    parsed = df["message"].apply(extract_pure_body)
    df["sender"] = [item[0] for item in parsed]
    df["pure_text"] = [item[1] for item in parsed]

    # 1. Filter by target authors and non-empty pure texts
    df = df[df["sender"].isin(target_authors)].copy()
    df = df[df["pure_text"].str.strip() != ""]

    # 2. Strict authorship verification
    authorship_mask = df.apply(
        lambda row: is_authored_by_owner(row["file"], row["sender"]), axis=1
    )
    df = df[authorship_mask].copy()

    # 3. DEDUPLICATION: Keep only the first occurrence of identical pure text per author
    df = df.drop_duplicates(subset=["sender", "pure_text"], keep="first")

    # Calculate word counts
    df["word_count"] = df["pure_text"].apply(lambda text: len(text.split()))

    # 4. Export per author
    exported_dfs = {}
    for author in target_authors:
        author_df = df[df["sender"] == author][
            ["file", "pure_text", "word_count"]
        ].reset_index(drop=True)

        clean_name = author.split("@")[0].replace(".", "_")
        csv_out = f"{output_prefix}_{clean_name}.csv"
        json_out = f"{output_prefix}_{clean_name}.json"

        author_df.to_csv(csv_out, index=False, encoding="utf-8")
        author_df.to_json(
            json_out, orient="records", indent=2, force_ascii=False
        )

        exported_dfs[author] = author_df
        print(
            f"Exported {len(author_df)} unique emails for {author} -> {csv_out}, {json_out}"
        )

    return exported_dfs


# Example Usage:
authors_to_extract = [
    "sally.beck@enron.com",
    "susan.scott@enron.com",
]

exported_data = export_multiple_authors_deduplicated(
    "leto/emails.csv", authors_to_extract
)
print(f"\nCompleted in: {time.time() - t0:.2f} seconds")