import urllib.request

# Official Gutenberg URL for The Federalist Papers
url = "https://www.gutenberg.org/cache/epub/18/pg18.txt"
filename = "federalist_papers.txt"

# Download the raw plain text file
print("Downloading...")
urllib.request.urlretrieve(url, filename)
print(f"Saved successfully as '{filename}'!")