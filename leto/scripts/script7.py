from matplotlib import pyplot as plt
import pandas as pd

word_count = pd.read_csv("leto/enron_emails/susan.csv")["word_count"]

plt.hist(word_count, bins=100)
plt.show()