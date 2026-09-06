filename = input()
with open("leto/enron_emails/"+filename+".txt") as f:
    print(len(f.read().split()))