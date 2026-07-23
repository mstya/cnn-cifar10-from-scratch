import kagglehub

kagglehub.login()

path = kagglehub.competition_download('cifar-10')

print("Path to competition files:", path)