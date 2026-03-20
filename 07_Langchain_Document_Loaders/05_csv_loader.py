from langchain_community.document_loaders import CSVLoader

# creates document object for every row 

loader = CSVLoader(file_path='05_Social_Network_Ads.csv')

data = loader.load()

print(data[1])