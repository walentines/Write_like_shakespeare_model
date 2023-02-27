from time import sleep
from progress.bar import Bar

with Bar('Processing...') as bar:
    for i in range(100):
        bar.next()