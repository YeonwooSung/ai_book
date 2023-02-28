# IoT Bot

[link](https://research.unsw.edu.au/projects/bot-iot-dataset)

The BoT-IoT dataset was created by designing a realistic network environment in the Cyber Range Lab of UNSW Canberra. The network environment incorporated a combination of normal and botnet traffic. The dataset’s source files are provided in different formats, including the original pcap files, the generated argus files and csv files. The files were separated, based on attack category and subcategory, to better assist in labeling process.

The captured pcap files are 69.3 GB in size, with more than 72.000.000 records. The extracted flow traffic, in csv format is 16.7 GB in size. The dataset includes DDoS, DoS, OS and Service Scan, Keylogging and Data exfiltration attacks, with the DDoS and DoS attacks further organized, based on the protocol used.

To ease the handling of the dataset, we extracted 5% of the original dataset via the use of select MySQL queries. The extracted 5%, is comprised of 4 files of approximately 1.07 GB total size, and about 3 million records.

## Notebooks

- [Simple Classification](./notebooks/Classification.ipynb)
- [Classification with best 6 features](./notebooks/ClassificationWIthBest6Features.ipynb)
- [Classification with best 8 features](./notebooks/ClassificationWIthBest8Features.ipynb)
