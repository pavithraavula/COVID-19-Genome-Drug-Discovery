# COVID-19-Genome-Drug-Discovery

This project explores the potential of using **COVID-19 genome sequencing** data to identify **drug targets** by analyzing viral mutations and their similarities. The goal is to develop a pipeline for predicting the most promising subsequences in the virus's genome that can be targeted for therapeutic drug development.

## Project Overview

In this project, we:
1. **Compare genome sequences** of different **COVID-19 mutations** to identify similarities.
2. **Reduce dimensionality** using **Principal Component Analysis (PCA)**.
3. **Apply K-means clustering** to find patterns in the data and trace the virus's lineage.
4. **Predict sequence similarity** to identify potential drug targets.

The aim is to help identify regions of the genome that can be used to develop drugs to fight the virus.

## Steps Involved

1. **Data Preprocessing**: Clean and scale genome sequence data for analysis.
2. **Principal Component Analysis (PCA)**: Reduce the number of features to focus on the most important ones.
3. **K-Means Clustering**: Group similar sequences together to trace the virus's mutations and lineage.
4. **Machine Learning Models**: Use regression models to predict bit scores, which reflect the similarity between sequences.

Conclusion
By analyzing viral genome sequences, reducing data complexity, and applying machine learning techniques, this project aims to identify key subsequences that could be targeted for developing effective COVID-19 drugs.
