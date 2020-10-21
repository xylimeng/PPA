# PPA
Principal Parcellation Analysis for Brain Connectomes and Multiple Traits

## Step 1: extract the ending points from fiber tracking results
```
Input: fiber tracking files
Usage: python PPA_Step_1.py
Output: end points files; number of fibers in each subject
```

## Step 2: mini_batch K-means on ending points
```
Input: end points files
Usage: python PPA_Step_2.py 100
       (100 is number of clusters, users could adjust here)
Output: cluster results
```

## Step 3: LASSO regression
```
Input: cluster results
Usage: python PPA_Step_3.py 100 PicVocab
       (100 is number of clusters, Picvocab is one human trait name, users could adjust here)
Output: prediction results
```
