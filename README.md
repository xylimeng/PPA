# PPA
Principal Parcellation Analysis for Brain Connectomes and Multiple Traits

## Step 1: extract endpoints from fiber tracking results
```
Input: fiber tracking files
Usage: python PPA_Step_1.py
Output: endpoints files; number of fibers in each subject
```

## Step 2: mini_batch K-means on fiber endpoints
```
Input: endpoints files
Usage: python PPA_Step_2.py 100
       (100 is the number of clusters; users can adjust this value)
Output: cluster results
```

## Step 3: LASSO regression
```
Input: cluster results
Usage: python PPA_Step_3.py 100 PicVocab
       (100 is the number of clusters, Picvocab specifies the selected human trait; users can adjust their values)
Output: prediction model
```
