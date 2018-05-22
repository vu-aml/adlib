#!/bin/bash

#script to create the index file needed for ENRON email dataset
#used in the enron directory
mkdir index_dir

for file in ham/*
do
    echo "ham ../${file}" >> index_dir/index
done
for file in spam/*
do 
	echo "spam ../${file}" >> index_dir/index
done
