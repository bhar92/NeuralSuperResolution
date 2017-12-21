#! /bin/bash

mkdir data
cd data
read -p "Downloading the Microsoft Coco dataset.. File size is 13.5GB! Proceed? [y/n] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
	wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
	unzip train2014.zip
fi

# Copy 10k images to data_dir
mkdir data_dir
i=0

for fname in $(find . -name *.jpg); do
	cp $fname data_dir/${i}_img.jpg
	i=$((i+1))
	if [[ $i -gt 10000 ]]; then
		break
	fi
done
