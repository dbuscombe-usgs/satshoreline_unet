

for file in *.tif; do gdal_translate -of JPEG -scale -co worldfile=yes $file "${file%.tif}.jpg"; done
rm *.wld
rm *.xml
rm *.tif
#72 x 305
for file in *.jpg; do convert $file -crop 100%x25% +repage "${file%.jpg}.jpg"; done

#
# cd labels
# for file in *.png; do convert $file -crop 100%x25% +repage "${file%.png}.jpg"; done
# rm *-4.jpg
# rm *.png
#
# for file in *.jpg; do cp $file ${file:0:40}${file:75:80}; done
#
#
# for file in *.tif; do gdal_translate -of JPEG -scale -co worldfile=yes $file "${file%.tif}.jpg"; done
# for file in *.jpg; do convert $file -crop 100%x25% +repage "${file%.jpg}.jpg"; done
