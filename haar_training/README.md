# HAAR Training 

## Create training set
```bash
opencv_createsamples -info annotations.txt -num 524 -w 16 -h 16 -vec positives.vec
opencv_createsamples -info annotations.txt -num 327 -w 20 -h 20 -vec positives.vec -maxxangle 0.1 -maxyangle 0.1 --maxzangle 0.1 
```

### Train
```bash
opencv_traincascade -data classifier -vec positives.vec -bg negatives.txt -numStages 10 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 1800 -numNeg 5000 -w 16 -h 16
opencv_traincascade -data data -vec positives.vec -bg negatives2.txt -numStages 5 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 5000 -numNeg 5000 -w 20 -h 20
```