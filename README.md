Goal: 
Naive Bayes Classification on visual pattern(digit)

Training: 
likelihood - calculated for every pixel locations of a given digit, Laplace smoothing is applied (constant k)
prior - calculated for every digits

Estimation: maximum a posteriori (MAP) classification

Data Set:
Digit - 0 represent backgound, 1 represent foreground, every 33 line one digit: 32 lines of binary pixels + last line comtain the corresponding digit
Face Image Set - space represent background, # represent edge, Size of one face image: 70 * 60
Face Tag Set - tagSet[index] = 0 if imageSet[index] is not a face, tagSet[index] = 1 if imageSet[index] is a face

Source of Data Set:
facedata & digitdata - https://courses.engr.illinois.edu/ece448/sp2018/mp3/mp3.html

To Run:
python naive.py
python face.py
