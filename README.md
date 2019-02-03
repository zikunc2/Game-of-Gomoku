<strong>Goal: </strong>

Naive Bayes Classification on visual pattern(digit)

<strong>Training: </strong>

likelihood - calculated for every pixel locations of a given digit, Laplace smoothing is applied (constant k)

prior - calculated for every digits

<strong>Estimation: </strong>

maximum a posteriori (MAP) classification

<strong>Data Set:</strong>

Digit - 0 represent backgound, 1 represent foreground, every 33 line one digit: 32 lines of binary pixels + last line comtain the corresponding digit

Face Image Set - space represent background, # represent edge, Size of one face image: 70 * 60

Face Tag Set - tagSet[index] = 0 if imageSet[index] is not a face, tagSet[index] = 1 if imageSet[index] is a face
