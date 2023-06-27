# Image_puzzle
Solving jumbled image puzzles using DEEP Learning

Data:
    -contains 6*6 jumbled images of 2 classes(faces, landmarks)
    -and csv corresponding to each image-id containing original position of the 36 tiles
Models:
    -Conv 2D lstm
    -Deep permumation network model
    -efficient-netB0 model
Final approach:
    -Constructed image corresponding to each test image by ranking neighbouring tiles to left, right, top and bottom by considering pearson   corellation with a combination of distance as a metric  

    