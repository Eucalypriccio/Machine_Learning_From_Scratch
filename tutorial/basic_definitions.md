# What is Machine Learning?

> "Field of study that gives computers the ability to learn without explicitly programmed." ---Arthur Samuel

---

## Classification

### Supervised Learning

get inputs `x`
->
give output labels `y`

the learning algorithms learn from being given "right answers"

**e.g.1**: regression---housing price prediction

given many pairs of house size(x) -> price(y)

to get an algorithm to systematically choose the most appropriate line, curve or other thing to fit to the dataset

* predict a number
* infinitely many possible outputs

**e.g.2** classification---cancer detection

to decide whether a lump is **benign** or **malignant** ?

"0": benign

"1": malignant

|tumor size|diagnosis|
|-|-|
|2|0|
|5|1|
|1|0|
|7|1|
|...|...|

* predict categories
* small number of possible outputs

### Unsupersived Learning

get inputs `x` \
but no output labels

let the algorithm to find something interesting (structure, pattern) in unlabeled data all by itself

**e.g.1**: clustering---Google news

find articles that mention similar words and group them into clusters

>**question**: clustering vs classification \
no one told the algorithm to find the articles that contain similar words and put them into the same cluster

**e.g.2**: anormaly detection

find unusual data points

**e.g.3**: dimensionality reduction

compress data using fewer numbers
