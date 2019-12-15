<img src="img/logo.png" alt="logo" width="200" heigth="180"/>

The purpose of this repository is my self practice of Naive Bayes theorem. Exercise from the Book "Data Science from Scratch"

I am a student that is learning, let me know if you find any errors,the code is inspired from examples and exercises found in books.

## A few important things that I learned

This model is part of a family of simple "probabilistic classifiers" based on Bayes' theorem

Bayes Theorem says that to find P(A|B) we need to look for
P(B|A)/(P(B|A)+P(B|-A)), i.e, is the ratio of the probability that B happens conditioning that A happens with the probability that B happens.

The exercise proposed by the book is to create a simple spam filter and that's exactly what you will find in the code.

We start  with the idea that we will look for the a bunch of words and try to discover the probability that a spam message contains such word P(Xi|S) and we will do the same for the scenario where it contains the word but isn't a spam P(Xi|-S) with this logic we can use Bayes Theorem to find out if giving the set of words is the message spam or not.

* P ( S | X = x ) = P ( X = x | S ) / [ P ( X = x | S ) + P ( X = x | ¬ S ) ]

Now, we need to find P(X|S) and P(X|¬S) for every word and multiply it all (Naive assumption that the probabilities are independent),but considering that multiplying a lot of probabilities can create a problem called underflow we will use the equivalent with exponentiation and logs;

* exp(log(p1) + ⋯ + log(pn))

Having a number of training messages labeled as spam and not spam and then calculate the probability of an word i is appears in a spam is not enough, take the scenario where in the training set the word i appears only in not spams (hams) and we would estimate that P(i|S) = O. Our classifier would always assign a spam probability of 0 to any message containing such word, even in the scenario where is clearly a spam.

To avoid this problem, we will use some kind of smoothing, a pseudo-count k that will allow us to at least not assign 0 probability for such cases or similar ones.

* P ( X i | S ) = (k + number of spams containing w_i) /( 2k + number of spams)

### todo
- [ ] add a use case for the model

## Resources that I used to learn about this fun topic:
* Book: Data Science from Scratch, Joel Grus
* Video: Bayes theorem, Khan Academy
