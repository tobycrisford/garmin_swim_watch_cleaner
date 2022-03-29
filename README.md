# Work in progress

A Garmin swim watch uses an accelerometer to detect when a swimmer has finished one length of the pool and started a new one. The length of the swimming pool can also be input manually into the watch, and this is used to calculate the total distance covered. However, the length detection is not perfect. Sometimes false length transitions are recorded, and it is also possible that a real transition might be missed.

I have some swim watch data from a swimmer (lets call her Alice) where it is clear that the total distance has been significantly overestimated, but it is not immediately obvious which length transitions are wrong and which are real. In this project I will use a Bayesian approach to estimate the number of lengths that Alice actually swam. I think that calculating the posterior distribution exactly will be too hard, so I'll use a Monte Carlo method to generate repeated random samples from it instead.
