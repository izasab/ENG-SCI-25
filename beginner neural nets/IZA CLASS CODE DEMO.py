from numpy import exp, array, random, dot


def sig(x):
    return 1 / (1 + exp(-x)) 


def sig_der(x):
    return x * (1 - x)


inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]) #TRAINING INPUT DATA 

outputs = array([[0, 1, 1, 0]]).T #TRAINING OUTPUT DATA

weights = 2 * random.random((3, 1)) - 1 #RANDOM NEURONS THAT ARE PUT IN THE MIDDLE OF the training input data and the output data (3 neurons)

print("dumb neurons: \n", weights)


for i in range(100000):

    out = sig(dot(inputs, weights))

    error = outputs - out

    adj = error * sig_der(out)

    weights += dot(inputs.T, adj) #NEURONS WEIGHTS HAVE JUST GOTTEN A BIT SMARTER


print ("\n \n \n iz test \n")

print (sig((dot(array([0, 0, 1]), weights))), '\n') #NOW LET'S TEST OUR SUPER SMART BRAIN

print (sig((dot(array([0, 1, 1]), weights))), '\n') #LET'S TEST IT MORE



print("\n smart neurons: \n",weights,"\n")
