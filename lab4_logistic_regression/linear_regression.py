from numpy import *

def RSS_GRADIENT (b_current, m_current, b_gradient, m_gradient, points):
    N = float(len(points))
    for i in range(0, len(points)):
        #transform the values of x and y to a smaller scale
        x = points[i][0]
        y = points[i][1]
                
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    return [b_gradient, m_gradient]

def compute_error_for_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]

        totalError += (y - (m * x + b))

    return(totalError/float(len(points)))

def step_gradient(b_current, m_current, points, learning_rate):
    #gradient_descendent
    b_gradient, m_gradient = RSS_GRADIENT(b_current, m_current, 0, 0, points)
    print(b_gradient, m_gradient)

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]

def gradient_descendent_runner(points, starting_b, starting_m, learning_rate, num_interactions):
    b = starting_b
    m = starting_m

    RSS_NORM = linalg.norm(RSS_GRADIENT(b, m, 0, 0, points))
    #for i in range(0, num_interactions):
    while (RSS_NORM > 1e-10) :
    #while (RSS_NORM > 0.0389) :
        b, m = step_gradient(b, m, array(points), learning_rate)
        RSS_NORM = linalg.norm(RSS_GRADIENT(b, m, 0, 0, points))

    return[b, m]
def RSS_GRADIENT (b_current, m_current, b_gradient, m_gradient, points):
    N = float(len(points))
    for i in range(0, len(points)):
        #transform the values of x and y to a smaller scale
        x = points[i][0]
        y = points[i][1]
                
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    return [b_gradient, m_gradient]

def compute_error_for_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]

        totalError += (y - (m * x + b))

    return(totalError/float(len(points)))

def step_gradient(b_current, m_current, points, learning_rate):
    #gradient_descendent
    b_gradient, m_gradient = RSS_GRADIENT(b_current, m_current, 0, 0, points)
    print(b_gradient, m_gradient)

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]

def gradient_descendent_runner(points, starting_b, starting_m, learning_rate, num_interactions):
    b = starting_b
    m = starting_m

    RSS_NORM = linalg.norm(RSS_GRADIENT(b, m, 0, 0, points))
    #for i in range(0, num_interactions):
    while (RSS_NORM > 1e-10) :
    #while (RSS_NORM > 0.0389) :
        b, m = step_gradient(b, m, array(points), learning_rate)
        RSS_NORM = linalg.norm(RSS_GRADIENT(b, m, 0, 0, points))

    return[b, m]