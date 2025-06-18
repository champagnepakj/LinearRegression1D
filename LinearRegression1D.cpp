// LinearRegression1D.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>








double predict(double x, double w, double b)
{
    // x = X[i]
    // w = weight(slope)
    // b = bias(y-intercept)
    

    return w * x + b;
}


double compute_loss(std::vector<int>& X, std::vector<int>& Y, double w, double b)
{
    double error;
    double squared_error;
    double sum{ 0 };


    for (int i = 0; i < X.size(); i++)
    {
        error = Y[i] - predict(X[i], w, b);
        squared_error = error * error;
        sum += squared_error;
    }

    return sum / X.size();

}












int main()
{
    const double weight{ 1.0 };
    const double bias{ 0.0 };

    std::vector<int> X{ 1, 2, 3, 4, 5 };
    std::vector<int> Y{ 2, 4, 6, 8, 10 }; // Actual

    
    for (int i = 0; i < X.size(); i++)
    {
        double element_prediction = predict(X[i], weight, bias);
        std::cout << "Input:" << X[i] << ", Actual: " << Y[i] << ", Predicted: " << element_prediction << '\n';
    }
    

    double loss{ compute_loss(X, Y, weight, bias) };
    std::cout << "MSE Loss: " << loss << '\n';

   

    return 0;
}


/*

    First Run: w(1.0), b(0.0)
    -------------------------------------
    Input:1, Actual: 2  ,  Predicted:  1
    Input:2, Actual: 4  ,  Predicted:  2
    Input:3, Actual: 6  ,  Predicted:  3
    Input:4, Actual: 8  ,  Predicted:  4
    Input:5, Actual: 10 ,  Predicted:  5
    -------------------------------------

    - By using weight(1.0), b(0.0), the prediction
      followed y^ = 1.0 * x + 0.0 -> just returns x.

    - Underestimating by a 1/2.

    ************************************************

    Calculate Loss (Error):
    -------------------------------------
    
    - To calculate Loss Mean Squared Error (MSE) will
      be used.


      Second Run: With MSE
    ------------------------------------- 
    Input:1, Actual: 2  ,  Predicted:  1  --------------|
    Input:2, Actual: 4  ,  Predicted:  2  --------------|
    Input:3, Actual: 6  ,  Predicted:  3  --------------|
    Input:4, Actual: 8  ,  Predicted:  4  --------------|
    Input:5, Actual: 10 ,  Predicted:  5  --------------|
    MSE Loss: 11                                        |
    -------------------------------------               |
                                                        |
    - The reason the MSE calculated 11 was becuase      |
      if we take the vector of predicted and square     |
      of each result and divide by the size of vector   |
      X: 1, 4, 9, 16, 25 = 55.                          |
      MSE = 55 / 5 -> X.size() <------------------------|
      = 11.                                             



*/