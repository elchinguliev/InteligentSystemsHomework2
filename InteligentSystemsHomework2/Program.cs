using System;
using System.Linq;

class LinearRegression
{
    private double w;  // slope
    private double b;  // intercept
    private double learningRate;
    private int iterations;

    // Constructor to initialize learning rate and iterations
    public LinearRegression(double learningRate, int iterations)
    {
        this.learningRate = learningRate;
        this.iterations = iterations;
    }

    // Method to train the model using gradient descent
    public void Train(double[] x, double[] y)
    {
        int n = x.Length;
        w = 0; // initial slope
        b = 0; // initial intercept

        for (int i = 0; i < iterations; i++)
        {
            double dw = 0; // gradient of w
            double db = 0; // gradient of b

            // Compute gradients
            for (int j = 0; j < n; j++)
            {
                double y_pred = w * x[j] + b;
                dw += -2 * x[j] * (y[j] - y_pred);
                db += -2 * (y[j] - y_pred);
            }

            // Update the weights
            w -= (dw / n) * learningRate;
            b -= (db / n) * learningRate;
        }
    }

    // Method to predict new values
    public double Predict(double x)
    {
        return w * x + b;
    }

    // Method to calculate the Mean Squared Error (MSE)
    public double CalculateMSE(double[] x, double[] y)
    {
        double mse = 0;
        int n = x.Length;

        for (int i = 0; i < n; i++)
        {
            double y_pred = Predict(x[i]);
            mse += Math.Pow(y[i] - y_pred, 2);
        }

        return mse / n;
    }
}

class Program
{
    static void Main(string[] args)
    {
        double[] x = { 1, 2, 3, 4, 5 };
        double[] y = { 1, 2, 3, 3.5, 5 };


        //check it with another x,y values for addition points
        //double[] x = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        //double[] y = { 1.2, 2.1, 3.5, 4.8, 4.9, 6.1, 6.8, 7.5, 9.1, 10.2 }; // Noisy linear data


        // Create and train the Linear Regression model
        LinearRegression model = new LinearRegression(learningRate: 0.01, iterations: 1000);
        model.Train(x, y);

        // Test the model on new data
        double[] testData = { 6, 7, 8 };
        Console.WriteLine("Predictions:");
        foreach (double input in testData)
        {
            double prediction = model.Predict(input);
            Console.WriteLine($"Input: {input}, Predicted Output: {prediction}");
        }

        // Calculate the Mean Squared Error (MSE) on training data
        double mse = model.CalculateMSE(x, y);
        Console.WriteLine($"Mean Squared Error: {mse}");
    }
}
