using CustomerChurnMLDemo;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.AutoML;
using ConfusionMatrix = CustomerChurnMLDemo.ConfusionMatrix;

class Program
{
    static void Main(string[] args)
    {
        ConfusionMatrix.CreateMatrix();
    }
}