Item Response Theory
====================

This software fits the 1-PL or 2-PL Item Response Theory model to observed responses.  It is structured as a .NET Core 2.1 Console Application, with a configuration file.  Run the application with no arguments to get instructions.

This software was used to generate the variational results in the paper ["Bayesian Prior Choice in IRT Estimation Using MCMC and Variational Bayes"](https://doi.org/10.3389/fpsyg.2016.01422).

Installation
============

The software runs on any platform that supports .NET Core 2.1.  The easiest way to build and run is with Visual Studio 2017.  Alternatively, you can build and run from the command line using the following .NET Core commands:
```
dotnet run IRT.sln <arguments>
```
The build process automatically downloads Infer.NET.

Tom Minka
