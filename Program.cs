using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Diagnostics;
using System.Reflection;
using System.Configuration;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Serialization;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Algorithms;

namespace IRT
{
    public enum PriorType
    {
        Standard, Hierarchical, Vague, StandardVague, VagueStandard, Standard5
    };

    public enum AlgorithmType
    {
        Variational, MCMC, EP
    };

    public class Options
    {
        public double credibleIntervalProbability = 1.0;
        public int numberOfSamples = 2000;
        public int burnIn = 100;
        public int numParams = 2;
    }

    class Program
    {
        static void Main(string[] args)
        {
            string conditionPrefix = (args.Length > 0) ? args[0] : "";
            if (args.Length < 4)
            {
                Console.WriteLine("Usage: irt.exe algorithm priorType responseFile outputFolder");
                Console.WriteLine("algorithm = Variational, MCMC, or EP");
                Console.WriteLine("priorType = Standard, Hierarchical, Vague, StandardVague (ability=Standard), VagueStandard (difficulty=Standard), or Standard5 (difficulty has stddev=5)");
                Console.WriteLine("responseFile contains a header row followed by rows of 0/1 entries, where the first column is the user ID");
                Console.WriteLine("(alternatively, if the first entry in responseFile is a number, then it is assumed to have no header row and no IDs)");
                Console.WriteLine("outputFolder will contain parameter estimates as ability.txt, difficulty.txt, p.txt");
                //Console.WriteLine("credibleIntervalProbability is only used by MCMC and specifies how much probability should be contained in the credible intervals (default 0.95)");
                return;
            }
            Options options = new Options();
            options.numParams = int.Parse(ConfigurationManager.AppSettings.Get("numberOfParameters"));
            options.numberOfSamples = int.Parse(ConfigurationManager.AppSettings.Get("numberOfSamples"));
            options.burnIn = int.Parse(ConfigurationManager.AppSettings.Get("burnIn"));
            options.credibleIntervalProbability = double.Parse(ConfigurationManager.AppSettings.Get("credibleIntervalProbability"));
            AlgorithmType algType = ParseEnum<AlgorithmType>(args[0]);
            PriorType priorType = ParseEnum<PriorType>(args[1]);
            Matrix responses = ReadResponseFile(args[2]);
            OneShot(priorType, algType, responses, args[3], options);
        }

        public static T ParseEnum<T>(string s)
        {
            return (T)Enum.Parse(typeof(T), s, true);
        }

        public static Matrix ReadResponseFile(string responseFile)
        {
            List<double[]> rows = new List<double[]>();
            using (StreamReader reader = new StreamReader(responseFile))
            {
                string[] delimiters = { " ", ",", ";", ":" };
                bool hasDelimiter = false;
                // read header line
                string header = reader.ReadLine().Trim();
                foreach (string delimiter in delimiters)
                {
                    if (header.Contains(delimiter)) hasDelimiter = true;
                }
                if (!hasDelimiter)
                {
                    string line = header;
                    while (!reader.EndOfStream)
                    {
                        if (line == null) line = reader.ReadLine().Trim();
                        if (line.Length > 0)
                        {
                            if (line.Length != header.Length) throw new Exception("The length of this row does not match the first row: " + line);
                            double[] row = Util.ArrayInit(line.Length, i => double.Parse(line[i].ToString()));
                            rows.Add(row);
                        }
                        line = null;
                    }
                }
                else
                {
                    string[] fields = header.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                    double dummy;
                    string line;
                    int offset;
                    if (double.TryParse(fields[0], out dummy))
                    {
                        line = header;
                        offset = 0;
                    }
                    else
                    {
                        line = null;
                        offset = 1;
                    }
                    while (!reader.EndOfStream)
                    {
                        if (line == null) line = reader.ReadLine();
                        if (line.Length > 0)
                        {
                            string[] substrings = line.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                            if (substrings.Length != fields.Length) throw new Exception("The number of columns in this row does not match the header row: " + line);
                            double[] row = Util.ArrayInit(substrings.Length - offset, i => double.Parse(substrings[offset + i]));
                            rows.Add(row);
                        }
                        line = null;
                    }
                }
            }
            Matrix m = new Matrix(rows.Count, (rows.Count == 0) ? 0 : (rows[0].Length));
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Cols; j++)
                {
                    m[i, j] = rows[i][j];
                }
            }
            return m;
        }

        public static void OneShot(PriorType priorType, AlgorithmType algType, Matrix responses, string outputFolder, Options options)
        {
            Directory.CreateDirectory(outputFolder);
            LogisticIrtModel train = new LogisticIrtModel(options.numParams, priorType);
            LogisticIrtTestModel test = new LogisticIrtTestModel(options.numParams);
            train.engine.Compiler.WriteSourceFiles = false;
            test.engine.Compiler.WriteSourceFiles = false;
            train.engine.ShowProgress = false;
            test.engine.ShowProgress = false;
            if (algType == AlgorithmType.Variational)
            {
                train.engine.Algorithm = new VariationalMessagePassing();
                test.engine.Algorithm = new VariationalMessagePassing();
            }
            Gaussian[] abilityPost, difficultyPost;
            Gamma[] discriminationPost = null;
            Beta[] guessProbPost = null;
            Matrix responseProbMean;
            Matrix abilityCred, difficultyCred;
            if (algType != AlgorithmType.MCMC)
            {
                train.ObserveResponses(responses);
                train.RunToConvergence();
                abilityPost = train.engine.Infer<Gaussian[]>(train.ability);
                difficultyPost = train.engine.Infer<Gaussian[]>(train.difficulty);
                if (options.numParams >= 2)
                {
                    discriminationPost = train.engine.Infer<Gamma[]>(train.discrimination);
                }
                if (options.numParams >= 3)
                {
                    guessProbPost = train.engine.Infer<Beta[]>(train.guessProb);
                }
                responseProbMean = test.GetResponseProbs(abilityPost, difficultyPost, discriminationPost, guessProbPost);
            }
            else
            { // MCMC
                LogisticIrtSampler sampler = new LogisticIrtSampler();
                sampler.abilityMeanPrior = Gaussian.FromMeanAndVariance(0, 1e6);
                sampler.abilityPrecPrior = Gamma.FromShapeAndRate(1, 1);
                sampler.difficultyMeanPrior = Gaussian.FromMeanAndVariance(0, 1e6);
                sampler.difficultyPrecPrior = Gamma.FromShapeAndRate(1, 1);
                sampler.discriminationMeanPrior = Gaussian.FromMeanAndVariance(0, 1e6);
                sampler.discriminationPrecPrior = Gamma.FromShapeAndRate(1, 1);
                if (train.abilityMean.IsObserved) sampler.abilityMeanPrior = Gaussian.PointMass(train.abilityMean.ObservedValue);
                if (train.abilityPrecision.IsObserved) sampler.abilityPrecPrior = Gamma.PointMass(train.abilityPrecision.ObservedValue);
                if (train.difficultyMean.IsObserved) sampler.difficultyMeanPrior = Gaussian.PointMass(train.difficultyMean.ObservedValue);
                if (train.difficultyPrecision.IsObserved) sampler.difficultyPrecPrior = Gamma.PointMass(train.difficultyPrecision.ObservedValue);
                if (train.discriminationMean.IsObserved) sampler.discriminationMeanPrior = Gaussian.PointMass(train.discriminationMean.ObservedValue);
                if (train.discriminationPrecision.IsObserved) sampler.discriminationPrecPrior = Gamma.PointMass(train.discriminationPrecision.ObservedValue);
                sampler.Sample(options, responses);
                abilityPost = sampler.abilityPost;
                difficultyPost = sampler.difficultyPost;
                responseProbMean = sampler.responseProbMean;
                discriminationPost = sampler.discriminationPost;
                abilityCred = sampler.abilityCred;
                difficultyCred = sampler.difficultyCred;
                WriteMatrix(abilityCred, outputFolder + @"\ability_ci.txt");
                WriteMatrix(difficultyCred, outputFolder + @"\difficulty_ci.txt");
            }
            bool showEstimates = false;
            if (showEstimates)
            {
                Console.WriteLine("abilityMean = {0}", train.engine.Infer(train.abilityMean));
                Console.WriteLine("abilityPrecision = {0}", train.engine.Infer(train.abilityPrecision));
                //Console.WriteLine("abilityMean2 = {0}", train.engine.Infer(train.abilityMean2));
                //Console.WriteLine("abilityPrecision2 = {0}", train.engine.Infer(train.abilityPrecision2));
                Console.WriteLine("difficultyMean = {0}", train.engine.Infer(train.difficultyMean));
                Console.WriteLine("difficultyPrecision = {0}", train.engine.Infer(train.difficultyPrecision));
            }
            WriteMatrix(ToMeanMatrix(abilityPost), outputFolder + @"\ability.txt");
            WriteMatrix(ToStddevMatrix(abilityPost), outputFolder + @"\ability_se.txt");
            WriteMatrix(ToMeanMatrix(difficultyPost), outputFolder + @"\difficulty.txt");
            WriteMatrix(ToStddevMatrix(difficultyPost), outputFolder + @"\difficulty_se.txt");
            WriteMatrix(responseProbMean, outputFolder + @"\p.txt");
            if (discriminationPost != null)
            {
                WriteMatrix(ToMeanMatrix(discriminationPost), outputFolder + @"\discrimination.txt");
                WriteMatrix(ToStddevMatrix(discriminationPost), outputFolder + @"\discrimination_se.txt");
            }
            if(guessProbPost != null)
            {
                WriteMatrix(ToMeanMatrix(guessProbPost), outputFolder + @"\guess.txt");
                WriteMatrix(ToStddevMatrix(guessProbPost), outputFolder + @"\guess_se.txt");
            }
        }

        public static void WriteMatrix(Matrix m, string file)
        {
            using (StreamWriter writer = new StreamWriter(file))
            {
                writer.WriteLine(m.ToString());
            }
        }

        public static void LogisticIrt(int numParams, PriorType priorType, AlgorithmType algType, string conditionPrefix = "")
        {
            // timing on Intel Core 2 Duo P9500 with 4GB RAM running Windows Vista
            // 10_250 trial 1:
            // Bayesian/Hierarchical 2000 = 5.4s inference only
            // Variational/Hierarchical 50 iter = 4.2s inference only
            // Variational/Hierarchical 10 iter = 0.85s inference only
            // Variational_JJ/Hierarchical 50 iter = 0.1s inference only
            // Variational_JJ/Hierarchical 10 iter = 0.04s inference only
            // time on desktop:
            // Variational/Hierarchical 10 iter = 0.75s inference only (including test)
            // Variational_JJ/Hierarchical 10 iter = 0.07s inference only (including test)
            LogisticIrtModel train = new LogisticIrtModel(numParams, priorType);
            //train.engine.NumberOfIterations = 100;
            //train.engine.ShowTimings = true;
            string logistic_type = "";
            //logistic_type = "JJ";
            if (logistic_type == "JJ")
            {
                train.engine.Compiler.GivePriorityTo(typeof(LogisticOp_JJ96));
            }
            bool specialInitialization = false;
            if (specialInitialization)
            {
                // change initialization
                train.abilityMean.InitialiseTo(new Gaussian(5, 10));
                train.abilityPrecision.InitialiseTo(new Gamma(1, 10));
                train.difficultyMean.InitialiseTo(new Gaussian(5, 10));
                train.difficultyPrecision.InitialiseTo(new Gamma(1, 10));
            }
            LogisticIrtTestModel test = new LogisticIrtTestModel(numParams);
            train.engine.ShowProgress = false;
            test.engine.ShowProgress = false;
            if (algType == AlgorithmType.Variational)
            {
                train.engine.Algorithm = new VariationalMessagePassing();
                test.engine.Algorithm = new VariationalMessagePassing();
            }
            bool showTiming = false;

            string baseFolder = @"..\..\";
            string modelName = numParams + "-PL";
            //modelName = "Mild_skew";
            //modelName = "Extreme_skew";
            //modelName = "Lsat";
            //modelName = "Wide_b";
            string modelFolder = baseFolder + @"Data_mat\" + modelName;
            DirectoryInfo modelDir = new DirectoryInfo(modelFolder);
            foreach (DirectoryInfo conditionDir in modelDir.GetDirectories())
            {
                string condition = conditionDir.Name;
                if (!condition.StartsWith(conditionPrefix))
                    continue;
                int trimStart = condition.Length - 1;
                string inputFolder = baseFolder + @"Data_mat\" + modelName + @"\" + condition;
                string alg;
                if (algType == AlgorithmType.Variational) alg = "Variational" + logistic_type + @"\" + priorType;
                else alg = algType + @"\" + priorType;
                string outputFolder = baseFolder + @"Estimates_mat\" + modelName + @"\" + alg + @"\" + condition;
                Console.WriteLine(outputFolder);
                DirectoryInfo outputDir = Directory.CreateDirectory(outputFolder);
                DirectoryInfo inputDir = new DirectoryInfo(inputFolder);
                foreach (FileInfo file in inputDir.GetFiles("*.mat"))
                {
                    string name = file.Name;
                    string number = name; //.Substring(trimStart);
                    string outputFileName = outputFolder + @"\" + number;
                    if (File.Exists(outputFileName)) continue;
                    Console.WriteLine(file.FullName);
                    Dictionary<string, object> dict = MatlabReader.Read(file.FullName);
                    Matrix m = (Matrix)dict["Y"];
                    Gaussian[] abilityPost, difficultyPost;
                    Gamma[] discriminationPost = null;
                    Beta[] guessProbPost = null;
                    Matrix responseProbMean;
                    if (algType != AlgorithmType.MCMC)
                    {
                        // VMP
                        Stopwatch watch = new Stopwatch();
                        watch.Start();
                        train.ObserveResponses(m);
                        train.RunToConvergence();
                        abilityPost = train.engine.Infer<Gaussian[]>(train.ability);
                        difficultyPost = train.engine.Infer<Gaussian[]>(train.difficulty);
                        if (numParams >= 2)
                        {
                            discriminationPost = train.engine.Infer<Gamma[]>(train.discrimination);
                        }
                        if (numParams >= 3)
                        {
                            guessProbPost = train.engine.Infer<Beta[]>(train.guessProb);
                        }
                        responseProbMean = test.GetResponseProbs(abilityPost, difficultyPost, discriminationPost, guessProbPost);
                        watch.Stop();
                        if (showTiming) Console.WriteLine(algType + " elapsed time = {0}ms", watch.ElapsedMilliseconds);
                    }
                    else
                    {
                        // sampler
                        LogisticIrtSampler sampler = new LogisticIrtSampler();
                        sampler.abilityMeanPrior = Gaussian.FromMeanAndVariance(0, 1e6);
                        sampler.abilityPrecPrior = Gamma.FromShapeAndRate(1, 1);
                        sampler.difficultyMeanPrior = Gaussian.FromMeanAndVariance(0, 1e6);
                        sampler.difficultyPrecPrior = Gamma.FromShapeAndRate(1, 1);
                        sampler.discriminationMeanPrior = Gaussian.FromMeanAndVariance(0, 1e6);
                        sampler.discriminationPrecPrior = Gamma.FromShapeAndRate(1, 1);
                        // for debugging
                        //sampler.abilityObserved = ((Matrix)dict["ability"]).ToArray<double>();
                        //sampler.difficultyObserved = ((Matrix)dict["difficulty"]).ToArray<double>();
                        //sampler.discriminationObserved = ((Matrix)dict["discrimination"]).ToArray<double>();
                        if (train.abilityMean.IsObserved) sampler.abilityMeanPrior = Gaussian.PointMass(train.abilityMean.ObservedValue);
                        if (train.abilityPrecision.IsObserved) sampler.abilityPrecPrior = Gamma.PointMass(train.abilityPrecision.ObservedValue);
                        if (train.difficultyMean.IsObserved) sampler.difficultyMeanPrior = Gaussian.PointMass(train.difficultyMean.ObservedValue);
                        if (train.difficultyPrecision.IsObserved) sampler.difficultyPrecPrior = Gamma.PointMass(train.difficultyPrecision.ObservedValue);
                        if (train.discriminationMean.IsObserved) sampler.discriminationMeanPrior = Gaussian.PointMass(train.discriminationMean.ObservedValue);
                        if (train.discriminationPrecision.IsObserved) sampler.discriminationPrecPrior = Gamma.PointMass(train.discriminationPrecision.ObservedValue);
                        Stopwatch watch = new Stopwatch();
                        watch.Start();
                        sampler.Sample(new Options(), m);
                        abilityPost = sampler.abilityPost;
                        difficultyPost = sampler.difficultyPost;
                        responseProbMean = sampler.responseProbMean;
                        discriminationPost = sampler.discriminationPost;
                        watch.Stop();
                        if (showTiming) Console.WriteLine("MCMC elapsed time = {0}ms", watch.ElapsedMilliseconds);
                    }
                    bool showEstimates = false;
                    if (showEstimates)
                    {
                        Console.WriteLine("abilityMean = {0}", train.engine.Infer(train.abilityMean));
                        Console.WriteLine("abilityPrecision = {0}", train.engine.Infer(train.abilityPrecision));
                        //Console.WriteLine("abilityMean2 = {0}", train.engine.Infer(train.abilityMean2));
                        //Console.WriteLine("abilityPrecision2 = {0}", train.engine.Infer(train.abilityPrecision2));
                        Console.WriteLine("difficultyMean = {0}", train.engine.Infer(train.difficultyMean));
                        Console.WriteLine("difficultyPrecision = {0}", train.engine.Infer(train.difficultyPrecision));
                    }
                    if (showEstimates)
                    {
                        for (int i = 0; i < 10; i++)
                        {
                            Console.WriteLine(responseProbMean[i]);
                        }
                        //Console.WriteLine(ToMeanMatrix(difficultyPost));
                    }
                    using (MatlabWriter writer = new MatlabWriter(outputFileName))
                    {
                        writer.Write("ability", ToMeanMatrix(abilityPost));
                        writer.Write("ability_se", ToStddevMatrix(abilityPost));
                        writer.Write("difficulty", ToMeanMatrix(difficultyPost));
                        writer.Write("difficulty_se", ToStddevMatrix(difficultyPost));
                        if (discriminationPost != null)
                        {
                            writer.Write("discrimination", ToMeanMatrix(discriminationPost));
                            writer.Write("discrimination_se", ToStddevMatrix(discriminationPost));
                        }
                        if (guessProbPost != null)
                            writer.Write("guessing", ToMeanAndStddevMatrix(guessProbPost));
                        writer.Write("p", responseProbMean);
                    }
                    //break;
                }
                //break;
            }
        }

        public static Matrix ToMatrix<T>(T[] dists, Converter<T, double> converter)
        {
            Matrix m = new Matrix(dists.Length, 1);
            for (int i = 0; i < m.Rows; i++)
            {
                m[i, 0] = converter(dists[i]);
            }
            return m;
        }
        public static Matrix ToMeanMatrix<T>(T[] dists)
          where T : CanGetMean<double>
        {
            return ToMatrix(dists, d => d.GetMean());
        }
        public static Matrix ToStddevMatrix<T>(T[] dists)
          where T : CanGetVariance<double>
        {
            return ToMatrix(dists, d => Math.Sqrt(d.GetVariance()));
        }
        public static Matrix ToMeanMatrix<T>(T[,] dists)
          where T : CanGetMean<double>
        {
            Matrix m = new Matrix(dists.GetLength(0), dists.GetLength(1));
            for (int i = 0; i < m.Rows; i++)
            {
                for (int j = 0; j < m.Cols; j++)
                {
                    m[i, j] = dists[i, j].GetMean();
                }
            }
            return m;
        }
        public static Matrix ToMeanAndStddevMatrix<T>(T[] dists)
          where T : CanGetMean<double>, CanGetVariance<double>
        {
            Matrix m = new Matrix(dists.Length, 2);
            for (int i = 0; i < m.Rows; i++)
            {
                m[i, 0] = dists[i].GetMean();
                m[i, 1] = Math.Sqrt(dists[i].GetVariance());
            }
            return m;
        }
        public static int[,] ConvertToInt(double[,] array)
        {
            int rows = array.GetLength(0);
            int cols = array.GetLength(1);
            int[,] result = new int[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = (int)array[i, j];
                }
            }
            return result;
        }

    }
}
