using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Utils;

namespace IRT
{
    public class LogisticIrtSampler
    {
        public Gaussian abilityMeanPrior;
        public Gamma abilityPrecPrior;
        public Gaussian difficultyMeanPrior;
        public Gamma difficultyPrecPrior;
        public Gaussian discriminationMeanPrior;
        public Gamma discriminationPrecPrior;

        public Gaussian[] abilityPost;
        public Gaussian[] difficultyPost;
        public Gamma[] discriminationPost;
        public Matrix responseProbMean;
        public Matrix abilityCred;
        public Matrix difficultyCred;

        public double[] abilityObserved, difficultyObserved, discriminationObserved;

        public void Sample(Options options, Matrix data)
        {
            if (options.numParams > 2) throw new Exception("numParams > 2");
            int numStudents = data.Rows;
            int numQuestions = data.Cols;
            // initialize the sampler at the mean of the priors (not sampling from the priors)
            double abilityMean = abilityMeanPrior.GetMean();
            double abilityPrec = abilityPrecPrior.GetMean();
            double difficultyMean = difficultyMeanPrior.GetMean();
            double difficultyPrec = difficultyPrecPrior.GetMean();
            double discriminationMean = discriminationMeanPrior.GetMean();
            double discriminationPrec = discriminationPrecPrior.GetMean();
            double[] ability = new double[numStudents];
            double[] difficulty = new double[numQuestions];
            List<double>[] difficultySamples = new List<double>[numQuestions];
            GaussianEstimator[] difficultyEstimator = new GaussianEstimator[numQuestions];
            for (int question = 0; question < numQuestions; question++)
            {
                difficultyEstimator[question] = new GaussianEstimator();
                difficultySamples[question] = new List<double>();
                if (difficultyObserved != null)
                {
                    difficulty[question] = difficultyObserved[question];
                    difficultyEstimator[question].Add(difficultyObserved[question]);
                    difficultySamples[question].Add(difficultyObserved[question]);
                }
            }
            List<double>[] abilitySamples = new List<double>[numStudents];
            GaussianEstimator[] abilityEstimator = new GaussianEstimator[ability.Length];
            for (int student = 0; student < abilityEstimator.Length; student++)
            {
                abilityEstimator[student] = new GaussianEstimator();
                abilitySamples[student] = new List<double>();
                if (abilityObserved != null)
                {
                    ability[student] = abilityObserved[student];
                    abilityEstimator[student].Add(abilityObserved[student]);
                    abilitySamples[student].Add(abilityObserved[student]);
                }
            }
            double[] discrimination = new double[numQuestions];
            List<double>[] discriminationSamples = new List<double>[numQuestions];
            GammaEstimator[] discriminationEstimator = new GammaEstimator[numQuestions];
            for (int question = 0; question < numQuestions; question++)
            {
                discriminationEstimator[question] = new GammaEstimator();
                discriminationSamples[question] = new List<double>();
                discrimination[question] = 1;
                if (discriminationObserved != null)
                {
                    discrimination[question] = discriminationObserved[question];
                    discriminationEstimator[question].Add(discriminationObserved[question]);
                    discriminationSamples[question].Add(discriminationObserved[question]);
                }
            }
            responseProbMean = new Matrix(numStudents, numQuestions);
            int niters = options.numberOfSamples;
            int burnin = options.burnIn;
            bool useGumbel = true;
            double logisticVariance = Math.PI * Math.PI / 3;
            double shape = 4.5;
            Gamma precPrior = Gamma.FromShapeAndRate(shape, (shape - 1) * logisticVariance);
            precPrior = Gamma.PointMass(1);
            double[,] prec = new double[numStudents, numQuestions];
            double[,] x = new double[numStudents, numQuestions];
            int numRejected = 0, numAttempts = 0;
            for (int iter = 0; iter < niters; iter++)
            {
                for (int student = 0; student < numStudents; student++)
                {
                    for (int question = 0; question < numQuestions; question++)
                    {
                        // sample prec given ability, difficulty, x
                        // N(x; ability-difficulty, 1/prec) = Gamma(prec; 1.5, (x-ability+difficulty)^2/2)
                        Gamma precPost = precPrior;
                        double xMean = (ability[student] - difficulty[question]) * discrimination[question];
                        double delta = x[student, question] - xMean;
                        Gamma like = Gamma.FromShapeAndRate(1.5, 0.5 * delta * delta);
                        precPost.SetToProduct(precPost, like);
                        prec[student, question] = precPost.Sample();
                        // sample x given ability, difficulty, prec, data
                        // using an independence chain MH
                        bool y = (data[student, question] > 0);
                        double sign = y ? 1.0 : -1.0;
                        Gaussian xPrior = Gaussian.FromMeanAndPrecision(xMean, prec[student, question]);
                        // we want to sample from xPrior*I(x>0)
                        // instead we sample from xPost
                        Gaussian xPost = xPrior * IsPositiveOp.XAverageConditional(y, xPrior);
                        double oldx = x[student, question];
                        double newx = xPost.Sample();
                        numAttempts++;
                        if (newx * sign < 0)
                        {
                            newx = oldx; // rejected
                            numRejected++;
                        }
                        else
                        {
                            // importance weights
                            double oldw = xPrior.GetLogProb(oldx) - xPost.GetLogProb(oldx);
                            double neww = xPrior.GetLogProb(newx) - xPost.GetLogProb(newx);
                            // acceptance ratio
                            double paccept = Math.Exp(neww - oldw);
                            if (paccept < 1 && Rand.Double() > paccept)
                            {
                                newx = oldx; // rejected
                                numRejected++;
                            }
                        }
                        x[student, question] = newx;
                        if (iter >= burnin)
                        {
                            double responseProb = MMath.Logistic(xMean);
                            responseProbMean[student, question] += responseProb;
                        }
                    }
                }
                if (abilityObserved == null)
                {
                    // sample ability given difficulty, prec, x
                    for (int student = 0; student < numStudents; student++)
                    {
                        Gaussian post = Gaussian.FromMeanAndPrecision(abilityMean, abilityPrec);
                        for (int question = 0; question < numQuestions; question++)
                        {
                            // N(x; disc*(ability-difficulty), 1/prec) =propto N(x/disc; ability-difficulty, 1/disc^2/prec) = N(ability; x/disc+difficulty, 1/disc^2/prec)
                            Gaussian abilityLike = Gaussian.FromMeanAndPrecision(x[student, question] / discrimination[question] + difficulty[question], prec[student, question] * discrimination[question] * discrimination[question]);
                            post.SetToProduct(post, abilityLike);
                        }
                        ability[student] = post.Sample();
                        if (iter >= burnin)
                        {
                            abilityEstimator[student].Add(post);
                            abilitySamples[student].Add(ability[student]);
                        }
                    }
                }
                // sample difficulty given ability, prec, x
                for (int question = 0; question < numQuestions; question++)
                {
                    Gaussian post = Gaussian.FromMeanAndPrecision(difficultyMean, difficultyPrec);
                    for (int student = 0; student < numStudents; student++)
                    {
                        // N(x; disc*(ability-difficulty), 1/prec) =propto N(x/disc; ability-difficulty, 1/disc^2/prec) = N(difficulty; ability-x/disc, 1/disc^2/prec)
                        if (discrimination[question] > 0)
                        {
                            Gaussian like = Gaussian.FromMeanAndPrecision(ability[student] - x[student, question] / discrimination[question], prec[student, question] * discrimination[question] * discrimination[question]);
                            post.SetToProduct(post, like);
                        }
                    }
                    difficulty[question] = post.Sample();
                    if (iter >= burnin)
                    {
                        //if (difficulty[question] > 100)
                        //    Console.WriteLine("difficulty[{0}] = {1}", question, difficulty[question]);
                        difficultyEstimator[question].Add(post);
                        difficultySamples[question].Add(difficulty[question]);
                    }
                }
                if (options.numParams > 1 && discriminationObserved == null)
                {
                    // sample discrimination given ability, difficulty, prec, x
                    for (int question = 0; question < numQuestions; question++)
                    {
                        // moment-matching on the prior
                        Gaussian approxPrior = Gaussian.FromMeanAndVariance(Math.Exp(discriminationMean + 0.5 / discriminationPrec), Math.Exp(2 * discriminationMean + 1 / discriminationPrec) * (Math.Exp(1 / discriminationPrec) - 1));
                        Gaussian post = approxPrior;
                        for (int student = 0; student < numStudents; student++)
                        {
                            // N(x; disc*delta, 1/prec) =propto N(x/delta; disc, 1/prec/delta^2)
                            double delta = ability[student] - difficulty[question];
                            if (delta > 0)
                            {
                                Gaussian like = Gaussian.FromMeanAndPrecision(x[student, question] / delta, prec[student, question] * delta * delta);
                                post.SetToProduct(post, like);
                            }
                        }
                        TruncatedGaussian postTrunc = new TruncatedGaussian(post, 0, double.PositiveInfinity);
                        double olddisc = discrimination[question];
                        double newdisc = postTrunc.Sample();
                        // importance weights
                        Func<double, double> priorLogProb = delegate (double d)
                        {
                            double logd = Math.Log(d);
                            return Gaussian.GetLogProb(logd, discriminationMean, 1 / discriminationPrec) - logd;
                        };
                        double oldw = priorLogProb(olddisc) - approxPrior.GetLogProb(olddisc);
                        double neww = priorLogProb(newdisc) - approxPrior.GetLogProb(newdisc);
                        // acceptance ratio
                        double paccept = Math.Exp(neww - oldw);
                        if (paccept < 1 && Rand.Double() > paccept)
                        {
                            // rejected
                        }
                        else
                            discrimination[question] = newdisc;
                        if (iter >= burnin)
                        {
                            discriminationEstimator[question].Add(discrimination[question]);
                            discriminationSamples[question].Add(discrimination[question]);
                        }
                    }
                }
                // sample abilityMean given ability, abilityPrec
                Gaussian abilityMeanPost = abilityMeanPrior;
                for (int student = 0; student < numStudents; student++)
                {
                    Gaussian like = GaussianOp.MeanAverageConditional(ability[student], abilityPrec);
                    abilityMeanPost *= like;
                }
                abilityMean = abilityMeanPost.Sample();
                // sample abilityPrec given ability, abilityMean
                Gamma abilityPrecPost = abilityPrecPrior;
                for (int student = 0; student < numStudents; student++)
                {
                    Gamma like = GaussianOp.PrecisionAverageConditional(ability[student], abilityMean);
                    abilityPrecPost *= like;
                }
                abilityPrec = abilityPrecPost.Sample();
                // sample difficultyMean given difficulty, difficultyPrec
                Gaussian difficultyMeanPost = difficultyMeanPrior;
                for (int question = 0; question < numQuestions; question++)
                {
                    Gaussian like = GaussianOp.MeanAverageConditional(difficulty[question], difficultyPrec);
                    difficultyMeanPost *= like;
                }
                difficultyMean = difficultyMeanPost.Sample();
                // sample difficultyPrec given difficulty, difficultyMean
                Gamma difficultyPrecPost = difficultyPrecPrior;
                for (int question = 0; question < numQuestions; question++)
                {
                    Gamma like = GaussianOp.PrecisionAverageConditional(difficulty[question], difficultyMean);
                    difficultyPrecPost *= like;
                }
                difficultyPrec = difficultyPrecPost.Sample();
                // sample discriminationMean given discrimination, discriminationPrec
                Gaussian discriminationMeanPost = discriminationMeanPrior;
                for (int question = 0; question < numQuestions; question++)
                {
                    Gaussian like = GaussianOp.MeanAverageConditional(Math.Log(discrimination[question]), discriminationPrec);
                    discriminationMeanPost *= like;
                }
                discriminationMean = discriminationMeanPost.Sample();
                // sample discriminationPrec given discrimination, discriminationMean
                Gamma discriminationPrecPost = discriminationPrecPrior;
                for (int question = 0; question < numQuestions; question++)
                {
                    Gamma like = GaussianOp.PrecisionAverageConditional(Math.Log(discrimination[question]), discriminationMean);
                    discriminationPrecPost *= like;
                }
                discriminationPrec = discriminationPrecPost.Sample();
                //if (iter % 1 == 0)
                //    Console.WriteLine("iter = {0}", iter);
            }
            //Console.WriteLine("abilityMean = {0}, abilityPrec = {1}", abilityMean, abilityPrec);
            //Console.WriteLine("difficultyMean = {0}, difficultyPrec = {1}", difficultyMean, difficultyPrec);
            int numSamplesUsed = niters - burnin;
            responseProbMean.Scale(1.0 / numSamplesUsed);
            //Console.WriteLine("acceptance rate = {0}", ((double)numAttempts - numRejected)/numAttempts);
            difficultyPost = Array.ConvertAll(difficultyEstimator, est => est.GetDistribution(Gaussian.Uniform()));
            abilityPost = Array.ConvertAll(abilityEstimator, est => est.GetDistribution(Gaussian.Uniform()));
            if (options.numParams > 1)
            {
                discriminationPost = Array.ConvertAll(discriminationEstimator, est => est.GetDistribution(new Gamma()));
            }
            abilityCred = GetCredibleIntervals(options.credibleIntervalProbability, abilitySamples);
            difficultyCred = GetCredibleIntervals(options.credibleIntervalProbability, difficultySamples);
            bool saveSamples = false;
            if (saveSamples)
            {
                using (MatlabWriter writer = new MatlabWriter(@"..\..\samples.mat"))
                {
                    int q = 11;
                    writer.Write("difficulty", difficultySamples[q]);
                    writer.Write("discrimination", discriminationSamples[q]);
                }
            }
        }

        public static Matrix GetCredibleIntervals(double prob, List<double>[] samples)
        {
            Matrix m = new Matrix(samples.Length, 2);
            for (int i = 0; i < samples.Length; i++)
            {
                double min, max;
                GetCredibleInterval(prob, samples[i], out min, out max);
                m[i, 0] = min;
                m[i, 1] = max;
            }
            return m;
        }

        public static void GetCredibleInterval(double prob, IList<double> samples, out double min, out double max)
        {
            double[] sorted = samples.ToArray();
            Array.Sort(sorted);
            int n = (int)Math.Ceiling(prob * samples.Count);
            if (n < 5) Console.WriteLine("warning: credible interval only contains " + n + " samples");
            int offset = n - 1;
            min = sorted[0];
            max = sorted[offset];
            double minWidth = max - min;
            for (int i = offset + 1; i < samples.Count; i++)
            {
                double width = sorted[i] - sorted[i - offset];
                if (width < minWidth)
                {
                    minWidth = width;
                    min = sorted[i - offset];
                    max = sorted[i];
                }
            }
        }
    }
}
