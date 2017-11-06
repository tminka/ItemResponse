using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Utils;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Factors;
using MicrosoftResearch.Infer.Maths;
using System.IO;
using MicrosoftResearch.Infer.Collections;

namespace IRT
{
	public class LogisticIrtModel
	{
		public Variable<int> numStudents;
		public Variable<double> abilityMean, abilityPrecision;
		public Variable<double> abilityMean2, abilityPrecision2;
		public VariableArray<bool> isExceptional;
		public Variable<IDistribution<bool[]>> isExceptionalInit;
		public VariableArray<double> ability;
		public Variable<int> numQuestions;
		public Variable<double> difficultyMean, difficultyPrecision;
		public VariableArray<double> difficulty;
		public Variable<double> discriminationMean, discriminationPrecision;
		public VariableArray<double> discrimination;
		public VariableArray<double> guessProb;
		public VariableArray2D<bool> response;
		public InferenceEngine engine;

		public LogisticIrtModel(int numParams, PriorType priorType)
		{
			numStudents = Variable.New<int>().Named("numStudents");
			Range student = new Range(numStudents);
			abilityMean = Variable.GaussianFromMeanAndVariance(0, 1e6).Named("abilityMean");
			abilityPrecision = Variable.GammaFromShapeAndRate(1, 1).Named("abilityPrecision");
			ability = Variable.Array<double>(student).Named("ability");
			if (true) {
				ability[student] = Variable.GaussianFromMeanAndPrecision(abilityMean, abilityPrecision).ForEach(student);
			} else if (true) {
				// truncated Gaussian prior for ability
				double threshold, m, v;
				if (false) {
					// matched to Mild_skew generator
					threshold = -1.6464;
					m = -0.4;
					v = 1.5;
				} else {
					// matched to Extreme_skew generator
					threshold = -1.0187;
					m = -10;
					v = 10;
				}
				VariableArray<double> abilityTrunc = Variable.Array<double>(student).Named("abilityTrunc");
				abilityTrunc[student] = Variable.TruncatedGaussian(m, v, threshold, double.PositiveInfinity).ForEach(student);
				ability[student] = Variable.Copy(abilityTrunc[student]);
				ability.AddAttribute(new MarginalPrototype(new Gaussian()));
			} else {
				// mixture
				abilityMean2 = Variable.GaussianFromMeanAndVariance(0, 1e6).Named("abilityMean2");
				abilityPrecision2 = Variable.GammaFromShapeAndRate(1, 1).Named("abilityPrecision2");
				Variable<double> weight2 = Variable.Beta(1, 1).Named("weight2");
				isExceptional = Variable.Array<bool>(student).Named("isExceptional");
				isExceptionalInit = Variable.New<IDistribution<bool[]>>();
				isExceptional.InitialiseTo(isExceptionalInit);
				using (Variable.ForEach(student)) {
					isExceptional[student] = Variable.Bernoulli(weight2);
					using (Variable.If(isExceptional[student])) {
						ability[student] = Variable.GaussianFromMeanAndPrecision(abilityMean2, abilityPrecision2);
					}
					using (Variable.IfNot(isExceptional[student])) {
						ability[student] = Variable.GaussianFromMeanAndPrecision(abilityMean, abilityPrecision);
					}
				}
			}
			numQuestions = Variable.New<int>().Named("numQuestions");
			Range question = new Range(numQuestions);
			difficultyMean = Variable.GaussianFromMeanAndVariance(0, 1e6).Named("difficultyMean");
			difficultyPrecision = Variable.GammaFromShapeAndRate(1, 1).Named("difficultyPrecision");
			difficulty = Variable.Array<double>(question).Named("difficulty");
			difficulty[question] = Variable.GaussianFromMeanAndPrecision(difficultyMean, difficultyPrecision).ForEach(question);
			discriminationMean = Variable.GaussianFromMeanAndVariance(0, 1e6).Named("discriminationMean");
			discriminationPrecision = Variable.GammaFromShapeAndRate(1, 1).Named("discriminationPrecision");
			discrimination = Variable.Array<double>(question).Named("discrimination");
			discrimination[question] = Variable.Exp(Variable.GaussianFromMeanAndPrecision(discriminationMean, discriminationPrecision).ForEach(question));
			guessProb = Variable.Array<double>(question).Named("guessProb");
			guessProb[question] = Variable.Beta(2, 12).ForEach(question);
			response = Variable.Array<bool>(student, question).Named("response");
            if (numParams == 1)
            {
                response[student, question] = Variable.BernoulliFromLogOdds(ability[student] - difficulty[question]);
            }
            else if (numParams == 2)
            {
                response[student, question] = Variable.BernoulliFromLogOdds(((ability[student] - difficulty[question]).Named("minus") * discrimination[question]).Named("product"));
            }
            else if (numParams == 3)
            {
                using (Variable.ForEach(student))
                {
                    using (Variable.ForEach(question))
                    {
                        Variable<bool> guess = Variable.Bernoulli(guessProb[question]);
                        using (Variable.If(guess))
                        {
                            response[student, question] = Variable.Bernoulli(1 - 1e-10);
                        }
                        using (Variable.IfNot(guess))
                        {
                            Variable<double> score = (ability[student] - difficulty[question]) * discrimination[question];
                            score.Name = "score";
                            // explicit MarginalPrototype is needed when ability and difficulty are observed
                            score.AddAttribute(new MarginalPrototype(new Gaussian()));
                            response[student, question] = Variable.BernoulliFromLogOdds(score);
                        }
                    }
                }
            }
            else throw new ArgumentException($"Unsupported number of parameters: {numParams}");
			if (priorType == PriorType.Standard) {
				// standard normal prior
				abilityMean.ObservedValue = 0;
				abilityPrecision.ObservedValue = 1;
				difficultyMean.ObservedValue = 0;
				difficultyPrecision.ObservedValue = 1;
				discriminationMean.ObservedValue = 0;
				discriminationPrecision.ObservedValue = 4*4;
			} else if (priorType == PriorType.Vague) {
				// vague prior 
				abilityMean.ObservedValue = 0;
				abilityPrecision.ObservedValue = 1e-6;
				difficultyMean.ObservedValue = 0;
				difficultyPrecision.ObservedValue = 1e-6;
				discriminationMean.ObservedValue = 0;
				// must have exp(var) be finite, i.e. var <= 709, precision > 1.5e-3
				discriminationPrecision.ObservedValue = 1.5e-2;
			} else if (priorType == PriorType.StandardVague) {
				abilityMean.ObservedValue = 0;
				abilityPrecision.ObservedValue = 1;
				difficultyMean.ObservedValue = 0;
				difficultyPrecision.ObservedValue = 1e-6;
				discriminationMean.ObservedValue = 0;
				discriminationPrecision.ObservedValue = 1.5e-2;
			} else if (priorType == PriorType.VagueStandard) {
				abilityMean.ObservedValue = 0;
				abilityPrecision.ObservedValue = 1e-6;
				difficultyMean.ObservedValue = 0;
				difficultyPrecision.ObservedValue = 1;
				discriminationMean.ObservedValue = 0;
				discriminationPrecision.ObservedValue = 4*4;
			} else if (priorType == PriorType.Standard5) {
				abilityMean.ObservedValue = 0;
				abilityPrecision.ObservedValue = 1;
				difficultyMean.ObservedValue = 0;
				difficultyPrecision.ObservedValue = 1.0/25;
				discriminationMean.ObservedValue = 0;
				discriminationPrecision.ObservedValue = 4*4;
			} else if (priorType == PriorType.Hierarchical) {
				// do nothing
			} else throw new ArgumentException($"priorType {priorType} is not supported");
			engine = new InferenceEngine();
		}
		public void ObserveResponses(Matrix data)
		{
			numStudents.ObservedValue = data.Rows;
			numQuestions.ObservedValue = data.Cols;
			response.ObservedValue = ConvertToBool(data.ToArray());
			//isExceptionalInit.ObservedValue = Distribution<bool>.Array(Util.ArrayInit(numStudents.ObservedValue, i => new Bernoulli(Rand.Double())));
		}
		public void RunToConvergence(double tolerance = 1e-4)
		{
			engine.Compiler.ReturnCopies = true;
			//engine.ShowProgress = true;
			Diffable oldPost = null;
			for (int i = 1; i < 1000; i++) {
				engine.NumberOfIterations = i;
				Diffable newPost = engine.Infer<Diffable>(difficulty);
				if (oldPost != null) {
					double activity = newPost.MaxDiff(oldPost);
					//Console.WriteLine(activity);
					if (activity < tolerance) break;
				}
				oldPost = newPost;
			}
		}
		public static bool[,] ConvertToBool(double[,] array)
		{
			int rows = array.GetLength(0);
			int cols = array.GetLength(1);
			bool[,] result = new bool[rows, cols];
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					result[i, j] = (array[i, j] > 0);
				}
			}
			return result;
		}
	}
	public class LogisticIrtTestModel
	{
		public Variable<int> numStudents;
		public VariableArray<double> ability;
		public VariableArray<Gaussian> abilityPriors;
		public Variable<int> numQuestions;
		public VariableArray<double> difficulty;
		public VariableArray<Gaussian> difficultyPriors;
		public VariableArray<double> discrimination;
		public VariableArray<Gamma> discriminationPriors;
		public VariableArray<double> guessProb;
		public VariableArray<Beta> guessProbPriors;
		public VariableArray2D<double> responseProb;
		public InferenceEngine engine;

		public LogisticIrtTestModel(int numParams)
		{
			numStudents = Variable.New<int>().Named("numStudents");
			numQuestions = Variable.New<int>().Named("numQuestions");
			Range student = new Range(numStudents);
			abilityPriors = Variable.Array<Gaussian>(student).Named("abilityPriors");
			ability = Variable.Array<double>(student).Named("ability");
			ability[student] = Variable.Random<double, Gaussian>(abilityPriors[student]);
			Range question = new Range(numQuestions);
			difficultyPriors = Variable.Array<Gaussian>(question).Named("difficultyPriors");
			difficulty = Variable.Array<double>(question).Named("difficulty");
			difficulty[question] = Variable.Random<double, Gaussian>(difficultyPriors[question]);
			discriminationPriors = Variable.Array<Gamma>(question).Named("discriminationPriors");
			discrimination = Variable.Array<double>(question).Named("discrimination");
			discrimination[question] = Variable.Random<double, Gamma>(discriminationPriors[question]);
			guessProbPriors = Variable.Array<Beta>(question).Named("guessProbPriors");
			guessProb = Variable.Array<double>(question).Named("guessProb");
			guessProb[question] = Variable.Random<double, Beta>(guessProbPriors[question]);
			engine = new InferenceEngine();
			//engine.NumberOfIterations = 2;
			responseProb = Variable.Array<double>(student, question).Named("prob");
			if (numParams == 1) {
				responseProb[student, question] = Variable.Logistic(ability[student] - difficulty[question]);
			} else if (numParams >= 2) {
				responseProb[student, question] = Variable.Logistic((ability[student] - difficulty[question])*discrimination[question]);
			}
		}

		public Matrix GetResponseProbs(Gaussian[] abilityDist, Gaussian[] difficultyDist, Gamma[] discriminationDist, Beta[] guessProbDist)
		{
			numStudents.ObservedValue = abilityDist.Length;
			numQuestions.ObservedValue = difficultyDist.Length;
			abilityPriors.ObservedValue = abilityDist;
			difficultyPriors.ObservedValue = difficultyDist;
			if (discriminationDist != null) discriminationPriors.ObservedValue = discriminationDist;
			if (guessProbDist != null) guessProbPriors.ObservedValue = guessProbDist;
			Beta[,] probPost = engine.Infer<Beta[,]>(responseProb);
			if (guessProbDist == null) {
				return Program.ToMeanMatrix(probPost);
			} else {
				Matrix m = Program.ToMeanMatrix(probPost);
				for (int i = 0; i < m.Rows; i++) {
					for (int j = 0; j < m.Cols; j++) {
						double guess = guessProbDist[j].GetMean();
						m[i, j] = guess + (1-guess)*m[i, j];
					}
				}
				return m;
			}
		}
	}

}
