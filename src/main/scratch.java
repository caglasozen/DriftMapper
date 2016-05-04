package main;

import main.generator.AbruptTreeDriftGenerator;
import main.generator.CategoricalDriftGenerator;
import com.opencsv.CSVWriter;
import main.models.JointModel;
import main.models.posterior.EnsembleClassifier;
import main.models.posterior.PosteriorModel;
import main.models.posterior.SingleClassifier;
import main.models.prior.BayesianNetwork;
import main.models.prior.Eclat;
import main.models.prior.PriorModel;
import main.models.sampling.AbstractSampler;
import main.models.sampling.AllSamples;
import main.models.sampling.RandomSamples;
import moa.classifiers.*;
import moa.classifiers.meta.WEKAClassifier;
import moa.tasks.WriteStreamToARFFFile;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.io.*;
import java.util.ArrayList;

/**
 * Created by Lee on 28/01/2016.
 **/

public class scratch {
    public static void main(String args[]) {
        // Set classifiers
        WEKAClassifier lowDriftClassifier = new WEKAClassifier();
        lowDriftClassifier.baseLearnerOption.setCurrentObject(new Logistic());

        WEKAClassifier highDriftClassifier = new WEKAClassifier();
        highDriftClassifier.baseLearnerOption.setCurrentObject(new J48());

        WEKAClassifier ensembleClassifier = new WEKAClassifier();
        ensembleClassifier.baseLearnerOption.setCurrentObject(new RandomTree());

        WEKAClassifier classifierBD = new WEKAClassifier();
        classifierBD.baseLearnerOption.setCurrentObject(new RandomForest());
        classifierBD.prepareForUse();
        WEKAClassifier classifierAD = new WEKAClassifier();
        classifierBD.baseLearnerOption.setCurrentObject(new RandomForest());
        classifierBD.prepareForUse();

        com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter conv = new com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter();
        AbruptTreeDriftGenerator gen = new AbruptTreeDriftGenerator();
        gen.prepareForUse();
        Instance inst = conv.wekaInstance(gen.nextInstance().instance);
        Instances dataset = new Instances(inst.dataset());
        dataset.add(conv.wekaInstance(gen.nextInstance().instance));
        PosteriorModel basePosterior = new SingleClassifier(
                ensembleClassifier.copy(),dataset);
        PriorModel basePrior = new BayesianNetwork(dataset);
        AbstractSampler sampler = new RandomSamples(dataset, 1000, 0L);
        //AbstractSampler sampler = new AllSamples(dataset);
        JointModel baseModel = new JointModel(basePrior, basePosterior, sampler);
        try {
            if (args[0].equals("test")) {
                System.out.println("Test");
            }

            else if (args[0].equals("HalfEach")) {
                String filename = args[1] + args[2];
                Instances allInstances = loadAnyDataSet(filename);
                Experiments experiment = new Experiments(baseModel, allInstances, allInstances.size()/2, true);
                String[] dist = experiment.distanceBetweenStartEnd();
                writeToCSV(new String[][]{dist}, new String[]{"p(X)", "p(y|X)"}, "./data_out/half_out/" + args[2] + ".txt");
            }

            else if (args[0].equals("StartEnd")) {
                String filename = args[1] + args[2];
                Instances allInstances = loadAnyDataSet(filename);
                Experiments experiment = new Experiments(baseModel, allInstances, 1000, true);
                String[] dist = experiment.distanceBetweenStartEnd();
                writeToCSV(new String[][]{dist}, new String[]{"p(X)", "p(y|X)"}, "./data_out/se_out/" + args[2] + ".txt");
            }

            else if (args[0].equals("AllData")) {
                int windowSize = 1000;
                String filename = args[1] + args[2];
                Instances allInstances = loadAnyDataSet(filename);
                Experiments experiment = new Experiments(baseModel, allInstances, windowSize, false);
                String[][] dists = experiment.distanceToStartOverInstances();
                writeToCSV(dists, new String[]{"p(X)", "p(y|X)", Integer.toString(windowSize)}, "./data_out/ad_out/" + args[2] + ".txt");
            }

            else if (args[0].equals("AllPrevData")) {
                int windowSize = 5000;
                String filename = args[1] + args[2];
                Instances allInstances = loadAnyDataSet(filename);
                Experiments experiment = new Experiments(baseModel, allInstances, windowSize, false);
                String[][] dists = experiment.distanceToPrevOverInstances();
                writeToCSV(dists, new String[]{"p(X)", "p(y|X)", Integer.toString(windowSize)}, "./data_out/apd_out/" + args[2] + ".txt");
            }

            else if (args[0].equals("modelTest")){
                // Set data generator to use
                AbruptTreeDriftGenerator dataStream = new AbruptTreeDriftGenerator();
                //AbruptDriftGenerator dataStream = new AbruptDriftGenerator();
                configureDataSet(dataStream);
                dataStream.prepareForUse();
                // Get distance(s)
                String[][] results = classifierDistanceTest(baseModel, dataStream);
                writeToCSV(results, new String[]{"100", "1000", "10000", "100000", "1000000"}, "EnsembleDistance.csv");
            }

            else if (args[0].equals("mapTest")){
                // Set data generator to use
                AbruptTreeDriftGenerator dataStream = new AbruptTreeDriftGenerator();
                //AbruptDriftGenerator dataStream = new AbruptDriftGenerator();
                configureDataSet(dataStream);
                dataStream.prepareForUse();
                // Get distance(s)

                Instances allInstance = Experiments.convertStreamToInstances(dataStream);
                Experiments experiment = new Experiments(baseModel, allInstance, 5000, false);
                String[][] results = experiment.distanceToStartOverInstances();
                writeToCSV(results, new String[]{"p(X)", "p(y|X)"}, "EnsembleDistance.csv");
            }
            else if (args[0].equals("genData")) {
                String filename = args[1];
                // Set data generator to use
                AbruptTreeDriftGenerator dataStream = new AbruptTreeDriftGenerator();
                //AbruptDriftGenerator dataStream = new AbruptDriftGenerator();
                configureDataSet(dataStream);
                String folder = "./datasets/";
                String extension = ".arff";

                dataStream.prepareForUse();
                dataStream.writeDataStreamToFile(folder+filename+extension);
            }
            else if (args[0].equals("genAllData")) {
                String filename;
                // Set data generator to use
                AbruptTreeDriftGenerator dataStream = new AbruptTreeDriftGenerator();
                //AbruptDriftGenerator dataStream = new AbruptDriftGenerator();
                configureDataSet(dataStream);

                int[] burnIns = new int[]{500, 1000, 2000, 5000, 8000};
                double[] magnitudes = new double[]{0.2, 0.5, 0.8};
                String folder = "./datasets/";
                String extension = ".arff";

                for (int burnIn : burnIns) {

                    dataStream.burnInNInstances.setValue(burnIn);
                    dataStream.driftConditional.setValue(false);
                    dataStream.driftPriors.setValue(false);
                    filename = "b" + Integer.toString(burnIn) + "_none";
                    dataStream.prepareForUse();
                    dataStream.writeDataStreamToFile(folder+filename+extension);

                    for (double magnitude : magnitudes) {
                        dataStream.driftMagnitudeConditional.setValue(magnitude);
                        dataStream.driftMagnitudePrior.setValue(magnitude);

                        dataStream.driftConditional.setValue(false);
                        dataStream.driftPriors.setValue(true);
                        filename = "b" + Integer.toString(burnIn) + "_m" + Double.toString(magnitude) + "_prior";
                        dataStream.prepareForUse();
                        dataStream.writeDataStreamToFile(folder+filename+extension);

                        dataStream.driftConditional.setValue(true);
                        dataStream.driftPriors.setValue(false);
                        filename = "b" + Integer.toString(burnIn) + "_m" + Double.toString(magnitude) + "_posterior";
                        dataStream.prepareForUse();
                        dataStream.writeDataStreamToFile(folder+filename+extension);

                        dataStream.driftConditional.setValue(true);
                        dataStream.driftPriors.setValue(true);
                        filename = "b" + Integer.toString(burnIn) + "_m" + Double.toString(magnitude) + "_both";
                        dataStream.prepareForUse();
                        dataStream.writeDataStreamToFile(folder+filename+extension);
                    }
                }
            }
            else if (args[0].equals("martvard")) {
                // Set data generator to use
                AbruptTreeDriftGenerator dataStream = new AbruptTreeDriftGenerator();
                //AbruptDriftGenerator dataStream = new AbruptDriftGenerator();
                configureDataSet(dataStream);
                dataStream.prepareForUse();
                // Get distance(s)

                //Instances allInstances = Experiments.convertStreamToInstances(dataStream);
                Instances allInstances = loadDataSet("./datasets/b8000_m0.8_prior.arff");
                Instances instances1 = new Instances(allInstances, 0, allInstances.size()/2);
                Instances instances2 = new Instances(allInstances, allInstances.size()/2 - 1, allInstances.size()/2);
                MarTVarD marTVarD = new MarTVarD(instances1, instances2);
                int[][] sets = marTVarD.findOrderedNPle(2);
                for (int[] pair : sets) {
                    for (int i : pair) {
                        System.out.print(i);
                    }
                    System.out.println("");
                }
            }
        }
        catch (IOException ex) {
            ex.printStackTrace();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    private static void writeDataStream(CategoricalDriftGenerator dataStream, String filename) {
        WriteStreamToARFFFile writer = new WriteStreamToARFFFile();
        writer.streamOption.setCurrentObject(dataStream);
        writer.arffFileOption.setValue("./datasets/" + filename + ".arff");
        writer.maxInstancesOption.setValue(dataStream.burnInNInstances.getValue()*2);
        writer.prepareForUse();
        writer.doTask();
    }

    private static String[][] classifierDistanceTest(JointModel baseModel,
                                                     CategoricalDriftGenerator dataStream) {
        double[] driftMags = new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
        int[] dataPoints = new int []{100, 1000, 10000, 100000};
        int nTests = 10;

        String[][] results = new String[driftMags.length][dataPoints.length];

        for (int i = 0; i < driftMags.length; i++) {
            double dm = driftMags[i];
            dataStream.driftMagnitudePrior.setValue(dm);
            dataStream.driftMagnitudeConditional.setValue(dm);

            for (int j = 0; j < dataPoints.length; j++) {
                int nData = dataPoints[j];
                dataStream.burnInNInstances.setValue(nData);
                //System.out.println("Drift Magnitude: " + dm);
                //System.out.println("No. Instances Before/After Drift: " + nData);

                double avgPygv = 0.0f;
                for (int k = 0; k < nTests; k++) {
                    //System.out.println("Run: " + (k+1) + "\t");
                    dataStream.restart();
                    dataStream.prepareForUse();
                    Experiments experiment = new Experiments(baseModel.copy(), dataStream);
                    avgPygv += Double.parseDouble(experiment.distanceBetweenStartEnd()[0]);
                }
                avgPygv /= nTests;
                //System.out.println("p(y|X) drift = " + avgPygv);
                results[i][j] = Double.toString(avgPygv);
                System.out.println("Estimated p(X): " + avgPygv);
            }
        }

        return results;
    }

    private static void configureDataSet(CategoricalDriftGenerator dataStream) {
        dataStream.nAttributes.setValue(20);
        dataStream.nValuesPerAttribute.setValue(2);
        dataStream.precisionDriftMagnitude.setValue(0.01);
        dataStream.driftPriors.setValue(true);
        dataStream.driftConditional.setValue(true);
        dataStream.driftMagnitudeConditional.setValue(0.9);
        dataStream.driftMagnitudePrior.setValue(0.5);
        dataStream.burnInNInstances.setValue(10000);
    }

    public static void writeToCSV(String[][] data, String[] header, String filename) throws IOException{
        CSVWriter writer = new CSVWriter(new FileWriter(filename), '\t');
        // feed in your array (or convert your data to an array)
        writer.writeNext(header);
        for (String[] dataLine : data) {
            writer.writeNext(dataLine);
        }
        writer.close();
    }

    public static void writeToFile(String[] data, String filename) throws IOException {
        PrintWriter writer = new PrintWriter(filename, "UTF-8");
        for (String line : data) {
            writer.println(line);
        }
        writer.close();
    }

    public static Instances loadDataSet(String filename) throws IOException{
        // Check if any attribute is numeric
        Instances result;
        BufferedReader reader;

        reader = new BufferedReader(new FileReader(filename));
        result = new Instances(reader);
        result.setClassIndex(result.numAttributes() - 1);
        reader.close();
        return result;
    }

    public static Instances loadAnyDataSet(String filename) throws Exception{
        Instances continuousData = loadDataSet(filename);
        if (filename.equals("../datasets/gas-sensor.arff")) continuousData.setClassIndex(0);

        ArrayList<Integer> continuousIndex = new ArrayList<>();
        for (int i = 0; i < continuousData.numAttributes(); i++) {
            if (continuousData.attribute(i).isNumeric()) continuousIndex.add(i);
        }
        int[] attIndex = new int[continuousIndex.size()];
        for (int i = 0; i < continuousIndex.size(); i++) {
            attIndex[i] = continuousIndex.get(i);
        }

        Discretize filter = new Discretize();
        filter.setUseEqualFrequency(true);
        filter.setBins(5);
        filter.setAttributeIndicesArray(attIndex);
        filter.setInputFormat(continuousData);

        return Filter.useFilter(continuousData, filter);
    }

}
