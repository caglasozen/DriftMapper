package main;

import main.generator.AbruptTreeDriftGenerator;
import main.generator.CategoricalDriftGenerator;
import main.models.EnsembleClassifierModel;
import com.opencsv.CSVWriter;
import moa.classifiers.*;
import moa.classifiers.meta.WEKAClassifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
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

        WEKAClassifier classifierBD = new WEKAClassifier();
        classifierBD.baseLearnerOption.setCurrentObject(new RandomForest());
        classifierBD.prepareForUse();
        WEKAClassifier classifierAD = new WEKAClassifier();
        classifierBD.baseLearnerOption.setCurrentObject(new RandomForest());
        classifierBD.prepareForUse();

        EnsembleClassifierModel baseModel = new EnsembleClassifierModel(
                new Classifier[]{lowDriftClassifier.copy(), highDriftClassifier.copy()},
                new double[]{0.6, 1.0}, new Instances("Empty", new ArrayList<>(), 0));
        try {
            if (args[0].equals("test")) {
                System.out.println("Test");
            }

            else if (args[0].equals("HalfEach")) {
                String filename = args[1] + args[2];
                Instances allInstances = loadAnyDataSet(filename);
                Experiments experiment = new Experiments(baseModel, allInstances, allInstances.size()/2, true);
                String[] dist = experiment.distanceBetweenStartEnd();
                writeToCSV(new String[][]{dist}, new String[]{"p(X)", "p(y|X)"}, "./half_out/" + args[2] + ".txt");
            }

            else if (args[0].equals("StartEnd")) {
                String filename = args[1] + args[2];
                Instances allInstances = loadAnyDataSet(filename);
                Experiments experiment = new Experiments(baseModel, allInstances, 1000, true);
                String[] dist = experiment.distanceBetweenStartEnd();
                writeToCSV(new String[][]{dist}, new String[]{"p(X)", "p(y|X)"}, "./se_out/" + args[2] + ".txt");
            }

            else if (args[0].equals("AllData")) {
                int windowSize = 1000;
                String filename = args[1] + args[2];
                Instances allInstances = loadAnyDataSet(filename);
                Experiments experiment = new Experiments(baseModel, allInstances, windowSize, false);
                String[][] dists = experiment.distanceOverInstances();
                writeToCSV(dists, new String[]{"p(X)", "p(y|X)", Integer.toString(windowSize)}, "./ad_out/" + args[2] + ".txt");
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
        }
        catch (IOException ex) {
            System.out.println(ex);
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    private static String[][] classifierDistanceTest(EnsembleClassifierModel baseModel,
                                                     CategoricalDriftGenerator dataStream) {
        double[] driftMags = new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
        int[] dataPoints = new int []{100, 1000, 10000, 100000, 1000000};
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
                    avgPygv += Double.parseDouble(experiment.distanceBetweenStartEnd()[1]);
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
        dataStream.nAttributes.setValue(5);
        dataStream.nValuesPerAttribute.setValue(3);
        dataStream.precisionDriftMagnitude.setValue(0.01);
        dataStream.driftPriors.setValue(false);
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
